//! Independent, frozen developmental analysis. No training or checkpoint writes.
mod metrics;
mod spectral;
use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use metrics::{difference, distance, full, norm, values};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use serde_json::{json, Value};
use spectral::{Band, Spectrum};
use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use titan_image::{
    corpus::ImageCorpus,
    dynamics::DynamicsSystem,
    flow::RectifiedFlowRenderer,
    optimizer::PersistentAdamW,
    persistence::{load_checkpoint_read_only, ArtifactPaths},
    render::ImplicitRenderer,
    state::WorldState,
    training_fork::sha256,
    ComputeBackend, RunConfig,
};

#[derive(Debug, Serialize)]
struct Options {
    config: PathBuf,
    output: PathBuf,
    steps: usize,
    seed: u64,
    low_cutoff: f64,
    mid_cutoff: f64,
    recurrence_stride: usize,
    recurrence_max: usize,
    perturb_epsilon: Option<f64>,
    perturb_band: Band,
    response_ages: Vec<usize>,
    response_epsilons: Vec<f64>,
    response_bands: Vec<Band>,
    fixed_rms: f64,
    fixed_window: usize,
}
const HELP: &str = "titan_develop --config CONFIG_OR_METADATA.json --output NEW_DIRECTORY [options]
Frozen CPU-only v9 import, fixed saved genome, NO target references, NO training writes.
  --steps N                    Developmental steps after saved state (default 16, max 100000)
  --seed N                     Perturbation seed (default 42; checkpoint seed retained separately)
  --low-cutoff F                Radial low band upper edge, cycles/pixel (default .125)
  --mid-cutoff F                Mid band upper edge (default .25; high is remainder)
  --recurrence-stride N         Save full states every N steps (default 1; includes endpoints)
  --recurrence-max N            Maximum saved states (default 256, max 512; budget 512 MiB)
  --perturb-epsilon E           Enable paired finite-time growth, initial full-state L2 E
  --perturb-band low|mid|high   Initial perturbation on micro field (default high)
  --response-ages 0,8,16        Enable +/- response at these OFFSETS from saved age
  --response-epsilons .01,.02   L2 perturbation amplitudes (default .01,.02)
  --response-bands low,mid,high Bands to probe (default low,mid,high)
  --fixed-rms E                 State stagnation threshold (default 1e-6)
  --fixed-window N              Consecutive small updates (default 8)
Outputs: manifest.json, trajectory.jsonl, response.jsonl, recurrence.f64le, recurrence.json,
         temporal.json, summary.json. Existing output directories are refused.
";
fn band(s: &str) -> Result<Band> {
    Ok(match s {
        "low" => Band::Low,
        "mid" => Band::Mid,
        "high" => Band::High,
        _ => bail!("invalid band {s}"),
    })
}
fn parse(args: &[String]) -> Result<Options> {
    let mut o = Options {
        config: PathBuf::new(),
        output: PathBuf::new(),
        steps: 16,
        seed: 42,
        low_cutoff: 0.125,
        mid_cutoff: 0.25,
        recurrence_stride: 1,
        recurrence_max: 256,
        perturb_epsilon: None,
        perturb_band: Band::High,
        response_ages: vec![],
        response_epsilons: vec![0.01, 0.02],
        response_bands: vec![Band::Low, Band::Mid, Band::High],
        fixed_rms: 1e-6,
        fixed_window: 8,
    };
    ensure!(
        args.len().is_multiple_of(2),
        "each option requires a value; see --help"
    );
    for pair in args.chunks(2) {
        let v = &pair[1];
        match pair[0].as_str() {
            "--config" => o.config = v.into(),
            "--output" => o.output = v.into(),
            "--steps" => o.steps = v.parse()?,
            "--seed" => o.seed = v.parse()?,
            "--low-cutoff" => o.low_cutoff = v.parse()?,
            "--mid-cutoff" => o.mid_cutoff = v.parse()?,
            "--recurrence-stride" => o.recurrence_stride = v.parse()?,
            "--recurrence-max" => o.recurrence_max = v.parse()?,
            "--perturb-epsilon" => o.perturb_epsilon = Some(v.parse()?),
            "--perturb-band" => o.perturb_band = band(v)?,
            "--response-ages" => {
                o.response_ages = v
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()?
            }
            "--response-epsilons" => {
                o.response_epsilons = v
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()?
            }
            "--response-bands" => {
                o.response_bands = v.split(',').map(band).collect::<Result<_>>()?
            }
            "--fixed-rms" => o.fixed_rms = v.parse()?,
            "--fixed-window" => o.fixed_window = v.parse()?,
            _ => bail!("unknown option {}; see --help", pair[0]),
        }
    }
    ensure!(
        !o.config.as_os_str().is_empty() && !o.output.as_os_str().is_empty(),
        "--config and --output required"
    );
    ensure!(
        (1..=100000).contains(&o.steps)
            && o.recurrence_stride > 0
            && (2..=512).contains(&o.recurrence_max),
        "invalid step/recurrence limits"
    );
    ensure!(
        o.steps.div_ceil(o.recurrence_stride) < o.recurrence_max,
        "increase recurrence stride or capacity"
    );
    ensure!(
        o.low_cutoff.is_finite()
            && o.mid_cutoff.is_finite()
            && o.low_cutoff > 0.
            && o.low_cutoff < o.mid_cutoff
            && o.mid_cutoff < 0.5,
        "require 0 < low < mid < .5"
    );
    ensure!(
        o.perturb_epsilon
            .iter()
            .chain(&o.response_epsilons)
            .all(|e| e.is_finite() && *e > 0.),
        "epsilons must be finite and positive"
    );
    ensure!(
        o.response_ages.iter().all(|a| *a <= o.steps)
            && o.response_ages.len() <= 64
            && o.response_epsilons.len() <= 8
            && o.response_bands.len() <= 3,
        "response panel too large or age beyond horizon"
    );
    ensure!(
        o.fixed_rms.is_finite() && o.fixed_rms >= 0. && o.fixed_window > 0,
        "invalid fixed-point thresholds"
    );
    Ok(o)
}
fn create(path: &Path) -> Result<BufWriter<File>> {
    Ok(BufWriter::new(
        OpenOptions::new().write(true).create_new(true).open(path)?,
    ))
}
fn save(path: &Path, v: &impl Serialize) -> Result<()> {
    let mut f = create(path)?;
    serde_json::to_writer_pretty(&mut f, v)?;
    f.write_all(b"\n")?;
    f.flush()?;
    Ok(())
}
fn line(f: &mut BufWriter<File>, v: &Value) -> Result<()> {
    serde_json::to_writer(&mut *f, v)?;
    f.write_all(b"\n")?;
    Ok(())
}
fn identities(paths: &[PathBuf]) -> Result<BTreeMap<String, String>> {
    paths
        .iter()
        .filter(|p| p.is_file())
        .map(|p| Ok((p.display().to_string(), sha256(p)?)))
        .collect()
}
fn advance(dynamics: &DynamicsSystem, x: &WorldState, genome: &Tensor) -> Result<WorldState> {
    let out = dynamics.step(x, genome, None, None, 0., false)?;
    ensure!(
        out.micro_reference_drive_rms == 0. && out.macro_reference_drive_rms == 0.,
        "unexpected reference drive"
    );
    Ok(out.world.detached())
}
fn perturbation(x: &WorldState, o: &Options, b: Band) -> Result<Vec<f64>> {
    let (_, _, h, w) = x.micro.dims4()?;
    let mut rng = ChaCha8Rng::seed_from_u64(o.seed);
    let noise: Vec<f64> = (0..x.micro.elem_count())
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let mut p = Spectrum::new(&noise, h, w)?.project(b, o.low_cutoff, o.mid_cutoff, true);
    let n = norm(&p);
    ensure!(n > 1e-12, "empty perturbation band on this grid");
    for v in &mut p {
        *v /= n;
    }
    Ok(p)
}
fn displaced(x: &WorldState, p: &[f64], epsilon: f64) -> Result<WorldState> {
    let v: Vec<_> = values(&x.micro)?
        .iter()
        .zip(p)
        .map(|(x, p)| (x + epsilon * p) as f32)
        .collect();
    let mut out = x.detached();
    out.micro = Tensor::from_vec(v, x.micro.shape(), x.micro.device())?;
    Ok(out)
}
fn low_response(
    plus: &Tensor,
    minus: &Tensor,
    base: &Tensor,
    epsilon: f64,
    o: &Options,
) -> Result<f64> {
    let (_, _, h, w) = base.dims4()?;
    let p = values(plus)?;
    let m = values(minus)?;
    let b = values(base)?;
    let q: Vec<_> = (0..b.len())
        .map(|i| (p[i] + m[i] - 2. * b[i]) / (2. * epsilon * epsilon))
        .collect();
    Ok(norm(&Spectrum::new(&q, h, w)?.project(
        Band::Low,
        o.low_cutoff,
        o.mid_cutoff,
        false,
    )))
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help") {
        print!("{HELP}");
        return Ok(());
    }
    run(parse(&args)?)
}
fn run(o: Options) -> Result<()> {
    let started = Instant::now();
    let input: Value = serde_json::from_slice(&std::fs::read(&o.config)?)?;
    let original: RunConfig =
        serde_json::from_value(input.get("config").unwrap_or(&input).clone())?;
    original.validate()?;
    let paths = ArtifactPaths::new(&original);
    ensure!(
        paths.checkpoint_complete(),
        "complete current checkpoint required; recovery is disabled"
    );
    ensure!(
        !original.output_dir.join(".fork-incomplete").exists(),
        "incomplete fork"
    );
    let protected = vec![
        paths.model.clone(),
        paths.optimizer.clone(),
        paths.world.clone(),
        paths.checkpoint_manifest.clone(),
        paths.metadata.clone(),
        paths.metrics.clone(),
        paths.events.clone(),
        o.config.clone(),
    ];
    let before = identities(&protected)?;
    std::fs::create_dir(&o.output)
        .context("output must be a new directory with an existing parent")?;
    let mut config = original.clone();
    config.compute_backend = ComputeBackend::Cpu;
    config.detail.cache_dir = Some(o.output.join("corpus_cache"));
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads.max(1))
        .build_global();
    let device = Device::Cpu;
    let mut corpus = ImageCorpus::new(&config, &device)?;
    let mut vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
    let dynamics = DynamicsSystem::new(&config, vb.pp("dynamics"), &device)?;
    let _renderer = ImplicitRenderer::new(&config, vb.pp("renderer"))?;
    let _flow = RectifiedFlowRenderer::new(&config, vb.pp("flow"))?;
    let mut optimizer = PersistentAdamW::new(&vars, &config)?;
    let (mut world, load) = load_checkpoint_read_only(
        &paths,
        &mut vars,
        &mut optimizer,
        &device,
        &config,
        corpus.fingerprint(),
    )?;
    ensure!(
        !load.model.grafted && load.optimizer_moments_new == 0,
        "analysis requires exact anatomy, no in-memory migration"
    );
    let sample = corpus.sample_index(world.target_index, &device)?;
    ensure!(
        identities(&protected)? == before,
        "checkpoint changed during import"
    );
    let initial = full(&world)?;
    let n = initial.len();
    let capacity = o.steps.div_ceil(o.recurrence_stride) + 1;
    ensure!(
        n.checked_mul(capacity)
            .and_then(|v| v.checked_mul(8))
            .is_some_and(|v| v <= 512 * 1024 * 1024),
        "recurrence state storage exceeds 512 MiB; increase stride"
    );
    let initial_step = world.step;
    let initial_age = world.age;
    let binary = std::env::current_exe()?;
    save(
        &o.output.join("manifest.json"),
        &json!({"diagnostics_schema":"titan.development.v1","complete":false,
        "build_commit":env!("TITAN_BUILD_COMMIT"),"build_dirty":env!("TITAN_BUILD_DIRTY"),"build_rustflags":env!("TITAN_BUILD_RUSTFLAGS"),
        "binary_sha256":sha256(&binary)?,"source_config":original,"effective_config":config,"options":o,
        "checkpoint_hashes_sha256":before,"load":load,"training_step":initial_step,"saved_developmental_age":initial_age,
        "conditioning":{"genome_source_index":sample.index,"source_fingerprint":sample.fingerprint,"references_present":false,"reference_fidelity":0.0},
        "enabled_operators":["legacy_G"],"state_order":["micro_NCHW","macro_NCHW","memory"],
        "shapes":[world.micro.dims(),world.macro_field.dims(),world.memory.dims()],
        "bands":{"domain":"periodic original latent grids; no windowing; cycles per cell; DC included in low energy",
            "radius":"sqrt((min(kx,W-kx)/W)^2+(min(ky,H-ky)/H)^2)","low":"0 <= f <= low_cutoff","mid":"low_cutoff < f <= mid_cutoff","high":"f > mid_cutoff",
            "DFT":"unnormalized forward f64; inverse /HW; Parseval energy sum(abs(F)^2)/HW"},
        "gradient_policy":"spatial gradient reported; parameter gradient null because no loss or backward pass",
        "clock_policy":"saved counters advance normally; same clocks in all pairs; distances exclude counters; conditional finite-time growth, no Jacobian or asymptotic claim",
        "perturbation_policy":"ChaCha8 seed; uniform[-1,1), exact band projection, per-channel DC removed, global micro L2 normalized to 1; cast displaced states to f32",
        "recurrence_retained_state_bytes":n*capacity*8}),
    )?;
    let mut trajectory = create(&o.output.join("trajectory.jsonl"))?;
    let mut responses = create(&o.output.join("response.jsonl"))?;
    let mut history = Vec::with_capacity(capacity);
    let mut ages = Vec::with_capacity(capacity);
    let mut energies = Vec::with_capacity(o.steps + 1);
    let mut growth = None;
    if let Some(e) = o.perturb_epsilon {
        let p = perturbation(&world, &o, o.perturb_band)?;
        let shifted = displaced(&world, &p, e)?;
        let delta = distance(&initial, &full(&shifted)?);
        ensure!(delta > 0., "perturbation vanished in f32");
        growth = Some((shifted, delta));
    }
    let mut previous: Option<Vec<f64>> = None;
    let mut previous_update = None;
    let (mut dynamics_seconds, mut diagnostics_seconds, mut probe_seconds) = (0., 0., 0.);
    let mut quiet = 0usize;
    for offset in 0..=o.steps {
        let tick = Instant::now();
        let x = full(&world)?;
        let update = previous.as_ref().map(|p| distance(&x, p));
        let rms = update.map(|u| u / (n as f64).sqrt());
        if rms.is_some_and(|r| r <= o.fixed_rms) {
            quiet += 1;
        } else {
            quiet = 0;
        }
        let e = norm(&x).powi(2) / (2. * n as f64);
        energies.push(e);
        let growth_row = if let Some((shifted, d0)) = &growth {
            let d = distance(&x, &full(shifted)?);
            json!({"initial_actual_l2":d0,"current_l2":d,"finite_time_rate":if offset>0 && d>0. {Some((d/d0).ln()/offset as f64)} else {None},"exact_coalescence":d==0.})
        } else {
            Value::Null
        };
        line(
            &mut trajectory,
            &json!({"offset":offset,"developmental_age":world.age,"clock_step":world.step,"training_step":initial_step,
            "full_state_energy_per_scalar":e,"full_state_l2":norm(&x),"update_l2":update,"update_energy":update.map(|u|u*u/2.),"update_rms":rms,
            "micro":metrics::field(&world.micro,o.low_cutoff,o.mid_cutoff)?,"macro":metrics::field(&world.macro_field,o.low_cutoff,o.mid_cutoff)?,
            "hidden_memory_l2":norm(&values(&world.memory)?),"parameter_gradient_l2":null,"perturbation":growth_row,
            "update_components":previous_update,"state_stagnation_candidate":quiet>=o.fixed_window,"consecutive_small_updates":quiet}),
        )?;
        if offset % o.recurrence_stride == 0 || offset == o.steps {
            history.push(x.clone());
            ages.push(world.age);
        }
        diagnostics_seconds += tick.elapsed().as_secs_f64();
        let need_next = offset < o.steps || o.response_ages.contains(&offset);
        let next = if need_next {
            let tick = Instant::now();
            let y = advance(&dynamics, &world, &sample.genome_tensor)?;
            dynamics_seconds += tick.elapsed().as_secs_f64();
            Some(y)
        } else {
            None
        };
        if o.response_ages.contains(&offset) {
            let tick = Instant::now();
            let base = next.as_ref().unwrap();
            let base_delta = distance(&full(base)?, &x);
            for &b in &o.response_bands {
                let p = perturbation(&world, &o, b)?;
                for &epsilon in &o.response_epsilons {
                    let xp = displaced(&world, &p, epsilon)?;
                    let xm = displaced(&world, &p, -epsilon)?;
                    let plus_delta = difference(&full(&xp)?, &x);
                    let minus_delta = difference(&full(&xm)?, &x);
                    ensure!(
                        norm(&plus_delta) > 0. && norm(&minus_delta) > 0.,
                        "response perturbation vanished in f32"
                    );
                    let gp = advance(&dynamics, &xp, &sample.genome_tensor)?;
                    let gm = advance(&dynamics, &xm, &sample.genome_tensor)?;
                    let qm = low_response(&gp.micro, &gm.micro, &base.micro, epsilon, &o)?;
                    let qa = low_response(
                        &gp.macro_field,
                        &gm.macro_field,
                        &base.macro_field,
                        epsilon,
                        &o,
                    )?;
                    let q = qm.hypot(qa);
                    line(
                        &mut responses,
                        &json!({"offset":offset,"developmental_age":world.age,"training_step":initial_step,"band":b,"epsilon":epsilon,
                        "actual_plus_l2":norm(&plus_delta),"actual_minus_l2":norm(&minus_delta),
                        "input_symmetry_residual_l2":norm(&plus_delta.iter().zip(&minus_delta).map(|(a,b)|a+b).collect::<Vec<_>>()),
                        "q_micro_l2":qm,"q_macro_l2":qa,"q_spatial_l2":q,"baseline_full_update_l2":base_delta,
                        "q_over_baseline_update":if base_delta>0. {Some(q/base_delta)} else {None}}),
                    )?;
                }
            }
            probe_seconds += tick.elapsed().as_secs_f64();
        }
        if offset < o.steps {
            if let Some((shifted, _)) = &mut growth {
                let tick = Instant::now();
                *shifted = advance(&dynamics, shifted, &sample.genome_tensor)?;
                probe_seconds += tick.elapsed().as_secs_f64();
            }
            let next = next.unwrap();
            let r = difference(&full(&next)?, &x);
            let zero = vec![0.; n];
            previous_update = Some(metrics::cancellation(&r, &zero, &zero, &zero));
            previous = Some(x);
            world = next;
        }
        if offset > 0 && offset % 64 == 0 {
            eprintln!(
                "development {offset}/{}; elapsed {:.1}s",
                o.steps,
                started.elapsed().as_secs_f64()
            );
        }
    }
    trajectory.flush()?;
    responses.flush()?;
    let tick = Instant::now();
    let mut recurrence = create(&o.output.join("recurrence.f64le"))?;
    for a in &history {
        for b in &history {
            recurrence.write_all(&distance(a, b).to_le_bytes())?;
        }
    }
    recurrence.flush()?;
    save(
        &o.output.join("recurrence.json"),
        &json!({"dtype":"little-endian f64","shape":[history.len(),history.len()],"layout":"row-major","metric":"unweighted full micro+macro+memory Euclidean distance","developmental_ages":ages}),
    )?;
    save(
        &o.output.join("temporal.json"),
        &metrics::temporal(&energies),
    )?;
    diagnostics_seconds += tick.elapsed().as_secs_f64();
    let after = identities(&protected)?;
    ensure!(
        before == after,
        "protected training files changed during analysis; results incomplete"
    );
    let artifacts = identities(
        &[
            "manifest.json",
            "trajectory.jsonl",
            "response.jsonl",
            "recurrence.f64le",
            "recurrence.json",
            "temporal.json",
        ]
        .map(|p| o.output.join(p)),
    )?;
    save(
        &o.output.join("summary.json"),
        &json!({"diagnostics_schema":"titan.development.v1","complete":true,
        "training_files_unchanged":true,"checkpoint_hashes_after":after,"artifacts_sha256":artifacts,
        "final_developmental_age":world.age,"final_state_energy_per_scalar":energies.last(),"retained_state_bytes":history.len()*n*8,
        "base_dynamics_seconds":dynamics_seconds,"diagnostics_seconds":diagnostics_seconds,"additional_probe_seconds":probe_seconds,
        "elapsed_seconds":started.elapsed().as_secs_f64(),"state_stagnation_candidate":quiet>=o.fixed_window}),
    )?;
    println!(
        "Frozen development complete: {} ({} steps, {:.2}s)",
        o.output.display(),
        o.steps,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_cli_is_rejected() {
        let args = |tail: &str| {
            format!("--config c --output o {tail}")
                .split_whitespace()
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        for s in [
            "--steps 0",
            "--recurrence-stride 0",
            "--perturb-epsilon NaN",
            "--steps 99999",
            "--response-ages 17",
            "--low-cutoff .3",
        ] {
            assert!(parse(&args(s)).is_err(), "{s}");
        }
    }
    #[test]
    fn symmetric_quadratic_response_and_linear_zero() -> Result<()> {
        let args = "--config c --output o"
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let o = parse(&args)?;
        let tensor = |x: Vec<f32>| Tensor::from_vec(x, (1, 1, 4, 4), &Device::Cpu).unwrap();
        let x = vec![2f32; 16];
        let p: Vec<_> = (0..16)
            .map(|i| if i % 2 == 0 { 0.25f32 } else { -0.25 })
            .collect();
        let (mut plus, mut minus) = (x.clone(), x.clone());
        for i in 0..16 {
            plus[i] += 0.5 * p[i];
            minus[i] -= 0.5 * p[i];
        }
        assert!(
            low_response(
                &tensor(plus.clone()),
                &tensor(minus.clone()),
                &tensor(x.clone()),
                0.5,
                &o
            )? < 1e-12
        );
        let square = |x: Vec<f32>| tensor(x.iter().map(|v| v * v).collect());
        assert!(
            (low_response(&square(plus), &square(minus), &square(x), 0.5, &o)? - 0.25).abs()
                < 1e-10
        );
        Ok(())
    }
}
