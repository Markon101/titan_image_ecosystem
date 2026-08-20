use crate::config::{RunConfig, SCHEMA_VERSION};
use crate::corpus::{ImageCorpus, TargetSample};
use crate::dynamics::DynamicsSystem;
use crate::metrics::{image_metrics, tensor_rms, MetricRecord};
use crate::objectives::{visual_loss, LossOutput};
use crate::optimizer::{OptimizerStats, PersistentAdamW};
use crate::persistence::{load_checkpoint, save_checkpoint, write_json_atomic, ArtifactPaths};
use crate::render::{
    save_contact_sheet, save_mastered_png, save_png, save_state_atlas, ImplicitRenderer, RenderPlan,
};
use crate::state::WorldState;
use crate::tensor_ops::splitmix64;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Default, Clone, Serialize)]
struct PhaseSeconds {
    corpus_startup: f64,
    tracked_dynamics: f64,
    detached_dynamics: f64,
    render_and_loss: f64,
    backward: f64,
    optimizer: f64,
    metrics_and_logging: f64,
    output_rendering: f64,
    checkpointing: f64,
    total: f64,
}

#[derive(Serialize)]
struct BuildProvenance {
    commit: &'static str,
    dirty: bool,
    release: bool,
    target_arch: &'static str,
}

#[derive(Serialize)]
struct RunMetadata<'a> {
    schema_version: u32,
    package_version: &'static str,
    run_id: String,
    build: BuildProvenance,
    invocation: Vec<String>,
    config: &'a RunConfig,
    effective_threads: usize,
    source_images: usize,
    cached_source_images: usize,
    corpus_fingerprint: String,
    config_signature: String,
    parameter_count: usize,
    resumed: bool,
    render_only: bool,
    metrics_continued: bool,
    loaded_optimizer_tensors: usize,
    start_world_step: u64,
    completed_world_step: u64,
    optimizer_updates_start: u64,
    optimizer_updates_end: u64,
    optimizer_windows: usize,
    full_core_windows: usize,
    decoder_only_windows: usize,
    final_target: &'a str,
    peak_rss_kib: Option<u64>,
    phase_seconds: PhaseSeconds,
    average_ms_per_development_step: f64,
    outputs: Vec<String>,
    claims: [&'static str; 5],
}

pub fn run(config: RunConfig) -> Result<()> {
    config.validate()?;
    let available_threads = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let effective_threads = config.threads.min(available_threads).max(1);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(effective_threads)
        .build_global();

    let started = Instant::now();
    let device = Device::Cpu;
    std::fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("cannot create {}", config.output_dir.display()))?;
    let paths = ArtifactPaths::new(&config);

    let corpus_started = Instant::now();
    let mut corpus = ImageCorpus::new(&config, &device)?;
    let mut phase = PhaseSeconds {
        corpus_startup: seconds(corpus_started.elapsed()),
        ..PhaseSeconds::default()
    };

    let mut vars = VarMap::new();
    let builder = VarBuilder::from_varmap(&vars, DType::F32, &device);
    let dynamics = DynamicsSystem::new(&config, builder.pp("dynamics"), &device)?;
    let renderer = ImplicitRenderer::new(&config, builder.pp("renderer"))?;
    deterministic_initialize(&vars, config.seed)?;
    let mut optimizer = PersistentAdamW::new(&vars, &config)?;
    let parameter_count = vars
        .data()
        .lock()
        .expect("VarMap mutex poisoned")
        .values()
        .map(|variable| variable.elem_count())
        .sum();

    if config.render_only && !paths.checkpoint_complete() {
        bail!(
            "--render-only requires a complete v5 checkpoint for this output directory and run tag"
        );
    }

    let checkpoint_available = paths.checkpoint_complete();
    if !config.fresh && paths.checkpoint_exists() && !checkpoint_available {
        bail!(
            "partial v5 checkpoint set in {}; use --fresh or restore all checkpoint files",
            config.output_dir.display()
        );
    }
    let (loaded_world, resumed, loaded_optimizer_tensors) = if !config.fresh && checkpoint_available
    {
        let (world, moments) = load_checkpoint(
            &paths,
            &mut vars,
            &mut optimizer,
            &device,
            &config,
            corpus.fingerprint(),
        )?;
        (Some(world), true, moments)
    } else {
        (None, false, 0)
    };

    let initial_episode = loaded_world.as_ref().map_or(0, |world| world.episode);
    let mut sample = corpus.sample(initial_episode, &device)?;
    let mut world = match loaded_world {
        Some(world) => {
            if world.target_index != sample.index {
                bail!(
                    "checkpoint target index {} disagrees with deterministic corpus schedule {}",
                    world.target_index,
                    sample.index
                );
            }
            world
        }
        None => {
            let mut fresh = WorldState::fresh(&config, config.seed ^ sample.fingerprint, &device)?;
            fresh.target_index = sample.index;
            fresh
        }
    };
    let start_world_step = world.step;
    let optimizer_updates_start = optimizer.updates();

    let metrics_continued = resumed && paths.metrics.exists();
    let mut metrics_writer = if config.render_only {
        None
    } else {
        let metrics_file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(metrics_continued)
            .truncate(!metrics_continued)
            .open(&paths.metrics)
            .with_context(|| format!("cannot open {}", paths.metrics.display()))?;
        let mut writer = BufWriter::new(metrics_file);
        if !metrics_continued {
            MetricRecord::write_header(&mut writer)?;
        }
        Some(writer)
    };

    println!(
        "TITAN Image v5: {} source image(s), {} cached, {} parameters, {} at world step {}",
        corpus.len(),
        corpus.cached_images(),
        parameter_count,
        if resumed {
            "resuming"
        } else {
            "starting fresh"
        },
        world.step
    );
    println!(
        "Profile {:?} / {:?}: {} threads, {}ch {}x{} + {}x{}, train {}px, endpoint loss every {} steps",
        config.profile,
        config.style,
        effective_threads,
        config.channels,
        config.micro_size,
        config.micro_size,
        config.macro_size,
        config.macro_size,
        config.train_resolution,
        config.bptt,
    );

    let train_plan = if config.render_only {
        None
    } else {
        Some(RenderPlan::new(&config, config.train_resolution, &device)?)
    };
    let mut output_plan: Option<RenderPlan> = None;
    let optimizer_windows = if config.render_only {
        0
    } else {
        config.steps / config.bptt
    };
    let mut full_core_windows = 0usize;
    let mut decoder_only_windows = 0usize;

    for local_window in 0..optimizer_windows {
        let window_started = Instant::now();
        select_episode_if_needed(
            &config,
            &mut corpus,
            &device,
            &mut world,
            &mut sample,
            metrics_writer.as_mut(),
        )?;
        let window_index = world.step / config.bptt as u64;
        let train_core = window_index.is_multiple_of(config.core_update_every as u64);
        if train_core {
            full_core_windows += 1;
        } else {
            decoder_only_windows += 1;
        }

        let mut movement_sum = 0.0f32;
        let mut movement_max = 0.0f32;
        let mut macro_updates = 0usize;
        for _ in 0..config.bptt {
            let tick = Instant::now();
            let stepped = dynamics.step(&world, &sample.genome_tensor, train_core)?;
            if train_core {
                phase.tracked_dynamics += seconds(tick.elapsed());
            } else {
                phase.detached_dynamics += seconds(tick.elapsed());
            }
            movement_sum += stepped.movement;
            movement_max = movement_max.max(stepped.movement);
            macro_updates += usize::from(stepped.macro_updated);
            world = stepped.world;
        }

        let tick = Instant::now();
        let rendered = renderer.render(
            &world.micro,
            &world.macro_field,
            &sample.genome_tensor,
            train_plan.as_ref().expect("training has a render plan"),
            true,
        )?;
        let losses = visual_loss(&rendered, &sample.image, &config)?;
        let loss_values = loss_scalars(&losses)?;
        phase.render_and_loss += seconds(tick.elapsed());

        let optimizer_stats = optimizer.backward_step(&losses.total)?;
        phase.backward += optimizer_stats.backward_seconds;
        phase.optimizer += optimizer_stats.step_seconds;
        world = world.detached();

        let tick = Instant::now();
        let diagnostics = image_metrics(&rendered.image.detach())?;
        let state_rms = tensor_rms(&world.micro)?;
        let record = metric_record(
            &world,
            &sample,
            optimizer.updates(),
            train_core,
            macro_updates,
            movement_sum / config.bptt as f32,
            movement_max,
            state_rms,
            &diagnostics,
            loss_values,
            &optimizer_stats,
            seconds(window_started.elapsed()),
        );
        record.write_csv(
            metrics_writer
                .as_mut()
                .expect("training has a metrics writer"),
        )?;
        if (local_window + 1).is_multiple_of(config.log_every)
            || local_window + 1 == optimizer_windows
        {
            println!(
                "step {:>7} | {} | loss {:.5} structure {:.5} | move {:.5} state {:.3} | grad {:.3} clip {:.3} | {:.1} ms/window",
                world.step,
                if train_core { "core" } else { "decode" },
                record.loss_total,
                record.loss_structure,
                record.movement_mean,
                record.state_rms,
                record.gradient_norm,
                record.gradient_clip_scale,
                1000.0 * record.window_seconds,
            );
        }
        phase.metrics_and_logging += seconds(tick.elapsed());

        if config.snapshot_every > 0 && world.step % config.snapshot_every as u64 == 0 {
            let tick = Instant::now();
            ensure_output_plan(&config, &device, &mut output_plan)?;
            let snapshot = renderer.render(
                &world.micro,
                &world.macro_field,
                &sample.genome_tensor,
                output_plan.as_ref().expect("output plan initialized"),
                false,
            )?;
            save_png(&snapshot.image, &snapshot_path(&config, world.step))?;
            phase.output_rendering += seconds(tick.elapsed());
        }
        if config.checkpoint_every > 0 && world.step % config.checkpoint_every as u64 == 0 {
            let tick = Instant::now();
            metrics_writer
                .as_mut()
                .expect("training has a metrics writer")
                .flush()?;
            save_checkpoint(
                &paths,
                &vars,
                &optimizer,
                &world,
                &config,
                corpus.fingerprint(),
            )?;
            phase.checkpointing += seconds(tick.elapsed());
        }
    }

    if let Some(writer) = metrics_writer.as_mut() {
        writer.flush()?;
        let tick = Instant::now();
        save_checkpoint(
            &paths,
            &vars,
            &optimizer,
            &world,
            &config,
            corpus.fingerprint(),
        )?;
        phase.checkpointing += seconds(tick.elapsed());
    }

    let output_tick = Instant::now();
    ensure_output_plan(&config, &device, &mut output_plan)?;
    let final_render = renderer.render(
        &world.micro.detach(),
        &world.macro_field.detach(),
        &sample.genome_tensor,
        output_plan.as_ref().expect("output plan initialized"),
        false,
    )?;
    save_png(&final_render.image, &paths.raw)?;
    save_mastered_png(
        &final_render.image,
        &paths.mastered,
        config.mastering_strength,
    )?;
    let mut outputs = vec![
        paths.raw.display().to_string(),
        paths.mastered.display().to_string(),
    ];
    if config.save_state_atlas {
        save_state_atlas(&world.micro, &paths.micro_state)?;
        save_state_atlas(&world.macro_field, &paths.macro_state)?;
        outputs.push(paths.micro_state.display().to_string());
        outputs.push(paths.macro_state.display().to_string());
    }
    let gallery_outputs = render_gallery(
        &config,
        &corpus,
        &dynamics,
        &renderer,
        output_plan.as_ref().expect("output plan initialized"),
        &device,
    )?;
    if !gallery_outputs.is_empty() {
        let mastered: Vec<PathBuf> = gallery_outputs
            .iter()
            .filter(|path| path.to_string_lossy().contains("_mastered"))
            .cloned()
            .collect();
        save_contact_sheet(&mastered, &paths.gallery)?;
        outputs.extend(
            gallery_outputs
                .iter()
                .map(|path| path.display().to_string()),
        );
        outputs.push(paths.gallery.display().to_string());
    }
    phase.output_rendering += seconds(output_tick.elapsed());
    phase.total = seconds(started.elapsed());

    let completed_steps = world.step.saturating_sub(start_world_step);
    let metadata = RunMetadata {
        schema_version: SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION"),
        run_id: run_id(),
        build: BuildProvenance {
            commit: env!("TITAN_BUILD_COMMIT"),
            dirty: env!("TITAN_BUILD_DIRTY") == "true",
            release: !cfg!(debug_assertions),
            target_arch: std::env::consts::ARCH,
        },
        invocation: std::env::args().collect(),
        config: &config,
        effective_threads,
        source_images: corpus.len(),
        cached_source_images: corpus.cached_images(),
        corpus_fingerprint: format!("{:016x}", corpus.fingerprint()),
        config_signature: format!("{:016x}", config.checkpoint_signature()),
        parameter_count,
        resumed,
        render_only: config.render_only,
        metrics_continued,
        loaded_optimizer_tensors,
        start_world_step,
        completed_world_step: world.step,
        optimizer_updates_start,
        optimizer_updates_end: optimizer.updates(),
        optimizer_windows,
        full_core_windows,
        decoder_only_windows,
        final_target: &sample.name,
        peak_rss_kib: peak_rss_kib(),
        phase_seconds: phase.clone(),
        average_ms_per_development_step: if completed_steps > 0 {
            1000.0 * phase.total / completed_steps as f64
        } else {
            0.0
        },
        outputs,
        claims: [
            "autonomous morphogenic image generator",
            "not an action-conditioned world model",
            "IFS and quasiperiodic fields are bounded target attractions",
            "finite raster outputs do not prove an exact fractal dimension",
            "profile timings are measurements of this invocation, not universal device constants",
        ],
    };
    let metadata_path = if config.render_only {
        &paths.render_metadata
    } else {
        &paths.metadata
    };
    write_json_atomic(metadata_path, &metadata)?;
    println!(
        "Finished at world step {} in {:.2}s. Raw: {}  Mastered: {}",
        world.step,
        phase.total,
        paths.raw.display(),
        paths.mastered.display()
    );
    if config.gallery > 0 {
        println!("Gallery: {}", paths.gallery.display());
    }
    Ok(())
}

fn select_episode_if_needed(
    config: &RunConfig,
    corpus: &mut ImageCorpus,
    device: &Device,
    world: &mut WorldState,
    sample: &mut TargetSample,
    metrics_writer: Option<&mut BufWriter<std::fs::File>>,
) -> Result<()> {
    let episode = world.step / config.episode_steps as u64;
    if episode == world.episode {
        return Ok(());
    }
    if let Some(writer) = metrics_writer {
        writer.flush()?;
    }
    let next = corpus.sample(episode, device)?;
    *world =
        world.reseed_for_episode(config, config.seed ^ next.fingerprint, episode, next.index)?;
    *sample = next;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn metric_record(
    world: &WorldState,
    sample: &TargetSample,
    optimizer_update: u64,
    core_trained: bool,
    macro_updates: usize,
    movement_mean: f32,
    movement_max: f32,
    state_rms: f32,
    image: &crate::metrics::ImageDiagnostics,
    loss: [f32; 6],
    optimizer: &OptimizerStats,
    window_seconds: f64,
) -> MetricRecord {
    MetricRecord {
        step: world.step,
        age: world.age,
        episode: world.episode,
        target_index: sample.index,
        optimizer_update,
        core_trained,
        macro_updates,
        movement_mean,
        movement_max,
        state_rms,
        image_variance: image.variance,
        seam_energy: image.seam,
        edge_energy: image.edge,
        gamut_excess: loss[5],
        red_mean: image.means[0],
        green_mean: image.means[1],
        blue_mean: image.means[2],
        red_variance: image.variances[0],
        green_variance: image.variances[1],
        blue_variance: image.variances[2],
        rg_correlation: image.correlations[0],
        rb_correlation: image.correlations[1],
        gb_correlation: image.correlations[2],
        loss_total: loss[0],
        loss_content: loss[1],
        loss_palette: loss[2],
        loss_structure: loss[3],
        loss_seam: loss[4],
        loss_gamut: loss[5],
        gradient_norm: optimizer.gradient_norm,
        gradient_clip_scale: optimizer.clip_scale,
        effective_learning_rate: optimizer.effective_learning_rate,
        window_seconds,
    }
}

fn loss_scalars(loss: &LossOutput) -> Result<[f32; 6]> {
    let values = Tensor::stack(
        &[
            &loss.total,
            &loss.content,
            &loss.palette,
            &loss.structure,
            &loss.seam,
            &loss.gamut,
        ],
        0,
    )?
    .to_vec1::<f32>()?;
    Ok(values.try_into().expect("six loss tensors were stacked"))
}

fn ensure_output_plan(
    config: &RunConfig,
    device: &Device,
    plan: &mut Option<RenderPlan>,
) -> Result<()> {
    if plan.is_none() {
        *plan = Some(RenderPlan::new(config, config.output_resolution, device)?);
    }
    Ok(())
}

fn render_gallery(
    config: &RunConfig,
    corpus: &ImageCorpus,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    plan: &RenderPlan,
    device: &Device,
) -> Result<Vec<PathBuf>> {
    let mut outputs = Vec::with_capacity(config.gallery * 2);
    for variant in 0..config.gallery {
        let genome = corpus.gallery_genome(variant, config.gallery_seed);
        let genome_tensor = Tensor::from_vec(genome, config.genome_dim, device)?;
        let variant_seed = splitmix64(config.seed ^ config.gallery_seed ^ variant as u64);
        let mut world = WorldState::fresh(config, variant_seed, device)?;
        let development_steps = config.gallery_steps + variant * config.gallery_stride;
        for _ in 0..development_steps {
            world = dynamics.step(&world, &genome_tensor, false)?.world;
        }
        let rendered = renderer.render(
            &world.micro,
            &world.macro_field,
            &genome_tensor,
            plan,
            false,
        )?;
        let raw = gallery_path(config, variant, false);
        let mastered = gallery_path(config, variant, true);
        save_png(&rendered.image, &raw)?;
        save_mastered_png(&rendered.image, &mastered, config.mastering_strength)?;
        outputs.push(raw);
        outputs.push(mastered);
    }
    Ok(outputs)
}

fn deterministic_initialize(varmap: &VarMap, seed: u64) -> Result<()> {
    let data = varmap.data().lock().expect("VarMap mutex poisoned");
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for name in names {
        let variable = &data[name];
        let dims = variable.dims();
        let count = variable.elem_count();
        let zero_nca_output = name.contains("micro_ca.output") || name.contains("macro_ca.output");
        let neutral_color_bias = name.contains("renderer.oklab.bias");
        let small_color_head = name.contains("renderer.oklab.weight");
        let mut rng = ChaCha8Rng::seed_from_u64(parameter_seed(seed, name));
        let values = if zero_nca_output || neutral_color_bias {
            vec![0.0f32; count]
        } else if name.ends_with(".weight") {
            let fan_in = *dims.get(1).unwrap_or(&1);
            let stdev = if small_color_head {
                0.04
            } else {
                (2.0f32 / fan_in as f32).sqrt()
            };
            normal_values(&mut rng, count, stdev)
        } else {
            let weight_name = format!("{}.weight", name.trim_end_matches(".bias"));
            let fan_in = data
                .get(&weight_name)
                .and_then(|weight| weight.dims().get(1).copied())
                .unwrap_or(1);
            let bound = 1.0f32 / (fan_in as f32).sqrt();
            (0..count).map(|_| rng.gen_range(-bound..bound)).collect()
        };
        let initialized = Tensor::from_vec(values, variable.shape().clone(), variable.device())?;
        variable.set(&initialized)?;
    }
    Ok(())
}

fn parameter_seed(seed: u64, name: &str) -> u64 {
    name.bytes()
        .fold(seed ^ 0xcbf2_9ce4_8422_2325, |hash, byte| {
            (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3)
        })
}

fn normal_values(rng: &mut ChaCha8Rng, count: usize, stdev: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let u1 = rng.gen_range(f32::EPSILON..1.0);
        let u2 = rng.gen_range(0.0..1.0);
        let radius = (-2.0 * u1.ln()).sqrt() * stdev;
        let angle = std::f32::consts::TAU * u2;
        values.push(radius * angle.cos());
        if values.len() < count {
            values.push(radius * angle.sin());
        }
    }
    values
}

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
}

fn run_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    format!(
        "{:016x}",
        splitmix64(nanos as u64 ^ std::process::id() as u64)
    )
}

fn peak_rss_kib() -> Option<u64> {
    let file = std::fs::File::open("/proc/self/status").ok()?;
    for line in BufReader::new(file).lines().map_while(|line| line.ok()) {
        if let Some(value) = line.strip_prefix("VmHWM:") {
            return value.split_whitespace().next()?.parse().ok();
        }
    }
    None
}

fn snapshot_path(config: &RunConfig, step: u64) -> PathBuf {
    config.output_dir.join(format!(
        "titan_image_snapshot_v5{}_{step:09}.png",
        config.suffix()
    ))
}

fn gallery_path(config: &RunConfig, variant: usize, mastered: bool) -> PathBuf {
    config.output_dir.join(format!(
        "titan_image_variant_v5{}_{:03}_{}.png",
        config.suffix(),
        variant + 1,
        if mastered { "mastered" } else { "raw" }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TrainingMode;

    fn tiny_config(corpus: PathBuf, output: PathBuf) -> RunConfig {
        RunConfig {
            corpus_dir: corpus,
            output_dir: output,
            mode: TrainingMode::Single,
            steps: 2,
            threads: 1,
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            genome_dim: 4,
            ca_hidden: 32,
            render_hidden: 32,
            render_blocks: 1,
            coord_bands: 2,
            train_resolution: 24,
            output_resolution: 24,
            episode_steps: 4,
            bptt: 1,
            snapshot_every: 0,
            checkpoint_every: 0,
            log_every: 1,
            gallery: 0,
            gallery_steps: 0,
            fresh: true,
            ..RunConfig::default()
        }
    }

    #[test]
    fn fresh_and_resumed_training_write_complete_artifacts() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "titan-image-v5-test-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        let corpus = root.join("corpus");
        let output = root.join("output");
        std::fs::create_dir_all(&corpus)?;
        let source = image::RgbImage::from_fn(24, 24, |x, y| {
            image::Rgb([(x * 10) as u8, (y * 10) as u8, ((x + y) * 5) as u8])
        });
        source.save(corpus.join("source.png"))?;
        let config = tiny_config(corpus, output.clone());
        run(config.clone())?;
        let paths = ArtifactPaths::new(&config);
        assert!(paths.checkpoint_complete());
        assert!(paths.raw.exists());
        assert!(paths.mastered.exists());
        assert!(paths.metrics.exists());
        assert!(paths.metadata.exists());
        let continued = RunConfig {
            fresh: false,
            ..config.clone()
        };
        run(continued)?;
        let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(&paths.metadata)?)?;
        assert_eq!(metadata["resumed"], true);
        assert_eq!(metadata["completed_world_step"], 4);
        assert_eq!(metadata["optimizer_updates_end"], 4);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn parameter_initialization_is_seed_deterministic() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig {
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            genome_dim: 4,
            ca_hidden: 32,
            render_hidden: 32,
            render_blocks: 1,
            coord_bands: 2,
            train_resolution: 24,
            output_resolution: 24,
            ..RunConfig::default()
        };
        let make = || -> Result<VarMap> {
            let vars = VarMap::new();
            let builder = VarBuilder::from_varmap(&vars, DType::F32, &device);
            let _dynamics = DynamicsSystem::new(&config, builder.pp("dynamics"), &device)?;
            let _renderer = ImplicitRenderer::new(&config, builder.pp("renderer"))?;
            deterministic_initialize(&vars, config.seed)?;
            Ok(vars)
        };
        let first = make()?;
        let second = make()?;
        let first_data = first.data().lock().expect("VarMap mutex poisoned");
        let second_data = second.data().lock().expect("VarMap mutex poisoned");
        for (name, variable) in first_data.iter() {
            let other = &second_data[name];
            assert_eq!(
                variable.flatten_all()?.to_vec1::<f32>()?,
                other.flatten_all()?.to_vec1::<f32>()?,
                "parameter {name} differed"
            );
        }
        Ok(())
    }
}
