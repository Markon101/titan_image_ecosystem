//! Fixed-panel fresh guided development, withdrawal and paired recovery.
//! Raw frames and every report are owned by the enclosing EvaluationArtifacts.
use super::*;
use crate::experiment::PanelConfig;
use serde_json::{json, Value};
use std::collections::VecDeque;

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    config: &RunConfig,
    panel: &PanelConfig,
    artifacts: &mut EvaluationArtifacts,
    corpus: &mut ImageCorpus,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    parent: &WorldState,
    device: &Device,
) -> Result<Value> {
    let started = std::time::Instant::now();
    let plan = RenderPlan::new(config, config.train_resolution, device)?;
    let mut points = Vec::new();
    let mut mature_images = Vec::new();
    for &target in &panel.targets {
        let sample = corpus.sample_index(target, device)?;
        for &seed in &panel.seeds {
            let mut world = WorldState::fresh(config, seed, device)?;
            world.target_index = target;
            world.morph_active_depth = parent.morph_active_depth;
            world.morph_generation = parent.morph_generation;
            world.morph_birth_generations = parent.morph_birth_generations.clone();
            let mut diagnostics = Vec::new();
            for age in 0..=panel.burn_in {
                if panel.diagnostic_ages.contains(&age) {
                    for fidelity in [0.0, config.reference_fidelity_max] {
                        diagnostics.push(dynamics.inspect_interface(
                            &world,
                            &sample.genome_tensor,
                            &sample.reference_micro,
                            &sample.reference_macro,
                            fidelity,
                        )?);
                    }
                }
                if age < panel.burn_in {
                    world =
                        advance(dynamics, &world, &sample, config.reference_fidelity_max, 0)?.world;
                }
            }
            let initial = render(config, renderer, &world, &sample, &plan)?;
            let initial_metrics = reconstruction_reference_metrics(&initial, &sample.image)?;
            mature_images.push((target, seed, initial.clone()));
            let stem = format!("panel_t{target}_s{seed}");
            artifacts.png(&initial, Path::new(&format!("{stem}_guided.png")))?;
            let recovery = recovery(
                config, panel, dynamics, renderer, &world, &sample, &plan, device,
            )?;
            let mut history = VecDeque::new();
            history.push_back(Frame::new(0, &world, &initial)?);
            let mut rollout = Vec::new();
            let horizon = *panel.horizons.iter().max().unwrap();
            for offset in 1..=horizon {
                let stepped = advance(dynamics, &world, &sample, 0.0, 0)?;
                anyhow::ensure!(
                    stepped.micro_reference_drive_rms == 0.0
                        && stepped.macro_reference_drive_rms == 0.0,
                    "autonomous panel reference drive was nonzero"
                );
                world = stepped.world;
                if offset % panel.stride != 0 && !panel.horizons.contains(&offset) {
                    continue;
                }
                let image = render(config, renderer, &world, &sample, &plan)?;
                let frame = Frame::new(offset, &world, &image)?;
                let lagged: Vec<_> = history.iter().map(|old| json!({
                    "lag_steps":offset-old.offset,"full_state_rms":vector_rms_distance(&frame.full, &old.full),
                    "spatial_latent_correlation":correlation(&frame.spatial, &old.spatial),
                    "image_correlation":correlation(&frame.image, &old.image),
                    "low_frequency_correlation":correlation(&frame.low, &old.low),
                    "edge_correlation":correlation(&frame.edge, &old.edge),
                    "brightness_normalized_rms":normalized_rms(&frame.image, &old.image),
                    "phase_motion":shift_correlation(&frame.image, &old.image, 32)
                })).collect();
                let raw = artifacts.png(
                    &image,
                    Path::new(&format!("{stem}_autonomous_{offset:05}.png")),
                )?;
                rollout.push(json!({"offset":offset,"age":world.age,"world_step":world.step,
                    "reference_fidelity":0.0,"runtime_references_present":false,
                    "micro_reference_drive_rms":stepped.micro_reference_drive_rms,
                    "macro_reference_drive_rms":stepped.macro_reference_drive_rms,
                    "micro":state_metrics(&world.micro, config.state_limit)?,
                    "macro":state_metrics(&world.macro_field, config.state_limit)?,
                    "image":image_metrics(&image)?, "target":reconstruction_reference_metrics(&image, &sample.image)?,
                    "micro_movement":stepped.micro_movement,"macro_movement":stepped.macro_movement,
                    "image_l1_from_withdrawal":mean_abs(&image.sub(&initial)?)?,
                    "lagged":lagged,"raw_frame":raw}));
                history.push_back(frame);
                if history.len() > panel.history_capacity {
                    history.pop_front();
                }
            }
            points.push(json!({"target":target,"source_fingerprint":format!("{:016x}",sample.fingerprint),
                "seed":seed,"burn_in":panel.burn_in,"burn_in_reference_fidelity":config.reference_fidelity_max,
                "initial_reconstruction":initial_metrics,"initial_image":image_metrics(&initial)?,
                "interface":diagnostics,"recovery":recovery,"autonomous":rollout}));
        }
    }
    let mut separability = Vec::new();
    for (i, (target, seed, image)) in mature_images.iter().enumerate() {
        for (other, other_seed, other_image) in mature_images.iter().skip(i + 1) {
            if seed == other_seed && target != other {
                separability.push(json!({"seed":seed,"target_a":target,"target_b":other,
                    "metrics":reconstruction_reference_metrics(image, other_image)?}));
            }
        }
    }
    let report = json!({"version":1,"initialization":"fresh deterministic world, actual guided burn-in; checkpoint anatomy retained",
        "reference_policy":"target retained for metrics/loss; both dynamics reference paths absent in autonomous phases",
        "recurrence_policy":"lagged full-state RMS and spatial metrics; moving recurrence is not filtered by tiny motion; no dynamical-class claim",
        "correlation_policy":"Pearson correlation; null for constant inputs; 32px image, 8px low frequency, 8px latent grids; phase search +/-2 pixels periodic",
        "recovery_policy":"same global steps and same clock sequence except explicitly named clock robustness cases",
        "panel":panel,"points":points,"target_separability":separability,
        "elapsed_seconds":started.elapsed().as_secs_f64()});
    artifacts.json(Path::new("experimental_panel.json"), &report)?;
    Ok(report)
}

pub(super) fn advance(
    dynamics: &DynamicsSystem,
    world: &WorldState,
    sample: &TargetSample,
    fidelity: f32,
    clock_seed_xor: u64,
) -> Result<crate::dynamics::StepOutput> {
    let micro = (fidelity > 0.0).then_some(&sample.reference_micro);
    let macro_field = (fidelity > 0.0).then_some(&sample.reference_macro);
    dynamics.step_ablated(
        world,
        &sample.genome_tensor,
        micro,
        macro_field,
        micro,
        macro_field,
        fidelity,
        false,
        &DynamicsAblation {
            clock_seed_xor,
            ..Default::default()
        },
    )
}
fn render(
    config: &RunConfig,
    renderer: &ImplicitRenderer,
    world: &WorldState,
    sample: &TargetSample,
    plan: &RenderPlan,
) -> Result<Tensor> {
    let (_, emergence) = config.developmental_schedule(
        (world.age as f32 / config.developmental_horizon() as f32).min(1.0),
    );
    Ok(renderer
        .render_with_emergence(
            &world.micro,
            &world.macro_field,
            &sample.genome_tensor,
            plan,
            emergence,
            false,
        )?
        .image)
}
struct Frame {
    offset: usize,
    full: Vec<f32>,
    spatial: Vec<f32>,
    image: Vec<f32>,
    low: Vec<f32>,
    edge: Vec<f32>,
}
impl Frame {
    fn new(offset: usize, world: &WorldState, image: &Tensor) -> Result<Self> {
        let mut full = world.micro.flatten_all()?.to_vec1::<f32>()?;
        full.extend(world.macro_field.flatten_all()?.to_vec1::<f32>()?);
        full.extend(world.memory.flatten_all()?.to_vec1::<f32>()?);
        let mut spatial = downsample(&world.micro, 8)?;
        spatial.extend(downsample(&world.macro_field, 8)?);
        let low = downsample(image, 8)?;
        let image = downsample(image, 32)?;
        let edge = edges(&image, 32);
        Ok(Self {
            offset,
            full,
            spatial,
            image,
            low,
            edge,
        })
    }
}
fn downsample(t: &Tensor, edge: usize) -> Result<Vec<f32>> {
    // Area pooling where possible; bilinear only for nondivisible shapes.
    let (_, _, h, w) = t.dims4()?;
    let t = if h >= edge && w >= edge && h % edge == 0 && w % edge == 0 {
        t.avg_pool2d((h / edge, w / edge))?
    } else {
        t.upsample_bilinear2d(edge, edge, false)?
    };
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}
fn edges(v: &[f32], n: usize) -> Vec<f32> {
    v.chunks(n * n)
        .flat_map(|c| {
            (0..n * n).map(move |i| {
                let x = i % n;
                let y = i / n;
                ((c[y * n + (x + 1).min(n - 1)] - c[i]).powi(2)
                    + (c[(y + 1).min(n - 1) * n + x] - c[i]).powi(2))
                .sqrt()
            })
        })
        .collect()
}
fn correlation(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let ma = a.iter().map(|v| f64::from(*v)).sum::<f64>() / a.len() as f64;
    let mb = b.iter().map(|v| f64::from(*v)).sum::<f64>() / b.len() as f64;
    let mut dot = 0.0;
    let mut aa = 0.0;
    let mut bb = 0.0;
    for (a, b) in a.iter().zip(b) {
        let a = f64::from(*a) - ma;
        let b = f64::from(*b) - mb;
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }
    (aa > 1e-20 && bb > 1e-20).then(|| (dot / (aa * bb).sqrt()).clamp(-1.0, 1.0))
}
fn normalized_rms(a: &[f32], b: &[f32]) -> Option<f64> {
    correlation(a, b).map(|c| (2.0 - 2.0 * c).max(0.0).sqrt())
}
fn shift_correlation(a: &[f32], b: &[f32], n: usize) -> Value {
    let mut best = None::<(f64, i32, i32)>;
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            let shifted: Vec<f32> = b
                .chunks(n * n)
                .flat_map(|c| {
                    (0..n * n).map(move |i| {
                        c[((i / n) as i32 + dy).rem_euclid(n as i32) as usize * n
                            + ((i % n) as i32 + dx).rem_euclid(n as i32) as usize]
                    })
                })
                .collect();
            if let Some(c) = correlation(a, &shifted) {
                if best.is_none_or(|old| c > old.0) {
                    best = Some((c, dx, dy));
                }
            }
        }
    }
    best.map_or(
        Value::Null,
        |(c, dx, dy)| json!({"correlation":c,"dx":dx,"dy":dy}),
    )
}

#[allow(clippy::too_many_arguments)]
fn recovery(
    config: &RunConfig,
    panel: &PanelConfig,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    initial: &WorldState,
    sample: &TargetSample,
    plan: &RenderPlan,
    device: &Device,
) -> Result<Vec<Value>> {
    if panel.recovery_horizon == 0 {
        return Ok(Vec::new());
    }
    let mut reports = Vec::new();
    for (mode, fidelity) in [
        ("guided", config.reference_fidelity_max),
        ("autonomous", 0.0),
    ] {
        let mut cases: Vec<_> = panel.recovery_cases.iter().map(|c| c.as_str()).collect();
        if panel.clock_robustness {
            cases.push("clock_sequence_only");
        }
        for case in cases {
            let clock_xor = if case == "clock_sequence_only" {
                0xa17e_51d5
            } else {
                0
            };
            let mut control = initial.detached();
            let mut damaged = perturb_field(
                initial,
                case == "micro_noise" || case == "combined",
                case == "macro_noise" || case == "combined",
                case == "memory_noise" || case == "combined",
                case == "micro_patch",
                case == "macro_patch",
                config,
                device,
            )?;
            let initial_distance = world_distance(&damaged, &control)?;
            let control_image = render(config, renderer, &control, sample, plan)?;
            let damaged_image = render(config, renderer, &damaged, sample, plan)?;
            let initial_render_error = mean_abs(&damaged_image.sub(&control_image)?)?;
            let mut half = None;
            let mut rendered_half = None;
            let mut trajectory = Vec::new();
            let mut macro_updates = 0usize;
            for offset in 1..=panel.recovery_horizon {
                anyhow::ensure!(
                    control.step == damaged.step && control.age == damaged.age,
                    "paired recovery clock/time drift"
                );
                let control_step = advance(dynamics, &control, sample, fidelity, 0)?;
                let damaged_step = advance(dynamics, &damaged, sample, fidelity, clock_xor)?;
                anyhow::ensure!(
                    control_step.macro_updated == damaged_step.macro_updated,
                    "paired macro cadence drift"
                );
                macro_updates += usize::from(control_step.macro_updated);
                control = control_step.world;
                damaged = damaged_step.world;
                let distance = world_distance(&damaged, &control)?;
                if initial_distance > 1e-12 && half.is_none() && distance <= 0.5 * initial_distance
                {
                    half = Some(offset);
                }
                if offset % panel.stride == 0 || offset == panel.recovery_horizon {
                    let a = render(config, renderer, &control, sample, plan)?;
                    let b = render(config, renderer, &damaged, sample, plan)?;
                    let l1 = mean_abs(&a.sub(&b)?)?;
                    if initial_render_error > 1e-12
                        && rendered_half.is_none()
                        && l1 <= 0.5 * initial_render_error
                    {
                        rendered_half = Some(offset);
                    }
                    trajectory.push(
                        json!({"offset":offset,"state_distance":distance,"rendered_l1":l1,"macro_updates":macro_updates,
                        "structural_recovery":reconstruction_reference_metrics(&b,&a)?,
                        "control_target":reconstruction_reference_metrics(&a,&sample.image)?,
                        "damaged_target":reconstruction_reference_metrics(&b,&sample.image)?}),
                    );
                }
            }
            reports.push(json!({"mode":mode,"case":case,"reference_fidelity":fidelity,
                "development_steps":panel.recovery_horizon,"macro_updates":macro_updates,
                "same_clock_sequence":clock_xor==0,"clock_seed_xor":clock_xor,
                "perturbation":"uniform noise +/-0.03 with legacy smooth bound, central third patch erase; combined=micro+macro+memory noise",
                "initial_state_distance":initial_distance,"initial_rendered_l1":initial_render_error,
                "state_time_to_half":half,"rendered_time_to_half_sampled":rendered_half,
                "failed_to_halve_state_within_horizon":initial_distance>1e-12 && half.is_none(),
                "trajectory":trajectory}));
        }
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn correlations_reject_constants_and_detect_brightness_normalized_structure() {
        assert_eq!(correlation(&[1.0; 4], &[1.0; 4]), None);
        assert!((correlation(&[0., 1., 3.], &[2., 4., 8.]).unwrap() - 1.).abs() < 1e-12);
        assert!(normalized_rms(&[0., 1., 3.], &[2., 4., 8.]).unwrap() < 1e-6);
        let a: Vec<f32> = (0..64).map(|i| ((i * 7) % 19) as f32).collect();
        let b: Vec<f32> = a
            .chunks(8)
            .flat_map(|r| (0..8).map(move |x| r[(x + 1) % 8]))
            .collect();
        assert!(
            shift_correlation(&a, &b, 8)["correlation"]
                .as_f64()
                .unwrap()
                > 0.999
        );
    }
}
