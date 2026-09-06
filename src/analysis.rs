use crate::benchmark::{
    benchmark_reference_report, benchmark_suite, pairwise_separability,
    reconstruction_reference_metrics, BenchmarkTarget, PairwiseSeparability,
};
use crate::config::{RunConfig, TrainingMode};
use crate::corpus::{ImageCorpus, TargetSample};
use crate::dynamics::{DynamicsAblation, DynamicsSystem};
use crate::flow::{flow_oklab_to_rgb, RectifiedFlowRenderer};
use crate::metrics::{image_metrics, state_metrics, tensor_rms};
use crate::objectives::{cross_resolution_consistency, visual_loss};
use crate::persistence::{write_json_atomic, ArtifactPaths};
use crate::probe::{run_natural_image_probes, NaturalImageProbeReport};
use crate::render::{
    save_contact_sheet_resized, save_png, ImplicitRenderer, RenderPlan, SpatialView,
};
use crate::state::WorldState;
use crate::telemetry::tensor_fingerprint;
use crate::tensor_ops::{mean_abs, smooth_limit, splitmix64};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::path::{Path, PathBuf};

mod provenance;
pub use provenance::AnalysisProvenance;

#[derive(Clone, Debug, Serialize)]
pub struct FrontierPoint {
    pub emergence_strength: f32,
    pub content_loss: f32,
    pub grounding_loss: f32,
    pub ground_coarse: f32,
    pub ground_mid: f32,
    pub ground_fine: f32,
    pub structure_loss: f32,
    pub emergent_residual_rms: f32,
    pub emergent_low_energy: f32,
    pub emergent_tv: f32,
    pub head_redundancy: f32,
    pub seam: f32,
    pub gamut: f32,
    pub image_variance: f32,
    pub edge_energy: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ResolutionConsistencyPoint {
    pub kind: &'static str,
    pub high_resolution: usize,
    pub low_resolution: usize,
    pub l1: f32,
    pub low_frequency_l1: f32,
    pub edge_l1: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct AttractorRecord {
    pub analysis_version: u32,
    pub reference_fidelity: f32,
    pub micro_reference_drive_rms: f32,
    pub macro_reference_drive_rms: f32,
    pub offset: usize,
    pub micro_movement: f32,
    pub macro_movement: f32,
    pub micro_rms: f32,
    pub macro_rms: f32,
    pub micro_near_bound_fraction: f32,
    pub macro_near_bound_fraction: f32,
    pub memory_rms: f32,
    pub image_delta_valid: bool,
    pub image_delta: f32,
    pub recurrence_distance_valid: bool,
    pub recurrence_distance: f32,
    pub output_fingerprint: String,
    pub approximate_cycle_candidate: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct PerturbationRecord {
    pub name: String,
    pub noise_distribution: &'static str,
    pub reference_fidelity: f32,
    pub initial_state_distance: f32,
    pub final_state_distance: f32,
    pub final_output_l1: f32,
    pub recovery_ratio: f32,
    pub time_to_half_recovery: Option<usize>,
    pub perturbation_recovery_observed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct DynamicsAblationRecord {
    pub name: String,
    pub horizon: usize,
    pub output_l1_from_full: f32,
    pub micro_l1_from_full: f32,
    pub macro_l1_from_full: f32,
    pub memory_l1_from_full: f32,
    pub output_fingerprint: String,
}
#[derive(Clone, Debug, Serialize)]
pub struct SeparabilityPair {
    pub first_target: usize,
    pub second_target: usize,
    pub output_l1: f32,
    pub low_frequency_l1: f32,
    pub edge_l1: f32,
    pub micro_l1: f32,
    pub macro_l1: f32,
    pub memory_l1: f32,
    pub emergent_l1: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SeparabilitySummary {
    pub targets: Vec<usize>,
    pub fixed_age: usize,
    pub pair_count: usize,
    pub mean_output_l1: f32,
    pub mean_low_frequency_l1: f32,
    pub mean_edge_l1: f32,
    pub minimum_output_l1: f32,
    pub nearest_pair: Option<(usize, usize)>,
    pub pairs: Vec<SeparabilityPair>,
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkAgePoint {
    pub age: usize,
    pub raw_l1: f32,
    pub coarse_spatial_l1: f32,
    pub edge_l1: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkTargetResult {
    pub id: String,
    pub convergence: Vec<BenchmarkAgePoint>,
    pub final_raw_l1: f32,
    pub final_coarse_spatial_l1: f32,
    pub final_edge_l1: f32,
    pub final_emergent_rms: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct BenchmarkRunReport {
    pub reference: crate::benchmark::BenchmarkReferenceReport,
    pub fixed_seed: u64,
    pub results: Vec<BenchmarkTargetResult>,
    pub candidate_separability: PairwiseSeparability,
}
#[derive(Clone, Debug, Serialize)]
pub struct AnalysisSummary {
    pub schema_version: u32,
    pub provenance: AnalysisProvenance,
    pub world_step: u64,
    pub interpretation_rule: &'static str,
    pub decomposition_labels: Vec<String>,
    pub decomposition_montage: Option<String>,
    pub frontier: Vec<FrontierPoint>,
    pub resolution_consistency: Vec<ResolutionConsistencyPoint>,
    pub target_separability: Option<SeparabilitySummary>,
    pub autonomous_rollout: Vec<AttractorRecord>,
    pub perturbations: Vec<PerturbationRecord>,
    pub flow_sample: Option<String>,
    pub benchmark: Option<BenchmarkRunReport>,
    pub natural_image_probe: Option<NaturalImageProbeReport>,
    pub dynamics_ablation: Vec<DynamicsAblationRecord>,
}

#[allow(clippy::too_many_arguments)]
pub fn run_checkpoint_analysis(
    config: &RunConfig,
    paths: &ArtifactPaths,
    corpus: &mut ImageCorpus,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    flow_renderer: &RectifiedFlowRenderer,
    world: &WorldState,
    sample: &TargetSample,
    device: &Device,
) -> Result<AnalysisSummary> {
    let mut summary = AnalysisSummary {
        schema_version: crate::config::SCHEMA_VERSION,
        provenance: AnalysisProvenance::new(config, paths, corpus, world, sample)?,
        world_step: world.step,
        interpretation_rule:
            "Operational diagnostics only; no automatic claim of strong emergence, homeostasis, strange attractors, or dynamical causation.",
        decomposition_labels: Vec::new(),
        decomposition_montage: None,
        frontier: Vec::new(),
        resolution_consistency: Vec::new(),
        target_separability: None,
        autonomous_rollout: Vec::new(),
        perturbations: Vec::new(),
        flow_sample: None,
        benchmark: None,
        natural_image_probe: None,
        dynamics_ablation: Vec::new(),
    };

    if config.analysis.render_attribution || config.analysis.emergence_gallery {
        let (labels, montage, frontier) =
            render_decomposition_frontier(config, paths, renderer, world, sample, device)?;
        summary.decomposition_labels = labels;
        summary.decomposition_montage = Some(montage.display().to_string());
        summary.frontier = frontier;
    }

    if config.analysis.only {
        summary.resolution_consistency =
            resolution_ladder(config, paths, renderer, world, sample, device)?;
        if config.mode == TrainingMode::Family {
            summary.target_separability = Some(target_separability(
                config, paths, corpus, dynamics, renderer, device,
            )?);
        }
    }
    if config.analysis.autonomous_horizon > 0 {
        summary.autonomous_rollout =
            autonomous_rollout(config, paths, dynamics, renderer, world, sample, device)?;
    }
    if config.analysis.perturbation_horizon > 0 {
        summary.perturbations =
            perturbation_recovery(config, dynamics, renderer, world, sample, device)?;
        write_json_atomic(&paths.perturbation_analysis, &summary.perturbations)?;
    }
    if config.analysis.dynamics_horizon > 0 {
        summary.dynamics_ablation =
            dynamics_ablation(config, dynamics, renderer, world, sample, device)?;
    }
    if config.analysis.only && config.objective.uses_flow() {
        let plan = RenderPlan::new(config, config.flow.resolution, device)?;
        let flow_output = flow_renderer.sample_midpoint_with_trajectory(
            &world.micro,
            &world.macro_field,
            &world.memory,
            &plan,
            splitmix64(config.seed ^ world.step ^ 0xf10a_0de5),
            config.flow.sample_steps,
            1.0,
            config.reference_fidelity_max,
            config.reconstruction.emergence_strength,
        )?;
        let image = flow_oklab_to_rgb(&flow_output.state)?;
        save_png(&image, &paths.flow_sample)?;
        write_json_atomic(&paths.flow_trajectory, &flow_output.trajectory)?;
        summary.flow_sample = Some(paths.flow_sample.display().to_string());
    }
    if config.analysis.benchmark {
        summary.benchmark = Some(run_reconstruction_benchmark(
            config, paths, dynamics, renderer, device,
        )?);
    }
    if config.analysis.probe_dir.is_some() {
        println!("PROBE: frozen held-out natural-image age/fidelity sweep");
        summary.natural_image_probe = Some(run_natural_image_probes(
            config, corpus, dynamics, renderer, world, device,
        )?);
    }
    summary.provenance.completed = provenance::completion_status(&summary);
    summary.provenance.artifacts = provenance::completed_artifacts(paths, &summary)?;
    let archive = PathBuf::from(&summary.provenance.archive);
    std::fs::create_dir_all(archive.parent().expect("analysis archive directory"))?;
    write_json_atomic(&archive, &summary)?;
    write_json_atomic(&paths.analysis, &summary)?;
    Ok(summary)
}

fn run_reconstruction_benchmark(
    config: &RunConfig,
    paths: &ArtifactPaths,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    device: &Device,
) -> Result<BenchmarkRunReport> {
    let resolution = 64usize;
    let suite = benchmark_suite(resolution)?;
    let reference = benchmark_reference_report(resolution)?;
    let plan = RenderPlan::new(config, resolution, device)?;
    let fixed_seed = config.seed ^ 0xb3ec_9001;
    let fixed_age = config.developmental_horizon().min(32);
    let checkpoints = [1usize, 8, 16, fixed_age];
    let mut results = Vec::new();
    let mut candidates = Vec::new();
    for target in suite {
        let reference_micro =
            target
                .image
                .upsample_bilinear2d(config.micro_size, config.micro_size, false)?;
        let reference_macro =
            target
                .image
                .upsample_bilinear2d(config.macro_size, config.macro_size, false)?;
        let genome = Tensor::zeros(config.genome_dim, DType::F32, device)?;
        let mut world = WorldState::fresh(config, fixed_seed, device)?;
        let mut convergence = Vec::new();
        let mut final_render = None;
        for age in 1..=fixed_age {
            world = dynamics
                .step(
                    &world,
                    &genome,
                    Some(&reference_micro),
                    Some(&reference_macro),
                    1.0,
                    false,
                )?
                .world;
            if checkpoints.contains(&age) {
                let rendered = renderer.render_with_emergence(
                    &world.micro,
                    &world.macro_field,
                    &genome,
                    &plan,
                    config.reconstruction.emergence_strength,
                    false,
                )?;
                let metrics = reconstruction_reference_metrics(&rendered.image, &target.image)?;
                convergence.push(BenchmarkAgePoint {
                    age,
                    raw_l1: metrics.raw_l1,
                    coarse_spatial_l1: metrics.coarse_spatial_l1,
                    edge_l1: metrics.edge_l1,
                });
                final_render = Some(rendered);
            }
        }
        let rendered = final_render.expect("benchmark includes final age checkpoint");
        let final_metrics = reconstruction_reference_metrics(&rendered.image, &target.image)?;
        results.push(BenchmarkTargetResult {
            id: target.metadata.id.to_owned(),
            convergence,
            final_raw_l1: final_metrics.raw_l1,
            final_coarse_spatial_l1: final_metrics.coarse_spatial_l1,
            final_edge_l1: final_metrics.edge_l1,
            final_emergent_rms: tensor_rms(&rendered.emergent_lab)?,
        });
        candidates.push(BenchmarkTarget {
            kind: target.kind,
            metadata: target.metadata,
            image: rendered.image,
        });
    }
    let report = BenchmarkRunReport {
        reference,
        fixed_seed,
        candidate_separability: pairwise_separability(&candidates)?,
        results,
    };
    write_json_atomic(&paths.benchmark, &report)?;
    Ok(report)
}
fn render_decomposition_frontier(
    config: &RunConfig,
    paths: &ArtifactPaths,
    renderer: &ImplicitRenderer,
    world: &WorldState,
    sample: &TargetSample,
    device: &Device,
) -> Result<(Vec<String>, PathBuf, Vec<FrontierPoint>)> {
    let plan = RenderPlan::new(config, config.train_resolution, device)?;
    let attribution = renderer.render_attribution(
        &world.micro,
        &world.macro_field,
        &sample.genome_tensor,
        &plan,
        config.reconstruction.emergence_strength,
    )?;
    let zero_micro = Tensor::zeros_like(&world.micro)?;
    let zero_macro = Tensor::zeros_like(&world.macro_field)?;
    let without_micro = renderer.render_with_emergence(
        &zero_micro,
        &world.macro_field,
        &sample.genome_tensor,
        &plan,
        config.reconstruction.emergence_strength,
        false,
    )?;
    let without_macro = renderer.render_with_emergence(
        &world.micro,
        &zero_macro,
        &sample.genome_tensor,
        &plan,
        config.reconstruction.emergence_strength,
        false,
    )?;
    let entries: Vec<(&str, &Tensor)> = vec![
        ("target", &sample.image),
        ("grounded", &attribution.grounded_image),
        ("emergent", &attribution.emergent_visual),
        ("composite", &attribution.image),
        (
            "state_only",
            attribution
                .state_only_image
                .as_ref()
                .expect("attribution state-only image"),
        ),
        (
            "learned_only",
            attribution
                .learned_only_image
                .as_ref()
                .expect("attribution learned-only image"),
        ),
        ("micro_zero", &without_micro.image),
        ("macro_zero", &without_macro.image),
    ];
    let mut labels = Vec::new();
    let mut images = Vec::new();
    for (label, image) in entries {
        let path = sibling_png(&paths.decomposition, label);
        save_png(image, &path)?;
        labels.push(format!("{label}:{}", path.display()));
        images.push(path);
    }

    let mut frontier = Vec::new();
    for (index, normalized) in [0.0f32, 0.25, 0.50, 0.75, 1.0].into_iter().enumerate() {
        let strength = config.reconstruction.emergence_strength * normalized;
        let rendered = renderer.render_with_emergence(
            &world.micro,
            &world.macro_field,
            &sample.genome_tensor,
            &plan,
            strength,
            false,
        )?;
        let loss = visual_loss(
            &rendered,
            &sample.image,
            &world.micro,
            &world.macro_field,
            &world.memory,
            config,
            config.reconstruction.grounding_strength,
            strength,
            config.detail.boundary,
        )?;
        let diagnostics = image_metrics(&rendered.image)?;
        frontier.push(FrontierPoint {
            emergence_strength: strength,
            content_loss: loss.content.to_scalar::<f32>()?,
            grounding_loss: loss.grounding.to_scalar::<f32>()?,
            ground_coarse: loss.ground_coarse.to_scalar::<f32>()?,
            ground_mid: loss.ground_mid.to_scalar::<f32>()?,
            ground_fine: loss.ground_fine.to_scalar::<f32>()?,
            structure_loss: loss.structure.to_scalar::<f32>()?,
            emergent_residual_rms: tensor_rms(&rendered.emergent_lab)?,
            emergent_low_energy: loss.emergent_low.to_scalar::<f32>()?,
            emergent_tv: loss.emergent_tv.to_scalar::<f32>()?,
            head_redundancy: loss.head_redundancy.to_scalar::<f32>()?,
            seam: diagnostics.seam,
            gamut: loss.gamut.to_scalar::<f32>()?,
            image_variance: diagnostics.variance,
            edge_energy: diagnostics.edge,
        });
        let path = sibling_png(
            &paths.decomposition,
            &format!("emergence_{index:02}_{normalized:.2}"),
        );
        save_png(&rendered.image, &path)?;
        labels.push(format!("emergence_{normalized:.2}:{}", path.display()));
        images.push(path);
    }
    save_contact_sheet_resized(&images, &paths.decomposition, 256)?;
    write_json_atomic(&paths.emergence_frontier, &frontier)?;
    Ok((labels, paths.decomposition.clone(), frontier))
}

fn resolution_ladder(
    config: &RunConfig,
    paths: &ArtifactPaths,
    renderer: &ImplicitRenderer,
    world: &WorldState,
    sample: &TargetSample,
    device: &Device,
) -> Result<Vec<ResolutionConsistencyPoint>> {
    let mut resolutions = vec![
        config.train_resolution,
        config.snapshot_resolution,
        config.output_resolution,
    ];
    let maximum = config.output_resolution.max(config.snapshot_resolution);
    for factor in [2usize, 4, 8] {
        let resolution = config.train_resolution.saturating_mul(factor);
        if resolution <= maximum {
            resolutions.push(resolution);
        }
    }
    resolutions.sort_unstable();
    resolutions.dedup();
    let mut renders = Vec::new();
    let mut images = Vec::new();
    for resolution in resolutions {
        let plan = RenderPlan::new(config, resolution, device)?;
        let rendered = renderer.render_with_emergence(
            &world.micro,
            &world.macro_field,
            &sample.genome_tensor,
            &plan,
            config.reconstruction.emergence_strength,
            false,
        )?;
        let path = sibling_png(&paths.resolution_ladder, &format!("{resolution:04}"));
        save_png(&rendered.image, &path)?;
        images.push(path);
        renders.push((resolution, rendered.image));
    }
    let mut consistency = Vec::new();
    for pair in renders.windows(2) {
        if pair[1].0 == 2 * pair[0].0 {
            let (l1, low, edge) = cross_resolution_consistency(&pair[1].1, &pair[0].1)?;
            consistency.push(ResolutionConsistencyPoint {
                kind: "whole_2x",
                high_resolution: pair[1].0,
                low_resolution: pair[0].0,
                l1: l1.to_scalar::<f32>()?,
                low_frequency_l1: low.to_scalar::<f32>()?,
                edge_l1: edge.to_scalar::<f32>()?,
            });
        }
    }
    if let Some((_, high_whole)) = renders
        .iter()
        .find(|(resolution, _)| *resolution == 2 * config.train_resolution)
    {
        let view = SpatialView {
            x: 0.25,
            y: 0.25,
            size: 0.5,
            zoom: 2.0,
        };
        let crop_plan = RenderPlan::new_view(config, config.train_resolution, view, device)?;
        let crop = renderer.render_with_emergence(
            &world.micro,
            &world.macro_field,
            &sample.genome_tensor,
            &crop_plan,
            config.reconstruction.emergence_strength,
            false,
        )?;
        let start = config.train_resolution / 2;
        let whole_overlap = high_whole
            .narrow(2, start, config.train_resolution)?
            .narrow(3, start, config.train_resolution)?
            .contiguous()?;
        let l1 = mean_abs(&crop.image.sub(&whole_overlap)?)?;
        let (low_frequency_l1, edge_l1) = pairwise_image_diagnostics(&crop.image, &whole_overlap)?;
        consistency.push(ResolutionConsistencyPoint {
            kind: "crop_whole_overlap",
            high_resolution: 2 * config.train_resolution,
            low_resolution: config.train_resolution,
            l1,
            low_frequency_l1,
            edge_l1,
        });
        let path = sibling_png(&paths.resolution_ladder, "crop_overlap");
        save_png(&crop.image, &path)?;
        images.push(path);
    }
    save_contact_sheet_resized(&images, &paths.resolution_ladder, 256)?;
    Ok(consistency)
}

fn autonomous_rollout(
    config: &RunConfig,
    paths: &ArtifactPaths,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    world: &WorldState,
    sample: &TargetSample,
    device: &Device,
) -> Result<Vec<AttractorRecord>> {
    let plan = RenderPlan::new(config, config.snapshot_resolution, device)?;
    let mut probe = world.clone();
    probe.age = probe.age.max(config.developmental_horizon() as u64);
    let mut prior_image: Option<Tensor> = None;
    let mut signatures: Vec<Vec<f32>> = Vec::new();
    let mut records = Vec::new();
    let mut images = Vec::new();
    let mut movement_samples = 0usize;
    let mut micro_movement_sum = 0.0f32;
    let mut macro_movement_sum = 0.0f32;
    let mut macro_updates = 0usize;
    for offset in 1..=config.analysis.autonomous_horizon {
        let stepped = dynamics.step(&probe, &sample.genome_tensor, None, None, 0.0, false)?;
        movement_samples += 1;
        micro_movement_sum += stepped.micro_movement;
        if stepped.macro_updated {
            macro_updates += 1;
            macro_movement_sum += stepped.macro_movement;
        }
        probe = stepped.world;
        if offset % config.analysis.stride != 0 && offset != config.analysis.autonomous_horizon {
            continue;
        }
        let rendered = renderer.render_with_emergence(
            &probe.micro,
            &probe.macro_field,
            &sample.genome_tensor,
            &plan,
            config.reconstruction.emergence_strength,
            false,
        )?;
        let (image_delta_valid, image_delta) = match prior_image.as_ref() {
            Some(previous) => (true, mean_abs(&rendered.image.sub(previous)?)?),
            None => (false, 0.0),
        };
        let signature = state_signature(&probe)?;
        let recurrence_distance_valid = !signatures.is_empty();
        let recurrence_distance = if recurrence_distance_valid {
            signatures
                .iter()
                .map(|previous| vector_rms_distance(previous, &signature))
                .fold(f32::INFINITY, f32::min)
        } else {
            0.0
        };
        signatures.push(signature);
        let micro = state_metrics(&probe.micro, config.state_limit)?;
        let macro_field = state_metrics(&probe.macro_field, config.state_limit)?;
        let micro_movement = micro_movement_sum / movement_samples.max(1) as f32;
        let macro_movement = if macro_updates > 0 {
            macro_movement_sum / macro_updates as f32
        } else {
            0.0
        };
        records.push(AttractorRecord {
            analysis_version: 2,
            reference_fidelity: 0.0,
            micro_reference_drive_rms: stepped.micro_reference_drive_rms,
            macro_reference_drive_rms: stepped.macro_reference_drive_rms,
            offset,
            micro_movement,
            macro_movement,
            micro_rms: micro.rms,
            macro_rms: macro_field.rms,
            micro_near_bound_fraction: micro.clamp_fraction,
            macro_near_bound_fraction: macro_field.clamp_fraction,
            memory_rms: tensor_rms(&probe.memory)?,
            image_delta_valid,
            image_delta,
            recurrence_distance_valid,
            recurrence_distance,
            output_fingerprint: tensor_fingerprint(&rendered.image)?,
            approximate_cycle_candidate: recurrence_distance_valid
                && recurrence_distance < 1e-3
                && micro_movement + macro_movement < 1e-3,
        });
        let path = sibling_png(&paths.attractor_analysis, &format!("{offset:05}"));
        save_png(&rendered.image, &path)?;
        images.push(path);
        prior_image = Some(rendered.image.detach());
        movement_samples = 0;
        micro_movement_sum = 0.0;
        macro_movement_sum = 0.0;
        macro_updates = 0;
    }
    save_contact_sheet_resized(
        &images,
        &paths.attractor_analysis.with_extension("png"),
        256,
    )?;
    write_json_atomic(&paths.attractor_analysis, &records)?;
    Ok(records)
}

fn perturbation_recovery(
    config: &RunConfig,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    world: &WorldState,
    sample: &TargetSample,
    device: &Device,
) -> Result<Vec<PerturbationRecord>> {
    let mut control = world.clone();
    control.age = control.age.max(config.developmental_horizon() as u64);
    let mut variants = vec![
        (
            "micro_gaussian".to_owned(),
            perturb_field(world, true, false, false, false, false, config, device)?,
        ),
        (
            "macro_gaussian".to_owned(),
            perturb_field(world, false, true, false, false, false, config, device)?,
        ),
        (
            "memory_gaussian".to_owned(),
            perturb_field(world, false, false, true, false, false, config, device)?,
        ),
        (
            "micro_patch_erased".to_owned(),
            perturb_field(world, false, false, false, true, false, config, device)?,
        ),
        (
            "macro_patch_erased".to_owned(),
            perturb_field(world, false, false, false, false, true, config, device)?,
        ),
    ];
    for (_, variant) in &mut variants {
        variant.age = control.age;
    }
    let initial: Vec<f32> = variants
        .iter()
        .map(|(_, variant)| world_distance(variant, &control))
        .collect::<Result<_>>()?;
    let mut half_recovery = vec![None; variants.len()];
    for offset in 1..=config.analysis.perturbation_horizon {
        control = dynamics
            .step(
                &control,
                &sample.genome_tensor,
                Some(&sample.reference_micro),
                Some(&sample.reference_macro),
                config.reference_fidelity_max,
                false,
            )?
            .world;
        for (index, (_, variant)) in variants.iter_mut().enumerate() {
            *variant = dynamics
                .step(
                    variant,
                    &sample.genome_tensor,
                    Some(&sample.reference_micro),
                    Some(&sample.reference_macro),
                    config.reference_fidelity_max,
                    false,
                )?
                .world;
            let distance = world_distance(variant, &control)?;
            if half_recovery[index].is_none() && distance <= 0.5 * initial[index] {
                half_recovery[index] = Some(offset);
            }
        }
    }
    let plan = RenderPlan::new(config, config.train_resolution, device)?;
    let control_image = renderer
        .render_with_emergence(
            &control.micro,
            &control.macro_field,
            &sample.genome_tensor,
            &plan,
            config.reconstruction.emergence_strength,
            false,
        )?
        .image;
    variants
        .into_iter()
        .enumerate()
        .map(|(index, (name, variant))| {
            let final_state_distance = world_distance(&variant, &control)?;
            let image = renderer
                .render_with_emergence(
                    &variant.micro,
                    &variant.macro_field,
                    &sample.genome_tensor,
                    &plan,
                    config.reconstruction.emergence_strength,
                    false,
                )?
                .image;
            let final_output_l1 = mean_abs(&image.sub(&control_image)?)?;
            let recovery_ratio = final_state_distance / initial[index].max(1e-8);
            Ok(PerturbationRecord {
                noise_distribution: if name.ends_with("_gaussian") {
                    "uniform[-0.03,0.03); legacy case name retained"
                } else {
                    "none; deterministic central patch erasure"
                },
                reference_fidelity: config.reference_fidelity_max,
                name,
                initial_state_distance: initial[index],
                final_state_distance,
                final_output_l1,
                recovery_ratio,
                time_to_half_recovery: half_recovery[index],
                perturbation_recovery_observed: recovery_ratio < 0.5,
            })
        })
        .collect()
}

fn dynamics_ablation(
    config: &RunConfig,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    world: &WorldState,
    sample: &TargetSample,
    device: &Device,
) -> Result<Vec<DynamicsAblationRecord>> {
    let variants = vec![
        ("full", DynamicsAblation::default()),
        (
            "interface_disabled",
            DynamicsAblation {
                disable_interface: true,
                ..DynamicsAblation::default()
            },
        ),
        (
            "micro_frozen",
            DynamicsAblation {
                freeze_micro: true,
                ..DynamicsAblation::default()
            },
        ),
        (
            "macro_frozen",
            DynamicsAblation {
                freeze_macro: true,
                ..DynamicsAblation::default()
            },
        ),
        (
            "nca_disabled",
            DynamicsAblation {
                disable_nca: true,
                ..DynamicsAblation::default()
            },
        ),
        (
            "reaction_disabled",
            DynamicsAblation {
                disable_reaction: true,
                ..DynamicsAblation::default()
            },
        ),
        (
            "phase_disabled",
            DynamicsAblation {
                disable_phase: true,
                ..DynamicsAblation::default()
            },
        ),
        (
            "cyclic_disabled",
            DynamicsAblation {
                disable_cyclic: true,
                ..DynamicsAblation::default()
            },
        ),
        (
            "forcing_disabled",
            DynamicsAblation {
                disable_forcing: true,
                ..DynamicsAblation::default()
            },
        ),
    ];
    let plan = RenderPlan::new(config, config.train_resolution.min(192), device)?;
    let mut outcomes = Vec::new();
    for (name, ablation) in variants {
        let mut state = world.clone();
        state.age = state.age.max(config.developmental_horizon() as u64);
        for _ in 0..config.analysis.dynamics_horizon {
            state = dynamics
                .step_ablated(
                    &state,
                    &sample.genome_tensor,
                    Some(&sample.reference_micro),
                    Some(&sample.reference_macro),
                    Some(&sample.reference_micro),
                    Some(&sample.reference_macro),
                    config.reference_fidelity_max,
                    false,
                    &ablation,
                )?
                .world;
        }
        let image = renderer
            .render_with_emergence(
                &state.micro,
                &state.macro_field,
                &sample.genome_tensor,
                &plan,
                config.reconstruction.emergence_strength,
                false,
            )?
            .image;
        outcomes.push((name.to_owned(), state, image));
    }
    let full = &outcomes[0];
    outcomes
        .iter()
        .map(|(name, state, image)| {
            Ok(DynamicsAblationRecord {
                name: name.clone(),
                horizon: config.analysis.dynamics_horizon,
                output_l1_from_full: mean_abs(&image.sub(&full.2)?)?,
                micro_l1_from_full: mean_abs(&state.micro.sub(&full.1.micro)?)?,
                macro_l1_from_full: mean_abs(&state.macro_field.sub(&full.1.macro_field)?)?,
                memory_l1_from_full: mean_abs(&state.memory.sub(&full.1.memory)?)?,
                output_fingerprint: tensor_fingerprint(image)?,
            })
        })
        .collect()
}
fn target_separability(
    config: &RunConfig,
    paths: &ArtifactPaths,
    corpus: &mut ImageCorpus,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    device: &Device,
) -> Result<SeparabilitySummary> {
    let target_count = corpus.len().min(8);
    let fixed_age = config.developmental_horizon().min(32);
    let plan = RenderPlan::new(config, config.train_resolution.min(192), device)?;
    let mut phenotypes = Vec::new();
    let mut montage = Vec::new();
    for index in 0..target_count {
        let sample = corpus.sample_index(index, device)?;
        let mut world = WorldState::fresh(config, config.seed ^ 0x5e9a_0001, device)?;
        world.target_index = index;
        for _ in 0..fixed_age {
            world = dynamics
                .step(
                    &world,
                    &sample.genome_tensor,
                    Some(&sample.reference_micro),
                    Some(&sample.reference_macro),
                    config.reference_fidelity_max,
                    false,
                )?
                .world;
        }
        let rendered = renderer.render_with_emergence(
            &world.micro,
            &world.macro_field,
            &sample.genome_tensor,
            &plan,
            config.reconstruction.emergence_strength,
            false,
        )?;
        let path = sibling_png(&paths.target_comparison, &format!("target_{index:03}"));
        save_png(&rendered.image, &path)?;
        montage.push(path);
        phenotypes.push((index, world, rendered.image, rendered.emergent_lab));
    }
    let mut pairs = Vec::new();
    for first in 0..phenotypes.len() {
        for second in first + 1..phenotypes.len() {
            let (low_frequency_l1, edge_l1) =
                pairwise_image_diagnostics(&phenotypes[first].2, &phenotypes[second].2)?;
            pairs.push(SeparabilityPair {
                first_target: phenotypes[first].0,
                second_target: phenotypes[second].0,
                output_l1: mean_abs(&phenotypes[first].2.sub(&phenotypes[second].2)?)?,
                low_frequency_l1,
                edge_l1,
                micro_l1: mean_abs(&phenotypes[first].1.micro.sub(&phenotypes[second].1.micro)?)?,
                macro_l1: mean_abs(
                    &phenotypes[first]
                        .1
                        .macro_field
                        .sub(&phenotypes[second].1.macro_field)?,
                )?,
                memory_l1: mean_abs(
                    &phenotypes[first]
                        .1
                        .memory
                        .sub(&phenotypes[second].1.memory)?,
                )?,
                emergent_l1: mean_abs(&phenotypes[first].3.sub(&phenotypes[second].3)?)?,
            });
        }
    }
    let mean_output_l1 =
        pairs.iter().map(|pair| pair.output_l1).sum::<f32>() / pairs.len().max(1) as f32;
    let mean_low_frequency_l1 =
        pairs.iter().map(|pair| pair.low_frequency_l1).sum::<f32>() / pairs.len().max(1) as f32;
    let mean_edge_l1 =
        pairs.iter().map(|pair| pair.edge_l1).sum::<f32>() / pairs.len().max(1) as f32;
    let nearest = pairs
        .iter()
        .min_by(|a, b| a.output_l1.total_cmp(&b.output_l1));
    let summary = SeparabilitySummary {
        targets: (0..target_count).collect(),
        fixed_age,
        pair_count: pairs.len(),
        mean_output_l1,
        mean_low_frequency_l1,
        mean_edge_l1,
        minimum_output_l1: nearest.map_or(0.0, |pair| pair.output_l1),
        nearest_pair: nearest.map(|pair| (pair.first_target, pair.second_target)),
        pairs,
    };
    save_contact_sheet_resized(&montage, &paths.target_comparison, 192)?;
    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn perturb_field(
    world: &WorldState,
    micro: bool,
    macro_field: bool,
    memory: bool,
    erase_micro_patch: bool,
    erase_macro_patch: bool,
    config: &RunConfig,
    device: &Device,
) -> Result<WorldState> {
    let mut output = world.clone();
    if micro {
        let noise = deterministic_noise(
            output.micro.dims4()?,
            splitmix64(config.seed ^ 0x51c0_0001),
            0.03,
            device,
        )?;
        output.micro = smooth_limit(&output.micro.add(&noise)?, config.state_limit)?;
    }
    if macro_field {
        let noise = deterministic_noise(
            output.macro_field.dims4()?,
            splitmix64(config.seed ^ 0x51c0_0002),
            0.03,
            device,
        )?;
        output.macro_field = smooth_limit(&output.macro_field.add(&noise)?, config.state_limit)?;
    }
    if memory {
        let dims = output.memory.dims2()?;
        let noise = deterministic_noise(
            (1, 1, dims.0, dims.1),
            splitmix64(config.seed ^ 0x51c0_0003),
            0.03,
            device,
        )?
        .reshape(dims)?;
        output.memory = smooth_limit(&output.memory.add(&noise)?, config.memory_limit)?;
    }
    if erase_micro_patch {
        let (_, channels, height, width) = output.micro.dims4()?;
        let mut mask = vec![1.0f32; channels * height * width];
        let y0 = height / 3;
        let y1 = (2 * height / 3).max(y0 + 1);
        let x0 = width / 3;
        let x1 = (2 * width / 3).max(x0 + 1);
        let plane = height * width;
        for channel in 0..channels {
            for y in y0..y1 {
                for x in x0..x1 {
                    mask[channel * plane + y * width + x] = 0.0;
                }
            }
        }
        output.micro = output.micro.mul(&Tensor::from_vec(
            mask,
            (1, channels, height, width),
            device,
        )?)?;
    }
    if erase_macro_patch {
        let (_, channels, height, width) = output.macro_field.dims4()?;
        let mut mask = vec![1.0f32; channels * height * width];
        let y0 = height / 3;
        let y1 = (2 * height / 3).max(y0 + 1);
        let x0 = width / 3;
        let x1 = (2 * width / 3).max(x0 + 1);
        let plane = height * width;
        for channel in 0..channels {
            for y in y0..y1 {
                for x in x0..x1 {
                    mask[channel * plane + y * width + x] = 0.0;
                }
            }
        }
        output.macro_field = output.macro_field.mul(&Tensor::from_vec(
            mask,
            (1, channels, height, width),
            device,
        )?)?;
    }
    Ok(output)
}

fn pairwise_image_diagnostics(first: &Tensor, second: &Tensor) -> Result<(f32, f32)> {
    let low_first = analysis_average_pool2(&analysis_average_pool2(first)?)?;
    let low_second = analysis_average_pool2(&analysis_average_pool2(second)?)?;
    let low_frequency_l1 = mean_abs(&low_first.sub(&low_second)?)?;
    let edge_first = analysis_edge_map(first)?;
    let edge_second = analysis_edge_map(second)?;
    let edge_l1 = mean_abs(&edge_first.sub(&edge_second)?)?;
    Ok((low_frequency_l1, edge_l1))
}

fn analysis_average_pool2(value: &Tensor) -> Result<Tensor> {
    let (batch, channels, height, width) = value.dims4()?;
    let height = height - height % 2;
    let width = width - width % 2;
    value
        .narrow(2, 0, height)?
        .narrow(3, 0, width)?
        .reshape((batch, channels, height / 2, 2, width / 2, 2))?
        .mean(5)?
        .mean(3)
        .map_err(Into::into)
}

fn analysis_edge_map(value: &Tensor) -> Result<Tensor> {
    let (_, _, height, width) = value.dims4()?;
    let dx = value
        .narrow(3, 1, width - 1)?
        .sub(&value.narrow(3, 0, width - 1)?)?
        .narrow(2, 0, height - 1)?;
    let dy = value
        .narrow(2, 1, height - 1)?
        .sub(&value.narrow(2, 0, height - 1)?)?
        .narrow(3, 0, width - 1)?;
    dx.sqr()?
        .add(&dy.sqr()?)?
        .affine(1.0, 1e-8)?
        .sqrt()
        .map_err(Into::into)
}

fn deterministic_noise(
    shape: (usize, usize, usize, usize),
    seed: u64,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let count = shape.0 * shape.1 * shape.2 * shape.3;
    let values: Vec<f32> = (0..count).map(|_| rng.gen_range(-scale..scale)).collect();
    Tensor::from_vec(values, shape, device).map_err(Into::into)
}

fn state_signature(world: &WorldState) -> Result<Vec<f32>> {
    let mut signature = Vec::new();
    signature.extend(world.micro.mean((2, 3))?.flatten_all()?.to_vec1::<f32>()?);
    signature.extend(
        world
            .macro_field
            .mean((2, 3))?
            .flatten_all()?
            .to_vec1::<f32>()?,
    );
    signature.extend(world.memory.flatten_all()?.to_vec1::<f32>()?);
    Ok(signature)
}

fn world_distance(first: &WorldState, second: &WorldState) -> Result<f32> {
    Ok(tensor_rms(&first.micro.sub(&second.micro)?)?
        + tensor_rms(&first.macro_field.sub(&second.macro_field)?)?
        + tensor_rms(&first.memory.sub(&second.memory)?)?)
}

fn vector_rms_distance(first: &[f32], second: &[f32]) -> f32 {
    if first.len() != second.len() || first.is_empty() {
        return f32::INFINITY;
    }
    (first
        .iter()
        .zip(second)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / first.len() as f32)
        .sqrt()
}

fn sibling_png(base: &Path, label: &str) -> PathBuf {
    let stem = base
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("titan_image_analysis_v9");
    base.with_file_name(format!("{stem}_{label}.png"))
}

#[cfg(test)]
mod regression;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_distance_is_deterministic() {
        let a = [0.0f32, 1.0, 2.0];
        let b = [0.0f32, 2.0, 2.0];
        let first = vector_rms_distance(&a, &b);
        assert_eq!(first, vector_rms_distance(&a, &b));
        assert!(first > 0.0);
    }
}
