use crate::analysis::run_checkpoint_analysis;
use crate::comparison::compare_v8_v9;
use crate::config::{
    BoundaryMode, ConditioningMode, MorphDepthMode, ObjectiveMode, RunConfig, SCHEMA_VERSION,
};
use crate::corpus::{CorpusSourceMetadata, CorpusSummary, ImageCorpus, TargetSample};
use crate::dynamics::DynamicsSystem;
use crate::flow::{build_training_sample, FlowLossOutput, RectifiedFlowRenderer};
use crate::metrics::{image_metrics, state_metrics, tensor_rms, MetricRecord, StateDiagnostics};
use crate::objectives::{cross_resolution_consistency, visual_loss, LossOutput};
use crate::optimizer::{OptimizerStats, PersistentAdamW};
use crate::persistence::{
    load_checkpoint, save_checkpoint, write_json_atomic, ArtifactPaths, CheckpointLoadReport,
};
use crate::render::{
    save_contact_sheet, save_mastered_png, save_png, save_state_atlas, ImplicitRenderer, RenderPlan,
};
use crate::state::WorldState;
use crate::telemetry::{
    append_jsonl, collect_model_statistics, tensor_fingerprint, AnatomySnapshot, GraftEvent,
    ImageSnapshot, MorphActivationEvent, StateSnapshot, SubsystemStatistics, TargetTelemetry,
};
use crate::tensor_ops::{mean_abs, splitmix64, variance};
use crate::terminal::TerminalReporter;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);
static INTERRUPT_HANDLER: OnceLock<std::result::Result<(), String>> = OnceLock::new();

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

struct DecompositionDiagnostics {
    grounded: crate::metrics::ImageDiagnostics,
    grounded_rms: f32,
    grounded_mean_abs: f32,
    emergent_rms: f32,
    emergent_mean_abs: f32,
    emergent_variance: f32,
    contribution_rms: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ObservationIdentity {
    target_index: usize,
    resolution: usize,
    detail: bool,
    fingerprint: u64,
}

struct CachedObservation {
    identity: ObservationIdentity,
    image: Tensor,
}

#[derive(Serialize)]
struct BuildProvenance {
    commit: &'static str,
    dirty: bool,
    release: bool,
    target_arch: &'static str,
    rustflags: &'static str,
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
    active_parameter_count: usize,
    inactive_reserve_parameters: usize,
    parameter_subsystems: Vec<SubsystemStatistics>,
    corpus_sources: Vec<CorpusSourceMetadata>,
    corpus_summary: CorpusSummary,
    corpus_fingerprint: String,
    config_signature: String,
    parameter_count: usize,
    requested_development_steps: usize,
    completed_development_steps: u64,
    interrupted: bool,
    stability_stopped: bool,
    resumed: bool,
    checkpoint_recovered: bool,
    render_only: bool,
    metrics_continued: bool,
    loaded_optimizer_tensors: usize,
    start_world_step: u64,
    completed_world_step: u64,
    optimizer_updates_start: u64,
    optimizer_updates_end: u64,
    optimizer_windows_requested: usize,
    optimizer_windows_completed: usize,
    full_core_windows: usize,
    decoder_only_windows: usize,
    gallery_requested: usize,
    gallery_completed: usize,
    final_target: &'a str,
    peak_rss_kib: Option<u64>,
    phase_seconds: PhaseSeconds,
    average_ms_per_development_step: f64,
    average_development_steps_per_second: f64,
    outputs: Vec<String>,
    claims: [&'static str; 5],
}

pub fn run(config: RunConfig) -> Result<()> {
    config.validate()?;
    let training_enabled = !config.render_only && !config.analysis.only;
    install_interrupt_handler()?;
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
    let target_statistics_path = config.output_dir.join(format!(
        "titan_image_target_statistics_v9{}.json",
        config.suffix()
    ));
    let mut terminal_reporter = TerminalReporter::new(config.terminal, config.research_preset);

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
    let flow_renderer = RectifiedFlowRenderer::new(&config, builder.pp("flow"))?;
    deterministic_initialize(&vars, config.seed)?;
    let mut optimizer = PersistentAdamW::new(&vars, &config)?;
    let parameter_count = learned_parameter_count(&vars);

    if config.render_only && !paths.recoverable_checkpoint_complete() {
        bail!(
            "--render-only requires a complete v9 checkpoint for this output directory and run tag"
        );
    }
    if config.analysis.probe_dir.is_some() && !paths.recoverable_checkpoint_complete() {
        bail!(
            "--probe-dir requires a complete v9 checkpoint for this output directory and run tag"
        );
    }

    let checkpoint_available = paths.recoverable_checkpoint_complete();
    if !config.fresh && paths.recoverable_checkpoint_exists() && !checkpoint_available {
        bail!(
            "partial v9 checkpoint set in {}; use --fresh or restore all checkpoint files",
            config.output_dir.display()
        );
    }
    let (loaded_world, resumed, checkpoint_load) = if !config.fresh && checkpoint_available {
        let (world, report) = load_checkpoint(
            &paths,
            &mut vars,
            &mut optimizer,
            &device,
            &config,
            corpus.fingerprint(),
        )?;
        (Some(world), true, report)
    } else {
        (None, false, CheckpointLoadReport::default())
    };
    let loaded_optimizer_tensors =
        checkpoint_load.optimizer_moments_preserved + checkpoint_load.optimizer_moments_new;
    if checkpoint_load.recovered_previous {
        terminal_reporter.event(
            "RECOVERY",
            format!(
                "restored the previous complete checkpoint generation at world step {}",
                loaded_world.as_ref().map_or(0, |world| world.step)
            ),
        );
    }

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
    let mut target_telemetry = TargetTelemetry::new(start_world_step);

    if checkpoint_load.model.grafted {
        let plan = RenderPlan::new(&config, config.train_resolution, &device)?;
        let age_phase = (world.age as f32 / config.episode_steps.max(1) as f32).clamp(0.0, 1.0);
        let (grounding_schedule, emergence_schedule) = config.developmental_schedule(age_phase);
        let mut old_world = world.clone();
        old_world.morph_active_depth = checkpoint_load.model.old_active_depth;
        let old_render = renderer.render_with_emergence(
            &old_world.micro,
            &old_world.macro_field,
            &sample.genome_tensor,
            &plan,
            emergence_schedule,
            false,
        )?;
        let new_render = renderer.render_with_emergence(
            &world.micro,
            &world.macro_field,
            &sample.genome_tensor,
            &plan,
            emergence_schedule,
            false,
        )?;
        let pre_loss = visual_loss(
            &old_render,
            &sample.image,
            &old_world.micro,
            &old_world.macro_field,
            &old_world.memory,
            &config,
            grounding_schedule,
            emergence_schedule,
            config.detail.boundary,
        )?;
        let pre_micro = state_metrics(&old_world.micro, config.state_limit)?;
        let pre_macro = state_metrics(&old_world.macro_field, config.state_limit)?;
        let pre_image = image_metrics(&old_render.image)?;
        let immediate_output_l1 = new_render
            .image
            .sub(&old_render.image)?
            .abs()?
            .mean_all()?
            .to_scalar::<f32>()?;
        let checkpoint_id_after = save_checkpoint(
            &paths,
            &vars,
            &optimizer,
            &world,
            &config,
            corpus.fingerprint(),
        )?;
        let old_generation = world.morph_generation.saturating_sub(1);
        let old_layers = checkpoint_load.model.old_morph_layers;
        let event = GraftEvent {
            event_type: "morphic_graft",
            schema_version: SCHEMA_VERSION,
            event_id: format!(
                "graft-{:016x}-g{}",
                checkpoint_id_after, world.morph_generation
            ),
            world_step: world.step,
            checkpoint_id_before: checkpoint_load.checkpoint_id,
            checkpoint_id_after,
            old_anatomy: AnatomySnapshot {
                physical_layers: old_layers,
                active_depth: checkpoint_load.model.old_active_depth,
                graft_generation: old_generation,
                block_birth_generations: world.morph_birth_generations[..old_layers].to_vec(),
            },
            new_anatomy: AnatomySnapshot {
                physical_layers: config.morph_layers,
                active_depth: world.morph_active_depth,
                graft_generation: world.morph_generation,
                block_birth_generations: world.morph_birth_generations.clone(),
            },
            copied_tensor_count: checkpoint_load.model.copied_tensors,
            copied_parameter_count: checkpoint_load.model.copied_parameters,
            new_tensor_count: checkpoint_load.model.new_tensors,
            new_parameter_count: checkpoint_load.model.new_parameters,
            resized_tensors: checkpoint_load.model.resized_tensors.clone(),
            skipped_tensors: checkpoint_load.model.skipped_tensors.clone(),
            optimizer_moments_preserved: checkpoint_load.optimizer_moments_preserved,
            optimizer_moments_new: checkpoint_load.optimizer_moments_new,
            optimizer_updates_preserved: checkpoint_load.optimizer_updates_preserved,
            pre_graft_loss_total: pre_loss.total.to_scalar::<f32>()?,
            pre_graft_loss_grounding: pre_loss.grounding.to_scalar::<f32>()?,
            pre_graft_micro: StateSnapshot::from(&pre_micro),
            pre_graft_macro: StateSnapshot::from(&pre_macro),
            pre_graft_memory_rms: tensor_rms(&old_world.memory)?,
            pre_graft_output_fingerprint: tensor_fingerprint(&old_render.image)?,
            pre_graft_image: ImageSnapshot::from(&pre_image),
            immediate_output_l1,
            immediate_memory_rms_delta: 0.0,
            committed: true,
        };
        append_jsonl(&paths.events, &event)?;
        write_json_atomic(&paths.graft_analysis, &event)?;
        terminal_reporter.event(
            "GRAFT",
            format!(
                "committed L{}/{} -> L{}/{} | copied {} params, born {} params | preservation L1 {:.8}",
            checkpoint_load.model.old_active_depth,
            old_layers,
            world.morph_active_depth,
            config.morph_layers,
            checkpoint_load.model.copied_parameters,
            checkpoint_load.model.new_parameters,
            immediate_output_l1,
            ),
        );
    } else if checkpoint_load.model.new_active_depth > checkpoint_load.model.old_active_depth {
        let mut old_world = world.clone();
        old_world.morph_active_depth = checkpoint_load.model.old_active_depth;
        let old_step = dynamics
            .step(
                &old_world,
                &sample.genome_tensor,
                Some(&sample.reference_micro),
                Some(&sample.reference_macro),
                config.reference_fidelity_max,
                false,
            )?
            .world;
        let activated_step = dynamics
            .step(
                &world,
                &sample.genome_tensor,
                Some(&sample.reference_micro),
                Some(&sample.reference_macro),
                config.reference_fidelity_max,
                false,
            )?
            .world;
        let preservation = mean_abs(&old_step.micro.sub(&activated_step.micro)?)?
            + mean_abs(&old_step.macro_field.sub(&activated_step.macro_field)?)?
            + mean_abs(&old_step.memory.sub(&activated_step.memory)?)?;
        if preservation > 1e-6 {
            bail!("startup morph activation was not function-preserving: delta {preservation}");
        }
        save_checkpoint(
            &paths,
            &vars,
            &optimizer,
            &world,
            &config,
            corpus.fingerprint(),
        )?;
        let event = MorphActivationEvent {
            event_type: "morph_activation",
            schema_version: SCHEMA_VERSION,
            world_step: world.step,
            old_active_depth: checkpoint_load.model.old_active_depth,
            new_active_depth: world.morph_active_depth,
            physical_layers: config.morph_layers,
            plateau_improvement: 0.0,
            seam_before: 0.0,
            function_preservation_l1: preservation,
        };
        append_jsonl(&paths.events, &event)?;
        terminal_reporter.event(
            "MORPH ACTIVATION",
            format!(
                "startup L{} -> L{} of {} | preservation {:.8}",
                checkpoint_load.model.old_active_depth,
                world.morph_active_depth,
                config.morph_layers,
                preservation
            ),
        );
    }

    let metrics_continued = training_enabled && resumed && paths.metrics.exists();
    let mut metrics_writer = if !training_enabled {
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

    terminal_reporter.event(
        "START",
        format!(
            "{} source image(s), {} cached, {} parameters, {} at world step {}",
            corpus.len(),
            corpus.cached_images(),
            parameter_count,
            if resumed {
                "resuming"
            } else {
                "starting fresh"
            },
            world.step
        ),
    );
    let corpus_summary = corpus.summary();
    terminal_reporter.event(
        "CORPUS",
        format!(
            "{}x{}..{}x{}, median {:.2} MP, total {:.1} MP, aspect {:.2}..{:.2}, pyramid cache {:.1} MiB",
            corpus_summary.min_width,
            corpus_summary.min_height,
            corpus_summary.max_width,
            corpus_summary.max_height,
            corpus_summary.median_megapixels,
            corpus_summary.total_megapixels,
            corpus_summary.min_aspect_ratio,
            corpus_summary.max_aspect_ratio,
            corpus_summary.cached_pyramid_bytes as f64 / (1024.0 * 1024.0),
        ),
    );
    terminal_reporter.event(
        "PROFILE",
        format!(
            "{:?}/{:?}/{:?}: {} threads, {}ch {}x{} + {}x{}, global {}px/detail {}px",
            config.profile,
            config.style,
            config.research_preset,
            effective_threads,
            config.channels,
            config.micro_size,
            config.micro_size,
            config.macro_size,
            config.macro_size,
            config.train_resolution,
            config.detail.resolution,
        ),
    );
    terminal_reporter.event(
        "ANATOMY",
        format!(
            "interface {}x{}, width {}, loops {}, morph L{}/{}, {:?} conditioning, {:?} optimizer",
            config.interface_grid,
            config.interface_grid,
            config.interface_width,
            config.interface_loops,
            world.morph_active_depth,
            config.morph_layers,
            config.conditioning,
            optimizer.kind(),
        ),
    );
    terminal_reporter.event(
        "SAFETY",
        "Ctrl-C finishes the active optimizer window, saves a checkpoint, and publishes final metadata.",
    );

    let train_plan = if !training_enabled {
        None
    } else {
        Some(RenderPlan::new(&config, config.train_resolution, &device)?)
    };
    let mut snapshot_plan: Option<RenderPlan> = None;
    let mut consistency_plan: Option<RenderPlan> = None;
    let mut output_plan: Option<RenderPlan> = None;
    let flow_plan = if config.objective.uses_flow() {
        Some(RenderPlan::new(&config, config.flow.resolution, &device)?)
    } else {
        None
    };
    let optimizer_windows = if !training_enabled {
        0
    } else {
        config.steps / config.bptt
    };
    let mut full_core_windows = 0usize;
    let mut decoder_only_windows = 0usize;
    let mut interrupted = false;
    let mut stability_stopped = false;
    let mut saturation_windows = 0usize;
    let mut previous_observation: Option<CachedObservation> = None;
    let mut morph_loss_history = VecDeque::with_capacity(config.morph_growth.plateau_window);

    for local_window in 0..optimizer_windows {
        if stop_requested() {
            interrupted = true;
            println!(
                "\nStop requested before the next optimizer window; saving the current organism."
            );
            break;
        }
        let window_started = Instant::now();
        let window_start_step = world.step;
        select_episode_if_needed(
            &config,
            &mut corpus,
            &device,
            &mut world,
            &mut sample,
            metrics_writer.as_mut(),
        )?;
        let episode_started = world.age == 0;
        if episode_started {
            terminal_reporter.event(
                "EPISODE",
                format!(
                    "episode {} target {} ({})",
                    world.episode, sample.index, sample.name
                ),
            );
        }
        let window_index = world.step / config.bptt as u64;
        let train_core = window_index.is_multiple_of(config.core_update_every as u64);
        if train_core {
            full_core_windows += 1;
        } else {
            decoder_only_windows += 1;
        }

        let mut micro_movement_sum = 0.0f32;
        let mut micro_movement_max = 0.0f32;
        let mut macro_movement_sum = 0.0f32;
        let mut macro_movement_max = 0.0f32;
        let mut micro_reference_drive_sum = 0.0f32;
        let mut macro_reference_drive_sum = 0.0f32;
        let mut macro_updates = 0usize;
        let reference_fidelity = reference_fidelity(&config, world.step);
        let age_phase_start =
            (world.age as f32 / config.episode_steps.max(1) as f32).clamp(0.0, 1.0);
        let detail = corpus.detail_observation(
            sample.index,
            world.step,
            age_phase_start,
            &config,
            &device,
        )?;
        for _ in 0..config.bptt {
            let tick = Instant::now();
            let stepped = dynamics.step_with_local_reference(
                &world,
                &sample.genome_tensor,
                Some(&sample.reference_micro),
                Some(&sample.reference_macro),
                detail.as_ref().map(|value| &value.local_reference_micro),
                detail.as_ref().map(|value| &value.local_reference_macro),
                reference_fidelity,
                train_core,
            )?;
            if train_core {
                phase.tracked_dynamics += seconds(tick.elapsed());
            } else {
                phase.detached_dynamics += seconds(tick.elapsed());
            }
            micro_movement_sum += stepped.micro_movement;
            micro_movement_max = micro_movement_max.max(stepped.micro_movement);
            macro_movement_sum += stepped.macro_movement;
            macro_movement_max = macro_movement_max.max(stepped.macro_movement);
            macro_updates += usize::from(stepped.macro_updated);
            micro_reference_drive_sum += stepped.micro_reference_drive_rms;
            macro_reference_drive_sum += stepped.macro_reference_drive_rms;
            world = stepped.world;
        }

        let tick = Instant::now();
        let detail_plan = detail
            .as_ref()
            .map(|observation| {
                RenderPlan::new_view(&config, config.detail.resolution, observation.view, &device)
            })
            .transpose()?;
        let active_plan = detail_plan
            .as_ref()
            .or(train_plan.as_ref())
            .expect("training render plan");
        let observation_identity = ObservationIdentity {
            target_index: sample.index,
            resolution: active_plan.resolution,
            detail: detail.is_some(),
            fingerprint: detail
                .as_ref()
                .map_or(sample.fingerprint, |observation| observation.fingerprint),
        };
        let target = detail
            .as_ref()
            .map_or(&sample.image, |observation| &observation.target);
        let boundary = if detail.is_some() {
            BoundaryMode::Crop
        } else {
            config.detail.boundary
        };
        let age_phase = (world.age as f32 / config.episode_steps.max(1) as f32).clamp(0.0, 1.0);
        let (grounding_schedule, emergence_schedule) = config.developmental_schedule(age_phase);
        let rendered = renderer.render_with_emergence(
            &world.micro,
            &world.macro_field,
            &sample.genome_tensor,
            active_plan,
            emergence_schedule,
            true,
        )?;
        let mut losses = visual_loss(
            &rendered,
            target,
            &world.micro,
            &world.macro_field,
            &world.memory,
            &config,
            grounding_schedule,
            emergence_schedule,
            boundary,
        )?;
        if detail.is_none()
            && config.reconstruction.loss_cross_resolution > 0.0
            && window_index.is_multiple_of(8)
            && config.train_resolution.is_multiple_of(2)
        {
            if consistency_plan.is_none() {
                consistency_plan = Some(RenderPlan::new(
                    &config,
                    config.train_resolution / 2,
                    &device,
                )?);
            }
            let low = renderer.render_with_emergence(
                &world.micro,
                &world.macro_field,
                &sample.genome_tensor,
                consistency_plan
                    .as_ref()
                    .expect("consistency plan initialized"),
                emergence_schedule,
                true,
            )?;
            let (l1, low_frequency, edge) =
                cross_resolution_consistency(&rendered.image, &low.image)?;
            let weighted = l1
                .add(&low_frequency.affine(0.5, 0.0)?)?
                .add(&edge.affine(0.25, 0.0)?)?;
            losses.total = losses
                .total
                .add(&weighted.affine(config.reconstruction.loss_cross_resolution as f64, 0.0)?)?;
            losses.cross_resolution = l1;
            losses.cross_resolution_low = low_frequency;
            losses.cross_resolution_edge = edge;
        }
        if config.objective == ObjectiveMode::HybridFlow {
            losses.total = losses
                .total
                .affine(config.flow.endpoint_weight as f64, 0.0)?;
        }
        let flow_active = config.objective.uses_flow()
            && (config.objective == ObjectiveMode::Flow
                || window_index.is_multiple_of(config.flow.cadence as u64));
        let flow_diagnostics = if flow_active {
            let time_seed = splitmix64(
                config.seed
                    ^ world.step.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ sample.fingerprint
                    ^ 0xf10a_0001,
            );
            let noise_seed = splitmix64(time_seed ^ 0xf10a_5eed);
            let flow_sample = build_training_sample(
                &sample.flow_image,
                time_seed,
                noise_seed,
                config.flow.min_time,
                config.flow.max_time,
            )?;
            let flow_loss = flow_renderer.training_loss(
                &flow_sample,
                &world.micro,
                &world.macro_field,
                &world.memory,
                flow_plan.as_ref().expect("flow plan initialized"),
                age_phase,
                reference_fidelity,
                emergence_schedule,
                true,
            )?;
            losses.total = losses
                .total
                .add(&flow_loss.loss.affine(config.flow.weight as f64, 0.0)?)?;
            Some(flow_loss)
        } else {
            None
        };
        let decomposition = DecompositionDiagnostics {
            grounded: image_metrics(&rendered.grounded_image.detach())?,
            grounded_rms: tensor_rms(&rendered.grounded_image.detach())?,
            grounded_mean_abs: mean_abs(&rendered.grounded_image.detach())?,
            emergent_rms: tensor_rms(&rendered.emergent_lab.detach())?,
            emergent_mean_abs: mean_abs(&rendered.emergent_lab.detach())?,
            emergent_variance: variance(&rendered.emergent_lab.detach())?.to_scalar::<f32>()?,
            contribution_rms: tensor_rms(
                &rendered
                    .image
                    .detach()
                    .sub(&rendered.grounded_image.detach())?,
            )?,
        };
        let loss_values = loss_scalars(&losses)?;
        phase.render_and_loss += seconds(tick.elapsed());

        let optimizer_stats = optimizer.backward_step(&losses.total)?;
        phase.backward += optimizer_stats.backward_seconds;
        phase.optimizer += optimizer_stats.step_seconds;
        world = world.detached();

        let tick = Instant::now();
        let diagnostics = image_metrics(&rendered.image.detach())?;
        let micro_state = state_metrics(&world.micro, config.state_limit)?;
        let macro_state = state_metrics(&world.macro_field, config.state_limit)?;
        let current_image = rendered.image.detach();
        let (image_delta_valid, image_delta_mean, image_delta_rms) = temporal_image_delta(
            previous_observation.as_ref(),
            observation_identity,
            &current_image,
        )?;
        let interface_memory_rms = tensor_rms(&world.memory)?;
        let stability_violation = micro_state.clamp_fraction > config.max_saturation_fraction
            || macro_state.clamp_fraction > config.max_saturation_fraction;
        saturation_windows = if stability_violation {
            saturation_windows + 1
        } else {
            0
        };
        previous_observation = Some(CachedObservation {
            identity: observation_identity,
            image: current_image,
        });
        let window_seconds = seconds(window_started.elapsed());
        let development_steps_per_second = config.bptt as f64 / window_seconds.max(1e-9);
        let record = metric_record(
            &world,
            &sample,
            optimizer.updates(),
            train_core,
            macro_updates,
            episode_started,
            micro_movement_sum / config.bptt as f32,
            micro_movement_max,
            if macro_updates > 0 {
                macro_movement_sum / macro_updates as f32
            } else {
                0.0
            },
            macro_movement_max,
            &micro_state,
            &macro_state,
            image_delta_valid,
            image_delta_mean,
            image_delta_rms,
            &diagnostics,
            loss_values,
            &optimizer_stats,
            reference_fidelity,
            interface_memory_rms,
            stability_violation,
            window_seconds,
            development_steps_per_second,
            &config,
            &decomposition,
            grounding_schedule,
            emergence_schedule,
            detail.as_ref(),
            micro_reference_drive_sum / config.bptt as f32,
            macro_reference_drive_sum / config.bptt as f32,
            flow_diagnostics.as_ref(),
        );
        target_telemetry.observe(&sample.name, config.episode_steps, &record);
        record.write_csv(
            metrics_writer
                .as_mut()
                .expect("training has a metrics writer"),
        )?;
        if (local_window + 1).is_multiple_of(config.log_every)
            || local_window + 1 == optimizer_windows
        {
            terminal_reporter.training(&record);
        }
        phase.metrics_and_logging += seconds(tick.elapsed());
        morph_loss_history.push_back(record.loss_grounding);
        while morph_loss_history.len() > config.morph_growth.plateau_window {
            morph_loss_history.pop_front();
        }
        maybe_activate_morph(
            &config,
            &dynamics,
            &sample,
            reference_fidelity,
            &optimizer_stats,
            &vars,
            &optimizer,
            corpus.fingerprint(),
            stability_violation,
            &morph_loss_history,
            &paths,
            &mut world,
        )?;
        if config.stability_patience > 0 && saturation_windows >= config.stability_patience {
            stability_stopped = true;
            interrupted = true;
            println!(
                "Stability watchdog stopped training at step {} after {} consecutive near-bound windows; saving a resumable checkpoint.",
                world.step,
                saturation_windows,
            );
            break;
        }

        if stop_requested() {
            interrupted = true;
            println!(
                "\nStop requested: completed optimizer window at step {}; saving the organism.",
                world.step
            );
            break;
        }

        if cadence_due(window_start_step, world.step, config.snapshot_every) {
            let tick = Instant::now();
            ensure_render_plan(
                &config,
                config.snapshot_resolution,
                &device,
                &mut snapshot_plan,
            )?;
            let snapshot = renderer.render(
                &world.micro,
                &world.macro_field,
                &sample.genome_tensor,
                snapshot_plan.as_ref().expect("snapshot plan initialized"),
                false,
            )?;
            let path = snapshot_path(&config, &world);
            save_png(&snapshot.image, &path)?;
            terminal_reporter.event(
                "SNAPSHOT",
                format!("step {} -> {}", world.step, path.display()),
            );
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
            let model_stats = collect_model_statistics(
                &vars,
                &optimizer,
                world.step,
                world.morph_active_depth,
                &world.morph_birth_generations,
                config.objective.uses_flow(),
            )?;
            write_json_atomic(&paths.model_stats, &model_stats)?;
            write_json_atomic(
                &target_statistics_path,
                &target_telemetry.report(world.step),
            )?;
            terminal_reporter.event("CHECKPOINT", format!("committed world step {}", world.step));
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
    let final_model_stats = collect_model_statistics(
        &vars,
        &optimizer,
        world.step,
        world.morph_active_depth,
        &world.morph_birth_generations,
        config.objective.uses_flow(),
    )?;
    write_json_atomic(&paths.model_stats, &final_model_stats)?;
    if training_enabled {
        write_json_atomic(
            &target_statistics_path,
            &target_telemetry.report(world.step),
        )?;
    }

    let output_tick = Instant::now();
    println!(
        "Rendering final raw/mastered output at {}px...",
        config.output_resolution
    );
    ensure_render_plan(&config, config.output_resolution, &device, &mut output_plan)?;
    let final_render = renderer.render(
        &world.micro.detach(),
        &world.macro_field.detach(),
        &sample.genome_tensor,
        output_plan.as_ref().expect("output plan initialized"),
        false,
    )?;
    let final_raw_path = if config.analysis.only {
        config.output_dir.join(format!(
            "titan_image_analysis_render_v9{}.png",
            config.suffix()
        ))
    } else {
        paths.raw.clone()
    };
    let final_mastered_path = if config.analysis.only {
        config.output_dir.join(format!(
            "titan_image_analysis_mastered_v9{}.png",
            config.suffix()
        ))
    } else {
        paths.mastered.clone()
    };
    let final_grounded_path = if config.analysis.only {
        config.output_dir.join(format!(
            "titan_image_analysis_grounded_v9{}.png",
            config.suffix()
        ))
    } else {
        paths.grounded.clone()
    };
    let final_emergent_path = if config.analysis.only {
        config.output_dir.join(format!(
            "titan_image_analysis_emergent_v9{}.png",
            config.suffix()
        ))
    } else {
        paths.emergent.clone()
    };
    save_png(&final_render.image, &final_raw_path)?;
    save_mastered_png(
        &final_render.image,
        &final_mastered_path,
        config.mastering_strength,
    )?;
    save_png(&final_render.grounded_image, &final_grounded_path)?;
    save_png(&final_render.emergent_visual, &final_emergent_path)?;
    let mut outputs = vec![
        final_raw_path.display().to_string(),
        final_mastered_path.display().to_string(),
        final_grounded_path.display().to_string(),
        final_emergent_path.display().to_string(),
    ];
    if training_enabled {
        outputs.push(target_statistics_path.display().to_string());
    }
    if config.save_state_atlas {
        println!("Writing micro/macro state atlases...");
        save_state_atlas(&world.micro, &paths.micro_state)?;
        save_state_atlas(&world.macro_field, &paths.macro_state)?;
        outputs.push(paths.micro_state.display().to_string());
        outputs.push(paths.macro_state.display().to_string());
    }
    if !stability_stopped && (config.analysis_requested() || config.analysis.emergence_gallery) {
        println!("ANALYSIS: decomposition, emergence frontier, and requested frozen-state probes");
        let analysis = run_checkpoint_analysis(
            &config,
            &paths,
            &mut corpus,
            &dynamics,
            &renderer,
            &flow_renderer,
            &world,
            &sample,
            &device,
        )?;
        outputs.push(paths.analysis.display().to_string());
        if let Some(probe) = &analysis.natural_image_probe {
            outputs.push(probe.report.clone());
            outputs.push(probe.montage.clone());
            for target in &probe.targets {
                outputs.extend(target.points.iter().map(|point| point.output.clone()));
            }
        }
        if config.analysis.render_attribution || config.analysis.emergence_gallery {
            outputs.push(paths.decomposition.display().to_string());
            outputs.push(paths.emergence_frontier.display().to_string());
        }
    }
    interrupted |= stop_requested();
    let (gallery_outputs, gallery_interrupted) = if stability_stopped || config.analysis.only {
        (Vec::new(), false)
    } else {
        render_gallery(
            &config,
            &corpus,
            &dynamics,
            &renderer,
            output_plan.as_ref().expect("output plan initialized"),
            &device,
        )?
    };
    interrupted |= gallery_interrupted || stop_requested();
    let gallery_completed = gallery_outputs.len() / 2;
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
    let average_development_steps_per_second = if phase.total > 0.0 {
        completed_steps as f64 / phase.total
    } else {
        0.0
    };
    let metadata = RunMetadata {
        schema_version: SCHEMA_VERSION,
        package_version: env!("CARGO_PKG_VERSION"),
        run_id: run_id(),
        build: BuildProvenance {
            commit: env!("TITAN_BUILD_COMMIT"),
            dirty: env!("TITAN_BUILD_DIRTY") == "true",
            release: !cfg!(debug_assertions),
            target_arch: std::env::consts::ARCH,
            rustflags: env!("TITAN_BUILD_RUSTFLAGS"),
        },
        invocation: std::env::args().collect(),
        config: &config,
        effective_threads,
        source_images: corpus.len(),
        cached_source_images: corpus.cached_images(),
        corpus_sources: corpus.source_manifest(),
        corpus_summary: corpus.summary_snapshot()?,
        corpus_fingerprint: format!("{:016x}", corpus.fingerprint()),
        config_signature: format!("{:016x}", config.checkpoint_signature()),
        parameter_count,
        active_parameter_count: final_model_stats.active_parameters,
        inactive_reserve_parameters: final_model_stats.inactive_reserve_parameters,
        parameter_subsystems: final_model_stats.subsystems.clone(),
        requested_development_steps: config.steps,
        completed_development_steps: completed_steps,
        interrupted,
        stability_stopped,
        resumed,
        checkpoint_recovered: checkpoint_load.recovered_previous,
        render_only: config.render_only,
        metrics_continued,
        loaded_optimizer_tensors,
        start_world_step,
        completed_world_step: world.step,
        optimizer_updates_start,
        optimizer_updates_end: optimizer.updates(),
        optimizer_windows_requested: optimizer_windows,
        optimizer_windows_completed: full_core_windows + decoder_only_windows,
        full_core_windows,
        decoder_only_windows,
        gallery_requested: config.gallery,
        gallery_completed,
        final_target: &sample.name,
        peak_rss_kib: peak_rss_kib(),
        phase_seconds: phase.clone(),
        average_ms_per_development_step: if completed_steps > 0 {
            1000.0 * phase.total / completed_steps as f64
        } else {
            0.0
        },
        outputs,
        average_development_steps_per_second,
        claims: [
            "autonomous morphogenic image generator",
            "not an action-conditioned world model",
            "IFS and quasiperiodic fields are bounded target attractions",
            "finite raster outputs do not prove an exact fractal dimension",
            "profile timings are measurements of this invocation, not universal device constants",
        ],
    };
    let metadata_path = if config.render_only || config.analysis.only {
        &paths.render_metadata
    } else {
        &paths.metadata
    };
    write_json_atomic(metadata_path, &metadata)?;
    if let Some(v8_dir) = config.analysis.compare_v8_dir.as_deref() {
        let comparison = compare_v8_v9(v8_dir, &paths)?;
        println!(
            "V8/V9 COMPARISON: v8 content {:?} -> v9 content {:?}, speed {:?} -> {:?}",
            comparison.v8.reconstruction_content,
            comparison.v9.reconstruction_content,
            comparison.v8.development_steps_per_second,
            comparison.v9.development_steps_per_second,
        );
    }
    println!(
        "{} at world step {} in {:.2}s ({:.2} development step/s). Raw: {}  Mastered: {}",
        if interrupted {
            "Stopped safely"
        } else {
            "Finished"
        },
        world.step,
        phase.total,
        average_development_steps_per_second,
        final_raw_path.display(),
        final_mastered_path.display(),
    );
    if gallery_completed > 0 {
        println!("Gallery: {}", paths.gallery.display());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maybe_activate_morph(
    config: &RunConfig,
    dynamics: &DynamicsSystem,
    sample: &TargetSample,
    reference_fidelity: f32,
    optimizer_stats: &OptimizerStats,
    vars: &VarMap,
    optimizer: &PersistentAdamW,
    corpus_fingerprint: u64,
    stability_violation: bool,
    history: &VecDeque<f32>,
    paths: &ArtifactPaths,
    world: &mut WorldState,
) -> Result<bool> {
    if config.morph_growth.mode != MorphDepthMode::Adaptive
        || world.morph_active_depth >= config.morph_growth.max_depth
        || world.morph_active_depth >= config.morph_layers
        || !world
            .step
            .is_multiple_of(config.morph_growth.interval as u64)
        || history.len() < config.morph_growth.plateau_window
        || stability_violation
        || optimizer_stats.clip_scale < 0.05
    {
        return Ok(false);
    }
    let split = history.len() / 2;
    let early = history.iter().take(split).sum::<f32>() / split.max(1) as f32;
    let late = history.iter().skip(split).sum::<f32>() / (history.len() - split).max(1) as f32;
    let improvement = (early - late) / early.abs().max(1e-6);
    if improvement > config.morph_growth.plateau_epsilon {
        return Ok(false);
    }

    let old_depth = world.morph_active_depth;
    let mut old_world = world.clone();
    old_world.morph_active_depth = old_depth;
    let mut activated_world = world.clone();
    activated_world.morph_active_depth = old_depth + 1;
    let old_step = dynamics
        .step(
            &old_world,
            &sample.genome_tensor,
            Some(&sample.reference_micro),
            Some(&sample.reference_macro),
            reference_fidelity,
            false,
        )?
        .world;
    let activated_step = dynamics
        .step(
            &activated_world,
            &sample.genome_tensor,
            Some(&sample.reference_micro),
            Some(&sample.reference_macro),
            reference_fidelity,
            false,
        )?
        .world;
    let preservation = mean_abs(&old_step.micro.sub(&activated_step.micro)?)?
        + mean_abs(&old_step.macro_field.sub(&activated_step.macro_field)?)?
        + mean_abs(&old_step.memory.sub(&activated_step.memory)?)?;
    if preservation > 1e-6 {
        bail!("reserved morph activation was not function-preserving: delta {preservation}");
    }
    world.morph_active_depth = old_depth + 1;
    save_checkpoint(paths, vars, optimizer, world, config, corpus_fingerprint)?;
    let event = MorphActivationEvent {
        event_type: "morph_activation",
        schema_version: SCHEMA_VERSION,
        world_step: world.step,
        old_active_depth: old_depth,
        new_active_depth: world.morph_active_depth,
        physical_layers: config.morph_layers,
        plateau_improvement: improvement,
        seam_before: 0.0,
        function_preservation_l1: preservation,
    };
    append_jsonl(&paths.events, &event)?;
    println!(
        "MORPH ACTIVATION: L{}/{} -> L{}/{} | plateau {:.5} | preservation {:.8}",
        old_depth,
        config.morph_layers,
        world.morph_active_depth,
        config.morph_layers,
        improvement,
        preservation,
    );
    Ok(true)
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

fn temporal_image_delta(
    previous: Option<&CachedObservation>,
    current_identity: ObservationIdentity,
    current_image: &Tensor,
) -> Result<(bool, f32, f32)> {
    let Some(previous) = previous else {
        return Ok((false, 0.0, 0.0));
    };
    if previous.identity != current_identity || previous.image.dims() != current_image.dims() {
        return Ok((false, 0.0, 0.0));
    }
    let delta = current_image.sub(&previous.image)?;
    Ok((true, mean_abs(&delta)?, tensor_rms(&delta)?))
}

#[allow(clippy::too_many_arguments)]
fn metric_record(
    world: &WorldState,
    sample: &TargetSample,
    optimizer_update: u64,
    core_trained: bool,
    macro_updates: usize,
    episode_started: bool,
    micro_movement_mean: f32,
    micro_movement_max: f32,
    macro_movement_mean: f32,
    macro_movement_max: f32,
    micro_state: &StateDiagnostics,
    macro_state: &StateDiagnostics,
    image_delta_valid: bool,
    image_delta_mean: f32,
    image_delta_rms: f32,
    image: &crate::metrics::ImageDiagnostics,
    loss: [f32; 21],
    optimizer: &OptimizerStats,
    reference_fidelity: f32,
    interface_memory_rms: f32,
    stability_violation: bool,
    window_seconds: f64,
    development_steps_per_second: f64,
    config: &RunConfig,
    decomposition: &DecompositionDiagnostics,
    grounding_schedule: f32,
    emergence_schedule: f32,
    detail: Option<&crate::corpus::DetailObservation>,
    micro_reference_drive_rms: f32,
    macro_reference_drive_rms: f32,
    flow: Option<&FlowLossOutput>,
) -> MetricRecord {
    let objective = match config.objective {
        ObjectiveMode::Endpoint => "endpoint",
        ObjectiveMode::ReconstructionPlus => "reconstruction-plus",
        ObjectiveMode::Flow => "flow",
        ObjectiveMode::HybridFlow => "hybrid-flow",
    };
    let flow_values = flow.map_or((false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), |value| {
        (
            true,
            value.loss_value,
            value.time,
            value.interpolant_rms,
            value.predicted_velocity_rms,
            value.target_velocity_rms,
            value.velocity_cosine,
            value.one_step_endpoint_l1,
            value.condition_rms,
        )
    });
    MetricRecord {
        step: world.step,
        age: world.age,
        episode: world.episode,
        target_index: sample.index,
        optimizer_update,
        core_trained,
        macro_updates,
        episode_started,
        micro_movement_mean,
        micro_movement_max,
        macro_movement_mean,
        macro_movement_max,
        micro_state_rms: micro_state.rms,
        macro_state_rms: macro_state.rms,
        micro_state_mean_abs: micro_state.mean_abs,
        macro_state_mean_abs: macro_state.mean_abs,
        micro_clamp_fraction: micro_state.clamp_fraction,
        macro_clamp_fraction: macro_state.clamp_fraction,
        micro_channel_rms_min: micro_state.channel_rms_min,
        micro_channel_rms_max: micro_state.channel_rms_max,
        macro_channel_rms_min: macro_state.channel_rms_min,
        macro_channel_rms_max: macro_state.channel_rms_max,
        image_delta_valid,
        image_delta_mean,
        image_delta_rms,
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
        gradient_rms: optimizer.gradient_rms,
        gradient_clip_scale: optimizer.clip_scale,
        updated_variables: optimizer.updated_variables,
        updated_parameters: optimizer.updated_parameters,
        effective_learning_rate: optimizer.effective_learning_rate,
        window_seconds,
        reference_fidelity,
        interface_memory_rms,
        muon_variables: optimizer.muon_variables,
        loss_state: loss[6],
        loss_memory: loss[7],
        core_gradient_rms: optimizer.core_gradient_rms,
        decoder_gradient_rms: optimizer.decoder_gradient_rms,
        core_updated_parameters: optimizer.core_updated_parameters,
        decoder_updated_parameters: optimizer.decoder_updated_parameters,
        development_steps_per_second,
        stability_violation,
        objective,
        supervision: if detail.is_some() { "crop" } else { "global" },
        active_morph_depth: world.morph_active_depth,
        physical_morph_layers: config.morph_layers,
        morph_generation: world.morph_generation,
        grounding_schedule,
        emergence_schedule,
        detail_zoom: detail.map_or(1.0, |value| value.zoom),
        pyramid_level: detail.map_or(0, |value| value.pyramid_level),
        micro_reference_drive_rms,
        macro_reference_drive_rms,
        loss_endpoint: loss[8],
        loss_grounding: loss[9],
        loss_ground_coarse: loss[10],
        loss_ground_mid: loss[11],
        loss_ground_fine: loss[12],
        loss_ssim: loss[13],
        loss_emergent_fit: loss[14],
        loss_emergent_low: loss[15],
        loss_emergent_tv: loss[16],
        head_redundancy: loss[17],
        cross_resolution_l1: loss[18],
        cross_resolution_low: loss[19],
        cross_resolution_edge: loss[20],
        grounded_output_rms: decomposition.grounded_rms,
        grounded_output_mean_abs: decomposition.grounded_mean_abs,
        grounded_output_variance: decomposition.grounded.variance,
        emergent_output_rms: decomposition.emergent_rms,
        emergent_output_mean_abs: decomposition.emergent_mean_abs,
        emergent_output_variance: decomposition.emergent_variance,
        emergent_contribution_rms: decomposition.contribution_rms,
        grounded_gradient_rms: optimizer.grounded_gradient_rms,
        emergent_gradient_rms: optimizer.emergent_gradient_rms,
        flow_gradient_rms: optimizer.flow_gradient_rms,
        grounded_update_rms: optimizer.grounded_update_rms,
        emergent_update_rms: optimizer.emergent_update_rms,
        grounded_update_weight_ratio: optimizer.grounded_update_weight_ratio,
        emergent_update_weight_ratio: optimizer.emergent_update_weight_ratio,
        flow_active: flow_values.0,
        flow_loss: flow_values.1,
        flow_time: flow_values.2,
        flow_interpolant_rms: flow_values.3,
        flow_pred_velocity_rms: flow_values.4,
        flow_target_velocity_rms: flow_values.5,
        flow_velocity_cosine: flow_values.6,
        flow_one_step_endpoint_l1: flow_values.7,
        flow_condition_rms: flow_values.8,
    }
}

fn loss_scalars(loss: &LossOutput) -> Result<[f32; 21]> {
    let values = Tensor::stack(
        &[
            &loss.total,
            &loss.content,
            &loss.palette,
            &loss.structure,
            &loss.seam,
            &loss.gamut,
            &loss.state,
            &loss.memory,
            &loss.endpoint,
            &loss.grounding,
            &loss.ground_coarse,
            &loss.ground_mid,
            &loss.ground_fine,
            &loss.ssim,
            &loss.emergent_fit,
            &loss.emergent_low,
            &loss.emergent_tv,
            &loss.head_redundancy,
            &loss.cross_resolution,
            &loss.cross_resolution_low,
            &loss.cross_resolution_edge,
        ],
        0,
    )?
    .to_vec1::<f32>()?;
    Ok(values
        .try_into()
        .expect("twenty-one loss tensors were stacked"))
}

fn ensure_render_plan(
    config: &RunConfig,
    resolution: usize,
    device: &Device,
    plan: &mut Option<RenderPlan>,
) -> Result<()> {
    if plan.is_none() {
        *plan = Some(RenderPlan::new(config, resolution, device)?);
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
) -> Result<(Vec<PathBuf>, bool)> {
    let mut outputs = Vec::with_capacity(config.gallery * 2);
    for variant in 0..config.gallery {
        if stop_requested() {
            println!(
                "Stop requested before gallery variant {}/{}; publishing completed outputs.",
                variant + 1,
                config.gallery
            );
            return Ok((outputs, true));
        }
        println!(
            "Rendering gallery variant {}/{} ({} development steps)...",
            variant + 1,
            config.gallery,
            config.gallery_steps + variant * config.gallery_stride
        );
        let genome = corpus.gallery_genome(variant, config.gallery_seed);
        let genome_tensor = Tensor::from_vec(genome, config.genome_dim, device)?;
        let variant_seed = splitmix64(config.seed ^ config.gallery_seed ^ variant as u64);
        let mut world = WorldState::fresh(config, variant_seed, device)?;
        let development_steps = config.gallery_steps + variant * config.gallery_stride;
        for _ in 0..development_steps {
            world = dynamics
                .step(&world, &genome_tensor, None, None, 0.0, false)?
                .world;
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
    Ok((outputs, stop_requested()))
}

fn zero_initialized_parameter(name: &str) -> bool {
    name.ends_with(".bias")
        || name.contains("micro_ca.output")
        || name.contains("renderer.emergent.weight")
        || name.contains("macro_ca.output")
        || name.contains("micro_write.weight")
        || name.contains("macro_write.weight")
        || name.contains("attention_output.weight")
        || name.contains("feedforward_contract.weight")
        || (name.contains(".morphic_") && name.contains(".contract.weight"))
}

fn unit_initialized_parameter(name: &str) -> bool {
    name.contains("norm.weight")
}

fn deterministic_initialize(varmap: &VarMap, seed: u64) -> Result<()> {
    let data = varmap.data().lock().expect("VarMap mutex poisoned");
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for name in names {
        let variable = &data[name];
        let dims = variable.dims();
        let count = variable.elem_count();
        let zero_initialized = zero_initialized_parameter(name);
        let unit_initialized = unit_initialized_parameter(name);
        let small_color_head = name.contains("renderer.grounded.weight");
        let mut rng = ChaCha8Rng::seed_from_u64(parameter_seed(seed, name));
        let values = if zero_initialized {
            vec![0.0f32; count]
        } else if unit_initialized {
            vec![1.0f32; count]
        } else if name.ends_with(".weight") {
            let fan_in = *dims.get(1).unwrap_or(&1);
            let stdev = if small_color_head {
                0.04
            } else {
                (2.0f32 / fan_in as f32).sqrt()
            };
            normal_values(&mut rng, count, stdev)
        } else {
            vec![0.0f32; count]
        };
        let initialized = Tensor::from_vec(values, variable.shape().clone(), variable.device())?;
        variable.set(&initialized)?;
    }
    Ok(())
}

fn learned_parameter_count(varmap: &VarMap) -> usize {
    varmap
        .data()
        .lock()
        .expect("VarMap mutex poisoned")
        .values()
        .map(|variable| variable.elem_count())
        .sum()
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

fn reference_fidelity(config: &RunConfig, step: u64) -> f32 {
    match config.conditioning {
        ConditioningMode::Generate => 0.0,
        ConditioningMode::Reconstruct => config.reference_fidelity_max,
        ConditioningMode::Hybrid => {
            let key = splitmix64(config.seed ^ step.wrapping_mul(0xd1b5_4a32_d192_ed03));
            let dropout = (key >> 40) as f32 / (1u32 << 24) as f32;
            if dropout < config.reference_dropout {
                return 0.0;
            }
            let unit = (splitmix64(key ^ 0xa17e_51d5) >> 40) as f32 / (1u32 << 24) as f32;
            config.reference_fidelity_min
                + unit * (config.reference_fidelity_max - config.reference_fidelity_min)
        }
    }
}

fn install_interrupt_handler() -> Result<()> {
    STOP_REQUESTED.store(false, Ordering::SeqCst);
    match INTERRUPT_HANDLER.get_or_init(|| {
        ctrlc::set_handler(|| STOP_REQUESTED.store(true, Ordering::SeqCst))
            .map_err(|error| error.to_string())
    }) {
        Ok(()) => Ok(()),
        Err(error) => bail!("cannot install Ctrl-C handler: {error}"),
    }
}

fn stop_requested() -> bool {
    STOP_REQUESTED.load(Ordering::SeqCst)
}

fn cadence_due(previous_step: u64, current_step: u64, cadence: usize) -> bool {
    cadence > 0 && previous_step / (cadence as u64) < current_step / (cadence as u64)
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

fn snapshot_path(config: &RunConfig, world: &WorldState) -> PathBuf {
    config.output_dir.join(format!(
        "titan_image_snapshot_v9{}_{:09}_ep{:05}_age{:04}_target{:04}.png",
        config.suffix(),
        world.step,
        world.episode,
        world.age,
        world.target_index,
    ))
}

fn gallery_path(config: &RunConfig, variant: usize, mastered: bool) -> PathBuf {
    config.output_dir.join(format!(
        "titan_image_variant_v9{}_{:03}_{}.png",
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
            snapshot_resolution: 24,
            episode_steps: 4,
            bptt: 1,
            snapshot_every: 0,
            interface_grid: 3,
            interface_width: 32,
            interface_loops: 1,
            morph_layers: 2,
            morph_depth: 1,
            morph_growth: crate::config::MorphGrowthConfig {
                min_depth: 1,
                max_depth: 2,
                ..RunConfig::default().morph_growth
            },
            flow: crate::config::FlowConfig {
                hidden: 16,
                resolution: 16,
                ..RunConfig::default().flow
            },
            detail: crate::config::DetailConfig {
                probability: 0.0,
                resolution: 32,
                ..RunConfig::default().detail
            },
            analysis: crate::config::AnalysisConfig {
                emergence_gallery: false,
                ..RunConfig::default().analysis
            },
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
            "titan-image-v9-test-{}-{}",
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
        assert!(
            metadata["average_development_steps_per_second"]
                .as_f64()
                .unwrap_or(0.0)
                > 0.0
        );
        let metrics = std::fs::read_to_string(&paths.metrics)?;
        let mut rows = metrics.lines();
        let header = rows.next().expect("metrics header");
        let columns = header.split(',').count();
        assert!(header.contains("core_gradient_rms"));
        assert!(header.contains("development_steps_per_second"));
        for row in rows {
            assert_eq!(row.split(',').count(), columns, "CSV column mismatch");
        }
        let checkpoint_before = [
            std::fs::read(&paths.model)?,
            std::fs::read(&paths.optimizer)?,
            std::fs::read(&paths.world)?,
            std::fs::read(&paths.checkpoint_manifest)?,
        ];
        let metrics_before = std::fs::read(&paths.metrics)?;
        let metadata_before = std::fs::read(&paths.metadata)?;
        let mut analysis = config.clone();
        analysis.fresh = false;
        analysis.analysis.only = true;
        analysis.analysis.render_attribution = true;
        run(analysis)?;
        assert_eq!(checkpoint_before[0], std::fs::read(&paths.model)?);
        assert_eq!(checkpoint_before[1], std::fs::read(&paths.optimizer)?);
        assert_eq!(checkpoint_before[2], std::fs::read(&paths.world)?);
        assert_eq!(
            checkpoint_before[3],
            std::fs::read(&paths.checkpoint_manifest)?
        );
        assert_eq!(metrics_before, std::fs::read(&paths.metrics)?);
        assert_eq!(metadata_before, std::fs::read(&paths.metadata)?);
        assert!(paths.analysis.exists());
        assert!(paths.render_metadata.exists());
        assert!(paths.previous_checkpoint().checkpoint_complete());

        let probe_dir = root.join("probes");
        std::fs::create_dir_all(&probe_dir)?;
        image::RgbImage::from_fn(28, 20, |x, y| {
            image::Rgb([
                ((3 * x + 5 * y) % 256) as u8,
                ((11 * x + y) % 256) as u8,
                ((x + 13 * y) % 256) as u8,
            ])
        })
        .save(probe_dir.join("unseen-a.png"))?;
        image::RgbImage::from_fn(22, 30, |x, y| {
            image::Rgb([
                ((17 * x + 2 * y) % 256) as u8,
                ((x + 7 * y) % 256) as u8,
                ((5 * x + 19 * y) % 256) as u8,
            ])
        })
        .save(probe_dir.join("unseen-b.png"))?;
        let mut probe = config.clone();
        probe.fresh = false;
        probe.analysis.only = true;
        probe.analysis.render_attribution = false;
        probe.analysis.probe_dir = Some(probe_dir);
        probe.analysis.probe_ages = vec![1, 2];
        probe.analysis.probe_reference_fidelities = vec![1.0, 0.0];
        run(probe)?;
        assert_eq!(checkpoint_before[0], std::fs::read(&paths.model)?);
        assert_eq!(checkpoint_before[1], std::fs::read(&paths.optimizer)?);
        assert_eq!(checkpoint_before[2], std::fs::read(&paths.world)?);
        assert_eq!(
            checkpoint_before[3],
            std::fs::read(&paths.checkpoint_manifest)?
        );
        assert_eq!(metrics_before, std::fs::read(&paths.metrics)?);
        assert_eq!(metadata_before, std::fs::read(&paths.metadata)?);
        let probe_summary: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&paths.analysis)?)?;
        let probe_report = &probe_summary["natural_image_probe"];
        assert_eq!(probe_report["held_out_by_source_bytes"], true);
        assert_eq!(probe_report["weights_frozen"], true);
        assert_eq!(probe_report["optimizer_steps"], 0);
        assert_eq!(probe_report["target_count"], 2);
        assert_eq!(probe_report["output_count"], 8);
        assert!(probe_report["fixed_world_seed"].as_u64().is_some());
        assert_eq!(probe_report["ages"], serde_json::json!([1, 2]));
        assert_eq!(
            probe_report["reference_fidelities"],
            serde_json::json!([1.0, 0.0])
        );
        let targets = probe_report["targets"]
            .as_array()
            .expect("probe target array");
        assert_eq!(targets.len(), 2);
        let mut reference_free_outputs = Vec::new();
        for target in targets {
            let points = target["points"].as_array().expect("probe points array");
            assert_eq!(points.len(), 4);
            let mut target_reference_free_outputs = Vec::new();
            for point in points {
                let output = point["output"].as_str().expect("probe output path");
                assert!(std::path::Path::new(output).exists());
                if point["reference_fidelity"] == 0.0 {
                    assert_eq!(point["micro_reference_drive_rms"], 0.0);
                    assert_eq!(point["macro_reference_drive_rms"], 0.0);
                    target_reference_free_outputs.push(std::fs::read(output)?);
                }
            }
            reference_free_outputs.push(target_reference_free_outputs);
        }
        assert_eq!(reference_free_outputs[0], reference_free_outputs[1]);
        assert!(std::path::Path::new(
            probe_report["montage"]
                .as_str()
                .expect("probe montage path")
        )
        .exists());
        assert!(
            std::path::Path::new(probe_report["report"].as_str().expect("probe report path"))
                .exists()
        );

        let overlap_dir = root.join("overlapping-probes");
        std::fs::create_dir_all(&overlap_dir)?;
        std::fs::copy(
            config.corpus_dir.join("source.png"),
            overlap_dir.join("copied-source.png"),
        )?;
        let mut overlap_probe = config.clone();
        overlap_probe.fresh = false;
        overlap_probe.analysis.only = true;
        overlap_probe.analysis.probe_dir = Some(overlap_dir);
        overlap_probe.analysis.probe_ages = vec![1];
        overlap_probe.analysis.probe_reference_fidelities = vec![1.0];
        let overlap_error = run(overlap_probe)
            .expect_err("training/probe byte overlap must be rejected")
            .to_string();
        assert!(overlap_error.contains("overlap the training corpus by source bytes"));
        assert_eq!(checkpoint_before[0], std::fs::read(&paths.model)?);
        assert_eq!(checkpoint_before[1], std::fs::read(&paths.optimizer)?);
        assert_eq!(checkpoint_before[2], std::fs::read(&paths.world)?);

        std::fs::write(&paths.model, b"interrupted checkpoint component")?;
        let mut recovery = config.clone();
        recovery.fresh = false;
        recovery.analysis.only = true;
        recovery.analysis.render_attribution = false;
        run(recovery)?;
        let recovery_metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&paths.render_metadata)?)?;
        assert_eq!(recovery_metadata["checkpoint_recovered"], true);
        assert_eq!(recovery_metadata["completed_world_step"], 2);
        let recovered_manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&paths.checkpoint_manifest)?)?;
        assert_eq!(recovered_manifest["world_step"], 2);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn append_only_morph_graft_preserves_model_and_optimizer_prefix() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "titan-image-v9-graft-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        let corpus = root.join("corpus");
        let output = root.join("output");
        std::fs::create_dir_all(&corpus)?;
        image::RgbImage::from_fn(24, 24, |x, y| {
            image::Rgb([(x * 7) as u8, (y * 9) as u8, ((x + 2 * y) * 3) as u8])
        })
        .save(corpus.join("source.png"))?;
        let config = tiny_config(corpus, output);
        run(config.clone())?;
        let paths = ArtifactPaths::new(&config);
        let old_model = candle_core::safetensors::load(&paths.model, &Device::Cpu)?;
        let old_optimizer = candle_core::safetensors::load(&paths.optimizer, &Device::Cpu)?;

        let grown = RunConfig {
            fresh: false,
            steps: 0,
            morph_layers: 3,
            morph_growth: crate::config::MorphGrowthConfig {
                max_depth: 3,
                ..config.morph_growth.clone()
            },
            analysis: crate::config::AnalysisConfig {
                only: true,
                emergence_gallery: false,
                ..config.analysis.clone()
            },
            ..config.clone()
        };
        run(grown)?;
        let new_model = candle_core::safetensors::load(&paths.model, &Device::Cpu)?;
        let new_optimizer = candle_core::safetensors::load(&paths.optimizer, &Device::Cpu)?;
        for (name, tensor) in &old_model {
            if name.starts_with("checkpoint.") {
                continue;
            }
            assert_eq!(
                tensor.flatten_all()?.to_vec1::<f32>()?,
                new_model[name].flatten_all()?.to_vec1::<f32>()?,
                "model tensor changed during append graft: {name}"
            );
        }
        for (name, tensor) in &old_optimizer {
            if !name.starts_with("optimizer.m.") && !name.starts_with("optimizer.v.") {
                continue;
            }
            assert_eq!(
                tensor.flatten_all()?.to_vec1::<f32>()?,
                new_optimizer[name].flatten_all()?.to_vec1::<f32>()?,
                "optimizer moment changed during append graft: {name}"
            );
        }
        let world = candle_core::safetensors::load(&paths.world, &Device::Cpu)?;
        assert_eq!(world["world.morph_generation"].to_scalar::<i64>()?, 1);
        assert_eq!(
            world["world.morph_birth_generations"].to_vec1::<i64>()?,
            vec![0, 0, 1]
        );
        let events = std::fs::read_to_string(&paths.events)?;
        assert!(events.contains("\"event_type\":\"morphic_graft\""));
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
            let values = variable.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(
                values,
                other.flatten_all()?.to_vec1::<f32>()?,
                "parameter {name} differed"
            );
            if zero_initialized_parameter(name) {
                assert!(
                    values.iter().all(|value| *value == 0.0),
                    "parameter {name} was not zero initialized"
                );
            } else if unit_initialized_parameter(name) {
                assert!(
                    values.iter().all(|value| *value == 1.0),
                    "parameter {name} was not unit initialized"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn balanced_profile_includes_recurrent_interface_capacity() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig::default();
        let vars = VarMap::new();
        let builder = VarBuilder::from_varmap(&vars, DType::F32, &device);
        let _dynamics = DynamicsSystem::new(&config, builder.pp("dynamics"), &device)?;
        let _renderer = ImplicitRenderer::new(&config, builder.pp("renderer"))?;
        let count = learned_parameter_count(&vars);
        assert_eq!(count, 815_606);
        assert!((4.6..=4.9).contains(&(count as f64 / 172_595.0)));
        Ok(())
    }

    #[test]
    fn nominal_snapshot_cadence_uses_safe_window_boundaries() {
        assert!(!cadence_due(48, 48, 50));
        assert!(cadence_due(48, 52, 50));
        assert!(cadence_due(96, 100, 50));
        assert!(!cadence_due(100, 104, 50));
        assert!(!cadence_due(48, 52, 0));
    }

    #[test]
    fn temporal_image_delta_invalidates_global_to_crop_resolution_transition() -> Result<()> {
        let global_identity = ObservationIdentity {
            target_index: 7,
            resolution: 192,
            detail: false,
            fingerprint: 0x701,
        };
        let previous = CachedObservation {
            identity: global_identity,
            image: Tensor::zeros((1, 3, 192, 192), DType::F32, &Device::Cpu)?,
        };
        let crop_identity = ObservationIdentity {
            target_index: 7,
            resolution: 128,
            detail: true,
            fingerprint: 0xc40,
        };
        let crop = Tensor::ones((1, 3, 128, 128), DType::F32, &Device::Cpu)?;
        assert_eq!(
            temporal_image_delta(Some(&previous), crop_identity, &crop)?,
            (false, 0.0, 0.0)
        );

        let changed_global = Tensor::ones((1, 3, 192, 192), DType::F32, &Device::Cpu)?;
        let (valid, mean, rms) =
            temporal_image_delta(Some(&previous), global_identity, &changed_global)?;
        assert!(valid);
        assert_eq!(mean, 1.0);
        assert_eq!(rms, 1.0);

        // Shape remains a second safety rail even if a future caller creates a
        // malformed identity with the wrong resolution.
        assert_eq!(
            temporal_image_delta(Some(&previous), global_identity, &crop)?,
            (false, 0.0, 0.0)
        );
        Ok(())
    }

    #[test]
    fn initialized_core_stays_bounded_over_long_autonomous_rollout() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig {
            style: crate::config::StylePreset::PureNca,
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            genome_dim: 4,
            ca_hidden: 32,
            interface_grid: 3,
            interface_width: 32,
            interface_loops: 2,
            morph_layers: 3,
            morph_depth: 2,
            reaction_gain: 0.0,
            phase_gain: 0.0,
            fractal_gain: 0.0,
            quasiperiodic_gain: 0.0,
            cyclic_gain: 0.0,
            train_resolution: 24,
            output_resolution: 24,
            snapshot_resolution: 24,
            ..RunConfig::default()
        };
        let vars = VarMap::new();
        let dynamics = DynamicsSystem::new(
            &config,
            VarBuilder::from_varmap(&vars, DType::F32, &device).pp("dynamics"),
            &device,
        )?;
        deterministic_initialize(&vars, config.seed)?;
        let genome = Tensor::zeros(config.genome_dim, DType::F32, &device)?;
        let mut world = WorldState::fresh(&config, config.seed, &device)?;
        for _ in 0..256 {
            world = dynamics
                .step(&world, &genome, None, None, 0.0, false)?
                .world;
        }
        let micro = state_metrics(&world.micro, config.state_limit)?;
        let macro_field = state_metrics(&world.macro_field, config.state_limit)?;
        assert!(micro.clamp_fraction < 1e-4);
        assert!(macro_field.clamp_fraction < 1e-4);
        assert!(world.memory.abs()?.max_all()?.to_scalar::<f32>()? < config.memory_limit);
        Ok(())
    }
}
