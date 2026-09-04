use crate::benchmark::{
    reconstruction_reference_metrics, target_reference_metrics, TargetReferenceMetrics,
};
use crate::config::{RunConfig, TrainingMode, SCHEMA_VERSION};
use crate::corpus::ImageCorpus;
use crate::dynamics::DynamicsSystem;
use crate::metrics::{image_metrics, state_metrics, tensor_rms};
use crate::persistence::write_json_atomic;
use crate::render::{save_contact_sheet_resized, save_png, ImplicitRenderer, RenderPlan};
use crate::state::WorldState;
use crate::tensor_ops::splitmix64;
use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use serde::Serialize;
use std::collections::HashSet;

const MAX_PROBE_IMAGES: usize = 256;
const MAX_PROBE_OUTPUTS: usize = 1024;

#[derive(Clone, Debug, Serialize)]
pub struct NaturalImageProbePoint {
    pub reference_fidelity: f32,
    pub age: usize,
    pub emergence_schedule: f32,
    pub output: String,
    pub raw_l1: f32,
    pub raw_l2: f32,
    pub coarse_spatial_l1: f32,
    pub edge_l1: f32,
    pub palette_mean_l1: f32,
    pub image_variance: f32,
    pub image_edge_energy: f32,
    pub seam_energy: f32,
    pub red_mean: f32,
    pub green_mean: f32,
    pub blue_mean: f32,
    pub micro_state_rms: f32,
    pub macro_state_rms: f32,
    pub memory_rms: f32,
    pub micro_near_bound_fraction: f32,
    pub macro_near_bound_fraction: f32,
    pub micro_reference_drive_rms: f32,
    pub macro_reference_drive_rms: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct NaturalImageProbeTarget {
    pub index: usize,
    pub name: String,
    pub source_fingerprint: String,
    pub source_width: u32,
    pub source_height: u32,
    pub reference: TargetReferenceMetrics,
    pub points: Vec<NaturalImageProbePoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct NaturalImageProbeReport {
    pub schema_version: u32,
    pub checkpoint_world_step: u64,
    pub training_corpus_fingerprint: String,
    pub probe_corpus_fingerprint: String,
    pub probe_dir: String,
    pub held_out_by_source_bytes: bool,
    pub weights_frozen: bool,
    pub optimizer_steps: usize,
    pub fixed_world_seed: u64,
    pub world_policy: &'static str,
    pub genome_policy: &'static str,
    pub reference_zero_policy: &'static str,
    pub resolution: usize,
    pub ages: Vec<usize>,
    pub reference_fidelities: Vec<f32>,
    pub target_count: usize,
    pub output_count: usize,
    pub montage: String,
    pub report: String,
    pub interpretation_rule: &'static str,
    pub targets: Vec<NaturalImageProbeTarget>,
}

pub fn run_natural_image_probes(
    config: &RunConfig,
    training_corpus: &ImageCorpus,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    checkpoint_world: &WorldState,
    device: &Device,
) -> Result<NaturalImageProbeReport> {
    let probe_dir = config
        .analysis
        .probe_dir
        .as_ref()
        .expect("probe runner requires --probe-dir");
    let mut probe_config = config.clone();
    probe_config.corpus_dir = probe_dir.clone();
    probe_config.mode = TrainingMode::Family;
    probe_config.detail.cache_dir = Some(
        config
            .output_dir
            .join(format!("probe_pyramid_cache_v9{}", config.suffix())),
    );
    let mut probe_corpus = ImageCorpus::new(&probe_config, device)?;
    if probe_corpus.len() > MAX_PROBE_IMAGES {
        bail!(
            "probe directory contains {} images; maximum is {MAX_PROBE_IMAGES}",
            probe_corpus.len()
        );
    }
    let requested_outputs = probe_corpus
        .len()
        .saturating_mul(config.analysis.probe_ages.len())
        .saturating_mul(config.analysis.probe_reference_fidelities.len());
    if requested_outputs > MAX_PROBE_OUTPUTS {
        bail!("probe sweep requests {requested_outputs} outputs; maximum is {MAX_PROBE_OUTPUTS}");
    }

    let training_manifest = training_corpus.source_manifest();
    let probe_manifest = probe_corpus.source_manifest();
    let training_fingerprints: HashSet<&str> = training_manifest
        .iter()
        .map(|source| source.fingerprint.as_str())
        .collect();
    let overlaps: Vec<&str> = probe_manifest
        .iter()
        .filter_map(|source| {
            training_fingerprints
                .contains(source.fingerprint.as_str())
                .then_some(source.fingerprint.as_str())
        })
        .collect();
    if !overlaps.is_empty() {
        bail!(
            "probe images overlap the training corpus by source bytes (fingerprints: {}); use genuinely held-out files",
            overlaps.join(", ")
        );
    }
    let unique_probe_fingerprints: HashSet<&str> = probe_manifest
        .iter()
        .map(|source| source.fingerprint.as_str())
        .collect();
    if unique_probe_fingerprints.len() != probe_manifest.len() {
        bail!(
            "probe directory contains duplicate image bytes; remove duplicates before evaluation"
        );
    }

    let plan = RenderPlan::new(config, config.train_resolution, device)?;
    let genome = Tensor::zeros(config.genome_dim, DType::F32, device)?;
    let fixed_world_seed = splitmix64(config.seed ^ 0x7072_6f62_655f_7639);
    let mut targets = Vec::with_capacity(probe_corpus.len());
    let mut output_paths = Vec::with_capacity(requested_outputs);
    for index in 0..probe_corpus.len() {
        let sample = probe_corpus.sample_index(index, device)?;
        let reference = target_reference_metrics(sample.name.clone(), &sample.image)?;
        let mut points = Vec::with_capacity(
            config.analysis.probe_ages.len() * config.analysis.probe_reference_fidelities.len(),
        );
        for &reference_fidelity in &config.analysis.probe_reference_fidelities {
            let mut world = WorldState::fresh(config, fixed_world_seed, device)?;
            world.target_index = index;
            world.morph_active_depth = checkpoint_world.morph_active_depth;
            world.morph_generation = checkpoint_world.morph_generation;
            world.morph_birth_generations = checkpoint_world.morph_birth_generations.clone();
            let max_age = *config
                .analysis
                .probe_ages
                .last()
                .expect("validated nonempty probe ages");
            for age in 1..=max_age {
                let (reference_micro, reference_macro) = if reference_fidelity == 0.0 {
                    (None, None)
                } else {
                    (Some(&sample.reference_micro), Some(&sample.reference_macro))
                };
                let advanced = dynamics.step(
                    &world,
                    &genome,
                    reference_micro,
                    reference_macro,
                    reference_fidelity,
                    false,
                )?;
                let micro_reference_drive_rms = advanced.micro_reference_drive_rms;
                let macro_reference_drive_rms = advanced.macro_reference_drive_rms;
                world = advanced.world;
                if !config.analysis.probe_ages.contains(&age) {
                    continue;
                }
                let age_phase =
                    (age as f32 / config.developmental_horizon().max(1) as f32).clamp(0.0, 1.0);
                let (_, emergence_schedule) = config.developmental_schedule(age_phase);
                let rendered = renderer.render_with_emergence(
                    &world.micro,
                    &world.macro_field,
                    &genome,
                    &plan,
                    emergence_schedule,
                    false,
                )?;
                let fidelity_code = (reference_fidelity * 1000.0).round() as u16;
                let output_path = config.output_dir.join(format!(
                    "titan_image_probe_v9{}_{index:03}_{:016x}_f{fidelity_code:04}_a{age:04}.png",
                    config.suffix(),
                    sample.fingerprint,
                ));
                save_png(&rendered.image, &output_path)?;
                let reconstruction =
                    reconstruction_reference_metrics(&rendered.image, &sample.image)?;
                let image = image_metrics(&rendered.image)?;
                let micro = state_metrics(&world.micro, config.state_limit)?;
                let macro_field = state_metrics(&world.macro_field, config.state_limit)?;
                points.push(NaturalImageProbePoint {
                    reference_fidelity,
                    age,
                    emergence_schedule,
                    output: output_path.display().to_string(),
                    raw_l1: reconstruction.raw_l1,
                    raw_l2: reconstruction.raw_l2,
                    coarse_spatial_l1: reconstruction.coarse_spatial_l1,
                    edge_l1: reconstruction.edge_l1,
                    palette_mean_l1: reconstruction.palette_mean_l1,
                    image_variance: image.variance,
                    image_edge_energy: image.edge,
                    seam_energy: image.seam,
                    red_mean: image.means[0],
                    green_mean: image.means[1],
                    blue_mean: image.means[2],
                    micro_state_rms: micro.rms,
                    macro_state_rms: macro_field.rms,
                    memory_rms: tensor_rms(&world.memory)?,
                    micro_near_bound_fraction: micro.clamp_fraction,
                    macro_near_bound_fraction: macro_field.clamp_fraction,
                    micro_reference_drive_rms,
                    macro_reference_drive_rms,
                });
                output_paths.push(output_path);
            }
        }
        targets.push(NaturalImageProbeTarget {
            index,
            name: sample.name,
            source_fingerprint: format!("{:016x}", sample.fingerprint),
            source_width: sample.source_width,
            source_height: sample.source_height,
            reference,
            points,
        });
    }

    let montage_path = config.output_dir.join(format!(
        "titan_image_probe_montage_v9{}.png",
        config.suffix()
    ));
    save_contact_sheet_resized(&output_paths, &montage_path, 192)?;
    let report_path = config.output_dir.join(format!(
        "titan_image_probe_report_v9{}.json",
        config.suffix()
    ));
    let report = NaturalImageProbeReport {
        schema_version: SCHEMA_VERSION,
        checkpoint_world_step: checkpoint_world.step,
        training_corpus_fingerprint: format!("{:016x}", training_corpus.fingerprint()),
        probe_corpus_fingerprint: format!("{:016x}", probe_corpus.fingerprint()),
        probe_dir: probe_dir.display().to_string(),
        held_out_by_source_bytes: true,
        weights_frozen: true,
        optimizer_steps: 0,
        fixed_world_seed,
        world_policy: "same fresh deterministic world seed per target and reference fidelity; checkpoint anatomy retained",
        genome_policy: "fixed zero genome for every target to isolate reference-driven transfer",
        reference_zero_policy: "fidelity zero supplies no reference tensors and no local reference drive",
        resolution: config.train_resolution,
        ages: config.analysis.probe_ages.clone(),
        reference_fidelities: config.analysis.probe_reference_fidelities.clone(),
        target_count: targets.len(),
        output_count: output_paths.len(),
        montage: montage_path.display().to_string(),
        report: report_path.display().to_string(),
        interpretation_rule: "Held-out reconstruction diagnostics measure transfer behavior; they do not by themselves establish generalization beyond this probe set.",
        targets,
    };
    write_json_atomic(&report_path, &report)?;
    Ok(report)
}
