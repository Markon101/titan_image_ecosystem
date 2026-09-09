use crate::config::{MorphDepthMode, RunConfig, SCHEMA_VERSION};
use crate::optimizer::{OptimizerMigrationReport, PersistentAdamW, OPTIMIZER_LAYOUT_VERSION};
use crate::state::WorldState;
use crate::tensor_ops::splitmix64;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct ArtifactPaths {
    pub model: PathBuf,
    pub optimizer: PathBuf,
    pub world: PathBuf,
    pub checkpoint_manifest: PathBuf,
    pub metrics: PathBuf,
    pub metadata: PathBuf,
    pub render_metadata: PathBuf,
    pub raw: PathBuf,
    pub mastered: PathBuf,
    pub gallery: PathBuf,
    pub micro_state: PathBuf,
    pub macro_state: PathBuf,
    pub events: PathBuf,
    pub analysis: PathBuf,
    pub model_stats: PathBuf,
    pub graft_analysis: PathBuf,
    pub attractor_analysis: PathBuf,
    pub perturbation_analysis: PathBuf,
    pub flow_trajectory: PathBuf,
    pub flow_sample: PathBuf,
    pub grounded: PathBuf,
    pub emergent: PathBuf,
    pub decomposition: PathBuf,
    pub emergence_frontier: PathBuf,
    pub resolution_ladder: PathBuf,
    pub target_comparison: PathBuf,
    pub benchmark: PathBuf,
    pub comparison: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CheckpointManifest {
    schema_version: u32,
    world_step: u64,
    checkpoint_id: u64,
    immutable_signature: u64,
    resolved_signature: u64,
    corpus_fingerprint: u64,
    morph_layers: usize,
    active_morph_depth: usize,
    morph_generation: u64,
    morph_birth_generations: Vec<u64>,
    morph_depth_mode: MorphDepthMode,
    morph_min_depth: usize,
    morph_max_depth: usize,
    morph_growth_interval: usize,
    morph_plateau_window: usize,
    morph_plateau_epsilon_bits: u32,
    morph_seam_threshold_bits: u32,
    optimizer_layout_version: u32,
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct ModelMigrationReport {
    pub grafted: bool,
    pub old_morph_layers: usize,
    pub new_morph_layers: usize,
    pub copied_tensors: usize,
    pub old_active_depth: usize,
    pub new_active_depth: usize,
    pub copied_parameters: usize,
    pub new_tensors: usize,
    pub new_parameters: usize,
    pub new_parameter_names: Vec<String>,
    pub resized_tensors: Vec<String>,
    pub skipped_tensors: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct CheckpointLoadReport {
    pub checkpoint_id: u64,
    pub recovered_previous: bool,
    pub model: ModelMigrationReport,
    pub optimizer_moments_preserved: usize,
    pub optimizer_moments_new: usize,
    pub optimizer_updates_preserved: u64,
}

impl ArtifactPaths {
    pub fn new(config: &RunConfig) -> Self {
        let suffix = config.suffix();
        let in_output = |name: &str, extension: &str| {
            config
                .output_dir
                .join(format!("{name}{suffix}.{extension}"))
        };
        Self {
            model: in_output("titan_image_model_v9", "safetensors"),
            optimizer: in_output("titan_image_optimizer_v9", "safetensors"),
            world: in_output("titan_image_world_v9", "safetensors"),
            checkpoint_manifest: in_output("titan_image_checkpoint_v9", "json"),
            metrics: in_output("titan_image_metrics_v9", "csv"),
            metadata: in_output("titan_image_run_metadata_v9", "json"),
            render_metadata: in_output("titan_image_render_metadata_v9", "json"),
            raw: in_output("titan_image_raw_v9", "png"),
            mastered: in_output("titan_image_mastered_v9", "png"),
            gallery: in_output("titan_image_gallery_v9", "png"),
            micro_state: in_output("titan_image_micro_state_v9", "png"),
            macro_state: in_output("titan_image_macro_state_v9", "png"),
            events: in_output("titan_image_events_v9", "jsonl"),
            analysis: in_output("titan_image_analysis_v9", "json"),
            model_stats: in_output("titan_image_model_stats_v9", "json"),
            graft_analysis: in_output("titan_image_graft_analysis_v9", "json"),
            attractor_analysis: in_output("titan_image_attractor_analysis_v9", "json"),
            perturbation_analysis: in_output("titan_image_perturbation_analysis_v9", "json"),
            flow_trajectory: in_output("titan_image_flow_trajectory_v9", "json"),
            flow_sample: in_output("titan_image_flow_sample_v9", "png"),
            grounded: in_output("titan_image_grounded_v9", "png"),
            emergent: in_output("titan_image_emergent_v9", "png"),
            decomposition: in_output("titan_image_decomposition_v9", "png"),
            emergence_frontier: in_output("titan_image_emergence_frontier_v9", "json"),
            resolution_ladder: in_output("titan_image_resolution_ladder_v9", "png"),
            target_comparison: in_output("titan_image_target_comparison_v9", "png"),
            benchmark: in_output("titan_image_benchmark_v9", "json"),
            comparison: in_output("titan_image_v8_v9_comparison", "json"),
        }
    }

    pub fn checkpoint_exists(&self) -> bool {
        self.model.exists()
            || self.optimizer.exists()
            || self.world.exists()
            || self.checkpoint_manifest.exists()
    }

    pub fn checkpoint_complete(&self) -> bool {
        self.model.exists()
            && self.optimizer.exists()
            && self.world.exists()
            && self.checkpoint_manifest.exists()
    }

    pub fn previous_checkpoint(&self) -> Self {
        let previous = |path: &Path| {
            let name = path
                .file_name()
                .and_then(|value| value.to_str())
                .expect("artifact path has a UTF-8 filename");
            path.with_file_name(format!("{name}.previous"))
        };
        let mut output = self.clone();
        output.model = previous(&self.model);
        output.optimizer = previous(&self.optimizer);
        output.world = previous(&self.world);
        output.checkpoint_manifest = previous(&self.checkpoint_manifest);
        output
    }

    pub fn recoverable_checkpoint_complete(&self) -> bool {
        self.checkpoint_complete() || self.previous_checkpoint().checkpoint_complete()
    }

    pub fn recoverable_checkpoint_exists(&self) -> bool {
        self.checkpoint_exists() || self.previous_checkpoint().checkpoint_exists()
    }
}

pub fn save_checkpoint(
    paths: &ArtifactPaths,
    varmap: &VarMap,
    optimizer: &PersistentAdamW,
    world: &WorldState,
    config: &RunConfig,
    corpus_fingerprint: u64,
) -> Result<u64> {
    if paths.checkpoint_complete() {
        publish_checkpoint_set(paths, &paths.previous_checkpoint())?;
    }
    let immutable_signature = config.checkpoint_signature();
    let resolved_signature = config.resolved_config_signature();
    let checkpoint_id = splitmix64(
        immutable_signature
            ^ resolved_signature.rotate_left(13)
            ^ world.step.rotate_left(27)
            ^ world.morph_generation.rotate_left(41)
            ^ (world.morph_active_depth as u64).rotate_left(53),
    );
    save_model(
        &paths.model,
        varmap,
        world.step,
        checkpoint_id,
        immutable_signature,
        resolved_signature,
        corpus_fingerprint,
        world.micro.device(),
    )?;
    save_world(
        &paths.world,
        world,
        checkpoint_id,
        immutable_signature,
        resolved_signature,
        corpus_fingerprint,
    )?;
    optimizer.save(
        &paths.optimizer,
        world.step,
        checkpoint_id,
        world.micro.device(),
    )?;
    write_json_atomic(
        &paths.checkpoint_manifest,
        &CheckpointManifest {
            schema_version: SCHEMA_VERSION,
            world_step: world.step,
            checkpoint_id,
            immutable_signature,
            resolved_signature,
            corpus_fingerprint,
            morph_layers: config.morph_layers,
            active_morph_depth: world.morph_active_depth,
            morph_generation: world.morph_generation,
            morph_birth_generations: world.morph_birth_generations.clone(),
            optimizer_layout_version: OPTIMIZER_LAYOUT_VERSION,
            morph_depth_mode: config.morph_growth.mode,
            morph_min_depth: config.morph_growth.min_depth,
            morph_max_depth: config.morph_growth.max_depth,
            morph_growth_interval: config.morph_growth.interval,
            morph_plateau_window: config.morph_growth.plateau_window,
            morph_plateau_epsilon_bits: config.morph_growth.plateau_epsilon.to_bits(),
            morph_seam_threshold_bits: config.morph_growth.seam_threshold.to_bits(),
        },
    )?;
    Ok(checkpoint_id)
}

pub fn load_checkpoint(
    paths: &ArtifactPaths,
    varmap: &mut VarMap,
    optimizer: &mut PersistentAdamW,
    device: &Device,
    config: &RunConfig,
    corpus_fingerprint: u64,
) -> Result<(WorldState, CheckpointLoadReport)> {
    match load_checkpoint_read_only(paths, varmap, optimizer, device, config, corpus_fingerprint) {
        Ok(result) => Ok(result),
        Err(primary_error) => {
            let previous = paths.previous_checkpoint();
            if !previous.checkpoint_complete() {
                return Err(primary_error);
            }
            match load_checkpoint_read_only(
                &previous,
                varmap,
                optimizer,
                device,
                config,
                corpus_fingerprint,
            ) {
                Ok((world, mut report)) => {
                    publish_checkpoint_set(&previous, paths)?;
                    report.recovered_previous = true;
                    Ok((world, report))
                }
                Err(previous_error) => Err(primary_error.context(format!(
                    "previous checkpoint generation was also unusable: {previous_error:#}"
                ))),
            }
        }
    }
}

pub(crate) fn load_checkpoint_read_only(
    paths: &ArtifactPaths,
    varmap: &mut VarMap,
    optimizer: &mut PersistentAdamW,
    device: &Device,
    config: &RunConfig,
    corpus_fingerprint: u64,
) -> Result<(WorldState, CheckpointLoadReport)> {
    if !paths.checkpoint_complete() {
        bail!(
            "incomplete v9 checkpoint set in {}; use --fresh or restore model, optimizer, world, and checkpoint manifest",
            paths.model.parent().unwrap_or(Path::new(".")).display()
        );
    }
    let manifest: CheckpointManifest =
        serde_json::from_slice(&std::fs::read(&paths.checkpoint_manifest).with_context(|| {
            format!(
                "cannot read checkpoint manifest {}",
                paths.checkpoint_manifest.display()
            )
        })?)?;
    if manifest.schema_version != SCHEMA_VERSION
        || manifest.immutable_signature != config.checkpoint_signature()
        || manifest.corpus_fingerprint != corpus_fingerprint
        || manifest.optimizer_layout_version != OPTIMIZER_LAYOUT_VERSION
    {
        bail!(
            "checkpoint manifest does not match v9 immutable evolution settings, optimizer layout, or corpus bytes"
        );
    }
    if manifest.morph_depth_mode != config.morph_growth.mode
        || manifest.morph_min_depth != config.morph_growth.min_depth
        || config.morph_growth.max_depth < manifest.morph_max_depth
        || manifest.morph_growth_interval != config.morph_growth.interval
        || manifest.morph_plateau_window != config.morph_growth.plateau_window
        || manifest.morph_plateau_epsilon_bits != config.morph_growth.plateau_epsilon.to_bits()
        || manifest.morph_seam_threshold_bits != config.morph_growth.seam_threshold.to_bits()
    {
        bail!("morph policy drift is not an approved v9 anatomy mutation");
    }
    if config.morph_layers < manifest.morph_layers {
        bail!("v9 supports append-only morph-layer growth, not physical layer removal");
    }
    if manifest.morph_birth_generations.len() != manifest.morph_layers {
        bail!("checkpoint morph birth registry does not match its anatomy");
    }
    let mut model = strict_load_model(
        &paths.model,
        varmap,
        device,
        config,
        &manifest,
        corpus_fingerprint,
    )?;
    let mut world = load_world(&paths.world, device, config, &manifest, corpus_fingerprint)?;
    if world.step != manifest.world_step {
        bail!(
            "checkpoint manifest/world mismatch: manifest {}, world {}",
            manifest.world_step,
            world.step
        );
    }
    model.old_active_depth = manifest.active_morph_depth;
    model.new_active_depth = world.morph_active_depth;
    let optimizer_report: OptimizerMigrationReport = optimizer.load_migrating(
        &paths.optimizer,
        world.step,
        manifest.checkpoint_id,
        &model.new_parameter_names,
        device,
    )?;
    if model.grafted {
        world.morph_generation = manifest.morph_generation + 1;
        world
            .morph_birth_generations
            .resize(config.morph_layers, world.morph_generation);
    }
    Ok((
        world,
        CheckpointLoadReport {
            checkpoint_id: manifest.checkpoint_id,
            recovered_previous: false,
            optimizer_moments_preserved: optimizer_report.preserved_moment_pairs,
            optimizer_moments_new: optimizer_report.new_moment_pairs,
            optimizer_updates_preserved: optimizer_report.updates_preserved,
            model,
        },
    ))
}

fn publish_checkpoint_set(source: &ArtifactPaths, destination: &ArtifactPaths) -> Result<()> {
    if !source.checkpoint_complete() {
        bail!("cannot publish an incomplete checkpoint generation");
    }
    let pairs = [
        (&source.model, &destination.model),
        (&source.optimizer, &destination.optimizer),
        (&source.world, &destination.world),
        (
            &source.checkpoint_manifest,
            &destination.checkpoint_manifest,
        ),
    ];
    let mut staged = Vec::with_capacity(pairs.len());
    for (source_path, destination_path) in pairs {
        let name = destination_path
            .file_name()
            .and_then(|value| value.to_str())
            .context("checkpoint destination has no UTF-8 filename")?;
        let stage = destination_path.with_file_name(format!("{name}.staging"));
        if stage.exists() {
            std::fs::remove_file(&stage)?;
        }
        if std::fs::hard_link(source_path, &stage).is_err() {
            std::fs::copy(source_path, &stage)?;
        }
        staged.push((stage, destination_path));
    }
    // Components are installed first; the manifest is the commit marker.
    for (stage, destination_path) in &staged[..3] {
        std::fs::rename(stage, destination_path)?;
    }
    std::fs::rename(&staged[3].0, staged[3].1)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn save_model(
    path: &Path,
    varmap: &VarMap,
    world_step: u64,
    checkpoint_id: u64,
    immutable_signature: u64,
    resolved_signature: u64,
    corpus_fingerprint: u64,
    device: &Device,
) -> Result<()> {
    let data = varmap.data().lock().expect("VarMap mutex poisoned");
    let mut tensors: HashMap<String, Tensor> = data
        .iter()
        .map(|(name, variable)| (name.clone(), variable.as_tensor().clone()))
        .collect();
    for (name, value) in [
        ("checkpoint.schema", SCHEMA_VERSION as u64),
        ("checkpoint.world_step", world_step),
        ("checkpoint.checkpoint_id", checkpoint_id),
        ("checkpoint.immutable_signature", immutable_signature),
        ("checkpoint.resolved_signature", resolved_signature),
        ("checkpoint.corpus_fingerprint", corpus_fingerprint),
    ] {
        tensors.insert(name.to_owned(), Tensor::new(value as i64, device)?);
    }
    atomic_safetensors(&tensors, path)
}

fn strict_load_model(
    path: &Path,
    varmap: &mut VarMap,
    device: &Device,
    config: &RunConfig,
    manifest: &CheckpointManifest,
    expected_corpus: u64,
) -> Result<ModelMigrationReport> {
    let tensors = candle_core::safetensors::load(path, device)
        .with_context(|| format!("cannot inspect model {}", path.display()))?;
    let scalar = |name: &str| -> Result<u64> {
        Ok(tensors
            .get(name)
            .with_context(|| format!("model missing {name}"))?
            .to_scalar::<i64>()? as u64)
    };
    if scalar("checkpoint.schema")? != SCHEMA_VERSION as u64
        || scalar("checkpoint.world_step")? != manifest.world_step
        || scalar("checkpoint.checkpoint_id")? != manifest.checkpoint_id
        || scalar("checkpoint.immutable_signature")? != manifest.immutable_signature
        || scalar("checkpoint.resolved_signature")? != manifest.resolved_signature
        || scalar("checkpoint.corpus_fingerprint")? != expected_corpus
    {
        bail!("model checkpoint markers do not match the committed manifest");
    }
    let data = varmap.data().lock().expect("VarMap mutex poisoned");
    let learned_saved: HashMap<&str, &Tensor> = tensors
        .iter()
        .filter_map(|(name, tensor)| {
            (!name.starts_with("checkpoint.")).then_some((name.as_str(), tensor))
        })
        .collect();
    for (name, saved) in &learned_saved {
        let current = data
            .get(*name)
            .with_context(|| format!("saved model tensor {name} was removed or renamed"))?;
        if saved.dims() != current.dims() || saved.dtype() != current.dtype() {
            bail!("saved model tensor shape/dtype mismatch for {name}");
        }
    }
    let mut report = ModelMigrationReport {
        grafted: config.morph_layers > manifest.morph_layers,
        old_morph_layers: manifest.morph_layers,
        new_morph_layers: config.morph_layers,
        old_active_depth: manifest.active_morph_depth,
        new_active_depth: manifest.active_morph_depth,
        ..ModelMigrationReport::default()
    };
    for (name, current) in data.iter() {
        if let Some(saved) = learned_saved.get(name.as_str()) {
            current.set(saved)?;
            report.copied_tensors += 1;
            report.copied_parameters += current.elem_count();
            continue;
        }
        let index = morphic_index(name).with_context(|| {
            format!("current model tensor {name} is missing from the checkpoint")
        })?;
        if index < manifest.morph_layers || index >= config.morph_layers {
            bail!("non-append or noncontiguous new morph tensor {name}");
        }
        if name.contains(".contract.") {
            let maximum = current.abs()?.max_all()?.to_scalar::<f32>()?;
            if maximum != 0.0 {
                bail!("new morph contract is not function-preserving for {name}");
            }
        }
        report.new_tensors += 1;
        report.new_parameters += current.elem_count();
        report.new_parameter_names.push(name.clone());
    }
    Ok(report)
}

fn morphic_index(name: &str) -> Option<usize> {
    let remainder = name.strip_prefix("dynamics.interface.morphic_")?;
    let (digits, suffix) = remainder.split_once('.')?;
    if digits.len() != 3 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    if !matches!(
        suffix,
        "norm.weight" | "expand.weight" | "expand.bias" | "contract.weight" | "contract.bias"
    ) {
        return None;
    }
    digits.parse().ok()
}

#[allow(clippy::too_many_arguments)]
fn save_world(
    path: &Path,
    world: &WorldState,
    checkpoint_id: u64,
    immutable_signature: u64,
    resolved_signature: u64,
    corpus_fingerprint: u64,
) -> Result<()> {
    let device = world.micro.device();
    let mut tensors = HashMap::new();
    tensors.insert("world.micro".to_owned(), world.micro.detach());
    tensors.insert("world.macro".to_owned(), world.macro_field.detach());
    tensors.insert("world.memory".to_owned(), world.memory.detach());
    tensors.insert(
        "world.morph_birth_generations".to_owned(),
        Tensor::from_vec(
            world
                .morph_birth_generations
                .iter()
                .map(|value| *value as i64)
                .collect::<Vec<_>>(),
            world.morph_birth_generations.len(),
            device,
        )?,
    );
    for (name, value) in [
        ("world.schema", SCHEMA_VERSION as u64),
        ("world.step", world.step),
        ("world.age", world.age),
        ("world.episode", world.episode),
        ("world.target_index", world.target_index as u64),
        ("world.checkpoint_id", checkpoint_id),
        ("world.immutable_signature", immutable_signature),
        ("world.resolved_signature", resolved_signature),
        ("world.corpus_fingerprint", corpus_fingerprint),
        ("world.morph_active_depth", world.morph_active_depth as u64),
        ("world.morph_generation", world.morph_generation),
        (
            "world.morph_physical_layers",
            world.morph_birth_generations.len() as u64,
        ),
    ] {
        tensors.insert(name.to_owned(), Tensor::new(value as i64, device)?);
    }
    atomic_safetensors(&tensors, path)
}

fn load_world(
    path: &Path,
    device: &Device,
    config: &RunConfig,
    manifest: &CheckpointManifest,
    expected_corpus: u64,
) -> Result<WorldState> {
    let tensors = candle_core::safetensors::load(path, device)
        .with_context(|| format!("cannot load world {}", path.display()))?;
    let scalar = |name: &str| -> Result<u64> {
        Ok(tensors
            .get(name)
            .with_context(|| format!("world missing {name}"))?
            .to_scalar::<i64>()? as u64)
    };
    let schema = scalar("world.schema")? as u32;
    if schema != SCHEMA_VERSION
        || scalar("world.checkpoint_id")? != manifest.checkpoint_id
        || scalar("world.immutable_signature")? != manifest.immutable_signature
        || scalar("world.resolved_signature")? != manifest.resolved_signature
        || scalar("world.corpus_fingerprint")? != expected_corpus
        || scalar("world.morph_physical_layers")? as usize != manifest.morph_layers
        || scalar("world.morph_active_depth")? as usize != manifest.active_morph_depth
        || scalar("world.morph_generation")? != manifest.morph_generation
    {
        bail!("world markers/anatomy do not match the committed v9 manifest");
    }
    let micro = tensors
        .get("world.micro")
        .context("world missing micro field")?
        .clone();
    let macro_field = tensors
        .get("world.macro")
        .context("world missing macro field")?
        .clone();
    let memory = tensors
        .get("world.memory")
        .context("world missing recurrent interface memory")?
        .clone();
    if micro.dims4()? != (1, config.channels, config.micro_size, config.micro_size)
        || macro_field.dims4()? != (1, config.channels, config.macro_size, config.macro_size)
        || memory.dims2()? != (1, config.interface_width)
    {
        bail!("world field shape does not match the requested v9 architecture");
    }
    let births: Vec<u64> = tensors
        .get("world.morph_birth_generations")
        .context("world missing morph birth registry")?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|value| value as u64)
        .collect();
    if births != manifest.morph_birth_generations {
        bail!("world morph birth registry disagrees with manifest");
    }
    let saved_active = scalar("world.morph_active_depth")? as usize;
    let active = match config.morph_growth.mode {
        crate::config::MorphDepthMode::Fixed => {
            if config.morph_depth < saved_active {
                bail!("active morph depth cannot decrease on resume");
            }
            config.morph_depth.max(saved_active)
        }
        crate::config::MorphDepthMode::Capacity => config.morph_growth.max_depth.max(saved_active),
        crate::config::MorphDepthMode::Adaptive => saved_active,
    };
    if active > config.morph_layers || active > config.morph_growth.max_depth {
        bail!("resolved active morph depth exceeds requested reserve capacity");
    }
    Ok(WorldState {
        micro,
        macro_field,
        step: scalar("world.step")?,
        memory,
        age: scalar("world.age")?,
        episode: scalar("world.episode")?,
        target_index: scalar("world.target_index")? as usize,
        morph_active_depth: active,
        morph_generation: scalar("world.morph_generation")?,
        morph_birth_generations: births,
    })
}

fn atomic_safetensors(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    let temporary = path.with_extension("safetensors.tmp");
    candle_core::safetensors::save(tensors, &temporary)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

pub fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(value)?;
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{Init, VarBuilder};

    fn persistence_test_path(label: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "titan-image-v9-persistence-{label}-{}-{nonce}.safetensors",
            std::process::id()
        ))
    }

    #[allow(clippy::field_reassign_with_default)]
    fn test_config(morph_layers: usize) -> RunConfig {
        let mut config = RunConfig::default();
        config.micro_size = 24;
        config.macro_size = 12;
        config.channels = 12;
        config.interface_width = 32;
        config.train_resolution = 24;
        config.output_resolution = 24;
        config.snapshot_resolution = 24;
        config.morph_layers = morph_layers;
        config.morph_depth = 1;
        config.morph_growth.min_depth = 1;
        config.morph_growth.max_depth = morph_layers;
        config
    }

    #[allow(clippy::too_many_arguments)]
    fn test_manifest(
        config: &RunConfig,
        step: u64,
        checkpoint_id: u64,
        corpus_fingerprint: u64,
        active_depth: usize,
        generation: u64,
        births: Vec<u64>,
    ) -> CheckpointManifest {
        CheckpointManifest {
            schema_version: SCHEMA_VERSION,
            world_step: step,
            checkpoint_id,
            immutable_signature: config.checkpoint_signature(),
            resolved_signature: config.resolved_config_signature(),
            corpus_fingerprint,
            morph_layers: config.morph_layers,
            active_morph_depth: active_depth,
            morph_generation: generation,
            morph_birth_generations: births,
            morph_depth_mode: config.morph_growth.mode,
            morph_min_depth: config.morph_growth.min_depth,
            morph_max_depth: config.morph_growth.max_depth,
            morph_growth_interval: config.morph_growth.interval,
            morph_plateau_window: config.morph_growth.plateau_window,
            morph_plateau_epsilon_bits: config.morph_growth.plateau_epsilon.to_bits(),
            morph_seam_threshold_bits: config.morph_growth.seam_threshold.to_bits(),
            optimizer_layout_version: OPTIMIZER_LAYOUT_VERSION,
        }
    }

    fn add_parameter(vars: &VarMap, name: &str, value: f64, device: &Device) -> Result<()> {
        let builder = VarBuilder::from_varmap(vars, DType::F32, device);
        let _ = builder.get_with_hints((2,), name, Init::Const(value))?;
        Ok(())
    }

    fn add_morphic_block(
        vars: &VarMap,
        index: usize,
        value: f64,
        contract_value: f64,
        device: &Device,
    ) -> Result<()> {
        let builder = VarBuilder::from_varmap(vars, DType::F32, device)
            .pp("dynamics")
            .pp("interface")
            .pp(format!("morphic_{index:03}"));
        let _ = builder.get_with_hints((2,), "norm.weight", Init::Const(value))?;
        let _ = builder.get_with_hints((4, 2), "expand.weight", Init::Const(value))?;
        let _ = builder.get_with_hints((4,), "expand.bias", Init::Const(value))?;
        let _ = builder.get_with_hints((2, 4), "contract.weight", Init::Const(contract_value))?;
        let _ = builder.get_with_hints((2,), "contract.bias", Init::Const(contract_value))?;
        Ok(())
    }

    fn parameter_values(vars: &VarMap, name: &str) -> Result<Vec<f32>> {
        let data = vars.data().lock().expect("VarMap mutex poisoned");
        Ok(data
            .get(name)
            .with_context(|| format!("missing test parameter {name}"))?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }

    #[test]
    fn model_append_copies_old_tensors_and_keeps_new_contracts_zero() -> Result<()> {
        let device = Device::Cpu;
        let old_config = test_config(1);
        let manifest = test_manifest(&old_config, 17, 29, 41, 1, 0, vec![0]);
        let old_vars = VarMap::new();
        add_parameter(&old_vars, "shared.weight", 0.75, &device)?;
        add_morphic_block(&old_vars, 0, 0.5, 0.25, &device)?;
        let path = persistence_test_path("append");
        save_model(
            &path,
            &old_vars,
            manifest.world_step,
            manifest.checkpoint_id,
            manifest.immutable_signature,
            manifest.resolved_signature,
            manifest.corpus_fingerprint,
            &device,
        )?;

        let new_config = test_config(2);
        let mut current = VarMap::new();
        add_parameter(&current, "shared.weight", -1.0, &device)?;
        add_morphic_block(&current, 0, -1.0, -1.0, &device)?;
        add_morphic_block(&current, 1, 0.9, 0.0, &device)?;
        let report = strict_load_model(
            &path,
            &mut current,
            &device,
            &new_config,
            &manifest,
            manifest.corpus_fingerprint,
        )?;
        assert!(report.grafted);
        assert_eq!(report.copied_tensors, 6);
        assert_eq!(report.new_tensors, 5);
        assert_eq!(report.copied_parameters, 26);
        assert_eq!(report.new_parameters, 24);
        assert_eq!(parameter_values(&current, "shared.weight")?, vec![0.75; 2]);
        assert_eq!(
            parameter_values(&current, "dynamics.interface.morphic_000.contract.weight")?,
            vec![0.25; 8]
        );
        assert!(
            parameter_values(&current, "dynamics.interface.morphic_001.contract.weight")?
                .iter()
                .all(|value| *value == 0.0)
        );
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn strict_model_load_rejects_unknown_missing_and_malformed_append_tensors() -> Result<()> {
        let device = Device::Cpu;
        let config = test_config(1);
        let manifest = test_manifest(&config, 3, 5, 7, 1, 0, vec![0]);

        let saved_with_unknown = VarMap::new();
        add_morphic_block(&saved_with_unknown, 0, 0.5, 0.0, &device)?;
        add_parameter(&saved_with_unknown, "retired.weight", 1.0, &device)?;
        let unknown_path = persistence_test_path("unknown");
        save_model(
            &unknown_path,
            &saved_with_unknown,
            3,
            5,
            manifest.immutable_signature,
            manifest.resolved_signature,
            7,
            &device,
        )?;
        let mut current = VarMap::new();
        add_morphic_block(&current, 0, -1.0, 0.0, &device)?;
        let error = strict_load_model(&unknown_path, &mut current, &device, &config, &manifest, 7)
            .unwrap_err()
            .to_string();
        assert!(error.contains("removed or renamed"), "{error}");
        std::fs::remove_file(unknown_path)?;

        let saved = VarMap::new();
        add_morphic_block(&saved, 0, 0.5, 0.0, &device)?;
        let missing_path = persistence_test_path("missing");
        save_model(
            &missing_path,
            &saved,
            3,
            5,
            manifest.immutable_signature,
            manifest.resolved_signature,
            7,
            &device,
        )?;
        let mut current = VarMap::new();
        add_morphic_block(&current, 0, -1.0, 0.0, &device)?;
        add_parameter(&current, "new_head.weight", 0.0, &device)?;
        let error = strict_load_model(&missing_path, &mut current, &device, &config, &manifest, 7)
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing from the checkpoint"), "{error}");

        let new_config = test_config(2);
        let mut malformed = VarMap::new();
        add_morphic_block(&malformed, 0, -1.0, 0.0, &device)?;
        add_parameter(
            &malformed,
            "dynamics.interface.morphic_001.evil.weight",
            0.0,
            &device,
        )?;
        let error = strict_load_model(
            &missing_path,
            &mut malformed,
            &device,
            &new_config,
            &manifest,
            7,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("missing from the checkpoint"), "{error}");
        std::fs::remove_file(missing_path)?;
        Ok(())
    }

    #[test]
    fn model_and_world_reject_mismatched_transaction_metadata() -> Result<()> {
        let device = Device::Cpu;
        let config = test_config(2);
        let mut world = WorldState::fresh(&config, 11, &device)?;
        world.step = 23;
        world.age = 7;
        world.episode = 2;
        world.target_index = 1;
        world.morph_active_depth = 1;
        world.morph_generation = 4;
        world.morph_birth_generations = vec![0, 4];
        let manifest = test_manifest(&config, 23, 31, 43, 1, 4, vec![0, 4]);

        let world_path = persistence_test_path("world-roundtrip");
        save_world(
            &world_path,
            &world,
            manifest.checkpoint_id,
            manifest.immutable_signature,
            manifest.resolved_signature,
            manifest.corpus_fingerprint,
        )?;
        let loaded = load_world(&world_path, &device, &config, &manifest, 43)?;
        assert_eq!(loaded.step, 23);
        assert_eq!(loaded.morph_generation, 4);
        assert_eq!(loaded.morph_birth_generations, vec![0, 4]);
        let mut tensors = candle_core::safetensors::load(&world_path, &device)?;
        tensors.insert(
            "world.checkpoint_id".to_owned(),
            Tensor::new(32i64, &device)?,
        );
        atomic_safetensors(&tensors, &world_path)?;
        let error = match load_world(&world_path, &device, &config, &manifest, 43) {
            Ok(_) => panic!("mismatched world transaction metadata was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("committed v9 manifest"), "{error}");
        std::fs::remove_file(world_path)?;

        let model_vars = VarMap::new();
        add_morphic_block(&model_vars, 0, 0.5, 0.0, &device)?;
        add_morphic_block(&model_vars, 1, 0.5, 0.0, &device)?;
        let model_path = persistence_test_path("model-signature");
        save_model(
            &model_path,
            &model_vars,
            23,
            31,
            manifest.immutable_signature,
            manifest.resolved_signature,
            43,
            &device,
        )?;
        let mut tensors = candle_core::safetensors::load(&model_path, &device)?;
        tensors.insert(
            "checkpoint.resolved_signature".to_owned(),
            Tensor::new((manifest.resolved_signature ^ 1) as i64, &device)?,
        );
        atomic_safetensors(&tensors, &model_path)?;
        let mut current = VarMap::new();
        add_morphic_block(&current, 0, -1.0, 0.0, &device)?;
        add_morphic_block(&current, 1, -1.0, 0.0, &device)?;
        let error = strict_load_model(&model_path, &mut current, &device, &config, &manifest, 43)
            .unwrap_err()
            .to_string();
        assert!(error.contains("committed manifest"), "{error}");
        std::fs::remove_file(model_path)?;
        Ok(())
    }
}
