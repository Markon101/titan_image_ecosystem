use crate::config::{RunConfig, SCHEMA_VERSION};
use crate::optimizer::PersistentAdamW;
use crate::state::WorldState;
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
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CheckpointManifest {
    schema_version: u32,
    world_step: u64,
    config_signature: u64,
    corpus_fingerprint: u64,
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
            model: in_output("titan_image_model_v8", "safetensors"),
            optimizer: in_output("titan_image_optimizer_v8", "safetensors"),
            world: in_output("titan_image_world_v8", "safetensors"),
            checkpoint_manifest: in_output("titan_image_checkpoint_v8", "json"),
            metrics: in_output("titan_image_metrics_v8", "csv"),
            metadata: in_output("titan_image_run_metadata_v8", "json"),
            render_metadata: in_output("titan_image_render_metadata_v8", "json"),
            raw: in_output("titan_image_raw_v8", "png"),
            mastered: in_output("titan_image_mastered_v8", "png"),
            gallery: in_output("titan_image_gallery_v8", "png"),
            micro_state: in_output("titan_image_micro_state_v8", "png"),
            macro_state: in_output("titan_image_macro_state_v8", "png"),
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
}

pub fn save_checkpoint(
    paths: &ArtifactPaths,
    varmap: &VarMap,
    optimizer: &PersistentAdamW,
    world: &WorldState,
    config: &RunConfig,
    corpus_fingerprint: u64,
) -> Result<()> {
    let signature = config.checkpoint_signature();
    save_model(
        &paths.model,
        varmap,
        world.step,
        signature,
        corpus_fingerprint,
        world.micro.device(),
    )?;
    save_world(&paths.world, world, signature, corpus_fingerprint)?;
    optimizer.save(&paths.optimizer, world.step, world.micro.device())?;
    // Publish the manifest last. A crash between tensor renames leaves a
    // detectable generation mismatch rather than a silently mixed checkpoint.
    write_json_atomic(
        &paths.checkpoint_manifest,
        &CheckpointManifest {
            schema_version: SCHEMA_VERSION,
            world_step: world.step,
            config_signature: signature,
            corpus_fingerprint,
        },
    )?;
    Ok(())
}

pub fn load_checkpoint(
    paths: &ArtifactPaths,
    varmap: &mut VarMap,
    optimizer: &mut PersistentAdamW,
    device: &Device,
    config: &RunConfig,
    corpus_fingerprint: u64,
) -> Result<(WorldState, usize)> {
    if !paths.checkpoint_complete() {
        bail!(
            "incomplete v8 checkpoint set in {}; use --fresh or restore model, optimizer, world, and checkpoint manifest",
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
    let signature = config.checkpoint_signature();
    if manifest.schema_version != SCHEMA_VERSION
        || manifest.config_signature != signature
        || manifest.corpus_fingerprint != corpus_fingerprint
    {
        bail!(
            "checkpoint manifest does not match v8 architecture, training settings, or corpus bytes; use the original inputs or a new --run-tag with --fresh"
        );
    }
    let world = load_world(&paths.world, device, config, signature, corpus_fingerprint)?;
    if world.step != manifest.world_step {
        bail!(
            "checkpoint manifest/world mismatch: manifest {}, world {}",
            manifest.world_step,
            world.step
        );
    }
    verify_model_markers(
        &paths.model,
        world.step,
        signature,
        corpus_fingerprint,
        device,
    )?;
    varmap
        .load(&paths.model)
        .with_context(|| format!("cannot load model {}", paths.model.display()))?;
    let moments = optimizer.load(&paths.optimizer, world.step, device)?;
    Ok((world, moments))
}

fn save_model(
    path: &Path,
    varmap: &VarMap,
    world_step: u64,
    config_signature: u64,
    corpus_fingerprint: u64,
    device: &Device,
) -> Result<()> {
    let data = varmap.data().lock().expect("VarMap mutex poisoned");
    let mut tensors: HashMap<String, Tensor> = data
        .iter()
        .map(|(name, variable)| (name.clone(), variable.as_tensor().clone()))
        .collect();
    tensors.insert(
        "checkpoint.schema".to_owned(),
        Tensor::new(SCHEMA_VERSION as i64, device)?,
    );
    tensors.insert(
        "checkpoint.world_step".to_owned(),
        Tensor::new(world_step as i64, device)?,
    );
    tensors.insert(
        "checkpoint.config_signature".to_owned(),
        Tensor::new(config_signature as i64, device)?,
    );
    tensors.insert(
        "checkpoint.corpus_fingerprint".to_owned(),
        Tensor::new(corpus_fingerprint as i64, device)?,
    );
    atomic_safetensors(&tensors, path)
}

fn verify_model_markers(
    path: &Path,
    expected_world_step: u64,
    expected_signature: u64,
    expected_corpus: u64,
    device: &Device,
) -> Result<()> {
    let tensors = candle_core::safetensors::load(path, device)
        .with_context(|| format!("cannot inspect model {}", path.display()))?;
    let scalar = |name: &str| -> Result<u64> {
        Ok(tensors
            .get(name)
            .with_context(|| format!("model missing {name}"))?
            .to_scalar::<i64>()? as u64)
    };
    if scalar("checkpoint.schema")? != SCHEMA_VERSION as u64 {
        bail!("model schema is not v{SCHEMA_VERSION}");
    }
    let model_step = scalar("checkpoint.world_step")?;
    if model_step != expected_world_step {
        bail!("model/world mismatch: model {model_step}, world {expected_world_step}");
    }
    if scalar("checkpoint.config_signature")? != expected_signature
        || scalar("checkpoint.corpus_fingerprint")? != expected_corpus
    {
        bail!("model checkpoint settings or corpus do not match the requested run");
    }
    Ok(())
}

fn save_world(
    path: &Path,
    world: &WorldState,
    config_signature: u64,
    corpus_fingerprint: u64,
) -> Result<()> {
    let device = world.micro.device();
    let mut tensors = HashMap::new();
    tensors.insert("world.micro".to_owned(), world.micro.detach());
    tensors.insert("world.macro".to_owned(), world.macro_field.detach());
    tensors.insert("world.memory".to_owned(), world.memory.detach());
    for (name, value) in [
        ("world.schema", SCHEMA_VERSION as u64),
        ("world.step", world.step),
        ("world.age", world.age),
        ("world.episode", world.episode),
        ("world.target_index", world.target_index as u64),
        ("world.config_signature", config_signature),
        ("world.corpus_fingerprint", corpus_fingerprint),
    ] {
        tensors.insert(name.to_owned(), Tensor::new(value as i64, device)?);
    }
    atomic_safetensors(&tensors, path)
}

fn load_world(
    path: &Path,
    device: &Device,
    config: &RunConfig,
    expected_signature: u64,
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
    if schema != SCHEMA_VERSION {
        bail!("world schema {schema} is incompatible with v{SCHEMA_VERSION}");
    }
    if scalar("world.config_signature")? != expected_signature {
        bail!("checkpoint configuration does not match requested training settings");
    }
    if scalar("world.corpus_fingerprint")? != expected_corpus {
        bail!("checkpoint corpus bytes do not match the current source set");
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
        bail!("world field shape does not match the requested v8 architecture");
    }
    Ok(WorldState {
        micro,
        macro_field,
        step: scalar("world.step")?,
        memory,
        age: scalar("world.age")?,
        episode: scalar("world.episode")?,
        target_index: scalar("world.target_index")? as usize,
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
