//! Publish an untrained, resumable checkpoint in a new directory, without training.
use crate::{
    config::RunConfig,
    corpus::ImageCorpus,
    dynamics::DynamicsSystem,
    flow::RectifiedFlowRenderer,
    optimizer::PersistentAdamW,
    persistence::{self, ArtifactPaths},
    render::ImplicitRenderer,
    run_lease::RunLease,
    state::WorldState,
    training_fork::checkpoint_hashes,
};
use anyhow::{ensure, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use serde_json::{json, Value};
use std::path::Path;

pub fn create(mut config: RunConfig) -> Result<Value> {
    config.validate()?;
    ensure!(
        !config.render_only && !config.analysis.only,
        "initialize requires a training configuration"
    );
    // A published initialization must resume safely on every later invocation.
    config.fresh = false;
    config.detail.cache_dir = Some(config.output_dir.join("cache"));
    let parent = config
        .output_dir
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    std::fs::create_dir(&config.output_dir)?; // Never overwrite an existing run.
    let lease = RunLease::acquire(&config, "initialize")?;
    let marker = config.output_dir.join(".fork-incomplete");
    std::fs::write(&marker, b"initialization in progress; do not resume")?;
    let device = Device::Cpu;
    let mut corpus = ImageCorpus::new(&config, &device)?;
    let mut vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
    let _dynamics = DynamicsSystem::new(&config, vb.pp("dynamics"), &device)?;
    let _renderer = ImplicitRenderer::new(&config, vb.pp("renderer"))?;
    let _flow = RectifiedFlowRenderer::new(&config, vb.pp("flow"))?;
    crate::engine::deterministic_initialize(&vars, config.seed)?;
    let mut optimizer = PersistentAdamW::new(&vars, &config)?;
    let sample = corpus.sample(0, &device)?;
    let mut world = WorldState::fresh(&config, config.seed ^ sample.fingerprint, &device)?;
    world.target_index = sample.index;
    let paths = ArtifactPaths::new(&config);
    persistence::save_checkpoint(
        &paths,
        &vars,
        &optimizer,
        &world,
        &config,
        corpus.fingerprint(),
    )?;
    let (loaded, _) = persistence::load_checkpoint_read_only(
        &paths,
        &mut vars,
        &mut optimizer,
        &device,
        &config,
        corpus.fingerprint(),
    )?;
    ensure!(
        loaded.step == 0 && loaded.age == 0 && optimizer.updates() == 0,
        "initialization unexpectedly advanced world or optimizer"
    );
    let report = json!({"schema_version":crate::config::SCHEMA_VERSION,
        "operation":"untrained_initialization", "invocation_id":lease.invocation_id,
        "config":config, "world_step":loaded.step,"world_age":loaded.age,
        "optimizer_updates":optimizer.updates(), "completed_development_steps":0,
        "checkpoint_hashes_sha256":checkpoint_hashes(&paths)?,
        "build":{"commit":env!("TITAN_BUILD_COMMIT"),"dirty":env!("TITAN_BUILD_DIRTY")},
        "interpretation":"one deterministic initial model/world/optimizer; no training or evaluation"});
    persistence::write_json_atomic(&paths.metadata, &report)?;
    persistence::write_json_atomic(&config.output_dir.join("initialization.json"), &report)?;
    persistence::write_json_atomic(&config.output_dir.join("config.json"), &config)?;
    std::fs::remove_file(marker)?;
    Ok(report)
}
