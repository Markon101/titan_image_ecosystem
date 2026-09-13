use anyhow::Result;
use candle_core::{DType, Device};
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
use titan_image::{
    config::{DetailConfig, MorphGrowthConfig},
    experiment::NormTraining,
    initialization,
    persistence::ArtifactPaths,
    run,
    run_lease::RunLease,
    training_fork::{self, ForkRequest, OptimizerPolicy, WorldPolicy},
    RunConfig, TrainingMode,
};

fn fixture() -> Result<(PathBuf, RunConfig)> {
    let root = std::env::temp_dir().join(format!(
        "titan-controlled-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    let corpus = root.join("corpus");
    std::fs::create_dir_all(&corpus)?;
    image::RgbImage::from_fn(24, 24, |x, y| {
        image::Rgb([(x * 9) as u8, (y * 9) as u8, 100])
    })
    .save(corpus.join("source.png"))?;
    let mut c = RunConfig {
        corpus_dir: corpus,
        output_dir: root.join("initial"),
        run_tag: Some("initial".into()),
        mode: TrainingMode::Single,
        steps: 4,
        threads: 1,
        micro_size: 24,
        macro_size: 12,
        channels: 12,
        genome_dim: 4,
        ca_hidden: 32,
        render_hidden: 32,
        render_blocks: 1,
        coord_bands: 2,
        interface_grid: 3,
        interface_width: 32,
        interface_loops: 1,
        morph_layers: 2,
        morph_depth: 1,
        train_resolution: 24,
        output_resolution: 24,
        snapshot_resolution: 24,
        episode_steps: 8,
        bptt: 1,
        core_update_every: 2,
        gallery: 0,
        snapshot_every: 0,
        checkpoint_every: 0,
        save_state_atlas: false,
        morph_growth: MorphGrowthConfig {
            min_depth: 1,
            max_depth: 2,
            ..RunConfig::default().morph_growth
        },
        detail: DetailConfig {
            probability: 0.,
            resolution: 32,
            ..RunConfig::default().detail
        },
        ..Default::default()
    };
    c.analysis.emergence_gallery = false;
    c.experiment.optimizer_diagnostics = true;
    Ok((root, c))
}

fn tensors_equal(a: &Path, b: &Path) -> Result<()> {
    let a = candle_core::safetensors::load(a, &Device::Cpu)?;
    let b = candle_core::safetensors::load(b, &Device::Cpu)?;
    assert_eq!(
        a.keys().collect::<BTreeSet<_>>(),
        b.keys().collect::<BTreeSet<_>>()
    );
    for (name, x) in &a {
        if name.ends_with(".checkpoint_id")
            || name.ends_with(".immutable_signature")
            || name.ends_with(".resolved_signature")
        {
            continue;
        }
        let y = &b[name];
        assert_eq!(x.dims(), y.dims(), "{name}");
        assert_eq!(x.dtype(), y.dtype(), "{name}");
        match x.dtype() {
            DType::F32 => assert_eq!(
                x.flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                y.flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "{name}"
            ),
            DType::I64 => assert_eq!(
                x.flatten_all()?.to_vec1::<i64>()?,
                y.flatten_all()?.to_vec1::<i64>()?,
                "{name}"
            ),
            other => panic!("unexpected checkpoint dtype {other:?}"),
        }
    }
    Ok(())
}

fn equal_checkpoints(a: &RunConfig, b: &RunConfig) -> Result<()> {
    let a = ArtifactPaths::new(a);
    let b = ArtifactPaths::new(b);
    tensors_equal(&a.model, &b.model)?;
    tensors_equal(&a.world, &b.world)?;
    tensors_equal(&a.optimizer, &b.optimizer)
}

fn child(parent: &RunConfig, name: &str, norm: NormTraining, every: usize) -> Result<RunConfig> {
    let mut c = parent.clone();
    c.output_dir = parent.output_dir.parent().unwrap().join(name);
    c.run_tag = Some(name.into());
    c.detail.cache_dir = Some(c.output_dir.join("cache"));
    c.experiment.norm = norm;
    c.experiment.write_diagnostics_every = every;
    training_fork::create(&ForkRequest {
        parent_metadata: ArtifactPaths::new(parent).metadata,
        destination: c.clone(),
        optimizer: OptimizerPolicy::Retain,
        world: WorldPolicy::Retain,
        warmup: "retain".into(),
    })?;
    Ok(c)
}

#[test]
fn initialization_is_untrained_strict_and_shared_across_norm_arms() -> Result<()> {
    let (root, c) = fixture()?;
    let initial = initialization::create(c.clone())?;
    assert_eq!(initial["world_step"], 0);
    assert_eq!(initial["optimizer_updates"], 0);
    let parent: RunConfig = serde_json::from_value(initial["config"].clone())?;
    assert!(!parent.fresh);
    let before = training_fork::checkpoint_hashes(&ArtifactPaths::new(&parent))?;
    assert!(initialization::create(c).is_err());
    let legacy = child(&parent, "legacy", NormTraining::Legacy, 0)?;
    let diff = child(&parent, "diff", NormTraining::Differentiable, 0)?;
    equal_checkpoints(&parent, &legacy)?;
    equal_checkpoints(&parent, &diff)?;
    assert_ne!(legacy.checkpoint_signature(), diff.checkpoint_signature());
    run(diff.clone())?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ArtifactPaths::new(&diff).metadata)?)?;
    assert_eq!(metadata["resumed"], true);
    assert_eq!(metadata["start_world_step"], 0);
    assert_eq!(metadata["completed_world_step"], 4);
    assert_eq!(metadata["full_core_windows"], 2);
    assert_eq!(
        before,
        training_fork::checkpoint_hashes(&ArtifactPaths::new(&parent))?
    );
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn sampled_diagnostics_preserve_training_and_identify_resumes() -> Result<()> {
    let (root, c) = fixture()?;
    let initial = initialization::create(c)?;
    let parent: RunConfig = serde_json::from_value(initial["config"].clone())?;
    let off = child(&parent, "off", NormTraining::Differentiable, 0)?;
    let on = child(&parent, "on", NormTraining::Differentiable, 1)?;
    assert_eq!(off.checkpoint_signature(), on.checkpoint_signature());
    run(off.clone())?;
    run(on.clone())?;
    equal_checkpoints(&off, &on)?;
    let paths = ArtifactPaths::new(&on);
    let first: serde_json::Value = serde_json::from_slice(&std::fs::read(&paths.metadata)?)?;
    let diagnostics = on.output_dir.join("write_diagnostics_on.jsonl");
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(&diagnostics)?
        .lines()
        .map(serde_json::from_str)
        .collect::<std::result::Result<_, _>>()?;
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["world_step"], 0);
    assert_eq!(rows[1]["world_step"], 2);
    for row in rows {
        assert_eq!(row["invocation_id"], first["invocation_id"]);
        assert!(row["writes"]["micro"]["mean_tanh_derivative"]
            .as_f64()
            .unwrap()
            .is_finite());
        assert!(row["writes"]["macro_field"]["spatial_write_variance"]
            .as_f64()
            .unwrap()
            .is_finite());
    }
    run(on.clone())?;
    let second: serde_json::Value = serde_json::from_slice(&std::fs::read(&paths.metadata)?)?;
    assert_ne!(first["invocation_id"], second["invocation_id"]);
    let text = std::fs::read_to_string(diagnostics)?;
    assert_eq!(text.lines().count(), 4);
    let last: serde_json::Value = serde_json::from_str(text.lines().last().unwrap())?;
    assert_eq!(last["invocation_id"], second["invocation_id"]);
    let optimizer_rows =
        std::fs::read_to_string(on.output_dir.join("training_diagnostics_on.jsonl"))?;
    let last: serde_json::Value = serde_json::from_str(optimizer_rows.lines().last().unwrap())?;
    assert_eq!(last["invocation_id"], second["invocation_id"]);
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn another_process_cannot_train_or_analyze_an_owned_run() -> Result<()> {
    let (root, c) = fixture()?;
    initialization::create(c.clone())?;
    let before = training_fork::checkpoint_hashes(&ArtifactPaths::new(&c))?;
    let lease = RunLease::acquire(&c, "integration test")?;
    for extra in [None, Some("--analysis-only")] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_titan_image"));
        command
            .arg("--config-json")
            .arg(c.output_dir.join("config.json"));
        if let Some(arg) = extra {
            command.arg(arg);
        }
        let output = command.output()?;
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("namespace is busy"));
        assert_eq!(
            before,
            training_fork::checkpoint_hashes(&ArtifactPaths::new(&c))?
        );
    }
    drop(lease);
    drop(RunLease::acquire(&c, "released")?);
    std::fs::remove_dir_all(root)?;
    Ok(())
}
