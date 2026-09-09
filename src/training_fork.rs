//! Explicit checkpoint import. Normal resume validation is never relaxed.
use crate::{
    config::RunConfig,
    corpus::ImageCorpus,
    dynamics::DynamicsSystem,
    flow::RectifiedFlowRenderer,
    optimizer::PersistentAdamW,
    persistence::{self, ArtifactPaths},
    render::ImplicitRenderer,
    state::WorldState,
};
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerPolicy {
    Retain,
    Reset,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorldPolicy {
    Retain,
    Reseed,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForkRequest {
    pub parent_metadata: std::path::PathBuf,
    pub destination: RunConfig,
    pub optimizer: OptimizerPolicy,
    pub world: WorldPolicy,
    /// Retain the saved update count's warmup position, or reset with optimizer.
    /// New warmup schedules require a separate optimizer-reset experiment.
    pub warmup: String,
}

/// Uses the platform SHA-256 implementation, no shell or filename interpolation.
/// Failure is fatal; provenance never substitutes a weaker hash silently.
pub fn sha256(path: &Path) -> Result<String> {
    let result = std::process::Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()?;
    ensure!(
        result.status.success(),
        "sha256sum failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let output = String::from_utf8(result.stdout)?;
    let hash = output
        .split_whitespace()
        .next()
        .context("missing SHA-256")?;
    ensure!(
        hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()),
        "invalid SHA-256"
    );
    Ok(hash.to_owned())
}
pub fn checkpoint_hashes(paths: &ArtifactPaths) -> Result<BTreeMap<String, String>> {
    [
        ("model", &paths.model),
        ("world", &paths.world),
        ("optimizer", &paths.optimizer),
        ("manifest", &paths.checkpoint_manifest),
    ]
    .into_iter()
    .map(|(name, path)| Ok((name.to_owned(), sha256(path)?)))
    .collect()
}
pub fn differences(parent: &Value, child: &Value) -> BTreeMap<String, Value> {
    fn walk(path: String, a: &Value, b: &Value, out: &mut BTreeMap<String, Value>) {
        if a == b {
            return;
        }
        if let (Some(a), Some(b)) = (a.as_object(), b.as_object()) {
            let keys: std::collections::BTreeSet<_> = a.keys().chain(b.keys()).collect();
            for key in keys {
                walk(
                    if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    },
                    a.get(key).unwrap_or(&Value::Null),
                    b.get(key).unwrap_or(&Value::Null),
                    out,
                );
            }
        } else {
            out.insert(path, json!({"parent": a, "fork": b}));
        }
    }
    let mut out = BTreeMap::new();
    walk(String::new(), parent, child, &mut out);
    out
}

pub fn create(request: &ForkRequest) -> Result<Value> {
    let metadata: Value = serde_json::from_slice(&std::fs::read(&request.parent_metadata)?)?;
    let parent: RunConfig = serde_json::from_value(metadata["config"].clone())?;
    let child = &request.destination;
    parent.validate()?;
    child.validate()?;
    ensure!(
        !child.fresh && !child.render_only && !child.analysis.only,
        "fork destination must be a training configuration"
    );
    ensure!(
        child.run_tag.is_some() && child.run_tag != parent.run_tag,
        "fork requires a distinct run tag"
    );
    ensure!(
        !child.output_dir.try_exists()?,
        "fork destination must be a new directory"
    );
    ensure!(
        request.warmup
            == match request.optimizer {
                OptimizerPolicy::Retain => "retain",
                OptimizerPolicy::Reset => "reset",
            },
        "warmup must explicitly match optimizer retain/reset policy"
    );
    ensure!(
        parent.morph_layers == child.morph_layers
            && parent.initial_morph_depth() == child.initial_morph_depth(),
        "fork v1 does not combine anatomy grafting with training changes"
    );
    if request.optimizer == OptimizerPolicy::Retain {
        ensure!(
            parent.optimizer == child.optimizer,
            "retaining moments requires the same optimizer kind"
        );
        ensure!(
            parent.warmup_updates == child.warmup_updates,
            "retained warmup must keep its configured horizon"
        );
    }
    if request.world == WorldPolicy::Retain {
        ensure!(parent.seed == child.seed && parent.bptt == child.bptt
            && parent.episode_steps == child.episode_steps && parent.age_min == child.age_min
            && parent.age_max == child.age_max && parent.age_curriculum_steps == child.age_curriculum_steps
            && parent.mode == child.mode && parent.episode_reset == child.episode_reset,
            "retain world requires identical seed, target schedule, BPTT alignment and episode policy; use explicit reseed");
    }
    let parent_paths = ArtifactPaths::new(&parent);
    let before = checkpoint_hashes(&parent_paths)?;
    let device = Device::Cpu;
    // Corpus decoding may create caches, so use the new destination's namespace.
    // Reserve first; a failed import leaves an explicitly incomplete directory.
    std::fs::create_dir(&child.output_dir)?;
    std::fs::write(
        child.output_dir.join(".fork-incomplete"),
        b"import in progress; do not resume",
    )?;
    let mut corpus = ImageCorpus::new(child, &device)?;
    let mut vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
    let _dynamics = DynamicsSystem::new(child, vb.pp("dynamics"), &device)?;
    let _renderer = ImplicitRenderer::new(child, vb.pp("renderer"))?;
    let _flow = RectifiedFlowRenderer::new(child, vb.pp("flow"))?;
    let mut optimizer = PersistentAdamW::new(&vars, &parent)?;
    // Strict, read-only loader: NEVER invoke automatic previous-generation recovery,
    // which republishes recovered files into the parent directory.
    let (mut world, loaded) = persistence::load_checkpoint_read_only(
        &parent_paths,
        &mut vars,
        &mut optimizer,
        &device,
        &parent,
        corpus.fingerprint(),
    )?;
    ensure!(
        !loaded.model.grafted && loaded.model.new_tensors == 0,
        "fork requires identical parameter names and shapes"
    );
    let parent_world = json!({"step":world.step,"age":world.age,"episode":world.episode,
        "target_index":world.target_index,"active_depth":world.morph_active_depth});
    let updates = optimizer.updates();
    if request.optimizer == OptimizerPolicy::Reset {
        optimizer = PersistentAdamW::new(&vars, child)?;
    }
    if request.world == WorldPolicy::Reseed {
        let sample = corpus.sample(0, &device)?;
        let mut fresh = WorldState::fresh(child, child.seed ^ sample.fingerprint, &device)?;
        fresh.target_index = sample.index;
        fresh.morph_active_depth = world.morph_active_depth;
        fresh.morph_generation = world.morph_generation;
        fresh.morph_birth_generations = world.morph_birth_generations.clone();
        world = fresh;
    }
    ensure!(
        checkpoint_hashes(&parent_paths)? == before,
        "parent changed during import"
    );
    let paths = ArtifactPaths::new(child);
    persistence::save_checkpoint(
        &paths,
        &vars,
        &optimizer,
        &world,
        child,
        corpus.fingerprint(),
    )?;
    // Re-read under destination semantics (including its optimizer hyperparameters).
    let mut verify_optimizer = PersistentAdamW::new(&vars, child)?;
    persistence::load_checkpoint_read_only(
        &paths,
        &mut vars,
        &mut verify_optimizer,
        &device,
        child,
        corpus.fingerprint(),
    )?;
    ensure!(
        verify_optimizer.updates()
            == match request.optimizer {
                OptimizerPolicy::Retain => updates,
                OptimizerPolicy::Reset => 0,
            },
        "optimizer policy mismatch"
    );
    ensure!(
        checkpoint_hashes(&parent_paths)? == before,
        "parent changed during fork publication"
    );
    let report = json!({"version":1,"operation":"explicit_training_fork", "request":request,
        "parent_run_tag":parent.run_tag,"parent_world":parent_world,
        "parent_optimizer_updates":updates,"parent_hashes_sha256":before,
        "parent_metadata_sha256":sha256(&request.parent_metadata)?,
        "parent_checkpoint_manifest":serde_json::from_slice::<Value>(&std::fs::read(&parent_paths.checkpoint_manifest)?)?,
        "fork_hashes_sha256":checkpoint_hashes(&paths)?,
        "changed_config":differences(&serde_json::to_value(&parent)?, &serde_json::to_value(child)?),
        "build":{"commit":env!("TITAN_BUILD_COMMIT"),"dirty":env!("TITAN_BUILD_DIRTY")},
        "optimizer_updates":verify_optimizer.updates(), "warmup":request.warmup,
        "world_policy":request.world,"seed":child.seed,"training_budget_steps":child.steps,
        "normal_resume":"strict destination signature; parent recovery disabled during import"});
    persistence::write_json_atomic(&child.output_dir.join("fork.json"), &report)?;
    persistence::write_json_atomic(&child.output_dir.join("config.json"), child)?;
    std::fs::remove_file(child.output_dir.join(".fork-incomplete"))?;
    Ok(report)
}

pub fn cli(args: &[String]) -> Result<()> {
    if args == ["--help"] {
        println!("Usage: titan_image fork REQUEST.json\nRequest fields: parent_metadata, destination (complete RunConfig), optimizer (retain|reset), world (retain|reseed), warmup (retain|reset).\nCreates a new directory and checkpoint, but does not train. Then run --config-json DEST/config.json.");
        return Ok(());
    }
    ensure!(
        args.len() == 1,
        "fork expects one request JSON; see fork --help"
    );
    let request = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let report = create(&request)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Frozen CPU gradient reachability probe on a strictly validated checkpoint.
/// It advances a cloned world for one BPTT window and never calls optimizer.step.
pub fn gradient_probe(config: &RunConfig) -> Result<Value> {
    let device = Device::Cpu;
    let paths = ArtifactPaths::new(config);
    let before = checkpoint_hashes(&paths)?;
    let mut corpus = ImageCorpus::new(config, &device)?;
    let mut vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
    let mut cpu_config = config.clone();
    cpu_config.compute_backend = crate::ComputeBackend::Cpu;
    let dynamics = DynamicsSystem::new(&cpu_config, vb.pp("dynamics"), &device)?;
    let renderer = ImplicitRenderer::new(&cpu_config, vb.pp("renderer"))?;
    let _flow = RectifiedFlowRenderer::new(config, vb.pp("flow"))?;
    let mut optimizer = PersistentAdamW::new(&vars, config)?;
    let (mut world, _) = persistence::load_checkpoint_read_only(
        &paths,
        &mut vars,
        &mut optimizer,
        &device,
        config,
        corpus.fingerprint(),
    )?;
    let sample = corpus.sample_index(world.target_index, &device)?;
    let start = world.step;
    let mut penalty_sum: Option<candle_core::Tensor> = None;
    for _ in 0..config.bptt {
        let stepped = dynamics.step(
            &world,
            &sample.genome_tensor,
            Some(&sample.reference_micro),
            Some(&sample.reference_macro),
            config.reference_fidelity_max,
            true,
        )?;
        if let Some(p) = stepped.saturation_penalty {
            penalty_sum = Some(match penalty_sum {
                Some(sum) => sum.add(&p)?,
                None => p,
            });
        }
        world = stepped.world;
    }
    let plan = crate::render::RenderPlan::new(config, config.train_resolution, &device)?;
    let (grounding, emergence) = config
        .developmental_schedule((world.age as f32 / config.developmental_horizon() as f32).min(1.));
    let rendered = renderer.render_with_emergence(
        &world.micro,
        &world.macro_field,
        &sample.genome_tensor,
        &plan,
        emergence,
        true,
    )?;
    let mut losses = crate::objectives::visual_loss(
        &rendered,
        &sample.image,
        &world.micro,
        &world.macro_field,
        &world.memory,
        config,
        grounding,
        emergence,
        config.detail.boundary,
    )?;
    let saturation_penalty_loss = if let Some(sum) = penalty_sum {
        let penalty = sum.affine(1.0 / config.bptt as f64, 0.0)?;
        let scalar = penalty.to_scalar::<f32>()?;
        ensure!(scalar.is_finite(), "non-finite saturation penalty");
        losses.total = losses.total.add(&penalty)?;
        scalar
    } else {
        0.0
    };
    let gradients = losses.total.backward()?;
    let mut groups = BTreeMap::<String, crate::gradient_diagnostics::GroupStats>::new();
    let mut norms = BTreeMap::new();
    for (name, var) in vars.data().lock().unwrap().iter() {
        let gradient = gradients.get(var.as_tensor());
        if let Some(g) = gradient {
            ensure!(
                g.flatten_all()?
                    .to_vec1::<f32>()?
                    .iter()
                    .all(|v| v.is_finite()),
                "nonfinite gradient {name}"
            );
        }
        groups
            .entry(crate::gradient_diagnostics::group(name).to_owned())
            .or_default()
            .observe(var.as_tensor(), gradient)?;
        if name.contains("norm.weight") {
            norms.insert(name.clone(),json!({"reachable":gradient.is_some(),
                "l2":gradient.map(crate::gradient_diagnostics::energy).transpose()?.map(f64::sqrt)}));
        }
    }
    let total = groups.values().map(|g| g.gradient_energy).sum();
    for group in groups.values_mut() {
        group.finish(total);
    }
    ensure!(
        checkpoint_hashes(&paths)? == before,
        "gradient probe changed checkpoint bytes"
    );
    Ok(
        json!({"version":1,"backend":"cpu","config":config,"checkpoint_hashes_sha256":before,
        "start_step":start,"bptt":config.bptt,"optimizer_updates_applied":0,
        "probe_reference_fidelity":config.reference_fidelity_max,"norms":norms,"groups":groups,
        "saturation_penalty_loss":saturation_penalty_loss,
        "loss":losses.total.to_scalar::<f32>()?}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict_forks_preserve_parent_and_apply_optimizer_policy() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "titan-fork-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir_all(root.join("corpus"))?;
        image::RgbImage::from_fn(24, 24, |x, y| {
            image::Rgb([(x * 10) as u8, (y * 10) as u8, 100])
        })
        .save(root.join("corpus/source.png"))?;
        let mut config = RunConfig {
            corpus_dir: root.join("corpus"),
            output_dir: root.join("parent"),
            run_tag: Some("parent".into()),
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            interface_grid: 3,
            interface_width: 32,
            interface_loops: 1,
            ca_hidden: 32,
            render_hidden: 32,
            render_blocks: 2,
            morph_layers: 3,
            morph_depth: 2,
            train_resolution: 24,
            output_resolution: 24,
            snapshot_resolution: 24,
            steps: 8,
            ..Default::default()
        };
        config.morph_growth.min_depth = 2;
        config.morph_growth.max_depth = 3;
        config.validate()?;
        std::fs::create_dir(&config.output_dir)?;
        let device = Device::Cpu;
        let mut corpus = ImageCorpus::new(&config, &device)?;
        let vars = VarMap::new();
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
        let _d = DynamicsSystem::new(&config, vb.pp("dynamics"), &device)?;
        let _r = ImplicitRenderer::new(&config, vb.pp("renderer"))?;
        let _f = RectifiedFlowRenderer::new(&config, vb.pp("flow"))?;
        let mut optimizer = PersistentAdamW::new(&vars, &config)?;
        let variable = vars.data().lock().unwrap().values().next().unwrap().clone();
        optimizer.backward_step(&variable.as_tensor().sum_all()?)?;
        let sample = corpus.sample(0, &device)?;
        let mut world = WorldState::fresh(&config, 42, &device)?;
        world.target_index = sample.index;
        world.age = 8;
        world.step = 8;
        let paths = ArtifactPaths::new(&config);
        persistence::save_checkpoint(
            &paths,
            &vars,
            &optimizer,
            &world,
            &config,
            corpus.fingerprint(),
        )?;
        persistence::write_json_atomic(&paths.metadata, &json!({"config":config}))?;
        let before = checkpoint_hashes(&paths)?;
        for policy in [OptimizerPolicy::Retain, OptimizerPolicy::Reset] {
            let mut destination = config.clone();
            destination.output_dir = root.join(format!("fork-{policy:?}"));
            destination.run_tag = Some(format!("fork-{policy:?}"));
            destination.experiment.norm = crate::experiment::NormTraining::Differentiable;
            let request = ForkRequest {
                parent_metadata: paths.metadata.clone(),
                destination: destination.clone(),
                optimizer: policy,
                world: WorldPolicy::Retain,
                warmup: if policy == OptimizerPolicy::Retain {
                    "retain"
                } else {
                    "reset"
                }
                .into(),
            };
            let report = create(&request)?;
            assert_eq!(
                report["optimizer_updates"],
                if policy == OptimizerPolicy::Retain {
                    1
                } else {
                    0
                }
            );
            assert!(report["changed_config"]["experiment.norm"].is_object());
            assert!(create(&request).is_err(), "must not overwrite a fork");
            let mut check_vars = vars.clone();
            let mut check_opt = PersistentAdamW::new(&check_vars, &destination)?;
            let child_paths = ArtifactPaths::new(&destination);
            let (restored, _) = persistence::load_checkpoint_read_only(
                &child_paths,
                &mut check_vars,
                &mut check_opt,
                &device,
                &destination,
                corpus.fingerprint(),
            )?;
            assert_eq!(restored.step, 8);
            assert_eq!(restored.age, 8);
            assert!(
                persistence::load_checkpoint_read_only(
                    &child_paths,
                    &mut check_vars,
                    &mut check_opt,
                    &device,
                    &config,
                    corpus.fingerprint()
                )
                .is_err(),
                "ordinary resume must reject norm mode drift"
            );
            let parent_moments = candle_core::safetensors::load(&paths.optimizer, &device)?;
            let child_moments = candle_core::safetensors::load(&child_paths.optimizer, &device)?;
            for (name, value) in &parent_moments {
                if value.dtype() == DType::F32 {
                    let actual = child_moments[name].flatten_all()?.to_vec1::<f32>()?;
                    if policy == OptimizerPolicy::Retain {
                        assert_eq!(actual, value.flatten_all()?.to_vec1::<f32>()?);
                    } else {
                        assert!(actual.iter().all(|v| *v == 0.0));
                    }
                }
            }
        }
        let mut incompatible = config.clone();
        incompatible.output_dir = root.join("bad");
        incompatible.run_tag = Some("bad".into());
        incompatible.channels += 1;
        assert!(create(&ForkRequest {
            parent_metadata: paths.metadata.clone(),
            destination: incompatible,
            optimizer: OptimizerPolicy::Retain,
            world: WorldPolicy::Retain,
            warmup: "retain".into()
        })
        .is_err());
        assert_eq!(checkpoint_hashes(&paths)?, before);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
