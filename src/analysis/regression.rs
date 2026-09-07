use super::*;
use candle_nn::{VarBuilder, VarMap};
use std::time::{SystemTime, UNIX_EPOCH};

fn fixture() -> Result<(
    RunConfig,
    VarMap,
    DynamicsSystem,
    ImplicitRenderer,
    WorldState,
    TargetSample,
)> {
    let device = Device::Cpu;
    let root = std::env::temp_dir().join(format!(
        "titan-autonomous-regression-{}-{}",
        std::process::id(),
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
    ));
    std::fs::create_dir_all(&root)?;
    let config = RunConfig {
        output_dir: root,
        micro_size: 8,
        macro_size: 4,
        channels: 12,
        genome_dim: 4,
        ca_hidden: 16,
        render_hidden: 16,
        render_blocks: 1,
        interface_grid: 2,
        interface_width: 16,
        interface_loops: 1,
        morph_layers: 1,
        morph_depth: 1,
        morph_growth: crate::config::MorphGrowthConfig {
            min_depth: 1,
            max_depth: 1,
            ..RunConfig::default().morph_growth
        },
        train_resolution: 16,
        snapshot_resolution: 16,
        output_resolution: 16,
        episode_steps: 4,
        reaction_gain: 0.0,
        phase_gain: 0.0,
        fractal_gain: 0.0,
        quasiperiodic_gain: 0.0,
        cyclic_gain: 0.0,
        analysis: crate::config::AnalysisConfig {
            autonomous_horizon: 4,
            stride: 1,
            ..RunConfig::default().analysis
        },
        ..RunConfig::default()
    };
    let vars = VarMap::new();
    let builder = VarBuilder::from_varmap(&vars, DType::F32, &device);
    let dynamics = DynamicsSystem::new(&config, builder.pp("dynamics"), &device)?;
    let renderer = ImplicitRenderer::new(&config, builder.pp("renderer"))?;
    let world = WorldState::fresh(&config, 42, &device)?;
    let sample = TargetSample {
        image: Tensor::ones((1, 3, 16, 16), DType::F32, &device)?,
        flow_image: Tensor::ones((1, 3, 16, 16), DType::F32, &device)?,
        reference_micro: Tensor::ones((1, 3, 8, 8), DType::F32, &device)?,
        reference_macro: Tensor::ones((1, 3, 4, 4), DType::F32, &device)?,
        genome: vec![0.25; 4],
        genome_tensor: Tensor::new(&[0.25f32; 4], &device)?,
        index: 0,
        name: "reference".into(),
        source_width: 16,
        source_height: 16,
        square_crop_x: 0,
        square_crop_y: 0,
        square_crop_size: 16,
        pyramid_levels: vec![16],
        fingerprint: 1,
    };
    Ok((config, vars, dynamics, renderer, world, sample))
}

#[test]
fn autonomous_rollout_withdraws_reference_and_preserves_saved_state() -> Result<()> {
    let (config, _vars, dynamics, renderer, mut world, mut sample) = fixture()?;
    let device = Device::Cpu;
    // First develop with guidance. Withdrawal must continue this saved organism,
    // not replace it with a new seed or remove its fixed genome.
    world = dynamics
        .step(
            &world,
            &sample.genome_tensor,
            Some(&sample.reference_micro),
            Some(&sample.reference_macro),
            1.0,
            false,
        )?
        .world;
    let initial_micro = tensor_fingerprint(&world.micro)?;
    let initial_macro = tensor_fingerprint(&world.macro_field)?;
    let initial_memory = tensor_fingerprint(&world.memory)?;
    let paths = ArtifactPaths::new(&config);
    let first = autonomous_rollout(
        &config,
        &paths,
        &mut EvaluationArtifacts::new(&config, world.step)?,
        &dynamics,
        &renderer,
        &world,
        &sample,
        &device,
    )?;
    let mut expected = world.clone();
    expected.age = expected.age.max(config.developmental_horizon() as u64);
    let guided = dynamics.step(
        &expected,
        &sample.genome_tensor,
        Some(&sample.reference_micro),
        Some(&sample.reference_macro),
        1.0,
        false,
    )?;
    let unconditioned = dynamics.step(&expected, &sample.genome_tensor, None, None, 0.0, false)?;
    assert!(
        world_distance(&guided.world, &unconditioned.world)? > 1e-6,
        "fixture must detect accidental reference conditioning"
    );
    for _ in 0..config.analysis.autonomous_horizon {
        expected = dynamics
            .step(&expected, &sample.genome_tensor, None, None, 0.0, false)?
            .world;
    }
    let plan = RenderPlan::new(&config, config.snapshot_resolution, &device)?;
    let expected_image = renderer.render_with_emergence(
        &expected.micro,
        &expected.macro_field,
        &sample.genome_tensor,
        &plan,
        config.reconstruction.emergence_strength,
        false,
    )?;
    assert_eq!(
        first.last().unwrap().output_fingerprint,
        tensor_fingerprint(&expected_image.image)?
    );
    // Replace references with conspicuously different data; output must not change.
    sample.reference_micro = sample.reference_micro.affine(-3.0, 0.0)?;
    sample.reference_macro = sample.reference_macro.affine(5.0, 0.0)?;
    let second = autonomous_rollout(
        &config,
        &paths,
        &mut EvaluationArtifacts::new(&config, world.step)?,
        &dynamics,
        &renderer,
        &world,
        &sample,
        &device,
    )?;
    assert_eq!(serde_json::to_vec(&first)?, serde_json::to_vec(&second)?);
    assert!(first.iter().all(|r| r.reference_fidelity == 0.0
        && r.micro_reference_drive_rms == 0.0
        && r.macro_reference_drive_rms == 0.0));
    assert_eq!(initial_micro, tensor_fingerprint(&world.micro)?);
    assert_eq!(initial_macro, tensor_fingerprint(&world.macro_field)?);
    assert_eq!(initial_memory, tensor_fingerprint(&world.memory)?);
    std::fs::remove_dir_all(&config.output_dir)?;
    Ok(())
}

#[test]
fn motionless_first_sample_cannot_claim_recurrence() -> Result<()> {
    let (mut config, vars, _, renderer, mut world, sample) = fixture()?;
    config.nca_gain = 0.0;
    config.interface_gain = 0.0;
    config.state_leak = 0.0;
    let device = Device::Cpu;
    let dynamics = DynamicsSystem::new(
        &config,
        VarBuilder::from_varmap(&vars, DType::F32, &device).pp("dynamics"),
        &device,
    )?;
    world.micro = world.micro.zeros_like()?;
    world.macro_field = world.macro_field.zeros_like()?;
    let records = autonomous_rollout(
        &config,
        &ArtifactPaths::new(&config),
        &mut EvaluationArtifacts::new(&config, world.step)?,
        &dynamics,
        &renderer,
        &world,
        &sample,
        &device,
    )?;
    assert_eq!(records[0].micro_movement, 0.0);
    assert_eq!(records[0].macro_movement, 0.0);
    assert!(!records[0].recurrence_distance_valid);
    assert!(!records[0].approximate_cycle_candidate);
    assert!(records[1].recurrence_distance_valid);
    std::fs::remove_dir_all(&config.output_dir)?;
    Ok(())
}

#[test]
fn fresh_evaluations_never_adopt_canonical_or_prior_outputs() -> Result<()> {
    let (mut config, vars, dynamics, renderer, world, _) = fixture()?;
    let root = config.output_dir.clone();
    config.corpus_dir = root.join("training");
    let probe_dir = root.join("probes");
    config.output_dir = root.join("output");
    std::fs::create_dir_all(&config.output_dir)?;
    std::fs::create_dir_all(&config.corpus_dir)?;
    std::fs::create_dir_all(&probe_dir)?;
    image::RgbImage::from_pixel(16, 16, image::Rgb([40, 90, 160]))
        .save(config.corpus_dir.join("training.png"))?;
    image::RgbImage::from_fn(16, 16, |x, y| {
        image::Rgb([(x * 13) as u8, (y * 11) as u8, 80])
    })
    .save(probe_dir.join("probe.png"))?;
    config.mode = crate::config::TrainingMode::Family;
    config.output_resolution = 32;
    config.analysis.only = true;
    config.analysis.render_attribution = true;
    config.analysis.perturbation_horizon = 2;
    config.analysis.dynamics_horizon = 1;
    config.analysis.benchmark = true;
    config.objective = crate::config::ObjectiveMode::HybridFlow;
    config.flow.sample_steps = 2;
    config.analysis.probe_dir = Some(probe_dir.clone());
    config.analysis.probe_ages = vec![1, 2];
    // These used to collide at f0500 within a single evaluation.
    config.analysis.probe_reference_fidelities = vec![0.5, 0.5001];
    let device = Device::Cpu;
    let mut corpus = ImageCorpus::new(&config, &device)?;
    let sample = corpus.sample_index(0, &device)?;
    let flow = RectifiedFlowRenderer::new(
        &config,
        VarBuilder::from_varmap(&vars, DType::F32, &device).pp("flow"),
    )?;
    let paths = ArtifactPaths::new(&config);
    let fingerprint = fnv(&std::fs::read(probe_dir.join("probe.png"))?);
    let legacy_probe = config.output_dir.join(format!(
        "titan_image_probe_v9{}_000_{fingerprint}_f0500_a0001.png",
        config.suffix()
    ));
    let modern_probe = config.output_dir.join(format!(
        "titan_image_probe_v9{}_000_{fingerprint}_f0500_b3f000000_a0001.png",
        config.suffix()
    ));
    let stale = b"deliberately invalid old PNG bytes";
    for path in [
        &legacy_probe,
        &modern_probe,
        &paths.decomposition,
        &paths.resolution_ladder,
        &paths.attractor_analysis,
        &paths.target_comparison,
        &paths.benchmark,
    ] {
        std::fs::write(path, stale)?;
    }
    let mut evaluations = Vec::new();
    let mut saved_bytes = Vec::new();
    for _ in 0..2 {
        let mut artifacts = EvaluationArtifacts::new(&config, world.step)?;
        let summary = run_checkpoint_analysis(
            &config,
            &paths,
            &mut artifacts,
            &mut corpus,
            &dynamics,
            &renderer,
            &flow,
            &world,
            &sample,
            &device,
        )?;
        let summary = artifacts.publish(summary, &paths.analysis)?;
        let archive = Path::new(&summary.provenance.archive);
        assert_eq!(summary.provenance.analysis_version, 3);
        assert_eq!(std::fs::read(archive)?, std::fs::read(&paths.analysis)?);
        let root = archive.parent().unwrap();
        for (path, identity) in &summary.provenance.artifacts {
            assert_eq!(Path::new(path).parent(), Some(root));
            assert_eq!(path, &identity.path);
            let bytes = std::fs::read(path)?;
            assert_ne!(bytes, stale);
            assert_eq!(identity.bytes, bytes.len() as u64);
            assert_eq!(identity.fnv1a64, fnv(&bytes));
            if path.ends_with(".png") {
                image::open(path)?;
            }
            saved_bytes.push((path.clone(), bytes));
        }
        for canonical in [
            &paths.decomposition,
            &paths.resolution_ladder,
            &paths.attractor_analysis,
            &paths.target_comparison,
            &paths.benchmark,
            &paths.perturbation_analysis,
            &paths.flow_sample,
            &paths.flow_trajectory,
            &sibling_png(&paths.resolution_ladder, "0016"),
            &sibling_png(&paths.target_comparison, "target_000"),
        ] {
            assert!(summary
                .provenance
                .artifacts
                .contains_key(root.join(canonical.file_name().unwrap()).to_str().unwrap()));
        }
        let probe = summary.natural_image_probe.as_ref().unwrap();
        assert_eq!(probe.output_count, 4);
        assert_eq!(
            serde_json::to_vec_pretty(probe)?,
            std::fs::read(&probe.report)?
        );
        for point in &probe.targets[0].points {
            assert!(summary.provenance.artifacts.contains_key(&point.output));
            let png = image::open(&point.output)?.to_rgb8();
            let n = (png.width() * png.height()) as f32;
            for (channel, expected) in [point.red_mean, point.green_mean, point.blue_mean]
                .into_iter()
                .enumerate()
            {
                let actual = png
                    .pixels()
                    .map(|p| f32::from(p[channel]) / 255.0)
                    .sum::<f32>()
                    / n;
                assert!(
                    (actual - expected).abs() <= 0.5 / 255.0 + 1e-6,
                    "PNG must encode the tensor used for fresh metrics"
                );
            }
        }
        assert!(summary
            .autonomous_rollout
            .iter()
            .all(|r| r.reference_fidelity == 0.0
                && r.micro_reference_drive_rms == 0.0
                && r.macro_reference_drive_rms == 0.0));
        saved_bytes.push((summary.provenance.archive.clone(), std::fs::read(archive)?));
        evaluations.push(summary);
    }
    assert_ne!(
        evaluations[0].provenance.evaluation_id,
        evaluations[1].provenance.evaluation_id
    );
    for path in evaluations[0].provenance.artifacts.keys() {
        assert!(!evaluations[1].provenance.artifacts.contains_key(path));
    }
    for (path, bytes) in saved_bytes {
        assert_eq!(std::fs::read(path)?, bytes);
    }
    for path in [
        &legacy_probe,
        &modern_probe,
        &paths.decomposition,
        &paths.resolution_ladder,
        &paths.attractor_analysis,
        &paths.target_comparison,
        &paths.benchmark,
    ] {
        assert_eq!(std::fs::read(path)?, stale);
    }
    // A summary cannot manufacture ownership from a pre-existing canonical path.
    let artifacts = EvaluationArtifacts::new(&config, world.step)?;
    let mut forged = evaluations[0].clone();
    forged.provenance.evaluation_id = artifacts.evaluation_id.clone();
    forged.provenance.archive = artifacts.archive().display().to_string();
    forged.decomposition_montage = Some(paths.decomposition.display().to_string());
    let missing_archive = artifacts.archive();
    assert!(artifacts.publish(forged, &paths.analysis).is_err());
    assert!(!missing_archive.exists());
    // A successfully emitted file that changes before publication is rejected too.
    let mut artifacts = EvaluationArtifacts::new(&config, world.step)?;
    let emitted = artifacts.json(Path::new("changed.json"), &true)?;
    std::fs::write(&emitted, b"false")?;
    let mut summary = evaluations[0].clone();
    summary.provenance.evaluation_id = artifacts.evaluation_id.clone();
    summary.provenance.archive = artifacts.archive().display().to_string();
    let missing_archive = artifacts.archive();
    let error = artifacts.publish(summary, &paths.analysis).unwrap_err();
    assert!(error
        .to_string()
        .contains("artifact changed before publication"));
    assert!(!missing_archive.exists());
    std::fs::remove_dir_all(root)?;
    Ok(())
}

fn fnv(bytes: &[u8]) -> String {
    format!(
        "{:016x}",
        bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| (hash
            ^ u64::from(*byte))
        .wrapping_mul(0x100000001b3))
    )
}
