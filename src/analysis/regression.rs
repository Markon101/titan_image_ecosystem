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
        &config, &paths, &dynamics, &renderer, &world, &sample, &device,
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
        &config, &paths, &dynamics, &renderer, &world, &sample, &device,
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
