use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::path::PathBuf;
use titan_image::config::{
    BoundaryMode, ConditioningMode, MorphDepthMode, ObjectiveMode, ResearchPreset, RunConfig,
    SCHEMA_VERSION,
};
use titan_image::corpus::ImageCorpus;
use titan_image::interface::RecurrentInterface;
use titan_image::objectives::{cross_resolution_consistency, visual_loss};
use titan_image::render::{ImplicitRenderer, RenderOutput, RenderPlan};

#[test]
fn v9_schema_presets_and_default_curriculum_are_explicit() -> anyhow::Result<()> {
    assert_eq!(SCHEMA_VERSION, 9);
    for (encoded, expected) in [
        ("\"strict-reconstruct\"", ResearchPreset::StrictReconstruct),
        (
            "\"reconstruction-plus\"",
            ResearchPreset::ReconstructionPlus,
        ),
        ("\"grounded-emergent\"", ResearchPreset::GroundedEmergent),
        ("\"free-morph\"", ResearchPreset::FreeMorph),
        ("\"flow-reconstruct\"", ResearchPreset::FlowReconstruct),
    ] {
        assert_eq!(serde_json::from_str::<ResearchPreset>(encoded)?, expected);
    }
    assert!(serde_json::from_str::<ResearchPreset>("\"strong-emergence\"").is_err());

    let config = RunConfig::default();
    assert_eq!(config.research_preset, ResearchPreset::ReconstructionPlus);
    assert_eq!(config.objective, ObjectiveMode::ReconstructionPlus);
    assert_eq!(config.conditioning, ConditioningMode::Reconstruct);
    assert_eq!(config.interface_grid, 8);
    let (early_ground, early_emergence) = config.developmental_schedule(0.0);
    let (late_ground, late_emergence) = config.developmental_schedule(1.0);
    assert_eq!(early_ground, config.reconstruction.grounding_strength);
    assert_eq!(early_emergence, 0.0);
    assert_eq!(
        late_ground,
        config.reconstruction.grounding_strength * config.reconstruction.grounding_floor
    );
    assert_eq!(late_emergence, config.reconstruction.emergence_strength);
    assert!(late_ground > 0.0);

    let grid_16 = RunConfig {
        interface_grid: 16,
        ..config
    };
    grid_16.validate()?;
    Ok(())
}

#[test]
fn morph_depth_modes_choose_only_allocated_capacity() {
    let fixed = RunConfig::default();
    assert_eq!(fixed.initial_morph_depth(), fixed.morph_depth);

    let mut capacity = fixed.clone();
    capacity.morph_growth.mode = MorphDepthMode::Capacity;
    capacity.morph_growth.max_depth = capacity.morph_layers;
    assert_eq!(capacity.initial_morph_depth(), capacity.morph_layers);

    let mut adaptive = fixed;
    adaptive.morph_growth.mode = MorphDepthMode::Adaptive;
    adaptive.morph_growth.min_depth = 2;
    assert_eq!(adaptive.initial_morph_depth(), 2);
}

#[test]
fn nonzero_emergent_head_has_zero_influence_at_strength_zero_and_stays_bounded(
) -> anyhow::Result<()> {
    let device = Device::Cpu;
    let mut config = small_render_config();
    config.reconstruction.emergent_limit = 0.30;
    config.reconstruction.emergence_low_budget = 1.0;
    config.reconstruction.emergence_mid_budget = 1.0;
    let variables = VarMap::new();
    let renderer = ImplicitRenderer::new(
        &config,
        VarBuilder::from_varmap(&variables, DType::F32, &device).pp("renderer"),
    )?;
    {
        let data = variables.data().lock().expect("VarMap mutex poisoned");
        let weight = data
            .get("renderer.emergent.weight")
            .expect("emergent weight exists");
        let bias = data
            .get("renderer.emergent.bias")
            .expect("emergent bias exists");
        weight.set(&Tensor::zeros(weight.shape(), DType::F32, &device)?)?;
        bias.set(&Tensor::new(&[0.9f32, -0.7, 0.5], &device)?)?;
    }
    let plan = RenderPlan::new(&config, 24, &device)?;
    let micro = Tensor::zeros((1, 12, 24, 24), DType::F32, &device)?;
    let macro_field = Tensor::zeros((1, 12, 12, 12), DType::F32, &device)?;
    let genome = Tensor::zeros(4, DType::F32, &device)?;
    let grounded =
        renderer.render_with_emergence(&micro, &macro_field, &genome, &plan, 0.0, false)?;
    let composite =
        renderer.render_with_emergence(&micro, &macro_field, &genome, &plan, 1.0, false)?;

    assert!(grounded.emergent_lab.abs()?.max_all()?.to_scalar::<f32>()? > 1e-3);
    assert_eq!(
        grounded
            .image
            .sub(&grounded.grounded_image)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?,
        0.0
    );
    assert!(
        composite
            .image
            .sub(&composite.grounded_image)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?
            > 1e-5
    );
    assert!(
        composite
            .emergent_lab
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?
            < config.reconstruction.emergent_limit
    );
    assert_tensor_is_finite_and_bounded(&composite.image, 0.0, 1.0)?;
    assert_tensor_is_finite_and_bounded(&composite.emergent_visual, 0.0, 1.0)?;
    Ok(())
}

#[test]
fn multiscale_grounding_distinguishes_fine_detail_from_body_plan() -> Result<()> {
    let device = Device::Cpu;
    let resolution = 16;
    let plane = resolution * resolution;
    let mut checker = vec![0.0f32; 3 * plane];
    for channel in 0..3 {
        for y in 0..resolution {
            for x in 0..resolution {
                checker[channel * plane + y * resolution + x] =
                    if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
            }
        }
    }
    let grounded = Tensor::from_vec(checker, (1, 3, resolution, resolution), &device)?;
    let target =
        Tensor::ones((1, 3, resolution, resolution), DType::F32, &device)?.affine(0.5, 0.0)?;
    let rendered = render_output(grounded.clone(), Tensor::zeros_like(&grounded)?)?;
    let state = Tensor::zeros((1, 12, resolution, resolution), DType::F32, &device)?;
    let memory = Tensor::zeros((1, 32), DType::F32, &device)?;
    let loss = visual_loss(
        &rendered,
        &target,
        &state,
        &state,
        &memory,
        &RunConfig::default(),
        1.0,
        0.0,
        BoundaryMode::Natural,
    )?;
    assert!((loss.ground_fine.to_scalar::<f32>()? - 0.5).abs() < 1e-7);
    assert!(loss.ground_mid.to_scalar::<f32>()? < 1e-7);
    assert!(loss.ground_coarse.to_scalar::<f32>()? < 1e-7);
    Ok(())
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn emergence_schedule_zero_disables_emergent_regularization() -> Result<()> {
    let device = Device::Cpu;
    let image = Tensor::ones((1, 3, 16, 16), DType::F32, &device)?.affine(0.5, 0.0)?;
    let emergent = Tensor::ones_like(&image)?.affine(0.25, 0.0)?;
    let rendered = render_output(image.clone(), emergent)?;
    let state = Tensor::zeros((1, 12, 16, 16), DType::F32, &device)?;
    let memory = Tensor::zeros((1, 32), DType::F32, &device)?;
    let mut config = RunConfig::default();
    config.loss_content = 0.0;
    config.loss_palette = 0.0;
    config.loss_structure = 0.0;
    config.loss_seam = 0.0;
    config.loss_gamut = 0.0;
    config.loss_state = 0.0;
    config.loss_memory = 0.0;
    config.reconstruction.loss_composite = 0.0;
    config.reconstruction.loss_ground_coarse = 0.0;
    config.reconstruction.loss_ground_mid = 0.0;
    config.reconstruction.loss_ground_fine = 0.0;
    config.reconstruction.loss_emergent_fit = 0.0;
    config.reconstruction.loss_emergent_low = 1.0;
    config.reconstruction.loss_emergent_tv = 1.0;
    config.reconstruction.loss_head_redundancy = 1.0;

    let disabled = visual_loss(
        &rendered,
        &image,
        &state,
        &state,
        &memory,
        &config,
        0.0,
        0.0,
        BoundaryMode::Natural,
    )?;
    let enabled = visual_loss(
        &rendered,
        &image,
        &state,
        &state,
        &memory,
        &config,
        0.0,
        1.0,
        BoundaryMode::Natural,
    )?;
    assert_eq!(disabled.total.to_scalar::<f32>()?, 0.0);
    assert!(enabled.total.to_scalar::<f32>()? > 0.0);
    Ok(())
}

#[test]
fn cross_resolution_allows_subpixel_detail_but_rejects_geometry_shift() -> Result<()> {
    let device = Device::Cpu;
    let low_resolution = 8;
    let mut low = vec![0.0f32; 3 * low_resolution * low_resolution];
    for channel in 0..3 {
        for y in 0..low_resolution {
            for x in 0..low_resolution {
                low[channel * 64 + y * 8 + x] = 0.2 + 0.5 * x as f32 / 7.0 + 0.1 * y as f32 / 7.0;
            }
        }
    }
    let high = repeat_2x_with_zero_mean_detail(&low, low_resolution, 0.04);
    let low = Tensor::from_vec(low, (1, 3, 8, 8), &device)?;
    let high = Tensor::from_vec(high, (1, 3, 16, 16), &device)?;
    let (l1, low_frequency, edge) = cross_resolution_consistency(&high, &low)?;
    assert!(l1.to_scalar::<f32>()? < 1e-6);
    assert!(low_frequency.to_scalar::<f32>()? < 1e-6);
    assert!(edge.to_scalar::<f32>()? < 1e-6);

    let shifted = roll_planar_x(&high.flatten_all()?.to_vec1::<f32>()?, 16, 4);
    let shifted = Tensor::from_vec(shifted, (1, 3, 16, 16), &device)?;
    let (l1, low_frequency, edge) = cross_resolution_consistency(&shifted, &low)?;
    assert!(l1.to_scalar::<f32>()? > 0.02);
    assert!(low_frequency.to_scalar::<f32>()? > 0.01);
    assert!(edge.to_scalar::<f32>()? > 0.005);
    Ok(())
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn native_detail_crop_is_deterministic_and_preserves_source_aspect_metadata() -> anyhow::Result<()>
{
    let root = scratch_directory("native-detail");
    std::fs::create_dir_all(&root)?;
    let source_path = root.join("wide-source.png");
    let source = image::RgbImage::from_fn(320, 192, |x, y| {
        image::Rgb([
            (x * 255 / 319) as u8,
            (y * 255 / 191) as u8,
            ((x + 2 * y) % 256) as u8,
        ])
    });
    source.save_with_format(&source_path, image::ImageFormat::Png)?;

    let mut config = RunConfig::default();
    config.corpus_dir = root.clone();
    config.output_dir = root.join("output");
    config.detail.cache_dir = Some(root.join("pyramid-cache"));
    config.mode = titan_image::config::TrainingMode::Single;
    config.train_resolution = 32;
    config.micro_size = 32;
    config.macro_size = 16;
    config.flow.resolution = 16;
    config.image_cache = 1;
    config.detail.probability = 1.0;
    config.detail.curriculum_start = 0.0;
    config.detail.resolution = 32;
    config.detail.min_zoom = 2.0;
    config.detail.max_zoom = 2.0;
    config.detail.cache_max_level = 384;

    let mut corpus = ImageCorpus::new(&config, &Device::Cpu)?;
    let manifest = corpus.source_manifest();
    assert_eq!(manifest.len(), 1);
    assert_eq!((manifest[0].width, manifest[0].height), (320, 192));
    assert!((manifest[0].aspect_ratio - 320.0 / 192.0).abs() < 1e-6);
    assert_eq!(manifest[0].square_crop_x, 64);
    assert_eq!(manifest[0].square_crop_y, 0);
    assert_eq!(manifest[0].square_crop_size, 192);
    assert_eq!(manifest[0].pyramid_levels, vec![192, 320]);
    let sample = corpus.sample(0, &Device::Cpu)?;
    assert_eq!(sample.image.dims4()?, (1, 3, 32, 32));

    let first = corpus
        .detail_observation(0, 17, 1.0, &config, &Device::Cpu)?
        .expect("probability-one mature detail crop");
    let second = corpus
        .detail_observation(0, 17, 1.0, &config, &Device::Cpu)?
        .expect("same deterministic detail crop");
    assert_eq!(first.view, second.view);
    assert_eq!(first.pyramid_level, 192);
    assert_eq!(first.fingerprint, second.fingerprint);
    assert_eq!(first.target.dims4()?, (1, 3, 32, 32));
    assert_eq!(
        first.target.flatten_all()?.to_vec1::<f32>()?,
        second.target.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        first
            .local_reference_micro
            .flatten_all()?
            .to_vec1::<f32>()?,
        second
            .local_reference_micro
            .flatten_all()?
            .to_vec1::<f32>()?
    );
    assert!(first.view.x + first.view.size <= 1.0 + 1e-6);
    assert!(first.view.y + first.view.size <= 1.0 + 1e-6);
    assert_eq!(first.zoom, 2.0);

    drop(corpus);
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn hierarchical_grid_16_and_new_morph_activation_are_function_preserving() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let mut config = RunConfig {
        micro_size: 32,
        macro_size: 16,
        channels: 12,
        genome_dim: 4,
        interface_grid: 16,
        interface_width: 32,
        interface_loops: 1,
        morph_layers: 3,
        morph_depth: 1,
        train_resolution: 32,
        output_resolution: 32,
        ..RunConfig::default()
    };
    config.morph_growth.min_depth = 1;
    config.morph_growth.max_depth = 3;
    config.validate()?;
    let variables = VarMap::new();
    let interface = RecurrentInterface::new(
        &config,
        VarBuilder::from_varmap(&variables, DType::F32, &device).pp("interface"),
        &device,
    )?;
    let micro = Tensor::ones((1, 12, 32, 32), DType::F32, &device)?.affine(0.10, 0.0)?;
    let macro_field = Tensor::ones((1, 12, 16, 16), DType::F32, &device)?.affine(-0.08, 0.0)?;
    let reference_micro = Tensor::ones((1, 3, 32, 32), DType::F32, &device)?.affine(0.30, 0.0)?;
    let reference_macro = Tensor::ones((1, 3, 16, 16), DType::F32, &device)?.affine(0.20, 0.0)?;
    let genome = Tensor::zeros(4, DType::F32, &device)?;
    let memory = Tensor::ones((1, 32), DType::F32, &device)?.affine(0.15, 0.0)?;
    let forward = |active_depth| {
        interface.forward(
            &micro,
            &macro_field,
            &reference_micro,
            &reference_macro,
            &genome,
            &memory,
            1.0,
            0.75,
            false,
            active_depth,
        )
    };
    let old_depth = forward(1)?;
    let activated = forward(2)?;
    assert_eq!(old_depth.micro_bias.dims4()?, (1, 12, 32, 32));
    assert_eq!(old_depth.macro_bias.dims4()?, (1, 12, 16, 16));
    assert_eq!(
        old_depth.memory.flatten_all()?.to_vec1::<f32>()?,
        activated.memory.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        old_depth.micro_bias.flatten_all()?.to_vec1::<f32>()?,
        activated.micro_bias.flatten_all()?.to_vec1::<f32>()?
    );
    assert_eq!(
        old_depth.macro_bias.flatten_all()?.to_vec1::<f32>()?,
        activated.macro_bias.flatten_all()?.to_vec1::<f32>()?
    );
    assert!(forward(config.morph_layers + 1).is_err());
    assert_tensor_is_finite_and_bounded(
        &old_depth.memory,
        -config.memory_limit,
        config.memory_limit,
    )?;
    Ok(())
}

fn small_render_config() -> RunConfig {
    RunConfig {
        micro_size: 24,
        macro_size: 12,
        channels: 12,
        genome_dim: 4,
        render_hidden: 32,
        render_blocks: 1,
        coord_bands: 2,
        train_resolution: 24,
        output_resolution: 24,
        ..RunConfig::default()
    }
}

fn render_output(image: Tensor, emergent_lab: Tensor) -> Result<RenderOutput> {
    Ok(RenderOutput {
        grounded_image: image.clone(),
        image: image.clone(),
        emergent_visual: Tensor::zeros_like(&image)?,
        grounded_lab: Tensor::zeros_like(&image)?,
        emergent_lab,
        emergence_strength: 0.0,
        gamut_excess: Tensor::new(0.0f32, image.device())?,
        state_only_image: None,
        learned_only_image: None,
    })
}

fn repeat_2x_with_zero_mean_detail(low: &[f32], resolution: usize, detail: f32) -> Vec<f32> {
    let high_resolution = resolution * 2;
    let low_plane = resolution * resolution;
    let high_plane = high_resolution * high_resolution;
    let mut high = vec![0.0f32; 3 * high_plane];
    for channel in 0..3 {
        for y in 0..resolution {
            for x in 0..resolution {
                let value = low[channel * low_plane + y * resolution + x];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sign = if (dx + dy) % 2 == 0 { -1.0 } else { 1.0 };
                        high[channel * high_plane + (2 * y + dy) * high_resolution + 2 * x + dx] =
                            value + sign * detail;
                    }
                }
            }
        }
    }
    high
}

fn roll_planar_x(values: &[f32], resolution: usize, shift: usize) -> Vec<f32> {
    let plane = resolution * resolution;
    let mut shifted = vec![0.0f32; values.len()];
    for channel in 0..3 {
        for y in 0..resolution {
            for x in 0..resolution {
                shifted[channel * plane + y * resolution + x] =
                    values[channel * plane + y * resolution + (x + shift) % resolution];
            }
        }
    }
    shifted
}

fn assert_tensor_is_finite_and_bounded(tensor: &Tensor, min: f32, max: f32) -> Result<()> {
    let values = tensor.flatten_all()?.to_vec1::<f32>()?;
    assert!(values.iter().all(|value| value.is_finite()));
    assert!(values.iter().all(|value| *value >= min && *value <= max));
    Ok(())
}

fn scratch_directory(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "titan-image-v9-{label}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ))
}
