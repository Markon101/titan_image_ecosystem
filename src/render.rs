use crate::config::{ComputeBackend, RunConfig, StylePreset};
use crate::tensor_ops::{
    broadcast_vector, coordinate_features, coordinate_features_window, periodic_shift,
    pixelwise_linear_mode, smooth_limit, PeriodicUpsampler,
};
use anyhow::{Context, Result};
#[cfg(feature = "opencl")]
use candle_core::{backprop::GradStore, Var};
use candle_core::{Device, Tensor, D};
use candle_nn::{Init, Linear, VarBuilder};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
#[cfg(feature = "opencl")]
use std::sync::Mutex;

#[cfg(feature = "opencl")]
struct OpenClSlot {
    attempted: bool,
    renderer: Option<crate::opencl::OpenClMlp>,
    failure: Option<String>,
    training_boundary: Option<Var>,
}

pub struct ImplicitRenderer {
    input: Linear,
    blocks: Vec<Linear>,
    grounded_head: Linear,
    emergent_head: Linear,
    default_emergence_strength: f32,
    emergent_limit: f32,
    emergence_low_budget: f32,
    emergence_mid_budget: f32,
    state_skip: f32,
    chroma: f32,
    gamma: f64,
    style: StylePreset,
    compute_backend: ComputeBackend,
    #[cfg(feature = "opencl")]
    opencl: Mutex<OpenClSlot>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpatialView {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub zoom: f32,
}

impl SpatialView {
    pub const fn full() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            size: 1.0,
            zoom: 1.0,
        }
    }

    pub fn is_full(self) -> bool {
        self.x == 0.0 && self.y == 0.0 && self.size == 1.0
    }
}

struct FieldObservation {
    start_x: usize,
    start_y: usize,
    cells: usize,
    upsampler: PeriodicUpsampler,
}

impl FieldObservation {
    fn new(
        field_size: usize,
        resolution: usize,
        view: SpatialView,
        device: &Device,
    ) -> Result<Self> {
        if view.is_full() {
            return Ok(Self {
                start_x: 0,
                start_y: 0,
                cells: field_size,
                upsampler: PeriodicUpsampler::new(
                    field_size, field_size, resolution, resolution, device,
                )?,
            });
        }
        let cells = ((field_size as f32 * view.size).round() as usize).clamp(2, field_size);
        let max_start = field_size - cells;
        let start_x = ((view.x * field_size as f32).round() as usize).min(max_start);
        let start_y = ((view.y * field_size as f32).round() as usize).min(max_start);
        Ok(Self {
            start_x,
            start_y,
            cells,
            upsampler: PeriodicUpsampler::new_bounded(
                cells, cells, resolution, resolution, device,
            )?,
        })
    }

    fn apply(&self, field: &Tensor) -> candle_core::Result<Tensor> {
        let cropped = field
            .narrow(2, self.start_y, self.cells)?
            .narrow(3, self.start_x, self.cells)?
            .contiguous()?;
        self.upsampler.apply(&cropped)
    }
}

pub struct RenderPlan {
    pub resolution: usize,
    micro: FieldObservation,
    macro_field: FieldObservation,
    coordinates: Tensor,
    lod: Tensor,
    pub view: SpatialView,
}

pub struct RenderOutput {
    pub image: Tensor,
    pub grounded_image: Tensor,
    pub emergent_visual: Tensor,
    pub grounded_lab: Tensor,
    pub emergent_lab: Tensor,
    pub emergence_strength: f32,
    /// Mean excursion of pre-clipped linear RGB outside [0, 1]. This stays in
    /// the loss graph so saturated output heads retain a corrective gradient.
    pub gamut_excess: Tensor,
    pub state_only_image: Option<Tensor>,
    pub learned_only_image: Option<Tensor>,
}

impl RenderPlan {
    pub fn new(config: &RunConfig, resolution: usize, device: &Device) -> Result<Self> {
        Self::new_view(config, resolution, SpatialView::full(), device)
    }

    pub fn new_view(
        config: &RunConfig,
        resolution: usize,
        view: SpatialView,
        device: &Device,
    ) -> Result<Self> {
        anyhow::ensure!(
            view.x >= 0.0
                && view.y >= 0.0
                && view.size > 0.0
                && view.x + view.size <= 1.0 + 1e-6
                && view.y + view.size <= 1.0 + 1e-6,
            "render view must lie inside normalized phenotype coordinates"
        );
        let coordinate_tensor = if view.is_full() {
            coordinate_features(resolution, config.coord_bands, device)?
        } else {
            coordinate_features_window(
                resolution,
                config.coord_bands,
                view.x,
                view.y,
                view.size,
                device,
            )?
        };
        let lod_value = view.zoom.max(1.0).log2() / 5.0;
        Ok(Self {
            resolution,
            micro: FieldObservation::new(config.micro_size, resolution, view, device)?,
            macro_field: FieldObservation::new(config.macro_size, resolution, view, device)?,
            coordinates: coordinate_tensor.affine(config.coord_gain as f64, 0.0)?,
            lod: Tensor::new(lod_value, device)?
                .reshape((1, 1, 1, 1))?
                .broadcast_as((1, 1, resolution, resolution))?,
            view,
        })
    }

    pub fn observe_fields(&self, micro: &Tensor, macro_field: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            self.micro.apply(micro)?,
            self.macro_field.apply(macro_field)?,
        ))
    }

    pub fn lod_value(&self) -> f32 {
        self.view.zoom.max(1.0).log2() / 5.0
    }
}
impl ImplicitRenderer {
    pub fn new(config: &RunConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let input_features = config.channels * 2 + config.genome_dim + config.coord_bands * 4 + 1;
        let input = candle_nn::linear(input_features, config.render_hidden, vb.pp("input"))?;
        let mut blocks = Vec::with_capacity(config.render_blocks);
        for index in 0..config.render_blocks {
            blocks.push(candle_nn::linear(
                config.render_hidden,
                config.render_hidden,
                vb.pp(format!("block_{index:03}")),
            )?);
        }
        Ok(Self {
            input,
            blocks,
            grounded_head: candle_nn::linear(config.render_hidden, 3, vb.pp("grounded"))?,
            emergent_head: zero_linear(config.render_hidden, 3, vb.pp("emergent"))?,
            state_skip: config.state_skip,
            chroma: config.chroma,
            gamma: config.gamma as f64,
            style: config.style,
            compute_backend: config.compute_backend,
            #[cfg(feature = "opencl")]
            opencl: Mutex::new(OpenClSlot {
                attempted: false,
                renderer: None,
                failure: None,
                training_boundary: None,
            }),
            default_emergence_strength: config.reconstruction.emergence_strength,
            emergent_limit: config.reconstruction.emergent_limit,
            emergence_low_budget: config.reconstruction.emergence_low_budget,
            emergence_mid_budget: config.reconstruction.emergence_mid_budget,
        })
    }

    pub fn prepare_inference_backend(&self) -> Result<Option<String>> {
        if self.compute_backend == ComputeBackend::Cpu {
            return Ok(None);
        }
        #[cfg(not(feature = "opencl"))]
        {
            match self.compute_backend {
                ComputeBackend::OpenCl => {
                    anyhow::bail!("--compute-backend opencl requires cargo build --features opencl")
                }
                ComputeBackend::Auto => {
                    eprintln!("OPENCL fallback: binary built without opencl feature");
                    Ok(None)
                }
                ComputeBackend::Cpu => Ok(None),
            }
        }
        #[cfg(feature = "opencl")]
        {
            let mut slot = self.opencl.lock().expect("OpenCL renderer mutex poisoned");
            self.initialize_opencl(&mut slot)?;
            Ok(slot
                .renderer
                .as_ref()
                .map(|renderer| renderer.info.to_string()))
        }
    }

    #[cfg(feature = "opencl")]
    fn initialize_opencl(&self, slot: &mut OpenClSlot) -> Result<()> {
        if slot.attempted {
            if self.compute_backend == ComputeBackend::OpenCl {
                if let Some(message) = &slot.failure {
                    anyhow::bail!("OpenCL initialization failed: {message}");
                }
            }
            return Ok(());
        }
        slot.attempted = true;
        let result = self
            .opencl_layers()
            .and_then(|layers| crate::opencl::OpenClMlp::new(&layers, self.blocks.len()));
        match result {
            Ok(renderer) => slot.renderer = Some(renderer),
            Err(error) if self.compute_backend == ComputeBackend::Auto => {
                let message = format!("{error:#}");
                eprintln!("OPENCL fallback: {message}");
                slot.failure = Some(message);
            }
            Err(error) => {
                slot.failure = Some(format!("{error:#}"));
                return Err(error);
            }
        }
        Ok(())
    }

    #[cfg(feature = "opencl")]
    fn opencl_layers(&self) -> Result<Vec<crate::opencl::LinearData>> {
        let mut layers = Vec::with_capacity(self.blocks.len() + 3);
        layers.push(linear_data(&self.input)?);
        for block in &self.blocks {
            layers.push(linear_data(block)?);
        }
        layers.push(linear_data(&self.grounded_head)?);
        layers.push(linear_data(&self.emergent_head)?);
        Ok(layers)
    }

    fn opencl_heads(
        &self,
        features: &Tensor,
        resolution: usize,
    ) -> Result<Option<(Tensor, Tensor)>> {
        if self.compute_backend == ComputeBackend::Cpu {
            return Ok(None);
        }
        #[cfg(not(feature = "opencl"))]
        {
            let _ = (features, resolution);
            self.prepare_inference_backend()?;
            Ok(None)
        }
        #[cfg(feature = "opencl")]
        {
            let mut slot = self.opencl.lock().expect("OpenCL renderer mutex poisoned");
            self.initialize_opencl(&mut slot)?;
            let Some(renderer) = slot.renderer.as_ref() else {
                return Ok(None);
            };
            let values = features.flatten_all()?.to_vec1::<f32>()?;
            let output = renderer.run(&values, resolution * resolution)?;
            let tensor =
                Tensor::from_vec(output, (1, 6, resolution, resolution), features.device())?;
            Ok(Some((tensor.narrow(1, 0, 3)?, tensor.narrow(1, 3, 3)?)))
        }
    }

    fn opencl_training_heads(
        &self,
        features: &Tensor,
        resolution: usize,
    ) -> Result<(Tensor, Tensor)> {
        #[cfg(not(feature = "opencl"))]
        {
            let _ = (features, resolution);
            anyhow::bail!("OpenCL renderer training requires cargo build --features opencl");
        }
        #[cfg(feature = "opencl")]
        {
            anyhow::ensure!(
                self.compute_backend == ComputeBackend::OpenCl,
                "OpenCL decoder training requires --compute-backend opencl"
            );
            let layers = self.opencl_layers()?;
            let values = features.flatten_all()?.to_vec1::<f32>()?;
            let mut slot = self.opencl.lock().expect("OpenCL renderer mutex poisoned");
            self.initialize_opencl(&mut slot)?;
            let renderer = slot
                .renderer
                .as_mut()
                .context("OpenCL renderer did not initialize")?;
            let output = renderer.training_forward(&values, resolution * resolution, &layers)?;
            let boundary =
                Var::from_vec(output, (1, 6, resolution, resolution), features.device())?;
            let grounded = boundary.narrow(1, 0, 3)?;
            let emergent = boundary.narrow(1, 3, 3)?;
            slot.training_boundary = Some(boundary);
            Ok((grounded, emergent))
        }
    }

    pub fn render_decoder_training(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        genome: &Tensor,
        plan: &RenderPlan,
        emergence_strength: f32,
    ) -> Result<RenderOutput> {
        self.render_internal(
            micro,
            macro_field,
            genome,
            plan,
            emergence_strength,
            true,
            false,
            true,
        )
    }

    #[cfg(feature = "opencl")]
    pub fn populate_opencl_training_gradients(&self, gradients: &mut GradStore) -> Result<bool> {
        if self.compute_backend != ComputeBackend::OpenCl {
            return Ok(false);
        }
        let mut slot = self.opencl.lock().expect("OpenCL renderer mutex poisoned");
        let Some(boundary) = slot.training_boundary.take() else {
            return Ok(false);
        };
        let head_gradient = gradients
            .get(boundary.as_tensor())
            .context("loss did not produce OpenCL renderer boundary gradients")?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let renderer = slot
            .renderer
            .as_mut()
            .context("OpenCL renderer did not initialize")?;
        let layer_gradients = renderer.training_backward(&head_gradient)?;
        anyhow::ensure!(
            layer_gradients.len() == self.blocks.len() + 3,
            "OpenCL renderer gradient layer count mismatch"
        );
        insert_linear_gradients(gradients, &self.input, &layer_gradients[0])?;
        for (index, block) in self.blocks.iter().enumerate() {
            insert_linear_gradients(gradients, block, &layer_gradients[index + 1])?;
        }
        insert_linear_gradients(
            gradients,
            &self.grounded_head,
            &layer_gradients[self.blocks.len() + 1],
        )?;
        insert_linear_gradients(
            gradients,
            &self.emergent_head,
            &layer_gradients[self.blocks.len() + 2],
        )?;
        Ok(true)
    }

    pub fn refresh_opencl_weights(&self) -> Result<bool> {
        if self.compute_backend != ComputeBackend::OpenCl {
            return Ok(false);
        }
        #[cfg(not(feature = "opencl"))]
        {
            anyhow::bail!("OpenCL renderer requires cargo build --features opencl");
        }
        #[cfg(feature = "opencl")]
        {
            let layers = self.opencl_layers()?;
            let mut slot = self.opencl.lock().expect("OpenCL renderer mutex poisoned");
            self.initialize_opencl(&mut slot)?;
            let renderer = slot
                .renderer
                .as_mut()
                .context("OpenCL renderer did not initialize")?;
            renderer.refresh_weights(&layers)?;
            Ok(true)
        }
    }
    pub fn render(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        genome: &Tensor,
        plan: &RenderPlan,
        tracked: bool,
    ) -> Result<RenderOutput> {
        self.render_with_emergence(
            micro,
            macro_field,
            genome,
            plan,
            self.default_emergence_strength,
            tracked,
        )
    }

    pub fn render_with_emergence(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        genome: &Tensor,
        plan: &RenderPlan,
        emergence_strength: f32,
        tracked: bool,
    ) -> Result<RenderOutput> {
        self.render_internal(
            micro,
            macro_field,
            genome,
            plan,
            emergence_strength,
            tracked,
            false,
            false,
        )
    }

    pub fn render_attribution(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        genome: &Tensor,
        plan: &RenderPlan,
        emergence_strength: f32,
    ) -> Result<RenderOutput> {
        self.render_internal(
            micro,
            macro_field,
            genome,
            plan,
            emergence_strength,
            false,
            true,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_internal(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        genome: &Tensor,
        plan: &RenderPlan,
        emergence_strength: f32,
        tracked: bool,
        include_attribution: bool,
        opencl_training: bool,
    ) -> Result<RenderOutput> {
        let micro_up = plan.micro.apply(micro)?;
        let macro_up = plan.macro_field.apply(macro_field)?;
        let genome_field = broadcast_vector(genome, plan.resolution, plan.resolution)?;
        let features = Tensor::cat(
            &[
                &micro_up,
                &macro_up,
                &plan.coordinates,
                &genome_field,
                &plan.lod,
            ],
            1,
        )?;
        let accelerated = if opencl_training {
            Some(self.opencl_training_heads(&features, plan.resolution)?)
        } else if tracked {
            None
        } else {
            self.opencl_heads(&features, plan.resolution)?
        };
        let (grounded_learned, emergent_logits) = if let Some(heads) = accelerated {
            heads
        } else {
            let mut hidden = swish(&pixelwise_linear_mode(&features, &self.input, tracked)?)?;
            for block in &self.blocks {
                let residual = swish(&pixelwise_linear_mode(&hidden, block, tracked)?)?;
                hidden = hidden.add(&residual.affine(0.5, 0.0)?)?;
            }
            (
                pixelwise_linear_mode(&hidden, &self.grounded_head, tracked)?,
                pixelwise_linear_mode(&hidden, &self.emergent_head, tracked)?,
            )
        };
        let emergent_raw = emergent_logits
            .tanh()?
            .affine(self.emergent_limit as f64, 0.0)?;
        // A bounded, parameter-free path makes the actual organism observable
        // and prevents a coordinate-only decoder from satisfying statistics
        // while ignoring morphogenesis.
        let micro_channels: Vec<Tensor> = (0..12)
            .map(|channel| micro_up.narrow(1, channel, 1))
            .collect::<candle_core::Result<_>>()?;
        let macro_channels: Vec<Tensor> = (0..12)
            .map(|channel| macro_up.narrow(1, channel, 1))
            .collect::<candle_core::Result<_>>()?;
        let (lightness_basis, a_basis, b_basis) = match self.style {
            StylePreset::AlienFluid => (
                combine(&[
                    (&micro_channels[0], 1.0),
                    (&micro_channels[1], -1.0),
                    (&micro_channels[6], 0.15),
                    (&macro_channels[0], 0.30),
                ])?,
                combine(&[
                    (&micro_channels[2], 0.75),
                    (&micro_channels[4], 0.45),
                    (&micro_channels[7], -0.35),
                    (&macro_channels[2], 0.20),
                ])?,
                combine(&[
                    (&micro_channels[3], 0.75),
                    (&micro_channels[5], 0.45),
                    (&micro_channels[8], 0.35),
                    (&macro_channels[3], 0.20),
                ])?,
            ),
            StylePreset::FractalFlame => (
                combine(&[
                    (&micro_channels[4], 1.15),
                    (&micro_channels[0], 0.25),
                    (&micro_channels[1], -0.25),
                    (&macro_channels[4], 0.35),
                ])?,
                combine(&[
                    (&micro_channels[4], 0.90),
                    (&micro_channels[2], 0.20),
                    (&micro_channels[7], -0.30),
                ])?,
                combine(&[
                    (&micro_channels[5], 0.55),
                    (&micro_channels[3], 0.20),
                    (&micro_channels[8], 0.45),
                    (&macro_channels[5], 0.25),
                ])?,
            ),
            StylePreset::ReactionGarden => (
                combine(&[
                    (&micro_channels[0], 1.20),
                    (&micro_channels[1], -1.20),
                    (&macro_channels[0], 0.40),
                    (&macro_channels[1], -0.40),
                ])?,
                combine(&[
                    (&micro_channels[1], 1.0),
                    (&micro_channels[6], 0.45),
                    (&micro_channels[7], -0.35),
                ])?,
                combine(&[
                    (&micro_channels[0], 0.55),
                    (&micro_channels[1], -0.75),
                    (&micro_channels[8], 0.50),
                ])?,
            ),
            StylePreset::Quasicrystal => (
                combine(&[
                    (&micro_channels[5], 0.85),
                    (&micro_channels[2], 0.25),
                    (&macro_channels[5], 0.35),
                ])?,
                combine(&[
                    (&micro_channels[5], 1.0),
                    (&micro_channels[2], 0.35),
                    (&micro_channels[7], -0.20),
                ])?,
                combine(&[
                    (&micro_channels[5], -0.70),
                    (&micro_channels[3], 0.35),
                    (&micro_channels[8], 0.25),
                ])?,
            ),
            StylePreset::PureNca => (
                combine(&[
                    (&micro_channels[9], 1.0),
                    (&micro_channels[10], 0.50),
                    (&macro_channels[9], 0.30),
                ])?,
                combine(&[(&micro_channels[10], 1.0), (&micro_channels[11], -0.65)])?,
                combine(&[(&micro_channels[11], 1.0), (&micro_channels[9], -0.65)])?,
            ),
        };
        let state_lab = Tensor::cat(&[&lightness_basis, &a_basis, &b_basis], 1)?
            .tanh()?
            .affine(self.state_skip as f64, 0.0)?;
        let grounded_lab = grounded_learned.add(&state_lab)?;
        let emergent_shaped = scale_dependent_residual(
            &emergent_raw,
            self.emergence_low_budget,
            self.emergence_mid_budget,
        )?;
        let emergent_lab = smooth_limit(&emergent_shaped, self.emergent_limit)?;
        let composite_lab =
            grounded_lab.add(&emergent_lab.affine(emergence_strength as f64, 0.0)?)?;
        let (grounded_image, _) = lab_to_image(&grounded_lab, self.chroma, self.gamma)?;
        let (image, gamut_excess) = lab_to_image(&composite_lab, self.chroma, self.gamma)?;
        let emergent_visual = emergent_lab
            .affine(0.5 / self.emergent_limit.max(1e-6) as f64, 0.5)?
            .clamp(0.0f32, 1.0f32)?;
        let (state_only_image, learned_only_image) = if include_attribution {
            let learned_lab =
                grounded_learned.add(&emergent_lab.affine(emergence_strength as f64, 0.0)?)?;
            (
                Some(lab_to_image(&state_lab, self.chroma, self.gamma)?.0),
                Some(lab_to_image(&learned_lab, self.chroma, self.gamma)?.0),
            )
        } else {
            (None, None)
        };
        Ok(RenderOutput {
            image,
            grounded_image,
            emergent_visual,
            grounded_lab,
            emergent_lab,
            emergence_strength,
            gamut_excess,
            state_only_image,
            learned_only_image,
        })
    }
}

fn scale_dependent_residual(
    residual: &Tensor,
    low_budget: f32,
    mid_budget: f32,
) -> candle_core::Result<Tensor> {
    let mid_lowpass = periodic_tensor_blur(residual)?;
    let mut low = mid_lowpass.clone();
    for _ in 0..3 {
        low = periodic_tensor_blur(&low)?;
    }
    let mid = mid_lowpass.sub(&low)?;
    let high = residual.sub(&mid_lowpass)?;
    low.affine(low_budget as f64, 0.0)?
        .add(&mid.affine(mid_budget as f64, 0.0)?)?
        .add(&high)
}

fn periodic_tensor_blur(value: &Tensor) -> candle_core::Result<Tensor> {
    let (_, _, height, width) = value.dims4()?;
    value
        .affine(0.5, 0.0)?
        .add(&periodic_shift(value, 0, 1)?.affine(0.125, 0.0)?)?
        .add(&periodic_shift(value, 0, width - 1)?.affine(0.125, 0.0)?)?
        .add(&periodic_shift(value, 1, 0)?.affine(0.125, 0.0)?)?
        .add(&periodic_shift(value, height - 1, 0)?.affine(0.125, 0.0)?)
}

fn lab_to_image(
    lab_raw: &Tensor,
    chroma: f32,
    gamma: f64,
) -> candle_core::Result<(Tensor, Tensor)> {
    let l = candle_nn::ops::sigmoid(&lab_raw.narrow(1, 0, 1)?)?.affine(0.84, 0.08)?;
    let a = lab_raw
        .narrow(1, 1, 1)?
        .tanh()?
        .affine(chroma as f64, 0.0)?;
    let b = lab_raw
        .narrow(1, 2, 1)?
        .tanh()?
        .affine(chroma as f64, 0.0)?;
    let rgb_linear = oklab_to_linear_rgb(&l, &a, &b)?;
    let clipped = rgb_linear.clamp(0.0f32, 1.0f32)?;
    let gamut_excess = rgb_linear.sub(&clipped)?.abs()?.mean_all()?;
    let image = clipped
        .affine(1.0, 1e-6)?
        .powf(1.0 / gamma)?
        .clamp(0.0f32, 1.0f32)?;
    Ok((image, gamut_excess))
}

#[cfg(feature = "opencl")]
fn insert_linear_gradients(
    gradients: &mut GradStore,
    linear: &Linear,
    values: &crate::opencl::LinearGradData,
) -> Result<()> {
    let weight = Tensor::from_vec(
        values.weight.clone(),
        linear.weight().shape().clone(),
        linear.weight().device(),
    )?;
    let weight = if let Some(existing) = gradients.get(linear.weight()) {
        existing.add(&weight)?
    } else {
        weight
    };
    gradients.insert(linear.weight(), weight);
    let bias = linear
        .bias()
        .context("OpenCL renderer requires linear biases")?;
    let bias_gradient = Tensor::from_vec(values.bias.clone(), bias.shape().clone(), bias.device())?;
    let bias_gradient = if let Some(existing) = gradients.get(bias) {
        existing.add(&bias_gradient)?
    } else {
        bias_gradient
    };
    gradients.insert(bias, bias_gradient);
    Ok(())
}
#[cfg(feature = "opencl")]
fn linear_data(linear: &Linear) -> Result<crate::opencl::LinearData> {
    let (output, input) = linear.weight().dims2()?;
    let weight = linear.weight().flatten_all()?.to_vec1::<f32>()?;
    let bias = linear
        .bias()
        .context("OpenCL renderer requires linear biases")?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(crate::opencl::LinearData {
        input,
        output,
        weight,
        bias,
    })
}

fn zero_linear(input: usize, output: usize, vb: VarBuilder<'_>) -> Result<Linear> {
    let weight = vb.get_with_hints((output, input), "weight", Init::Const(0.0))?;
    let bias = vb.get_with_hints(output, "bias", Init::Const(0.0))?;
    Ok(Linear::new(weight, Some(bias)))
}
fn swish(x: &Tensor) -> candle_core::Result<Tensor> {
    x.mul(&candle_nn::ops::sigmoid(x)?)
}

fn combine(terms: &[(&Tensor, f64)]) -> candle_core::Result<Tensor> {
    let mut output = terms[0].0.affine(terms[0].1, 0.0)?;
    for (tensor, gain) in &terms[1..] {
        output = output.add(&tensor.affine(*gain, 0.0)?)?;
    }
    Ok(output)
}

fn oklab_to_linear_rgb(l: &Tensor, a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    let lp = l
        .add(&a.affine(0.396_337_78, 0.0)?)?
        .add(&b.affine(0.215_803_76, 0.0)?)?;
    let mp = l
        .sub(&a.affine(0.105_561_346, 0.0)?)?
        .sub(&b.affine(0.063_854_17, 0.0)?)?;
    let sp = l
        .sub(&a.affine(0.089_484_18, 0.0)?)?
        .sub(&b.affine(1.291_485_5, 0.0)?)?;
    let ll = lp.sqr()?.mul(&lp)?;
    let mm = mp.sqr()?.mul(&mp)?;
    let ss = sp.sqr()?.mul(&sp)?;
    let r = ll
        .affine(4.076_741_7, 0.0)?
        .sub(&mm.affine(3.307_711_6, 0.0)?)?
        .add(&ss.affine(0.230_969_94, 0.0)?)?;
    let g = ll
        .affine(-1.268_438, 0.0)?
        .add(&mm.affine(2.609_757_4, 0.0)?)?
        .sub(&ss.affine(0.341_319_4, 0.0)?)?;
    let blue = ll
        .affine(-0.004_196_086_3, 0.0)?
        .sub(&mm.affine(0.703_418_6, 0.0)?)?
        .add(&ss.affine(1.707_614_7, 0.0)?)?;
    Tensor::cat(&[&r, &g, &blue], 1)
}

pub fn tensor_to_planar_rgb(image: &Tensor) -> Result<Vec<f32>> {
    let (_, channels, h, w) = image.dims4()?;
    anyhow::ensure!(channels == 3, "renderer returned {channels} channels");
    Ok(image.reshape((channels * h * w,))?.to_vec1::<f32>()?)
}

pub fn save_png(image: &Tensor, path: &Path) -> Result<()> {
    let (_, channels, h, w) = image.dims4()?;
    anyhow::ensure!(channels == 3, "expected RGB image");
    let values = tensor_to_planar_rgb(image)?;
    save_planar_png(&values, w, h, path)
}

pub fn save_mastered_png(image: &Tensor, path: &Path, strength: f32) -> Result<()> {
    let (_, _, h, w) = image.dims4()?;
    let mut values = tensor_to_planar_rgb(image)?;
    toroidal_master(&mut values, w, h, strength);
    save_planar_png(&values, w, h, path)
}

/// Save every recurrent channel as a signed-color tile. This is diagnostic
/// evidence, not a mastered artwork: cyan and magenta indicate opposite signs,
/// and every channel is normalized independently around its own mean.
pub fn save_state_atlas(field: &Tensor, path: &Path) -> Result<()> {
    let (_, channels, height, width) = field.dims4()?;
    let values = field.detach().flatten_all()?.to_vec1::<f32>()?;
    let columns = (channels as f64).sqrt().ceil() as usize;
    let rows = channels.div_ceil(columns);
    let mut output = image::RgbImage::new((columns * width) as u32, (rows * height) as u32);
    let plane = width * height;
    for channel in 0..channels {
        let slice = &values[channel * plane..(channel + 1) * plane];
        let mean = slice.iter().sum::<f32>() / plane as f32;
        let rms = (slice
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / plane as f32)
            .sqrt()
            .max(1e-6);
        let tile_x = channel % columns;
        let tile_y = channel / columns;
        for y in 0..height {
            for x in 0..width {
                let z = ((slice[y * width + x] - mean) / (2.5 * rms)).clamp(-1.0, 1.0);
                let magnitude = z.abs();
                let rgb = [
                    (0.5 + 0.48 * z).clamp(0.0, 1.0),
                    (0.52 - 0.30 * magnitude).clamp(0.0, 1.0),
                    (0.5 - 0.48 * z).clamp(0.0, 1.0),
                ];
                output.put_pixel(
                    (tile_x * width + x) as u32,
                    (tile_y * height + y) as u32,
                    image::Rgb([
                        (rgb[0] * 255.0).round() as u8,
                        (rgb[1] * 255.0).round() as u8,
                        (rgb[2] * 255.0).round() as u8,
                    ]),
                );
            }
        }
    }
    let temporary = png_temporary_path(path);
    output.save_with_format(&temporary, image::ImageFormat::Png)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

fn save_planar_png(values: &[f32], width: usize, height: usize, path: &Path) -> Result<()> {
    let plane = width * height;
    let mut output = image::RgbImage::new(width as u32, height as u32);
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let channel =
                |c: usize| (values[c * plane + index].clamp(0.0, 1.0) * 255.0).round() as u8;
            output.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([channel(0), channel(1), channel(2)]),
            );
        }
    }
    let temporary = png_temporary_path(path);
    output
        .save_with_format(&temporary, image::ImageFormat::Png)
        .with_context(|| format!("cannot write temporary PNG {}", temporary.display()))?;
    std::fs::rename(&temporary, path)
        .with_context(|| format!("cannot publish PNG {}", path.display()))?;
    Ok(())
}

fn png_temporary_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".tmp");
    PathBuf::from(name)
}

fn toroidal_master(rgb: &mut [f32], width: usize, height: usize, strength: f32) {
    if strength <= 0.0 {
        return;
    }
    let plane = width * height;
    let mut luma = vec![0.0f32; plane];
    for index in 0..plane {
        luma[index] =
            0.299 * rgb[index] + 0.587 * rgb[plane + index] + 0.114 * rgb[2 * plane + index];
    }
    let mut local = luma.clone();
    let mut glow: Vec<f32> = luma.iter().map(|value| (value - 0.56).max(0.0)).collect();
    for _ in 0..3 {
        local = periodic_blur(&local, width, height);
    }
    for _ in 0..5 {
        glow = periodic_blur(&glow, width, height);
    }
    for channel in 0..3 {
        for index in 0..plane {
            let detail = 0.16 * strength * (luma[index] - local[index]);
            let bloom = 0.30 * strength * glow[index];
            let value = (rgb[channel * plane + index] + detail + bloom).max(0.0);
            let shoulder = 0.86 + 0.14 * value;
            rgb[channel * plane + index] = (value / shoulder).clamp(0.0, 1.0);
        }
    }
}

fn periodic_blur(values: &[f32], width: usize, height: usize) -> Vec<f32> {
    let mut output = vec![0.0; values.len()];
    for y in 0..height {
        for x in 0..width {
            let left = values[y * width + (x + width - 1) % width];
            let right = values[y * width + (x + 1) % width];
            let up = values[((y + height - 1) % height) * width + x];
            let down = values[((y + 1) % height) * width + x];
            output[y * width + x] =
                0.5 * values[y * width + x] + 0.125 * (left + right + up + down);
        }
    }
    output
}

pub fn save_contact_sheet(images: &[PathBuf], path: &Path) -> Result<()> {
    if images.is_empty() {
        return Ok(());
    }
    let decoded: Vec<image::RgbImage> = images
        .iter()
        .map(|image_path| {
            image::open(image_path)
                .with_context(|| format!("cannot reopen gallery image {}", image_path.display()))
                .map(|image| image.to_rgb8())
        })
        .collect::<Result<_>>()?;
    let width = decoded[0].width();
    let height = decoded[0].height();
    let columns = (images.len() as f64).sqrt().ceil() as u32;
    let rows = (images.len() as u32).div_ceil(columns);
    let mut sheet = image::RgbImage::new(width * columns, height * rows);
    for (index, image) in decoded.iter().enumerate() {
        anyhow::ensure!(
            image.width() == width && image.height() == height,
            "gallery dimensions differ"
        );
        let x = index as u32 % columns;
        let y = index as u32 / columns;
        image::imageops::overlay(
            &mut sheet,
            image,
            i64::from(x * width),
            i64::from(y * height),
        );
    }
    let temporary = png_temporary_path(path);
    sheet.save_with_format(&temporary, image::ImageFormat::Png)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}

pub fn save_contact_sheet_resized(images: &[PathBuf], path: &Path, tile_size: u32) -> Result<()> {
    if images.is_empty() {
        return Ok(());
    }
    let decoded: Vec<image::RgbImage> = images
        .iter()
        .map(|image_path| {
            image::open(image_path)
                .with_context(|| format!("cannot reopen diagnostic image {}", image_path.display()))
                .map(|image| {
                    image
                        .resize_to_fill(tile_size, tile_size, image::imageops::FilterType::Lanczos3)
                        .to_rgb8()
                })
        })
        .collect::<Result<_>>()?;
    let columns = (images.len() as f64).sqrt().ceil() as u32;
    let rows = (images.len() as u32).div_ceil(columns);
    let mut sheet = image::RgbImage::new(tile_size * columns, tile_size * rows);
    for (index, image) in decoded.iter().enumerate() {
        let x = index as u32 % columns;
        let y = index as u32 / columns;
        image::imageops::overlay(
            &mut sheet,
            image,
            i64::from(x * tile_size),
            i64::from(y * tile_size),
        );
    }
    let temporary = png_temporary_path(path);
    sheet.save_with_format(&temporary, image::ImageFormat::Png)?;
    std::fs::rename(temporary, path)?;
    Ok(())
}
pub fn seam_energy(image: &Tensor) -> candle_core::Result<Tensor> {
    let (_, _, h, w) = image.dims4()?;
    let horizontal = image
        .narrow(3, 0, 1)?
        .sub(&image.narrow(3, w - 1, 1)?)?
        .sqr()?
        .mean_all()?;
    let vertical = image
        .narrow(2, 0, 1)?
        .sub(&image.narrow(2, h - 1, 1)?)?
        .sqr()?
        .mean_all()?;
    horizontal.add(&vertical)?.affine(0.5, 0.0)
}

pub fn channel_moments(image: &Tensor) -> candle_core::Result<(Tensor, Tensor)> {
    let (_, channels, h, w) = image.dims4()?;
    let flat = image.reshape((channels, h * w))?;
    let mean = flat.mean(D::Minus1)?;
    let variance = flat
        .broadcast_sub(&mean.unsqueeze(1)?)?
        .sqr()?
        .mean(D::Minus1)?;
    Ok((mean, variance))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn constant_image_has_zero_seam_energy() -> candle_core::Result<()> {
        let image = Tensor::ones((1, 3, 16, 16), DType::F32, &Device::Cpu)?;
        assert!(seam_energy(&image)?.to_scalar::<f32>()? < 1e-8);
        Ok(())
    }

    #[test]
    fn zero_emergence_has_exactly_zero_composite_influence() -> Result<()> {
        let config = RunConfig {
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
        };
        let variables = candle_nn::VarMap::new();
        let renderer = ImplicitRenderer::new(
            &config,
            candle_nn::VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu).pp("renderer"),
        )?;
        let plan = RenderPlan::new(&config, 24, &Device::Cpu)?;
        let micro = Tensor::zeros((1, 12, 24, 24), DType::F32, &Device::Cpu)?;
        let macro_field = Tensor::zeros((1, 12, 12, 12), DType::F32, &Device::Cpu)?;
        let genome = Tensor::zeros(4, DType::F32, &Device::Cpu)?;
        let rendered =
            renderer.render_with_emergence(&micro, &macro_field, &genome, &plan, 0.0, true)?;
        assert_eq!(
            rendered
                .image
                .sub(&rendered.grounded_image)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?,
            0.0
        );
        assert_eq!(
            rendered.emergent_lab.abs()?.max_all()?.to_scalar::<f32>()?,
            0.0
        );
        Ok(())
    }

    #[test]
    fn crop_plan_uses_global_coordinates_and_requested_shape() -> Result<()> {
        let config = RunConfig {
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
        };
        let view = SpatialView {
            x: 0.5,
            y: 0.25,
            size: 0.25,
            zoom: 4.0,
        };
        let plan = RenderPlan::new_view(&config, 32, view, &Device::Cpu)?;
        let micro = Tensor::zeros((1, 12, 24, 24), DType::F32, &Device::Cpu)?;
        let macro_field = Tensor::zeros((1, 12, 12, 12), DType::F32, &Device::Cpu)?;
        let (micro, macro_field) = plan.observe_fields(&micro, &macro_field)?;
        assert_eq!(micro.dims4()?, (1, 12, 32, 32));
        assert_eq!(macro_field.dims4()?, (1, 12, 32, 32));
        assert_eq!(plan.view, view);
        Ok(())
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_renderer_matches_cpu() -> Result<()> {
        if std::env::var_os("TITAN_OPENCL_TEST").is_none() {
            return Ok(());
        }
        let base = RunConfig {
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            genome_dim: 4,
            render_hidden: 32,
            render_blocks: 2,
            coord_bands: 2,
            train_resolution: 24,
            output_resolution: 24,
            ..RunConfig::default()
        };
        let vars = candle_nn::VarMap::new();
        let cpu = ImplicitRenderer::new(
            &base,
            candle_nn::VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("renderer"),
        )?;
        let mut gpu_config = base.clone();
        gpu_config.compute_backend = ComputeBackend::OpenCl;
        let gpu = ImplicitRenderer::new(
            &gpu_config,
            candle_nn::VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("renderer"),
        )?;
        {
            let data = vars.data().lock().expect("VarMap mutex poisoned");
            for (name, variable) in data.iter() {
                let values = (0..variable.elem_count())
                    .map(|index| ((index as f32 * 0.017 + name.len() as f32).sin()) * 0.04)
                    .collect::<Vec<_>>();
                variable.set(&Tensor::from_vec(
                    values,
                    variable.shape().clone(),
                    &Device::Cpu,
                )?)?;
            }
        }
        let plan = RenderPlan::new(&base, 24, &Device::Cpu)?;
        let micro_values = (0..12 * 24 * 24)
            .map(|i| (i as f32 * 0.013).sin() * 0.3)
            .collect::<Vec<_>>();
        let macro_values = (0..12 * 12 * 12)
            .map(|i| (i as f32 * 0.019).cos() * 0.2)
            .collect::<Vec<_>>();
        let micro = Tensor::from_vec(micro_values, (1, 12, 24, 24), &Device::Cpu)?;
        let macro_field = Tensor::from_vec(macro_values, (1, 12, 12, 12), &Device::Cpu)?;
        let genome = Tensor::new(&[0.1f32, -0.2, 0.3, -0.4], &Device::Cpu)?;
        let cpu = cpu.render(&micro, &macro_field, &genome, &plan, false)?;
        let info = gpu
            .prepare_inference_backend()?
            .context("OpenCL did not initialize")?;
        let gpu = gpu.render(&micro, &macro_field, &genome, &plan, false)?;
        let difference = gpu.image.sub(&cpu.image)?.abs()?;
        let max_abs = difference.max_all()?.to_scalar::<f32>()?;
        let mean_abs = difference.mean_all()?.to_scalar::<f32>()?;
        let rms = difference.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
        eprintln!(
            "OPENCL parity | {info} | max_abs={max_abs:.8} mean_abs={mean_abs:.8} rms={rms:.8}"
        );
        assert!(max_abs <= 2e-4, "OpenCL max abs drift {max_abs}");
        assert!(mean_abs <= 2e-5, "OpenCL mean abs drift {mean_abs}");
        Ok(())
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_checkpoint_renderer_parity_and_benchmark() -> Result<()> {
        let Some(model_path) = std::env::var_os("TITAN_OPENCL_CHECKPOINT_MODEL") else {
            return Ok(());
        };
        let world_path = std::env::var_os("TITAN_OPENCL_CHECKPOINT_WORLD")
            .context("TITAN_OPENCL_CHECKPOINT_WORLD is required")?;
        let base = RunConfig {
            output_resolution: 384,
            snapshot_resolution: 384,
            ..RunConfig::default()
        };
        let vars = candle_nn::VarMap::new();
        let cpu = ImplicitRenderer::new(
            &base,
            candle_nn::VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("renderer"),
        )?;
        let mut gpu_config = base.clone();
        gpu_config.compute_backend = ComputeBackend::OpenCl;
        let gpu = ImplicitRenderer::new(
            &gpu_config,
            candle_nn::VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("renderer"),
        )?;
        let saved = candle_core::safetensors::load(model_path, &Device::Cpu)?;
        {
            let data = vars.data().lock().expect("VarMap mutex poisoned");
            for (name, variable) in data.iter() {
                variable.set(
                    saved
                        .get(name)
                        .with_context(|| format!("checkpoint missing {name}"))?,
                )?;
            }
        }
        let world = candle_core::safetensors::load(world_path, &Device::Cpu)?;
        let micro = world.get("world.micro").context("world missing micro")?;
        let macro_field = world.get("world.macro").context("world missing macro")?;
        let genome = Tensor::zeros(base.genome_dim, DType::F32, &Device::Cpu)?;
        let info = gpu
            .prepare_inference_backend()?
            .context("OpenCL did not initialize")?;
        for resolution in [192usize, 384] {
            let plan = RenderPlan::new(&base, resolution, &Device::Cpu)?;
            let started = std::time::Instant::now();
            let cpu_output = cpu.render(micro, macro_field, &genome, &plan, false)?;
            let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = std::time::Instant::now();
            let gpu_output = gpu.render(micro, macro_field, &genome, &plan, false)?;
            let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;
            let difference = gpu_output.image.sub(&cpu_output.image)?.abs()?;
            let max_abs = difference.max_all()?.to_scalar::<f32>()?;
            let mean_abs = difference.mean_all()?.to_scalar::<f32>()?;
            let rms = difference.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
            eprintln!("OPENCL checkpoint benchmark | {resolution}px | cpu_ms={cpu_ms:.3} gpu_ms={gpu_ms:.3} speedup={:.3}x max_abs={max_abs:.8} mean_abs={mean_abs:.8} rms={rms:.8}", cpu_ms / gpu_ms);
            assert!(max_abs <= 5e-4, "OpenCL max abs drift {max_abs}");
            assert!(mean_abs <= 5e-5, "OpenCL mean abs drift {mean_abs}");
        }
        eprintln!("OPENCL checkpoint device | {info}");
        Ok(())
    }
}
