use crate::config::RunConfig;
use crate::tensor_ops::{smooth_limit, PeriodicUpsampler};
use anyhow::Result;
use candle_core::{Device, Tensor, D};
use candle_nn::{Init, Linear, Module, RmsNorm, VarBuilder};

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Distribution {
    pub rms: f64,
    pub min: f32,
    pub p01: f32,
    pub p05: f32,
    pub median: f32,
    pub p95: f32,
    pub p99: f32,
    pub max: f32,
}
impl Distribution {
    pub fn of(value: &Tensor) -> Result<Self> {
        let mut v = value.flatten_all()?.to_vec1::<f32>()?;
        anyhow::ensure!(
            !v.is_empty() && v.iter().all(|x| x.is_finite()),
            "invalid diagnostic tensor"
        );
        let rms = (v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        v.sort_by(f32::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        Ok(Self {
            rms,
            min: v[0],
            p01: q(0.01),
            p05: q(0.05),
            median: q(0.5),
            p95: q(0.95),
            p99: q(0.99),
            max: v[v.len() - 1],
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct WriteStatistics {
    pub logits: Distribution,
    pub absolute_logits: Distribution,
    /// |logit| > atanh(0.99).
    pub saturated_fraction: f64,
    pub mean_tanh_derivative: f64,
    pub derivative_below_001: f64,
    pub derivative_below_0001: f64,
    /// Mean over channels of spatial variance; excludes between-channel bias.
    pub spatial_write_variance: f64,
}
impl WriteStatistics {
    fn new(logits: &Tensor, write: &Tensor) -> Result<Self> {
        let v = logits.flatten_all()?.to_vec1::<f32>()?;
        let derivatives: Vec<f64> = v
            .iter()
            .map(|x| 1.0 - f64::from(*x).tanh().powi(2))
            .collect();
        let fraction = |threshold: f64| {
            derivatives.iter().filter(|x| **x < threshold).count() as f64 / v.len() as f64
        };
        let (_, channels, height, width) = write.dims4()?;
        let values = write.flatten_all()?.to_vec1::<f32>()?;
        let spatial_write_variance = values
            .chunks(height * width)
            .map(|c| {
                let mean = c.iter().map(|x| f64::from(*x)).sum::<f64>() / c.len() as f64;
                c.iter()
                    .map(|x| (f64::from(*x) - mean).powi(2))
                    .sum::<f64>()
                    / c.len() as f64
            })
            .sum::<f64>()
            / channels as f64;
        Ok(Self {
            logits: Distribution::of(logits)?,
            absolute_logits: Distribution::of(&logits.abs()?)?,
            saturated_fraction: v.iter().filter(|x| x.abs() > 2.6466525).count() as f64
                / v.len() as f64,
            mean_tanh_derivative: derivatives.iter().sum::<f64>() / v.len() as f64,
            derivative_below_001: fraction(0.01),
            derivative_below_0001: fraction(0.001),
            spatial_write_variance,
        })
    }
}

#[derive(Default, Serialize)]
pub struct InterfaceTrace {
    pub micro: Option<WriteStatistics>,
    pub macro_field: Option<WriteStatistics>,
    pub token_rms_per_loop: Vec<f32>,
    pub gru_reset: Vec<Distribution>,
    pub gru_update: Vec<Distribution>,
    #[serde(skip)]
    micro_logits: Option<Tensor>,
    #[serde(skip)]
    macro_logits: Option<Tensor>,
}

pub struct InterfaceOutput {
    /// Weighted pre-tanh loss; absent outside opted-in tracked execution.
    pub saturation_penalty: Option<Tensor>,
    pub micro_bias: Tensor,
    pub macro_bias: Tensor,
    pub memory: Tensor,
}

struct GruCell {
    reset_input: Linear,
    reset_memory: Linear,
    update_input: Linear,
    update_memory: Linear,
    candidate_input: Linear,
    candidate_memory: Linear,
}

impl GruCell {
    fn new(width: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            reset_input: candle_nn::linear(width, width, vb.pp("reset_input"))?,
            reset_memory: candle_nn::linear(width, width, vb.pp("reset_memory"))?,
            update_input: candle_nn::linear(width, width, vb.pp("update_input"))?,
            update_memory: candle_nn::linear(width, width, vb.pp("update_memory"))?,
            candidate_input: candle_nn::linear(width, width, vb.pp("candidate_input"))?,
            candidate_memory: candle_nn::linear(width, width, vb.pp("candidate_memory"))?,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        memory: &Tensor,
        tracked: bool,
        trace: Option<&mut InterfaceTrace>,
    ) -> Result<Tensor> {
        let reset = candle_nn::ops::sigmoid(
            &linear_mode(&self.reset_input, input, tracked)?.add(&linear_mode(
                &self.reset_memory,
                memory,
                tracked,
            )?)?,
        )?;
        let update = candle_nn::ops::sigmoid(
            &linear_mode(&self.update_input, input, tracked)?.add(&linear_mode(
                &self.update_memory,
                memory,
                tracked,
            )?)?,
        )?;
        if let Some(trace) = trace {
            trace.gru_reset.push(Distribution::of(&reset)?);
            trace.gru_update.push(Distribution::of(&update)?);
        }
        let candidate = linear_mode(&self.candidate_input, input, tracked)?
            .add(&reset.mul(&linear_mode(&self.candidate_memory, memory, tracked)?)?)?
            .tanh()?;
        update
            .affine(-1.0, 1.0)?
            .mul(memory)?
            .add(&update.mul(&candidate)?)
            .map_err(Into::into)
    }
}

struct MorphicBlock {
    norm: RmsNorm,
    expand: Linear,
    contract: Linear,
}

impl MorphicBlock {
    fn new(width: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            norm: candle_nn::rms_norm(width, 1e-5, vb.pp("norm"))?,
            expand: candle_nn::linear(width, width * 2, vb.pp("expand"))?,
            contract: zero_linear(width * 2, width, vb.pp("contract"))?,
        })
    }

    fn forward(
        &self,
        value: &Tensor,
        index: usize,
        residual_gain: f32,
        tracked: bool,
        differentiable: bool,
    ) -> Result<Tensor> {
        let hidden = swish(&linear_mode(
            &self.expand,
            &norm_mode(&self.norm, value, tracked && differentiable)?,
            tracked,
        )?)?;
        let gain = residual_gain as f64 / ((index + 1) as f64).sqrt();
        value
            .add(&linear_mode(&self.contract, &hidden, tracked)?.affine(gain, 0.0)?)
            .map_err(Into::into)
    }
}

/// A phone-sized recurrent interface between dense spatial state and global
/// reasoning. Expensive self-attention is confined to a small pooled token
/// grid; the same transformer block is looped with shared weights.
pub struct RecurrentInterface {
    micro_projection: Linear,
    macro_projection: Linear,
    conditioning_projection: Linear,
    attention_norm: RmsNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    attention_output: Linear,
    feedforward_norm: RmsNorm,
    feedforward_expand: Linear,
    feedforward_contract: Linear,
    gru: GruCell,
    morphic: Vec<MorphicBlock>,
    micro_write: Linear,
    macro_write: Linear,
    micro_upsampler: PeriodicUpsampler,
    macro_upsampler: PeriodicUpsampler,
    grid: usize,
    width: usize,
    loops: usize,
    attention_grid: usize,
    morph_residual_gain: f32,
    memory_limit: f32,
    gain: f32,
    differentiable_norm: bool,
    saturation_penalty: Option<crate::experiment::SaturationPenalty>,
}

impl RecurrentInterface {
    pub fn new(config: &RunConfig, vb: VarBuilder<'_>, device: &Device) -> Result<Self> {
        let spatial_input = config.channels + 3;
        let micro_projection = candle_nn::linear(
            spatial_input,
            config.interface_width,
            vb.pp("micro_projection"),
        )?;
        let macro_projection = candle_nn::linear(
            spatial_input,
            config.interface_width,
            vb.pp("macro_projection"),
        )?;
        let conditioning_projection = candle_nn::linear(
            config.genome_dim + 2,
            config.interface_width,
            vb.pp("conditioning_projection"),
        )?;
        let attention_norm =
            candle_nn::rms_norm(config.interface_width, 1e-5, vb.pp("attention_norm"))?;
        let query = candle_nn::linear(
            config.interface_width,
            config.interface_width,
            vb.pp("query"),
        )?;
        let key = candle_nn::linear(config.interface_width, config.interface_width, vb.pp("key"))?;
        let value = candle_nn::linear(
            config.interface_width,
            config.interface_width,
            vb.pp("value"),
        )?;
        let attention_output = candle_nn::linear(
            config.interface_width,
            config.interface_width,
            vb.pp("attention_output"),
        )?;
        let feedforward_norm =
            candle_nn::rms_norm(config.interface_width, 1e-5, vb.pp("feedforward_norm"))?;
        let feedforward_expand = candle_nn::linear(
            config.interface_width,
            config.interface_width * 2,
            vb.pp("feedforward_expand"),
        )?;
        let feedforward_contract = candle_nn::linear(
            config.interface_width * 2,
            config.interface_width,
            vb.pp("feedforward_contract"),
        )?;
        let gru = GruCell::new(config.interface_width, vb.pp("gru"))?;
        let mut morphic = Vec::with_capacity(config.morph_layers);
        for index in 0..config.morph_layers {
            morphic.push(MorphicBlock::new(
                config.interface_width,
                vb.pp(format!("morphic_{index:03}")),
            )?);
        }
        let micro_write = zero_linear(
            config.interface_width,
            config.channels,
            vb.pp("micro_write"),
        )?;
        let macro_write = zero_linear(
            config.interface_width,
            config.channels,
            vb.pp("macro_write"),
        )?;
        Ok(Self {
            micro_projection,
            macro_projection,
            conditioning_projection,
            attention_norm,
            query,
            key,
            value,
            attention_output,
            feedforward_norm,
            feedforward_expand,
            feedforward_contract,
            gru,
            morphic,
            micro_write,
            macro_write,
            micro_upsampler: PeriodicUpsampler::new(
                config.interface_grid,
                config.interface_grid,
                config.micro_size,
                config.micro_size,
                device,
            )?,
            macro_upsampler: PeriodicUpsampler::new(
                config.interface_grid,
                config.interface_grid,
                config.macro_size,
                config.macro_size,
                device,
            )?,
            grid: config.interface_grid,
            width: config.interface_width,
            loops: config.interface_loops,
            attention_grid: config.interface_grid.min(8),
            gain: config.interface_gain,
            saturation_penalty: config.experiment.active_saturation_penalty().copied(),
            differentiable_norm: config.experiment.norm
                == crate::experiment::NormTraining::Differentiable,
            morph_residual_gain: config.morph_residual_gain,
            memory_limit: config.memory_limit,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        reference_micro: &Tensor,
        reference_macro: &Tensor,
        genome: &Tensor,
        memory: &Tensor,
        fidelity: f32,
        age_phase: f32,
        tracked: bool,
        active_morph_depth: usize,
    ) -> Result<InterfaceOutput> {
        self.forward_impl(
            micro,
            macro_field,
            reference_micro,
            reference_macro,
            genome,
            memory,
            fidelity,
            age_phase,
            tracked,
            active_morph_depth,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inspect(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        reference_micro: &Tensor,
        reference_macro: &Tensor,
        genome: &Tensor,
        memory: &Tensor,
        fidelity: f32,
        age_phase: f32,
        active_morph_depth: usize,
    ) -> Result<(InterfaceOutput, InterfaceTrace)> {
        let mut trace = InterfaceTrace::default();
        let output = self.forward_impl(
            micro,
            macro_field,
            reference_micro,
            reference_macro,
            genome,
            memory,
            fidelity,
            age_phase,
            false,
            active_morph_depth,
            Some(&mut trace),
        )?;
        trace.micro = Some(WriteStatistics::new(
            &trace.micro_logits.take().unwrap(),
            &output.micro_bias,
        )?);
        trace.macro_field = Some(WriteStatistics::new(
            &trace.macro_logits.take().unwrap(),
            &output.macro_bias,
        )?);
        Ok((output, trace))
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_impl(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        reference_micro: &Tensor,
        reference_macro: &Tensor,
        genome: &Tensor,
        memory: &Tensor,
        fidelity: f32,
        age_phase: f32,
        tracked: bool,
        active_morph_depth: usize,
        mut trace: Option<&mut InterfaceTrace>,
    ) -> Result<InterfaceOutput> {
        let reference_micro = reference_micro.affine(fidelity as f64, 0.0)?;
        let reference_macro = reference_macro.affine(fidelity as f64, 0.0)?;
        let micro_input = Tensor::cat(&[micro, &reference_micro], 1)?;
        let macro_input = Tensor::cat(&[macro_field, &reference_macro], 1)?;
        let micro_tokens = linear_mode(
            &self.micro_projection,
            &spatial_tokens(&micro_input, self.grid)?,
            tracked,
        )?;
        let macro_tokens = linear_mode(
            &self.macro_projection,
            &spatial_tokens(&macro_input, self.grid)?,
            tracked,
        )?;
        let condition_scalars = Tensor::from_vec(vec![fidelity, age_phase], (2,), genome.device())?;
        let condition =
            Tensor::cat(&[genome, &condition_scalars], 0)?.reshape((1, genome.dim(0)? + 2))?;
        let condition = linear_mode(&self.conditioning_projection, &condition, tracked)?;

        anyhow::ensure!(
            active_morph_depth <= self.morphic.len(),
            "active morph depth exceeds physical MorphicStack capacity"
        );
        let local_tokens = micro_tokens
            .add(&macro_tokens)?
            .affine(0.5, 0.0)?
            .broadcast_add(&condition)?
            .broadcast_add(memory)?;
        let mut tokens = if self.grid > self.attention_grid {
            pool_token_grid(&local_tokens, self.grid, self.attention_grid, self.width)?
        } else {
            local_tokens.clone()
        };
        let mut next_memory = memory.clone();
        for _ in 0..self.loops {
            let normalized = norm_mode(
                &self.attention_norm,
                &tokens,
                tracked && self.differentiable_norm,
            )?;
            let query = linear_mode(&self.query, &normalized, tracked)?;
            let key = linear_mode(&self.key, &normalized, tracked)?;
            let value = linear_mode(&self.value, &normalized, tracked)?;
            let attention = candle_nn::ops::softmax(
                &query
                    .matmul(&key.t()?)?
                    .affine(1.0 / (self.width as f64).sqrt(), 0.0)?,
                D::Minus1,
            )?;
            let routed = linear_mode(&self.attention_output, &attention.matmul(&value)?, tracked)?;
            tokens = tokens.add(&routed.affine(0.25, 0.0)?)?;

            let hidden = swish(&linear_mode(
                &self.feedforward_expand,
                &norm_mode(
                    &self.feedforward_norm,
                    &tokens,
                    tracked && self.differentiable_norm,
                )?,
                tracked,
            )?)?;
            tokens = tokens.add(
                &linear_mode(&self.feedforward_contract, &hidden, tracked)?.affine(0.25, 0.0)?,
            )?;

            next_memory = self.gru.forward(
                &tokens.mean(0)?.reshape((1, self.width))?,
                &next_memory,
                tracked,
                trace.as_deref_mut(),
            )?;
            next_memory = smooth_limit(&next_memory, self.memory_limit)?;
            for (index, block) in self.morphic.iter().take(active_morph_depth).enumerate() {
                next_memory = block.forward(
                    &next_memory,
                    index,
                    self.morph_residual_gain,
                    tracked,
                    self.differentiable_norm,
                )?;
            }
            tokens = tokens.broadcast_add(&next_memory.affine(0.10, 0.0)?)?;
            next_memory = smooth_limit(&next_memory, self.memory_limit)?;
            if let Some(trace) = trace.as_deref_mut() {
                trace
                    .token_rms_per_loop
                    .push(crate::metrics::tensor_rms(&tokens)?);
            }
        }

        let write_tokens = if self.grid > self.attention_grid {
            let global = upsample_token_grid(&tokens, self.attention_grid, self.grid, self.width)?;
            local_tokens
                .affine(0.5, 0.0)?
                .add(&global.affine(0.5, 0.0)?)?
        } else {
            tokens
        };
        let micro_logits = linear_mode(&self.micro_write, &write_tokens, tracked)?;
        let macro_logits = linear_mode(&self.macro_write, &write_tokens, tracked)?;
        if let Some(trace) = trace {
            trace.micro_logits = Some(micro_logits.detach());
            trace.macro_logits = Some(macro_logits.detach());
        }
        let saturation_penalty = if tracked {
            self.saturation_penalty
                .map(|p| -> Result<Tensor> {
                    write_saturation_loss(&micro_logits, &macro_logits, p.threshold)?
                        .affine(p.weight as f64, 0.0)
                        .map_err(Into::into)
                })
                .transpose()?
        } else {
            None
        };
        let micro_grid = micro_logits.tanh()?.t()?.reshape((
            1,
            self.micro_write.weight().dim(0)?,
            self.grid,
            self.grid,
        ))?;
        let macro_grid = macro_logits.tanh()?.t()?.reshape((
            1,
            self.macro_write.weight().dim(0)?,
            self.grid,
            self.grid,
        ))?;
        let micro_bias = self
            .micro_upsampler
            .apply(&micro_grid)?
            .affine(self.gain as f64, 0.0)?;
        let macro_bias = self
            .macro_upsampler
            .apply(&macro_grid)?
            .affine(self.gain as f64, 0.0)?;
        if !tracked {
            next_memory = next_memory.detach();
        }
        Ok(InterfaceOutput {
            saturation_penalty,
            micro_bias,
            macro_bias,
            memory: next_memory,
        })
    }
}

fn norm_mode(norm: &RmsNorm, value: &Tensor, differentiable: bool) -> candle_core::Result<Tensor> {
    if differentiable {
        norm.forward_diff(value)
    } else {
        norm.forward(value)
    }
}

// Mean squared excess above a logit threshold; bypasses the saturated tanh derivative.
fn write_saturation_loss(micro: &Tensor, macro_field: &Tensor, threshold: f32) -> Result<Tensor> {
    let excess = |x: &Tensor| {
        x.abs()?
            .affine(1.0, -(threshold as f64))?
            .relu()?
            .sqr()?
            .mean_all()
    };
    excess(micro)?
        .add(&excess(macro_field)?)?
        .affine(0.5, 0.0)
        .map_err(Into::into)
}

fn spatial_tokens(field: &Tensor, grid: usize) -> candle_core::Result<Tensor> {
    let (batch, channels, height, width) = field.dims4()?;
    if batch != 1 || !height.is_multiple_of(grid) || !width.is_multiple_of(grid) {
        candle_core::bail!(
            "cannot pool {:?} into a {grid}x{grid} recurrent interface",
            field.dims()
        );
    }
    field
        .reshape((1, channels, grid, height / grid, grid, width / grid))?
        .mean(5)?
        .mean(3)?
        .permute((0, 2, 3, 1))?
        .reshape((grid * grid, channels))
}

fn pool_token_grid(
    tokens: &Tensor,
    source_grid: usize,
    target_grid: usize,
    width: usize,
) -> candle_core::Result<Tensor> {
    if source_grid == target_grid {
        return Ok(tokens.clone());
    }
    if !source_grid.is_multiple_of(target_grid) {
        candle_core::bail!("hierarchical token grid must divide the source grid");
    }
    let factor = source_grid / target_grid;
    tokens
        .reshape((target_grid, factor, target_grid, factor, width))?
        .mean(3)?
        .mean(1)?
        .reshape((target_grid * target_grid, width))
}

fn upsample_token_grid(
    tokens: &Tensor,
    source_grid: usize,
    target_grid: usize,
    width: usize,
) -> candle_core::Result<Tensor> {
    if source_grid == target_grid {
        return Ok(tokens.clone());
    }
    if !target_grid.is_multiple_of(source_grid) {
        candle_core::bail!("hierarchical token grid must divide the target grid");
    }
    let factor = target_grid / source_grid;
    tokens
        .reshape((source_grid, source_grid, width))?
        .unsqueeze(1)?
        .unsqueeze(3)?
        .broadcast_as((source_grid, factor, source_grid, factor, width))?
        .reshape((target_grid * target_grid, width))
}
fn linear_mode(linear: &Linear, input: &Tensor, tracked: bool) -> candle_core::Result<Tensor> {
    if tracked {
        return linear.forward(input);
    }
    Linear::new(linear.weight().detach(), linear.bias().map(Tensor::detach)).forward(input)
}

fn zero_linear(input: usize, output: usize, vb: VarBuilder<'_>) -> Result<Linear> {
    let weight = vb.get_with_hints((output, input), "weight", Init::Const(0.0))?;
    let bias = vb.get_with_hints(output, "bias", Init::Const(0.0))?;
    Ok(Linear::new(weight, Some(bias)))
}

fn swish(value: &Tensor) -> candle_core::Result<Tensor> {
    value.mul(&candle_nn::ops::sigmoid(value)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn interface_is_shape_safe_and_zero_write_initialized() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig {
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            genome_dim: 4,
            interface_grid: 3,
            interface_width: 32,
            interface_loops: 2,
            morph_layers: 3,
            morph_depth: 2,
            train_resolution: 24,
            output_resolution: 24,
            ..RunConfig::default()
        };
        let variables = VarMap::new();
        let interface = RecurrentInterface::new(
            &config,
            VarBuilder::from_varmap(&variables, DType::F32, &device),
            &device,
        )?;
        let micro = Tensor::zeros((1, 12, 24, 24), DType::F32, &device)?;
        let macro_field = Tensor::zeros((1, 12, 12, 12), DType::F32, &device)?;
        let reference_micro = Tensor::ones((1, 3, 24, 24), DType::F32, &device)?;
        let reference_macro = Tensor::ones((1, 3, 12, 12), DType::F32, &device)?;
        let genome = Tensor::zeros(4, DType::F32, &device)?;
        let memory = Tensor::zeros((1, 32), DType::F32, &device)?;
        let output = interface.forward(
            &micro,
            &macro_field,
            &reference_micro,
            &reference_macro,
            &genome,
            &memory,
            0.5,
            0.25,
            true,
            config.morph_depth,
        )?;
        assert_eq!(output.micro_bias.dims4()?, (1, 12, 24, 24));
        assert_eq!(output.macro_bias.dims4()?, (1, 12, 12, 12));
        assert_eq!(output.memory.dims2()?, (1, 32));
        assert!(output.micro_bias.abs()?.max_all()?.to_scalar::<f32>()? < 1e-7);
        Ok(())
    }

    #[test]
    fn recurrent_interface_remains_bounded_over_long_rollout() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig {
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            genome_dim: 4,
            interface_grid: 3,
            interface_width: 32,
            interface_loops: 3,
            morph_layers: 4,
            morph_depth: 3,
            memory_limit: 2.0,
            train_resolution: 24,
            output_resolution: 24,
            ..RunConfig::default()
        };
        let variables = VarMap::new();
        let interface = RecurrentInterface::new(
            &config,
            VarBuilder::from_varmap(&variables, DType::F32, &device),
            &device,
        )?;
        let micro = Tensor::zeros((1, 12, 24, 24), DType::F32, &device)?;
        let macro_field = Tensor::zeros((1, 12, 12, 12), DType::F32, &device)?;
        let reference_micro = Tensor::zeros((1, 3, 24, 24), DType::F32, &device)?;
        let reference_macro = Tensor::zeros((1, 3, 12, 12), DType::F32, &device)?;
        let genome = Tensor::zeros(4, DType::F32, &device)?;
        let mut memory = Tensor::zeros((1, 32), DType::F32, &device)?;
        for step in 0..256 {
            let output = interface.forward(
                &micro,
                &macro_field,
                &reference_micro,
                &reference_macro,
                &genome,
                &memory,
                0.0,
                (step % 64) as f32 / 64.0,
                false,
                config.morph_depth,
            )?;
            memory = output.memory;
            assert!(
                memory.abs()?.max_all()?.to_scalar::<f32>()? < config.memory_limit,
                "memory escaped its smooth bound at step {step}"
            );
            assert!(
                output.micro_bias.abs()?.max_all()?.to_scalar::<f32>()?
                    <= config.interface_gain + 1e-6,
                "micro writeback escaped its tanh bound at step {step}"
            );
        }
        Ok(())
    }

    #[test]
    fn zero_contract_activation_is_function_preserving() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig {
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            genome_dim: 4,
            interface_grid: 3,
            interface_width: 32,
            interface_loops: 2,
            morph_layers: 3,
            morph_depth: 1,
            train_resolution: 24,
            output_resolution: 24,
            ..RunConfig::default()
        };
        let variables = VarMap::new();
        let interface = RecurrentInterface::new(
            &config,
            VarBuilder::from_varmap(&variables, DType::F32, &device),
            &device,
        )?;
        let micro = Tensor::zeros((1, 12, 24, 24), DType::F32, &device)?;
        let macro_field = Tensor::zeros((1, 12, 12, 12), DType::F32, &device)?;
        let reference_micro = Tensor::zeros((1, 3, 24, 24), DType::F32, &device)?;
        let reference_macro = Tensor::zeros((1, 3, 12, 12), DType::F32, &device)?;
        let genome = Tensor::zeros(4, DType::F32, &device)?;
        let memory = Tensor::zeros((1, 32), DType::F32, &device)?;
        let first = interface.forward(
            &micro,
            &macro_field,
            &reference_micro,
            &reference_macro,
            &genome,
            &memory,
            0.0,
            1.0,
            false,
            1,
        )?;
        let activated = interface.forward(
            &micro,
            &macro_field,
            &reference_micro,
            &reference_macro,
            &genome,
            &memory,
            0.0,
            1.0,
            false,
            2,
        )?;
        assert_eq!(
            first.memory.flatten_all()?.to_vec1::<f32>()?,
            activated.memory.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            first.micro_bias.flatten_all()?.to_vec1::<f32>()?,
            activated.micro_bias.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            first.macro_bias.flatten_all()?.to_vec1::<f32>()?,
            activated.macro_bias.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}

#[cfg(test)]
mod gradient_tests {
    use super::*;
    use candle_core::{DType, Var};
    use candle_nn::VarMap;
    #[test]
    fn rmsnorm_forward_and_finite_difference_gradients() -> Result<()> {
        for width in [4, 32, 160] {
            for scale in [1e-4f32, 0.3, 10.0] {
                let device = Device::Cpu;
                let values: Vec<_> = (0..3 * width)
                    .map(|i| ((i as f32 + 0.3) * 0.71).sin() * scale)
                    .collect();
                let input = Var::from_tensor(&Tensor::from_vec(values, (3, width), &device)?)?;
                let weight = Var::from_tensor(&Tensor::from_vec(
                    (0..width)
                        .map(|i| 0.8 + i as f32 / width as f32)
                        .collect::<Vec<_>>(),
                    width,
                    &device,
                )?)?;
                let norm = RmsNorm::new(weight.as_tensor().clone(), 1e-5);
                let fast = norm_mode(&norm, input.as_tensor(), false)?;
                let diff = norm_mode(&norm, input.as_tensor(), true)?;
                let max = fast.sub(&diff)?.abs()?.max_all()?.to_scalar::<f32>()?;
                assert!(
                    max < 3e-6,
                    "forward drift width={width} scale={scale}: {max}"
                );
                let grads = diff.sum_all()?.backward()?;
                for (name, var) in [("input", &input), ("scale", &weight)] {
                    let g = grads
                        .get(var.as_tensor())
                        .unwrap_or_else(|| panic!("missing {name} gradient"));
                    assert!(g
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|x| x.is_finite()));
                }
            }
        }
        // Noncontiguous normalization also retains the same numerical contract.
        let t = Tensor::from_vec(
            (0..24).map(|i| (i as f32 * 0.3).sin()).collect::<Vec<_>>(),
            (4, 6),
            &Device::Cpu,
        )?
        .t()?;
        let norm = RmsNorm::new(Tensor::ones(4, DType::F32, &Device::Cpu)?, 1e-5);
        assert!(!t.is_contiguous());
        assert!(
            norm.forward(&t)?
                .sub(&norm.forward_diff(&t)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?
                < 3e-6
        );
        let device = Device::Cpu;
        let xs = vec![0.2f32, -0.4, 0.7, 1.3];
        let ws = vec![0.8f32, 1.1, 1.2, 0.9];
        let x = Var::from_tensor(&Tensor::from_vec(xs.clone(), (1, 4), &device)?)?;
        let w = Var::from_tensor(&Tensor::from_vec(ws.clone(), 4, &device)?)?;
        let norm = RmsNorm::new(w.as_tensor().clone(), 1e-5);
        let gradients = norm.forward_diff(x.as_tensor())?.sum_all()?.backward()?;
        let gx = gradients
            .get(x.as_tensor())
            .unwrap()
            .flatten_all()?
            .to_vec1::<f32>()?;
        let gw = gradients.get(w.as_tensor()).unwrap().to_vec1::<f32>()?;
        let f = |x: Vec<f32>, w: Vec<f32>| -> Result<f32> {
            Ok(RmsNorm::new(Tensor::from_vec(w, 4, &device)?, 1e-5)
                .forward_diff(&Tensor::from_vec(x, (1, 4), &device)?)?
                .sum_all()?
                .to_scalar::<f32>()?)
        };
        for i in 0..4 {
            for input in [true, false] {
                let mut xp = xs.clone();
                let mut xm = xs.clone();
                let mut wp = ws.clone();
                let mut wm = ws.clone();
                if input {
                    xp[i] += 0.001;
                    xm[i] -= 0.001;
                } else {
                    wp[i] += 0.001;
                    wm[i] -= 0.001;
                }
                let finite = (f(xp, wp)? - f(xm, wm)?) / 0.002;
                let analytic = if input { gx[i] } else { gw[i] };
                assert!(
                    (finite - analytic).abs() < 4e-4,
                    "finite={finite} analytic={analytic}"
                );
            }
        }
        let legacy = norm.forward(x.as_tensor())?.sum_all()?.backward()?;
        assert!(legacy.get(x.as_tensor()).is_none() && legacy.get(w.as_tensor()).is_none());
        Ok(())
    }
    #[test]
    fn every_interface_normalization_scale_is_reachable_and_instrumentation_is_passive(
    ) -> Result<()> {
        let mut config = RunConfig {
            micro_size: 16,
            macro_size: 8,
            channels: 12,
            interface_grid: 2,
            interface_width: 16,
            interface_loops: 2,
            morph_layers: 2,
            morph_depth: 2,
            ..Default::default()
        };
        config.experiment.norm = crate::experiment::NormTraining::Differentiable;
        let device = Device::Cpu;
        let vars = VarMap::new();
        let interface = RecurrentInterface::new(
            &config,
            VarBuilder::from_varmap(&vars, DType::F32, &device),
            &device,
        )?;
        // Nonzero contracts/writes let the fixture test normalized-branch reachability,
        // instead of being masked by fresh zero-initialized residual outputs.
        for (name, var) in vars.data().lock().unwrap().iter() {
            if name.contains("contract.weight") || name.contains("write.weight") {
                let v: Vec<f32> = (0..var.elem_count())
                    .map(|i| ((i as f32 + 1.) * 0.41).sin() * 0.03)
                    .collect();
                var.set(&Tensor::from_vec(v, var.shape().clone(), &device)?)?;
            }
        }
        let micro = Tensor::ones((1, 12, 16, 16), DType::F32, &device)?.affine(0.2, 0.)?;
        let macro_field = Tensor::ones((1, 12, 8, 8), DType::F32, &device)?.affine(0.3, 0.)?;
        let rm = Tensor::ones((1, 3, 16, 16), DType::F32, &device)?;
        let ra = Tensor::ones((1, 3, 8, 8), DType::F32, &device)?;
        let genome = Tensor::ones(config.genome_dim, DType::F32, &device)?;
        let memory = Tensor::ones((1, 16), DType::F32, &device)?.affine(0.1, 0.)?;
        let output = interface.forward(
            &micro,
            &macro_field,
            &rm,
            &ra,
            &genome,
            &memory,
            1.,
            0.5,
            true,
            2,
        )?;
        let grads = output
            .memory
            .sqr()?
            .sum_all()?
            .add(&output.micro_bias.sqr()?.sum_all()?)?
            .backward()?;
        let mut count = 0;
        for (name, var) in vars.data().lock().unwrap().iter() {
            if name.contains("norm.weight") {
                let values = grads
                    .get(var.as_tensor())
                    .unwrap_or_else(|| panic!("missing {name}"))
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(values.iter().all(|x| x.is_finite()));
                assert!(
                    values.iter().any(|x| x.abs() > 1e-12),
                    "zero gradient {name}"
                );
                count += 1;
            }
        }
        assert_eq!(count, 4);
        let plain = interface.forward(
            &micro,
            &macro_field,
            &rm,
            &ra,
            &genome,
            &memory,
            1.,
            0.5,
            false,
            2,
        )?;
        let (inspected, trace) =
            interface.inspect(&micro, &macro_field, &rm, &ra, &genome, &memory, 1., 0.5, 2)?;
        assert_eq!(
            plain.micro_bias.flatten_all()?.to_vec1::<f32>()?,
            inspected.micro_bias.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(trace.token_rms_per_loop.len(), 2);
        assert_eq!(trace.gru_reset.len(), 2);
        let stats = WriteStatistics::new(
            &Tensor::new(&[0f32, 10., -10., 0.], &device)?,
            &Tensor::new(&[0f32, 1., -1., 0.], &device)?.reshape((1, 1, 2, 2))?,
        )?;
        assert_eq!(stats.saturated_fraction, 0.5);
        assert!((stats.mean_tanh_derivative - 0.5).abs() < 1e-6);
        assert!((stats.spatial_write_variance - 0.5).abs() < 1e-6);
        Ok(())
    }
}

#[cfg(test)]
mod saturation_tests {
    use super::*;
    use candle_core::{DType, Var};
    use candle_nn::VarMap;
    #[test]
    fn pre_tanh_penalty_matches_finite_differences_and_descends_when_tanh_is_flat() -> Result<()> {
        let d = Device::Cpu;
        let values = vec![-20f32, -3., -0.5, 0.5, 3., 20.];
        let x = Var::from_tensor(&Tensor::from_vec(values.clone(), 6, &d)?)?;
        let zero = Tensor::zeros(6, DType::F32, &d)?;
        let loss = write_saturation_loss(x.as_tensor(), &zero, 2.)?;
        let gradients = loss.backward()?;
        let g = gradients.get(x.as_tensor()).unwrap().to_vec1::<f32>()?;
        let f = |v: Vec<f32>| -> Result<f32> {
            Ok(
                write_saturation_loss(&Tensor::from_vec(v, 6, &d)?, &zero, 2.)?
                    .to_scalar::<f32>()?,
            )
        };
        for i in 0..6 {
            let mut a = values.clone();
            let mut b = values.clone();
            a[i] += 0.01;
            b[i] -= 0.01;
            let finite = (f(a)? - f(b)?) / 0.02;
            assert!(
                (g[i] - finite).abs() < 0.002,
                "index {i}: {} vs {finite}",
                g[i]
            );
        }
        assert_eq!(&g[2..4], &[0., 0.]);
        let tanh_grad = x.as_tensor().tanh()?.sum_all()?.backward()?;
        let tg = tanh_grad.get(x.as_tensor()).unwrap().to_vec1::<f32>()?;
        assert_eq!(tg[0], 0.);
        assert_eq!(tg[5], 0.);
        assert!(g[0] < 0. && g[5] > 0. && g.iter().all(|v| v.is_finite()));
        assert!(
            f(values.iter().zip(&g).map(|(v, g)| v - 0.01 * g).collect())?
                < loss.to_scalar::<f32>()?
        );
        Ok(())
    }
    #[test]
    fn opt_in_penalty_preserves_forward_values_and_zero_weight_has_no_graph() -> Result<()> {
        let d = Device::Cpu;
        let vars = VarMap::new();
        let mut c = RunConfig {
            micro_size: 8,
            macro_size: 4,
            channels: 12,
            interface_grid: 2,
            interface_width: 16,
            interface_loops: 1,
            morph_layers: 1,
            morph_depth: 1,
            ..Default::default()
        };
        let vb = VarBuilder::from_varmap(&vars, DType::F32, &d);
        let baseline = RecurrentInterface::new(&c, vb.clone(), &d)?;
        c.experiment.saturation_penalty = Some(crate::experiment::SaturationPenalty {
            weight: 0.01,
            threshold: 2.,
        });
        let active = RecurrentInterface::new(&c, vb.clone(), &d)?;
        c.experiment.saturation_penalty.as_mut().unwrap().weight = 0.;
        let disabled = RecurrentInterface::new(&c, vb, &d)?;
        for (name, var) in vars.data().lock().unwrap().iter() {
            if name.contains("write.bias") {
                var.set(&Tensor::ones(var.shape().clone(), DType::F32, &d)?.affine(20., 0.)?)?;
            }
        }
        let micro = Tensor::ones((1, 12, 8, 8), DType::F32, &d)?;
        let macro_field = Tensor::ones((1, 12, 4, 4), DType::F32, &d)?;
        let rm = Tensor::zeros((1, 3, 8, 8), DType::F32, &d)?;
        let ra = Tensor::zeros((1, 3, 4, 4), DType::F32, &d)?;
        let genome = Tensor::ones(c.genome_dim, DType::F32, &d)?;
        let memory = Tensor::zeros((1, 16), DType::F32, &d)?;
        let run = |i: &RecurrentInterface, tracked| {
            i.forward(
                &micro,
                &macro_field,
                &rm,
                &ra,
                &genome,
                &memory,
                0.,
                0.,
                tracked,
                1,
            )
        };
        let a = run(&baseline, true)?;
        let b = run(&active, true)?;
        let z = run(&disabled, true)?;
        assert!(a.saturation_penalty.is_none() && z.saturation_penalty.is_none());
        assert!(run(&active, false)?.saturation_penalty.is_none());
        for (a, b, z) in [
            (&a.micro_bias, &b.micro_bias, &z.micro_bias),
            (&a.macro_bias, &b.macro_bias, &z.macro_bias),
            (&a.memory, &b.memory, &z.memory),
        ] {
            assert_eq!(
                a.flatten_all()?.to_vec1::<f32>()?,
                b.flatten_all()?.to_vec1::<f32>()?
            );
            assert_eq!(
                a.flatten_all()?.to_vec1::<f32>()?,
                z.flatten_all()?.to_vec1::<f32>()?
            );
        }
        let loss = b.saturation_penalty.unwrap();
        assert!((loss.to_scalar::<f32>()? - 3.24).abs() < 1e-5);
        let grad = loss.backward()?;
        let mut reached = 0;
        for (name, var) in vars.data().lock().unwrap().iter() {
            if name.contains("write.bias") {
                let g = grad
                    .get(var.as_tensor())
                    .unwrap()
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert!(g.iter().all(|x| x.is_finite() && *x > 0.));
                reached += 1;
            }
        }
        assert_eq!(reached, 2);
        Ok(())
    }
}
