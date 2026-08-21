use crate::config::RunConfig;
use crate::tensor_ops::PeriodicUpsampler;
use anyhow::Result;
use candle_core::{Device, Tensor, D};
use candle_nn::{Init, Linear, Module, RmsNorm, VarBuilder};

pub struct InterfaceOutput {
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

    fn forward(&self, input: &Tensor, memory: &Tensor, tracked: bool) -> Result<Tensor> {
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
            contract: candle_nn::linear(width * 2, width, vb.pp("contract"))?,
        })
    }

    fn forward(&self, value: &Tensor, index: usize, tracked: bool) -> Result<Tensor> {
        let hidden = swish(&linear_mode(
            &self.expand,
            &self.norm.forward(value)?,
            tracked,
        )?)?;
        let gain = 0.30 / ((index + 1) as f64).sqrt();
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
    morph_depth: usize,
    gain: f32,
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
            morph_depth: config.morph_depth,
            gain: config.interface_gain,
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

        let mut tokens = micro_tokens
            .add(&macro_tokens)?
            .affine(0.5, 0.0)?
            .broadcast_add(&condition)?
            .broadcast_add(memory)?;
        let mut next_memory = memory.clone();
        for _ in 0..self.loops {
            let normalized = self.attention_norm.forward(&tokens)?;
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
                &self.feedforward_norm.forward(&tokens)?,
                tracked,
            )?)?;
            tokens = tokens.add(
                &linear_mode(&self.feedforward_contract, &hidden, tracked)?.affine(0.25, 0.0)?,
            )?;

            next_memory = self.gru.forward(
                &tokens.mean(0)?.reshape((1, self.width))?,
                &next_memory,
                tracked,
            )?;
            for (index, block) in self.morphic.iter().take(self.morph_depth).enumerate() {
                next_memory = block.forward(&next_memory, index, tracked)?;
            }
            tokens = tokens.broadcast_add(&next_memory.affine(0.10, 0.0)?)?;
        }

        let micro_grid = linear_mode(&self.micro_write, &tokens, tracked)?
            .t()?
            .reshape((1, self.micro_write.weight().dim(0)?, self.grid, self.grid))?;
        let macro_grid = linear_mode(&self.macro_write, &tokens, tracked)?
            .t()?
            .reshape((1, self.macro_write.weight().dim(0)?, self.grid, self.grid))?;
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
            micro_bias,
            macro_bias,
            memory: next_memory,
        })
    }
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
        )?;
        assert_eq!(output.micro_bias.dims4()?, (1, 12, 24, 24));
        assert_eq!(output.macro_bias.dims4()?, (1, 12, 12, 12));
        assert_eq!(output.memory.dims2()?, (1, 32));
        assert!(output.micro_bias.abs()?.max_all()?.to_scalar::<f32>()? < 1e-7);
        Ok(())
    }
}
