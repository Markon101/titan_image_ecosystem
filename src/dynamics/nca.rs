use crate::config::RunConfig;
use crate::tensor_ops::{
    deterministic_clock_mask, perceive_multiscale, perception_kernel, pixelwise_linear_mode,
    splitmix64, PERCEPTION_FEATURES_PER_CHANNEL,
};
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Init, Linear, VarBuilder};

const CLOCK_MASKS: usize = 8;

pub struct NeuralCa {
    perception: Tensor,
    input: Linear,
    hidden: Linear,
    output: Linear,
    clock_masks: Vec<Tensor>,
    clock_salt: u64,
    channels: usize,
    gain: f32,
}

impl NeuralCa {
    pub fn new(
        config: &RunConfig,
        size: usize,
        clock_salt: u64,
        vb: VarBuilder<'_>,
        device: &Device,
    ) -> Result<Self> {
        let in_features =
            config.channels * PERCEPTION_FEATURES_PER_CHANNEL + config.channels + config.genome_dim;
        let input = candle_nn::linear(in_features, config.ca_hidden, vb.pp("input"))?;
        let hidden = candle_nn::linear(config.ca_hidden, config.ca_hidden, vb.pp("hidden"))?;
        let out_vb = vb.pp("output");
        let weight = out_vb.get_with_hints(
            (config.channels, config.ca_hidden),
            "weight",
            Init::Const(0.0),
        )?;
        let bias = out_vb.get_with_hints(config.channels, "bias", Init::Const(0.0))?;
        let mut clock_masks = Vec::with_capacity(CLOCK_MASKS);
        for index in 0..CLOCK_MASKS {
            clock_masks.push(deterministic_clock_mask(
                size,
                size,
                clock_salt,
                index as u64,
                config.clock_probability,
                device,
            )?);
        }
        Ok(Self {
            perception: perception_kernel(config.channels, device)?,
            input,
            hidden,
            output: Linear::new(weight, Some(bias)),
            clock_masks,
            clock_salt,
            channels: config.channels,
            gain: config.nca_gain,
        })
    }

    pub fn delta(
        &self,
        field: &Tensor,
        macro_context: &Tensor,
        genome_field: &Tensor,
        seed: u64,
        step: u64,
        tracked: bool,
    ) -> Result<Tensor> {
        let perceived = perceive_multiscale(field, &self.perception, self.channels)?;
        let features = Tensor::cat(&[&perceived, macro_context, genome_field], 1)?;
        let h1 = swish(&pixelwise_linear_mode(&features, &self.input, tracked)?)?;
        let residual = swish(&pixelwise_linear_mode(&h1, &self.hidden, tracked)?)?;
        let hidden = h1.add(&residual.affine(0.5, 0.0)?)?;
        let raw = pixelwise_linear_mode(&hidden, &self.output, tracked)?.tanh()?;
        let clock = splitmix64(seed ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ self.clock_salt)
            as usize
            % self.clock_masks.len();
        Ok(raw
            .broadcast_mul(&self.clock_masks[clock])?
            .affine(self.gain as f64, 0.0)?)
    }
}

fn swish(x: &Tensor) -> candle_core::Result<Tensor> {
    x.mul(&candle_nn::ops::sigmoid(x)?)
}
