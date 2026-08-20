use crate::config::RunConfig;
use crate::tensor_ops::{
    deterministic_clock_mask, perceive_multiscale, perception_kernel, pixelwise_linear_mode,
    PERCEPTION_FEATURES_PER_CHANNEL,
};
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Init, Linear, VarBuilder};

pub struct NeuralCa {
    perception: Tensor,
    input: Linear,
    hidden: Linear,
    output: Linear,
    channels: usize,
    clock_probability: f32,
    gain: f32,
}

impl NeuralCa {
    pub fn new(config: &RunConfig, vb: VarBuilder<'_>, device: &Device) -> Result<Self> {
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
        Ok(Self {
            perception: perception_kernel(config.channels, device)?,
            input,
            hidden,
            output: Linear::new(weight, Some(bias)),
            channels: config.channels,
            clock_probability: config.clock_probability,
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
        let (_, _, h, w) = field.dims4()?;
        let mask =
            deterministic_clock_mask(h, w, seed, step, self.clock_probability, field.device())?;
        Ok(raw.broadcast_mul(&mask)?.affine(self.gain as f64, 0.0)?)
    }
}

fn swish(x: &Tensor) -> candle_core::Result<Tensor> {
    x.mul(&candle_nn::ops::sigmoid(x)?)
}
