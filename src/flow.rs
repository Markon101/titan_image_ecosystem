use crate::config::RunConfig;
use crate::render::RenderPlan;
use crate::tensor_ops::{broadcast_vector, pixelwise_linear_mode, splitmix64};
use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Init, Linear, VarBuilder};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

const FLOW_FINITE_CEILING: f32 = 1.0e6;
const FLOW_ODE_STATE_CEILING: f32 = 64.0;

pub struct FlowTrainingSample {
    pub time: f32,
    pub endpoint: Tensor,
    pub noise: Tensor,
    pub interpolant: Tensor,
    pub target_velocity: Tensor,
}

pub struct FlowLossOutput {
    pub loss: Tensor,
    /// Detached host scalar for CSV/terminal telemetry.
    pub loss_value: f32,
    pub time: f32,
    pub interpolant_rms: f32,
    pub predicted_velocity_rms: f32,
    pub target_velocity_rms: f32,
    pub velocity_cosine: f32,
    pub one_step_endpoint_l1: f32,
    pub condition_rms: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct FlowTrajectoryPoint {
    pub integration_step: usize,
    pub time: f32,
    pub state_rms: f32,
    pub state_max_abs: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct FlowTrajectory {
    pub solver: &'static str,
    pub integration_steps: usize,
    pub neural_function_evaluations: usize,
    pub seed: u64,
    pub points: Vec<FlowTrajectoryPoint>,
}

pub struct FlowOdeOutput {
    pub state: Tensor,
    pub trajectory: FlowTrajectory,
}

pub struct RectifiedFlowRenderer {
    state_projection: Linear,
    input_projection: Linear,
    condition_projection: Linear,
    blocks: Vec<Linear>,
    velocity_output: Linear,
    hidden: usize,
    memory_limit: f32,
}

impl RectifiedFlowRenderer {
    pub fn new(config: &RunConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let hidden = config.flow.hidden;
        Ok(Self {
            state_projection: candle_nn::linear(
                2 * config.channels,
                hidden,
                vb.pp("state_projection"),
            )?,
            input_projection: candle_nn::linear(3, hidden, vb.pp("input_projection"))?,
            condition_projection: candle_nn::linear(
                config.interface_width + 10,
                hidden,
                vb.pp("condition_projection"),
            )?,
            blocks: (0..2)
                .map(|index| candle_nn::linear(hidden, hidden, vb.pp(format!("block_{index:03}"))))
                .collect::<candle_core::Result<_>>()?,
            velocity_output: zero_linear(hidden, 3, vb.pp("velocity_output"))?,
            hidden,
            memory_limit: config.memory_limit,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn predict_velocity(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        memory: &Tensor,
        plan: &RenderPlan,
        interpolant: &Tensor,
        time: f32,
        age_phase: f32,
        fidelity: f32,
        emergence: f32,
        tracked: bool,
    ) -> Result<(Tensor, f32)> {
        validate_unit_interval("flow time", time)?;
        for (name, value) in [
            ("flow age phase", age_phase),
            ("flow reference fidelity", fidelity),
            ("flow emergence strength", emergence),
            ("flow LOD", plan.lod_value()),
        ] {
            ensure_finite_scalar(name, value)?;
        }
        validate_flow_image("flow interpolant", interpolant, plan.resolution)?;

        // A non-tracked call is a genuinely frozen read. Detach a potentially
        // tracked recurrent world as well as the weights so checkpoint-time ODE
        // analysis cannot retain or extend the training graph.
        let micro = if tracked {
            micro.clone()
        } else {
            micro.detach()
        };
        let macro_field = if tracked {
            macro_field.clone()
        } else {
            macro_field.detach()
        };
        let memory = if tracked {
            memory.clone()
        } else {
            memory.detach()
        };
        let interpolant = if tracked {
            interpolant.clone()
        } else {
            interpolant.detach()
        };
        ensure_finite("flow micro condition", &micro)?;
        ensure_finite("flow macro condition", &macro_field)?;
        ensure_finite("flow memory condition", &memory)?;
        ensure_finite("flow interpolant", &interpolant)?;

        let (micro, macro_field) = plan.observe_fields(&micro, &macro_field)?;
        let state = Tensor::cat(&[&micro.tanh()?, &macro_field.tanh()?], 1)?;
        let state_hidden = pixelwise_linear_mode(&state, &self.state_projection, tracked)?;
        let input_hidden = pixelwise_linear_mode(&interpolant, &self.input_projection, tracked)?;
        let features = time_features(
            &memory,
            self.memory_limit,
            time,
            age_phase,
            fidelity,
            emergence,
            plan.lod_value(),
        )?;
        let condition_rms = rms(&features)?;
        ensure_finite("flow condition features", &features)?;
        let condition = linear_mode(&self.condition_projection, &features, tracked)?;
        let condition = broadcast_vector(
            &condition.reshape((self.hidden,))?,
            plan.resolution,
            plan.resolution,
        )?;
        let mut hidden = swish(&state_hidden.add(&input_hidden)?.add(&condition)?)?;
        for block in &self.blocks {
            let residual = swish(&pixelwise_linear_mode(&hidden, block, tracked)?)?;
            hidden = hidden.add(&residual.affine(0.25, 0.0)?)?;
        }
        let velocity = pixelwise_linear_mode(&hidden, &self.velocity_output, tracked)?;
        ensure_finite("predicted flow velocity", &velocity)?;
        Ok((velocity, condition_rms))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn training_loss(
        &self,
        sample: &FlowTrainingSample,
        micro: &Tensor,
        macro_field: &Tensor,
        memory: &Tensor,
        plan: &RenderPlan,
        age_phase: f32,
        fidelity: f32,
        emergence: f32,
        tracked: bool,
    ) -> Result<FlowLossOutput> {
        for (name, value) in [
            ("flow endpoint", &sample.endpoint),
            ("flow noise", &sample.noise),
            ("flow interpolant", &sample.interpolant),
            ("flow target velocity", &sample.target_velocity),
        ] {
            validate_flow_image(name, value, plan.resolution)?;
            ensure_finite(name, value)?;
        }
        let (prediction, condition_rms) = self.predict_velocity(
            micro,
            macro_field,
            memory,
            plan,
            &sample.interpolant,
            sample.time,
            age_phase,
            fidelity,
            emergence,
            tracked,
        )?;
        let error = prediction.sub(&sample.target_velocity)?;
        let loss = error.sqr()?.mean_all()?;
        let predicted_velocity_rms = rms(&prediction)?;
        let target_velocity_rms = rms(&sample.target_velocity)?;
        let dot = prediction
            .mul(&sample.target_velocity)?
            .mean_all()?
            .to_scalar::<f32>()?;
        let velocity_cosine =
            (dot / (predicted_velocity_rms * target_velocity_rms).max(1e-8)).clamp(-1.0, 1.0);
        let estimated_endpoint = sample
            .interpolant
            .add(&prediction.affine((1.0 - sample.time) as f64, 0.0)?)?;
        let one_step_endpoint_l1 = estimated_endpoint
            .sub(&sample.endpoint)?
            .abs()?
            .mean_all()?
            .to_scalar::<f32>()?;
        let loss_scalar = loss.to_scalar::<f32>()?;
        ensure_finite_scalar("conditional flow-matching loss", loss_scalar)?;
        ensure_finite_scalar("flow velocity cosine", velocity_cosine)?;
        ensure_finite_scalar("flow one-step endpoint L1", one_step_endpoint_l1)?;
        Ok(FlowLossOutput {
            loss,
            loss_value: loss_scalar,
            time: sample.time,
            interpolant_rms: rms(&sample.interpolant)?,
            predicted_velocity_rms,
            target_velocity_rms,
            velocity_cosine,
            one_step_endpoint_l1,
            condition_rms,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_midpoint(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        memory: &Tensor,
        plan: &RenderPlan,
        seed: u64,
        steps: usize,
        age_phase: f32,
        fidelity: f32,
        emergence: f32,
    ) -> Result<Tensor> {
        Ok(self
            .sample_midpoint_with_trajectory(
                micro,
                macro_field,
                memory,
                plan,
                seed,
                steps,
                age_phase,
                fidelity,
                emergence,
            )?
            .state)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sample_midpoint_with_trajectory(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        memory: &Tensor,
        plan: &RenderPlan,
        seed: u64,
        steps: usize,
        age_phase: f32,
        fidelity: f32,
        emergence: f32,
    ) -> Result<FlowOdeOutput> {
        if steps == 0 {
            bail!("flow ODE sampling requires at least one integration step");
        }
        if steps > 4096 {
            bail!("flow ODE sampling step count {steps} exceeds the safety limit 4096");
        }
        ensure_finite("flow ODE micro condition", micro)?;
        ensure_finite("flow ODE macro condition", macro_field)?;
        ensure_finite("flow ODE memory condition", memory)?;
        let mut state = gaussian_noise(
            (1, 3, plan.resolution, plan.resolution),
            seed,
            micro.device(),
        )?;
        ensure_ode_state("initial flow ODE state", &state)?;
        let mut points = vec![flow_trajectory_point(0, 0.0, &state)?];
        let step_size = 1.0 / steps as f64;
        for index in 0..steps {
            let time = index as f32 / steps as f32;
            let (first, _) = self.predict_velocity(
                micro,
                macro_field,
                memory,
                plan,
                &state,
                time,
                age_phase,
                fidelity,
                emergence,
                false,
            )?;
            let midpoint = state.add(&first.affine(0.5 * step_size, 0.0)?)?;
            ensure_ode_state("flow ODE midpoint", &midpoint)?;
            let (second, _) = self.predict_velocity(
                micro,
                macro_field,
                memory,
                plan,
                &midpoint,
                time + 0.5 / steps as f32,
                age_phase,
                fidelity,
                emergence,
                false,
            )?;
            state = state.add(&second.affine(step_size, 0.0)?)?;
            ensure_ode_state("flow ODE state", &state)?;
            points.push(flow_trajectory_point(
                index + 1,
                (index + 1) as f32 / steps as f32,
                &state,
            )?);
        }
        Ok(FlowOdeOutput {
            state: state.detach(),
            trajectory: FlowTrajectory {
                solver: "midpoint",
                integration_steps: steps,
                neural_function_evaluations: 2 * steps,
                seed,
                points,
            },
        })
    }
}

fn flow_trajectory_point(
    integration_step: usize,
    time: f32,
    state: &Tensor,
) -> Result<FlowTrajectoryPoint> {
    Ok(FlowTrajectoryPoint {
        integration_step,
        time,
        state_rms: rms(state)?,
        state_max_abs: state.abs()?.max_all()?.to_scalar::<f32>()?,
    })
}

pub fn build_training_sample(
    endpoint_rgb: &Tensor,
    time_seed: u64,
    noise_seed: u64,
    min_time: f32,
    max_time: f32,
) -> Result<FlowTrainingSample> {
    validate_time_range(min_time, max_time)?;
    let (height, width) = validate_color_image("flow target image", endpoint_rgb)?;
    let endpoint = rgb_to_flow_oklab(endpoint_rgb)?.detach();
    let time = deterministic_time(time_seed, min_time, max_time);
    let noise = gaussian_noise((1, 3, height, width), noise_seed, endpoint.device())?;
    let (interpolant, target_velocity) = rectified_path(&endpoint, &noise, time)?;
    ensure_finite("flow endpoint", &endpoint)?;
    ensure_finite("flow noise", &noise)?;
    ensure_finite("flow interpolant", &interpolant)?;
    ensure_finite("flow target velocity", &target_velocity)?;
    Ok(FlowTrainingSample {
        time,
        endpoint,
        noise,
        interpolant,
        target_velocity,
    })
}
fn rectified_path(endpoint: &Tensor, noise: &Tensor, time: f32) -> Result<(Tensor, Tensor)> {
    validate_unit_interval("flow time", time)?;
    if endpoint.dims() != noise.dims() {
        bail!(
            "flow endpoint/noise shape mismatch: {:?} versus {:?}",
            endpoint.dims(),
            noise.dims()
        );
    }
    let interpolant = noise
        .affine((1.0 - time) as f64, 0.0)?
        .add(&endpoint.affine(time as f64, 0.0)?)?;
    let target_velocity = endpoint.sub(noise)?;
    Ok((interpolant, target_velocity))
}

pub fn deterministic_time(seed: u64, min_time: f32, max_time: f32) -> f32 {
    let bits = (splitmix64(seed) >> 40) as u32;
    let unit = (bits as f32 + 0.5) / (1u32 << 24) as f32;
    min_time + unit * (max_time - min_time)
}

pub fn rgb_to_flow_oklab(rgb: &Tensor) -> Result<Tensor> {
    validate_color_image("RGB flow endpoint", rgb)?;
    ensure_finite("RGB flow endpoint", rgb)?;
    let linear = rgb.clamp(0.0f32, 1.0f32)?.powf(2.2)?;
    let red = linear.narrow(1, 0, 1)?;
    let green = linear.narrow(1, 1, 1)?;
    let blue = linear.narrow(1, 2, 1)?;
    let l = red
        .affine(0.412_221_46, 0.0)?
        .add(&green.affine(0.536_332_55, 0.0)?)?
        .add(&blue.affine(0.051_445_995, 0.0)?)?
        .clamp(0.0f32, f32::MAX)?
        .powf(1.0 / 3.0)?;
    let m = red
        .affine(0.211_903_5, 0.0)?
        .add(&green.affine(0.680_699_5, 0.0)?)?
        .add(&blue.affine(0.107_396_96, 0.0)?)?
        .clamp(0.0f32, f32::MAX)?
        .powf(1.0 / 3.0)?;
    let s = red
        .affine(0.088_302_46, 0.0)?
        .add(&green.affine(0.281_718_85, 0.0)?)?
        .add(&blue.affine(0.629_978_7, 0.0)?)?
        .clamp(0.0f32, f32::MAX)?
        .powf(1.0 / 3.0)?;
    let lightness = l
        .affine(0.210_454_26, 0.0)?
        .add(&m.affine(0.793_617_8, 0.0)?)?
        .sub(&s.affine(0.004_072_047, 0.0)?)?
        .affine(2.0, -1.0)?;
    let a = l
        .affine(1.977_998_5, 0.0)?
        .sub(&m.affine(2.428_592_2, 0.0)?)?
        .add(&s.affine(0.450_593_7, 0.0)?)?
        .affine(2.5, 0.0)?;
    let b = l
        .affine(0.025_904_037, 0.0)?
        .add(&m.affine(0.782_771_77, 0.0)?)?
        .sub(&s.affine(0.808_675_77, 0.0)?)?
        .affine(2.5, 0.0)?;
    Tensor::cat(&[&lightness, &a, &b], 1).map_err(Into::into)
}

pub fn flow_oklab_to_rgb(flow: &Tensor) -> Result<Tensor> {
    validate_color_image("Oklab flow image", flow)?;
    ensure_finite("Oklab flow image", flow)?;
    let lightness = flow.narrow(1, 0, 1)?.affine(0.5, 0.5)?;
    let a = flow.narrow(1, 1, 1)?.affine(0.4, 0.0)?;
    let b = flow.narrow(1, 2, 1)?.affine(0.4, 0.0)?;
    let l = lightness
        .add(&a.affine(0.396_337_78, 0.0)?)?
        .add(&b.affine(0.215_803_76, 0.0)?)?;
    let m = lightness
        .sub(&a.affine(0.105_561_346, 0.0)?)?
        .sub(&b.affine(0.063_854_17, 0.0)?)?;
    let s = lightness
        .sub(&a.affine(0.089_484_18, 0.0)?)?
        .sub(&b.affine(1.291_485_5, 0.0)?)?;
    let l = l.sqr()?.mul(&l)?;
    let m = m.sqr()?.mul(&m)?;
    let s = s.sqr()?.mul(&s)?;
    let red = l
        .affine(4.076_741_7, 0.0)?
        .sub(&m.affine(3.307_711_6, 0.0)?)?
        .add(&s.affine(0.230_969_94, 0.0)?)?;
    let green = l
        .affine(-1.268_438, 0.0)?
        .add(&m.affine(2.609_757_4, 0.0)?)?
        .sub(&s.affine(0.341_319_4, 0.0)?)?;
    let blue = l
        .affine(-0.004_196_086_3, 0.0)?
        .sub(&m.affine(0.703_418_6, 0.0)?)?
        .add(&s.affine(1.707_614_7, 0.0)?)?;
    Tensor::cat(&[&red, &green, &blue], 1)?
        .clamp(0.0f32, 1.0f32)?
        .powf(1.0 / 2.2)
        .map_err(Into::into)
}

fn time_features(
    memory: &Tensor,
    memory_limit: f32,
    time: f32,
    age_phase: f32,
    fidelity: f32,
    emergence: f32,
    lod: f32,
) -> Result<Tensor> {
    let pi = std::f32::consts::PI;
    let scalars = Tensor::new(
        &[
            time,
            1.0 - time,
            (pi * time).sin(),
            (pi * time).cos(),
            (2.0 * pi * time).sin(),
            (2.0 * pi * time).cos(),
            age_phase,
            fidelity,
            emergence,
            lod,
        ],
        memory.device(),
    )?
    .reshape((1, 10))?;
    Tensor::cat(
        &[
            &memory.affine(1.0 / memory_limit.max(1e-6) as f64, 0.0)?,
            &scalars,
        ],
        1,
    )
    .map_err(Into::into)
}

fn validate_time_range(min_time: f32, max_time: f32) -> Result<()> {
    validate_unit_interval("minimum flow time", min_time)?;
    validate_unit_interval("maximum flow time", max_time)?;
    if min_time >= max_time {
        bail!("flow time range must satisfy min_time < max_time");
    }
    Ok(())
}

fn validate_unit_interval(name: &str, value: f32) -> Result<()> {
    ensure_finite_scalar(name, value)?;
    if !(0.0..=1.0).contains(&value) {
        bail!("{name} must be in [0, 1], got {value}");
    }
    Ok(())
}

fn validate_color_image(name: &str, value: &Tensor) -> Result<(usize, usize)> {
    let (batch, channels, height, width) = value.dims4()?;
    if batch != 1 || channels != 3 {
        bail!(
            "{name} must have shape [1, 3, H, W], got {:?}",
            value.dims()
        );
    }
    if height == 0 || width == 0 {
        bail!("{name} must have nonzero spatial dimensions");
    }
    Ok((height, width))
}

fn validate_flow_image(name: &str, value: &Tensor, resolution: usize) -> Result<()> {
    let (height, width) = validate_color_image(name, value)?;
    if height != resolution || width != resolution {
        bail!(
            "{name} must match the {resolution}x{resolution} flow observation, got {height}x{width}"
        );
    }
    Ok(())
}

fn ensure_finite_scalar(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() {
        bail!("{name} is non-finite");
    }
    Ok(())
}

fn ensure_ode_state(name: &str, value: &Tensor) -> Result<()> {
    ensure_finite(name, value)?;
    let maximum = value.abs()?.max_all()?.to_scalar::<f32>()?;
    if maximum > FLOW_ODE_STATE_CEILING {
        bail!("{name} exceeded the flow ODE safety ceiling {FLOW_ODE_STATE_CEILING}: {maximum}");
    }
    Ok(())
}

fn gaussian_noise(
    shape: (usize, usize, usize, usize),
    seed: u64,
    device: &Device,
) -> Result<Tensor> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let count = shape.0 * shape.1 * shape.2 * shape.3;
    let mut values = Vec::with_capacity(count);
    while values.len() < count {
        let u1 = rng.gen_range(f32::EPSILON..1.0);
        let u2 = rng.gen_range(0.0..1.0);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f32::consts::TAU * u2;
        values.push(radius * angle.cos());
        if values.len() < count {
            values.push(radius * angle.sin());
        }
    }
    Tensor::from_vec(values, shape, device).map_err(Into::into)
}

fn linear_mode(linear: &Linear, input: &Tensor, tracked: bool) -> candle_core::Result<Tensor> {
    if tracked {
        candle_nn::Module::forward(linear, input)
    } else {
        let detached = Linear::new(linear.weight().detach(), linear.bias().map(Tensor::detach));
        candle_nn::Module::forward(&detached, input)
    }
}

fn zero_linear(input: usize, output: usize, vb: VarBuilder<'_>) -> Result<Linear> {
    let weight = vb.get_with_hints((output, input), "weight", Init::Const(0.0))?;
    let bias = vb.get_with_hints(output, "bias", Init::Const(0.0))?;
    Ok(Linear::new(weight, Some(bias)))
}

fn swish(value: &Tensor) -> candle_core::Result<Tensor> {
    value.mul(&candle_nn::ops::sigmoid(value)?)
}

fn rms(value: &Tensor) -> Result<f32> {
    Ok(value.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?)
}

fn ensure_finite(name: &str, value: &Tensor) -> Result<()> {
    let rms = rms(value)?;
    let maximum = value.abs()?.max_all()?.to_scalar::<f32>()?;
    if !rms.is_finite() || !maximum.is_finite() || maximum > FLOW_FINITE_CEILING {
        bail!("{name} is non-finite or exceeded the finite-value safety ceiling");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::VarMap;

    #[test]
    fn rectified_sample_math_is_exact() -> Result<()> {
        let endpoint = Tensor::new(&[[[[1.0f32]], [[2.0]], [[3.0]]]], &Device::Cpu)?;
        let noise = Tensor::new(&[[[[-1.0f32]], [[0.0]], [[1.0]]]], &Device::Cpu)?;
        let time = 0.25;
        let (interpolant, velocity) = rectified_path(&endpoint, &noise, time)?;
        assert_eq!(
            interpolant.flatten_all()?.to_vec1::<f32>()?,
            vec![-0.5, 0.5, 1.5]
        );
        assert_eq!(
            velocity.flatten_all()?.to_vec1::<f32>()?,
            vec![2.0, 2.0, 2.0]
        );
        let (start, _) = rectified_path(&endpoint, &noise, 0.0)?;
        let (finish, _) = rectified_path(&endpoint, &noise, 1.0)?;
        assert_eq!(
            start.flatten_all()?.to_vec1::<f32>()?,
            noise.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            finish.flatten_all()?.to_vec1::<f32>()?,
            endpoint.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn deterministic_flow_time_is_open_interval() {
        let first = deterministic_time(7, 0.0, 1.0);
        let second = deterministic_time(7, 0.0, 1.0);
        assert_eq!(first, second);
        assert!(first > 0.0 && first < 1.0);
    }

    #[test]
    fn flow_oklab_round_trip_is_close() -> Result<()> {
        let rgb = Tensor::new(&[[[[0.2f32]], [[0.5]], [[0.8]]]], &Device::Cpu)?;
        let restored = flow_oklab_to_rgb(&rgb_to_flow_oklab(&rgb)?)?;
        let error = restored.sub(&rgb)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(error < 3e-3, "round-trip error {error}");
        Ok(())
    }

    #[test]
    fn zero_velocity_head_is_function_preserving() -> Result<()> {
        let config = RunConfig {
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            interface_width: 32,
            flow: crate::config::FlowConfig {
                hidden: 16,
                resolution: 16,
                ..RunConfig::default().flow
            },
            train_resolution: 24,
            output_resolution: 24,
            ..RunConfig::default()
        };
        let variables = VarMap::new();
        let renderer = RectifiedFlowRenderer::new(
            &config,
            VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu).pp("flow"),
        )?;
        let plan = RenderPlan::new(&config, 16, &Device::Cpu)?;
        let micro = Tensor::zeros((1, 12, 24, 24), DType::F32, &Device::Cpu)?;
        let macro_field = Tensor::zeros((1, 12, 12, 12), DType::F32, &Device::Cpu)?;
        let memory = Tensor::zeros((1, 32), DType::F32, &Device::Cpu)?;
        let interpolant = Tensor::zeros((1, 3, 16, 16), DType::F32, &Device::Cpu)?;
        let (velocity, _) = renderer.predict_velocity(
            &micro,
            &macro_field,
            &memory,
            &plan,
            &interpolant,
            0.5,
            1.0,
            1.0,
            0.5,
            true,
        )?;
        assert_eq!(velocity.abs()?.max_all()?.to_scalar::<f32>()?, 0.0);
        Ok(())
    }

    fn tiny_flow_fixture() -> Result<(
        RunConfig,
        VarMap,
        RectifiedFlowRenderer,
        RenderPlan,
        Tensor,
        Tensor,
        Tensor,
    )> {
        let config = RunConfig {
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            interface_width: 32,
            flow: crate::config::FlowConfig {
                hidden: 16,
                resolution: 16,
                ..RunConfig::default().flow
            },
            train_resolution: 24,
            output_resolution: 24,
            ..RunConfig::default()
        };
        let variables = VarMap::new();
        let renderer = RectifiedFlowRenderer::new(
            &config,
            VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu).pp("flow"),
        )?;
        let plan = RenderPlan::new(&config, 16, &Device::Cpu)?;
        let micro = Tensor::zeros((1, 12, 24, 24), DType::F32, &Device::Cpu)?;
        let macro_field = Tensor::zeros((1, 12, 12, 12), DType::F32, &Device::Cpu)?;
        let memory = Tensor::zeros((1, 32), DType::F32, &Device::Cpu)?;
        Ok((
            config,
            variables,
            renderer,
            plan,
            micro,
            macro_field,
            memory,
        ))
    }

    fn variable_snapshot(variables: &VarMap) -> Result<Vec<(String, Vec<f32>)>> {
        let data = variables.data().lock().expect("VarMap mutex poisoned");
        let mut snapshot = data
            .iter()
            .map(|(name, variable)| Ok((name.clone(), variable.flatten_all()?.to_vec1::<f32>()?)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        snapshot.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(snapshot)
    }

    #[test]
    fn deterministic_flow_noise_and_training_sample_are_repeatable() -> Result<()> {
        let first_noise = gaussian_noise((1, 3, 8, 8), 17, &Device::Cpu)?;
        let second_noise = gaussian_noise((1, 3, 8, 8), 17, &Device::Cpu)?;
        let different_noise = gaussian_noise((1, 3, 8, 8), 18, &Device::Cpu)?;
        assert_eq!(
            first_noise.flatten_all()?.to_vec1::<f32>()?,
            second_noise.flatten_all()?.to_vec1::<f32>()?
        );
        assert_ne!(
            first_noise.flatten_all()?.to_vec1::<f32>()?,
            different_noise.flatten_all()?.to_vec1::<f32>()?
        );
        ensure_finite("test Gaussian flow noise", &first_noise)?;

        let rgb = Tensor::ones((1, 3, 8, 8), DType::F32, &Device::Cpu)?.affine(0.4, 0.0)?;
        let first = build_training_sample(&rgb, 29, 31, 0.0, 1.0)?;
        let second = build_training_sample(&rgb, 29, 31, 0.0, 1.0)?;
        assert_eq!(first.time, second.time);
        assert_eq!(
            first.noise.flatten_all()?.to_vec1::<f32>()?,
            second.noise.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            first.interpolant.flatten_all()?.to_vec1::<f32>()?,
            second.interpolant.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn flow_loss_exposes_exact_finite_scalar_telemetry() -> Result<()> {
        let (_config, _variables, renderer, plan, micro, macro_field, memory) =
            tiny_flow_fixture()?;
        let rgb = Tensor::ones((1, 3, 16, 16), DType::F32, &Device::Cpu)?.affine(0.4, 0.0)?;
        let sample = build_training_sample(&rgb, 41, 43, 0.0, 1.0)?;
        let output = renderer.training_loss(
            &sample,
            &micro,
            &macro_field,
            &memory,
            &plan,
            0.5,
            1.0,
            0.25,
            true,
        )?;
        let tensor_value = output.loss.to_scalar::<f32>()?;
        assert!(output.loss_value.is_finite());
        assert_eq!(output.loss_value, tensor_value);
        // The velocity head is exactly zero at birth, so MSE is target RMS².
        assert!(
            (output.loss_value - output.target_velocity_rms.powi(2)).abs() < 1e-5,
            "{} versus {}",
            output.loss_value,
            output.target_velocity_rms.powi(2)
        );
        assert_eq!(output.predicted_velocity_rms, 0.0);
        assert_eq!(output.velocity_cosine, 0.0);
        Ok(())
    }

    #[test]
    fn midpoint_sampling_is_deterministic_and_does_not_mutate_model_or_world() -> Result<()> {
        let (config, variables, renderer, plan, micro, macro_field, memory) = tiny_flow_fixture()?;
        let optimizer = crate::optimizer::PersistentAdamW::new(&variables, &config)?;
        let updates_before = optimizer.updates();
        let variables_before = variable_snapshot(&variables)?;
        let micro_before = micro.flatten_all()?.to_vec1::<f32>()?;
        let macro_before = macro_field.flatten_all()?.to_vec1::<f32>()?;
        let memory_before = memory.flatten_all()?.to_vec1::<f32>()?;

        let first =
            renderer.sample_midpoint(&micro, &macro_field, &memory, &plan, 53, 4, 1.0, 1.0, 0.5)?;
        let second =
            renderer.sample_midpoint(&micro, &macro_field, &memory, &plan, 53, 4, 1.0, 1.0, 0.5)?;
        let diagnosed = renderer.sample_midpoint_with_trajectory(
            &micro,
            &macro_field,
            &memory,
            &plan,
            53,
            4,
            1.0,
            1.0,
            0.5,
        )?;
        assert_eq!(
            first.flatten_all()?.to_vec1::<f32>()?,
            second.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(
            first.flatten_all()?.to_vec1::<f32>()?,
            diagnosed.state.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(diagnosed.trajectory.solver, "midpoint");
        assert_eq!(diagnosed.trajectory.neural_function_evaluations, 8);
        assert_eq!(diagnosed.trajectory.points.len(), 5);
        assert_eq!(diagnosed.trajectory.points.last().unwrap().time, 1.0);
        assert_eq!(variables_before, variable_snapshot(&variables)?);
        assert_eq!(micro_before, micro.flatten_all()?.to_vec1::<f32>()?);
        assert_eq!(macro_before, macro_field.flatten_all()?.to_vec1::<f32>()?);
        assert_eq!(memory_before, memory.flatten_all()?.to_vec1::<f32>()?);
        assert_eq!(updates_before, optimizer.updates());
        Ok(())
    }

    #[test]
    fn flow_ode_rejects_zero_steps_and_runaway_states() -> Result<()> {
        let (_config, variables, renderer, plan, micro, macro_field, memory) = tiny_flow_fixture()?;
        assert!(renderer
            .sample_midpoint(&micro, &macro_field, &memory, &plan, 59, 0, 1.0, 1.0, 0.5,)
            .is_err());

        {
            let data = variables.data().lock().expect("VarMap mutex poisoned");
            let bias = data
                .get("flow.velocity_output.bias")
                .expect("flow velocity bias");
            bias.set(&Tensor::full(200.0f32, 3, &Device::Cpu)?)?;
        }
        let error = renderer
            .sample_midpoint(&micro, &macro_field, &memory, &plan, 61, 1, 1.0, 1.0, 0.5)
            .expect_err("runaway ODE state should be rejected");
        assert!(error.to_string().contains("ODE safety ceiling"));
        Ok(())
    }

    #[test]
    fn flow_rejects_invalid_time_nonfinite_target_and_shortcut_parameters() -> Result<()> {
        let (_config, variables, _renderer, _plan, _micro, _macro_field, _memory) =
            tiny_flow_fixture()?;
        let rgb = Tensor::zeros((1, 3, 4, 4), DType::F32, &Device::Cpu)?;
        assert!(build_training_sample(&rgb, 1, 2, 0.5, 0.5).is_err());
        assert!(build_training_sample(&rgb, 1, 2, -0.1, 1.0).is_err());

        let mut values = vec![0.0f32; 3 * 4 * 4];
        values[7] = f32::NAN;
        let nonfinite = Tensor::from_vec(values, (1, 3, 4, 4), &Device::Cpu)?;
        assert!(build_training_sample(&nonfinite, 1, 2, 0.0, 1.0).is_err());

        let names: Vec<String> = variables
            .data()
            .lock()
            .expect("VarMap mutex poisoned")
            .keys()
            .cloned()
            .collect();
        for forbidden in ["reference", "target", "genome", "encoder"] {
            assert!(
                names.iter().all(|name| !name.contains(forbidden)),
                "flow parameter namespace unexpectedly contains {forbidden}"
            );
        }
        Ok(())
    }
}
