use crate::render::{channel_moments, seam_energy};
use anyhow::Result;
use candle_core::Tensor;
use serde::Serialize;
use std::io::Write;

#[derive(Clone, Debug, Serialize)]
pub struct MetricRecord {
    pub step: u64,
    pub age: u64,
    pub episode: u64,
    pub target_index: usize,
    pub optimizer_update: u64,
    pub core_trained: bool,
    pub macro_updates: usize,
    pub episode_started: bool,
    pub micro_movement_mean: f32,
    pub micro_movement_max: f32,
    pub macro_movement_mean: f32,
    pub macro_movement_max: f32,
    pub micro_state_rms: f32,
    pub macro_state_rms: f32,
    pub micro_state_mean_abs: f32,
    pub macro_state_mean_abs: f32,
    pub micro_clamp_fraction: f32,
    pub macro_clamp_fraction: f32,
    pub micro_channel_rms_min: f32,
    pub micro_channel_rms_max: f32,
    pub macro_channel_rms_min: f32,
    pub macro_channel_rms_max: f32,
    pub image_delta_valid: bool,
    pub image_delta_mean: f32,
    pub image_delta_rms: f32,
    pub image_variance: f32,
    pub seam_energy: f32,
    pub edge_energy: f32,
    pub gamut_excess: f32,
    pub red_mean: f32,
    pub green_mean: f32,
    pub blue_mean: f32,
    pub red_variance: f32,
    pub green_variance: f32,
    pub blue_variance: f32,
    pub rg_correlation: f32,
    pub rb_correlation: f32,
    pub gb_correlation: f32,
    pub loss_total: f32,
    pub loss_content: f32,
    pub loss_palette: f32,
    pub loss_structure: f32,
    pub loss_seam: f32,
    pub loss_gamut: f32,
    pub gradient_norm: f32,
    pub gradient_rms: f32,
    pub gradient_clip_scale: f32,
    pub updated_variables: usize,
    pub updated_parameters: usize,
    pub effective_learning_rate: f64,
    pub window_seconds: f64,
    pub reference_fidelity: f32,
    pub interface_memory_rms: f32,
    pub muon_variables: usize,
    pub loss_state: f32,
    pub loss_memory: f32,
    pub core_gradient_rms: f32,
    pub decoder_gradient_rms: f32,
    pub core_updated_parameters: usize,
    pub decoder_updated_parameters: usize,
    pub stability_violation: bool,
    pub development_steps_per_second: f64,
}

#[derive(Clone, Debug)]
pub struct ImageDiagnostics {
    pub variance: f32,
    pub seam: f32,
    pub edge: f32,
    pub means: [f32; 3],
    pub variances: [f32; 3],
    pub correlations: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct StateDiagnostics {
    pub rms: f32,
    pub mean_abs: f32,
    pub clamp_fraction: f32,
    pub channel_rms_min: f32,
    pub channel_rms_max: f32,
}

impl MetricRecord {
    pub fn write_header(mut writer: impl Write) -> Result<()> {
        writeln!(writer, "step,age,episode,target_index,optimizer_update,core_trained,macro_updates,episode_started,micro_movement_mean,micro_movement_max,macro_movement_mean,macro_movement_max,micro_state_rms,macro_state_rms,micro_state_mean_abs,macro_state_mean_abs,micro_clamp_fraction,macro_clamp_fraction,micro_channel_rms_min,micro_channel_rms_max,macro_channel_rms_min,macro_channel_rms_max,image_delta_valid,image_delta_mean,image_delta_rms,image_variance,seam_energy,edge_energy,gamut_excess,red_mean,green_mean,blue_mean,red_variance,green_variance,blue_variance,rg_correlation,rb_correlation,gb_correlation,loss_total,loss_content,loss_palette,loss_structure,loss_seam,loss_gamut,gradient_norm,gradient_rms,gradient_clip_scale,updated_variables,updated_parameters,effective_learning_rate,window_seconds,reference_fidelity,interface_memory_rms,muon_variables,loss_state,loss_memory,core_gradient_rms,decoder_gradient_rms,core_updated_parameters,decoder_updated_parameters,stability_violation,development_steps_per_second")?;
        Ok(())
    }

    pub fn write_csv(&self, mut writer: impl Write) -> Result<()> {
        write!(
            writer,
            "{},{},{},{},{},{},{},{},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.9},{:.7},{},{},{:.9},{:.6},{:.7},{:.7},{}",
            self.step,
            self.age,
            self.episode,
            self.target_index,
            self.optimizer_update,
            self.core_trained,
            self.macro_updates,
            self.episode_started,
            self.micro_movement_mean,
            self.micro_movement_max,
            self.macro_movement_mean,
            self.macro_movement_max,
            self.micro_state_rms,
            self.macro_state_rms,
            self.micro_state_mean_abs,
            self.macro_state_mean_abs,
            self.micro_clamp_fraction,
            self.macro_clamp_fraction,
            self.micro_channel_rms_min,
            self.micro_channel_rms_max,
            self.macro_channel_rms_min,
            self.macro_channel_rms_max,
            self.image_delta_valid,
            self.image_delta_mean,
            self.image_delta_rms,
            self.image_variance,
            self.seam_energy,
            self.edge_energy,
            self.gamut_excess,
            self.red_mean,
            self.green_mean,
            self.blue_mean,
            self.red_variance,
            self.green_variance,
            self.blue_variance,
            self.rg_correlation,
            self.rb_correlation,
            self.gb_correlation,
            self.loss_total,
            self.loss_content,
            self.loss_palette,
            self.loss_structure,
            self.loss_seam,
            self.loss_gamut,
            self.gradient_norm,
            self.gradient_rms,
            self.gradient_clip_scale,
            self.updated_variables,
            self.updated_parameters,
            self.effective_learning_rate,
            self.window_seconds,
            self.reference_fidelity,
            self.interface_memory_rms,
            self.muon_variables,
        )?;
        writeln!(
            writer,
            ",{:.7},{:.7},{:.9},{:.9},{},{},{},{:.3}",
            self.loss_state,
            self.loss_memory,
            self.core_gradient_rms,
            self.decoder_gradient_rms,
            self.core_updated_parameters,
            self.decoder_updated_parameters,
            self.stability_violation,
            self.development_steps_per_second,
        )?;
        Ok(())
    }
}

pub fn image_metrics(image: &Tensor) -> candle_core::Result<ImageDiagnostics> {
    let (_, channels, h, w) = image.dims4()?;
    debug_assert_eq!(channels, 3);
    let seams = seam_energy(image)?;
    let dx = image
        .narrow(3, 1, w - 1)?
        .sub(&image.narrow(3, 0, w - 1)?)?;
    let dy = image
        .narrow(2, 1, h - 1)?
        .sub(&image.narrow(2, 0, h - 1)?)?;
    let edge = dx
        .sqr()?
        .mean_all()?
        .add(&dy.sqr()?.mean_all()?)?
        .affine(0.5, 0.0)?;
    let (means, variances) = channel_moments(image)?;
    let image_variance = variances.mean_all()?;
    let flat = image.reshape((3, h * w))?;
    let centered = flat.broadcast_sub(&means.unsqueeze(1)?)?;
    let correlation = |first: usize, second: usize| -> candle_core::Result<Tensor> {
        let covariance = centered
            .narrow(0, first, 1)?
            .mul(&centered.narrow(0, second, 1)?)?
            .mean_all()?;
        let scale = variances
            .narrow(0, first, 1)?
            .mul(&variances.narrow(0, second, 1)?)?
            .affine(1.0, 1e-8)?
            .sqrt()?
            .squeeze(0)?;
        covariance.div(&scale)
    };
    let scalar_values = Tensor::stack(
        &[
            &image_variance,
            &seams,
            &edge,
            &correlation(0, 1)?,
            &correlation(0, 2)?,
            &correlation(1, 2)?,
        ],
        0,
    )?
    .to_vec1::<f32>()?;
    let mean_values: [f32; 3] = means.to_vec1::<f32>()?.try_into().unwrap_or([0.0; 3]);
    let variance_values: [f32; 3] = variances.to_vec1::<f32>()?.try_into().unwrap_or([0.0; 3]);
    Ok(ImageDiagnostics {
        variance: scalar_values[0],
        seam: scalar_values[1],
        edge: scalar_values[2],
        means: mean_values,
        variances: variance_values,
        correlations: [scalar_values[3], scalar_values[4], scalar_values[5]],
    })
}

pub fn tensor_rms(tensor: &Tensor) -> candle_core::Result<f32> {
    tensor.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()
}

pub fn state_metrics(tensor: &Tensor, state_limit: f32) -> candle_core::Result<StateDiagnostics> {
    let (_, channels, height, width) = tensor.dims4()?;
    let absolute = tensor.abs()?;
    let threshold = state_limit * 0.99;
    let clamp_fraction = absolute
        .ge(threshold)?
        .to_dtype(candle_core::DType::F32)?
        .mean_all()?
        .to_scalar::<f32>()?;
    let channel_rms = tensor
        .sqr()?
        .reshape((channels, height * width))?
        .mean(1)?
        .sqrt()?
        .to_vec1::<f32>()?;
    Ok(StateDiagnostics {
        rms: tensor_rms(tensor)?,
        mean_abs: absolute.mean_all()?.to_scalar::<f32>()?,
        clamp_fraction,
        channel_rms_min: channel_rms.iter().copied().fold(f32::INFINITY, f32::min),
        channel_rms_max: channel_rms.iter().copied().fold(0.0, f32::max),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn state_metrics_report_scale_channels_and_bound_occupancy() -> candle_core::Result<()> {
        let values = vec![0.0f32, 1.0, -2.0, 3.5, 0.0, 0.0, 0.0, 0.0];
        let tensor = Tensor::from_vec(values, (1, 2, 2, 2), &Device::Cpu)?;
        let metrics = state_metrics(&tensor, 3.5)?;
        assert!((metrics.clamp_fraction - 0.125).abs() < 1e-6);
        assert!(metrics.channel_rms_max > metrics.channel_rms_min);
        assert!(metrics.rms.is_finite());
        Ok(())
    }

    #[test]
    fn spatial_variance_rejects_uniform_color_false_positive() -> candle_core::Result<()> {
        let red = Tensor::zeros((1, 1, 4, 4), candle_core::DType::F32, &Device::Cpu)?;
        let cyan = Tensor::ones((1, 1, 4, 4), candle_core::DType::F32, &Device::Cpu)?;
        let image = Tensor::cat(&[&red, &cyan, &cyan], 1)?;
        let metrics = image_metrics(&image)?;
        assert!(metrics.variance < 1e-7);
        assert!(metrics.edge < 1e-7);
        assert_eq!(metrics.means, [0.0, 1.0, 1.0]);
        Ok(())
    }
}
