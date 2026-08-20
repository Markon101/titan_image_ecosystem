use crate::render::{channel_moments, seam_energy};
use crate::tensor_ops::variance;
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
    pub movement_mean: f32,
    pub movement_max: f32,
    pub state_rms: f32,
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
    pub gradient_clip_scale: f32,
    pub effective_learning_rate: f64,
    pub window_seconds: f64,
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

impl MetricRecord {
    pub fn write_header(mut writer: impl Write) -> Result<()> {
        writeln!(writer, "step,age,episode,target_index,optimizer_update,core_trained,macro_updates,movement_mean,movement_max,state_rms,image_variance,seam_energy,edge_energy,gamut_excess,red_mean,green_mean,blue_mean,red_variance,green_variance,blue_variance,rg_correlation,rb_correlation,gb_correlation,loss_total,loss_content,loss_palette,loss_structure,loss_seam,loss_gamut,gradient_norm,gradient_clip_scale,effective_learning_rate,window_seconds")?;
        Ok(())
    }

    pub fn write_csv(&self, mut writer: impl Write) -> Result<()> {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.7},{:.9},{:.6}",
            self.step,
            self.age,
            self.episode,
            self.target_index,
            self.optimizer_update,
            self.core_trained,
            self.macro_updates,
            self.movement_mean,
            self.movement_max,
            self.state_rms,
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
            self.gradient_clip_scale,
            self.effective_learning_rate,
            self.window_seconds,
        )?;
        Ok(())
    }
}

pub fn image_metrics(image: &Tensor) -> candle_core::Result<ImageDiagnostics> {
    let (_, channels, h, w) = image.dims4()?;
    debug_assert_eq!(channels, 3);
    let image_variance = variance(image)?;
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
