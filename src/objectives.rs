use crate::config::{BoundaryMode, ObjectiveMode, RunConfig, TrainingMode};
use crate::render::{channel_moments, seam_energy, RenderOutput};
use crate::tensor_ops::periodic_shift;
use candle_core::{Result, Tensor};

pub struct LossOutput {
    pub total: Tensor,
    pub endpoint: Tensor,
    pub content: Tensor,
    pub palette: Tensor,
    pub structure: Tensor,
    pub seam: Tensor,
    pub gamut: Tensor,
    pub state: Tensor,
    pub memory: Tensor,
    pub grounding: Tensor,
    pub ground_coarse: Tensor,
    pub ground_mid: Tensor,
    pub ground_fine: Tensor,
    pub ssim: Tensor,
    pub emergent_fit: Tensor,
    pub emergent_low: Tensor,
    pub emergent_tv: Tensor,
    pub head_redundancy: Tensor,
    pub cross_resolution: Tensor,
    pub cross_resolution_low: Tensor,
    pub cross_resolution_edge: Tensor,
}

#[allow(clippy::too_many_arguments)]
pub fn visual_loss(
    rendered: &RenderOutput,
    target: &Tensor,
    micro: &Tensor,
    macro_field: &Tensor,
    memory: &Tensor,
    config: &RunConfig,
    grounding_schedule: f32,
    emergence_schedule: f32,
    boundary: BoundaryMode,
) -> Result<LossOutput> {
    let image = &rendered.image;
    let target = target.detach();
    let content = match config.mode {
        TrainingMode::Single | TrainingMode::Family => image.sub(&target)?.abs()?.mean_all()?,
        TrainingMode::Texture => texture_loss(image, &target)?,
    };
    let palette = moment_loss(image, &target)?;
    let structure = gradient_distribution_loss(image, &target)?;
    let seam = if boundary == BoundaryMode::Periodic {
        seam_energy(image)?
    } else {
        Tensor::new(0.0f32, image.device())?
    };
    let gamut = rendered.gamut_excess.clone();
    let state = stability_barrier(micro, config.state_soft_limit)?
        .add(&stability_barrier(macro_field, config.state_soft_limit)?.affine(0.5, 0.0)?)?;
    let memory = stability_barrier(memory, 0.5 * config.memory_limit)?;
    let (ground_fine, ground_mid, ground_coarse) =
        multiscale_l1(&rendered.grounded_image, &target)?;
    let ssim = ssim_loss(&rendered.grounded_image, &target)?;
    let grounding = ground_coarse
        .affine(config.reconstruction.loss_ground_coarse as f64, 0.0)?
        .add(&ground_mid.affine(config.reconstruction.loss_ground_mid as f64, 0.0)?)?
        .add(&ground_fine.affine(config.reconstruction.loss_ground_fine as f64, 0.0)?)?
        .add(&ssim.affine(0.2, 0.0)?)?;
    let actual_effect = image.sub(&rendered.grounded_image)?;
    let desired_effect = target.sub(&rendered.grounded_image.detach())?;
    let emergent_fit = actual_effect.sub(&desired_effect)?.abs()?.mean_all()?;
    let emergent_low = low_frequency_energy(&rendered.emergent_lab)?;
    let emergent_tv = total_variation(&rendered.emergent_lab)?;
    let head_redundancy =
        absolute_correlation(&rendered.grounded_lab.detach(), &rendered.emergent_lab)?;

    let endpoint = content
        .affine(config.loss_content as f64, 0.0)?
        .add(&palette.affine(config.loss_palette as f64, 0.0)?)?
        .add(&structure.affine(config.loss_structure as f64, 0.0)?)?
        .add(&seam.affine(config.loss_seam as f64, 0.0)?)?
        .add(&gamut.affine(config.loss_gamut as f64, 0.0)?)?;
    let endpoint_gain = f64::from(config.objective.uses_endpoint());
    let reconstruction_plus = matches!(
        config.objective,
        ObjectiveMode::ReconstructionPlus | ObjectiveMode::HybridFlow
    );
    let reconstruction_gain = f64::from(reconstruction_plus);
    let emergence_gain = reconstruction_gain * emergence_schedule as f64;
    let total = endpoint
        .affine(
            endpoint_gain * config.reconstruction.loss_composite as f64,
            0.0,
        )?
        .add(&grounding.affine(reconstruction_gain * grounding_schedule as f64, 0.0)?)?
        .add(&emergent_fit.affine(
            emergence_gain * config.reconstruction.loss_emergent_fit as f64,
            0.0,
        )?)?
        .add(&emergent_low.affine(
            emergence_gain * config.reconstruction.loss_emergent_low as f64,
            0.0,
        )?)?
        .add(&emergent_tv.affine(
            emergence_gain * config.reconstruction.loss_emergent_tv as f64,
            0.0,
        )?)?
        .add(&head_redundancy.affine(
            emergence_gain * config.reconstruction.loss_head_redundancy as f64,
            0.0,
        )?)?
        .add(&state.affine(config.loss_state as f64, 0.0)?)?
        .add(&memory.affine(config.loss_memory as f64, 0.0)?)?;
    Ok(LossOutput {
        cross_resolution: Tensor::new(0.0f32, image.device())?,
        cross_resolution_low: Tensor::new(0.0f32, image.device())?,
        cross_resolution_edge: Tensor::new(0.0f32, image.device())?,
        total,
        endpoint,
        content,
        palette,
        structure,
        seam,
        gamut,
        state,
        memory,
        grounding,
        ground_coarse,
        ground_mid,
        ground_fine,
        ssim,
        emergent_fit,
        emergent_low,
        emergent_tv,
        head_redundancy,
    })
}

pub fn cross_resolution_consistency(
    high_resolution: &Tensor,
    low_resolution: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let pooled = average_pool2(high_resolution)?;
    if pooled.dims4()? != low_resolution.dims4()? {
        candle_core::bail!("cross-resolution comparison requires exactly 2x matching views");
    }
    let l1 = pooled.sub(low_resolution)?.abs()?.mean_all()?;
    let low = average_pool2(&pooled)?
        .sub(&average_pool2(low_resolution)?)?
        .abs()?
        .mean_all()?;
    let edge = edge_map(&pooled)?
        .sub(&edge_map(low_resolution)?)?
        .abs()?
        .mean_all()?;
    Ok((l1, low, edge))
}

fn edge_map(value: &Tensor) -> Result<Tensor> {
    let (_, _, height, width) = value.dims4()?;
    let dx = value
        .narrow(3, 1, width - 1)?
        .sub(&value.narrow(3, 0, width - 1)?)?
        .narrow(2, 0, height - 1)?;
    let dy = value
        .narrow(2, 1, height - 1)?
        .sub(&value.narrow(2, 0, height - 1)?)?
        .narrow(3, 0, width - 1)?;
    dx.sqr()?.add(&dy.sqr()?)?.affine(1.0, 1e-8)?.sqrt()
}
fn multiscale_l1(image: &Tensor, target: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
    let fine = image.sub(target)?.abs()?.mean_all()?;
    let image_mid = average_pool2(image)?;
    let target_mid = average_pool2(target)?;
    let mid = image_mid.sub(&target_mid)?.abs()?.mean_all()?;
    let image_coarse = average_pool2(&image_mid)?;
    let target_coarse = average_pool2(&target_mid)?;
    let coarse = image_coarse.sub(&target_coarse)?.abs()?.mean_all()?;
    Ok((fine, mid, coarse))
}

fn average_pool2(value: &Tensor) -> Result<Tensor> {
    let (batch, channels, height, width) = value.dims4()?;
    let height = height - height % 2;
    let width = width - width % 2;
    value
        .narrow(2, 0, height)?
        .narrow(3, 0, width)?
        .reshape((batch, channels, height / 2, 2, width / 2, 2))?
        .mean(5)?
        .mean(3)
}

fn ssim_loss(image: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (_, channels, height, width) = image.dims4()?;
    let flat_image = image.reshape((channels, height * width))?;
    let flat_target = target.reshape((channels, height * width))?;
    let mean_image = flat_image.mean(1)?;
    let mean_target = flat_target.mean(1)?;
    let centered_image = flat_image.broadcast_sub(&mean_image.unsqueeze(1)?)?;
    let centered_target = flat_target.broadcast_sub(&mean_target.unsqueeze(1)?)?;
    let variance_image = centered_image.sqr()?.mean(1)?;
    let variance_target = centered_target.sqr()?.mean(1)?;
    let covariance = centered_image.mul(&centered_target)?.mean(1)?;
    let numerator = mean_image
        .mul(&mean_target)?
        .affine(2.0, 1e-4)?
        .mul(&covariance.affine(2.0, 9e-4)?)?;
    let denominator = mean_image
        .sqr()?
        .add(&mean_target.sqr()?)?
        .affine(1.0, 1e-4)?
        .mul(&variance_image.add(&variance_target)?.affine(1.0, 9e-4)?)?;
    numerator.div(&denominator)?.mean_all()?.affine(-1.0, 1.0)
}

fn low_frequency_energy(value: &Tensor) -> Result<Tensor> {
    average_pool2(&average_pool2(value)?)?.sqr()?.mean_all()
}

fn total_variation(value: &Tensor) -> Result<Tensor> {
    let (_, _, height, width) = value.dims4()?;
    let dx = value
        .narrow(3, 1, width - 1)?
        .sub(&value.narrow(3, 0, width - 1)?)?
        .abs()?
        .mean_all()?;
    let dy = value
        .narrow(2, 1, height - 1)?
        .sub(&value.narrow(2, 0, height - 1)?)?
        .abs()?
        .mean_all()?;
    dx.add(&dy)?.affine(0.5, 0.0)
}

fn absolute_correlation(first: &Tensor, second: &Tensor) -> Result<Tensor> {
    let first = first.flatten_all()?;
    let second = second.flatten_all()?;
    let first = first.broadcast_sub(&first.mean_all()?)?;
    let second = second.broadcast_sub(&second.mean_all()?)?;
    let covariance = first.mul(&second)?.mean_all()?;
    let scale = first
        .sqr()?
        .mean_all()?
        .mul(&second.sqr()?.mean_all()?)?
        .affine(1.0, 1e-8)?
        .sqrt()?;
    covariance.div(&scale)?.abs()
}
/// Color covariance plus multi-lag spatial autocorrelation. The v4 RGB Gram
/// loss was blind to arrangement; lags 1/2/4/8 make texture scale observable
/// while remaining much cheaper than a learned perceptual network on a phone.
fn texture_loss(image: &Tensor, target: &Tensor) -> Result<Tensor> {
    gram_loss(image, target)?.add(&autocorrelation_loss(image, target)?.affine(0.7, 0.0)?)
}

fn stability_barrier(value: &Tensor, soft_limit: f32) -> Result<Tensor> {
    value
        .abs()?
        .affine(1.0, -(soft_limit as f64))?
        .clamp(0.0f32, f32::MAX)?
        .sqr()?
        .mean_all()
}

fn gram_loss(image: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (_, channels, h, w) = image.dims4()?;
    let n = (h * w) as f64;
    let normalized = |value: &Tensor| -> Result<Tensor> {
        let flat = value.reshape((channels, h * w))?;
        let centered = flat.broadcast_sub(&flat.mean(1)?.unsqueeze(1)?)?;
        let scale = centered
            .sqr()?
            .mean(1)?
            .affine(1.0, 1e-5)?
            .sqrt()?
            .unsqueeze(1)?;
        centered.broadcast_div(&scale)
    };
    let image_flat = normalized(image)?;
    let target_flat = normalized(target)?;
    let image_gram = image_flat.matmul(&image_flat.t()?)?.affine(1.0 / n, 0.0)?;
    let target_gram = target_flat
        .matmul(&target_flat.t()?)?
        .affine(1.0 / n, 0.0)?;
    image_gram.sub(&target_gram)?.sqr()?.mean_all()
}

fn autocorrelation_loss(image: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (_, channels, h, w) = image.dims4()?;
    let image_centered = image.broadcast_sub(&image.mean_keepdim((2, 3))?)?;
    let target_centered = target.broadcast_sub(&target.mean_keepdim((2, 3))?)?;
    let image_variance = image_centered
        .sqr()?
        .reshape((channels, h * w))?
        .mean(1)?
        .affine(1.0, 1e-5)?;
    let target_variance = target_centered
        .sqr()?
        .reshape((channels, h * w))?
        .mean(1)?
        .affine(1.0, 1e-5)?;
    let mut total: Option<Tensor> = None;
    let mut terms = 0usize;
    for lag in [1usize, 2, 4, 8] {
        if lag >= h || lag >= w {
            continue;
        }
        for (dy, dx) in [(0, lag), (lag, 0)] {
            let image_correlation = image_centered
                .mul(&periodic_shift(&image_centered, dy, dx)?)?
                .reshape((channels, h * w))?
                .mean(1)?
                .div(&image_variance)?;
            let target_correlation = target_centered
                .mul(&periodic_shift(&target_centered, dy, dx)?)?
                .reshape((channels, h * w))?
                .mean(1)?
                .div(&target_variance)?;
            let term = image_correlation
                .sub(&target_correlation)?
                .sqr()?
                .mean_all()?;
            total = Some(match total {
                Some(previous) => previous.add(&term)?,
                None => term,
            });
            terms += 1;
        }
    }
    total
        .expect("at least one autocorrelation lag fits validated training resolution")
        .affine(1.0 / terms as f64, 0.0)
}

fn moment_loss(image: &Tensor, target: &Tensor) -> Result<Tensor> {
    moment_distance(image, target)
}

fn moment_distance(image: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (image_mean, image_variance) = channel_moments(image)?;
    let (target_mean, target_variance) = channel_moments(target)?;
    let mean_loss = image_mean.sub(&target_mean)?.sqr()?.mean_all()?;
    let contrast_loss = image_variance
        .affine(1.0, 1e-5)?
        .sqrt()?
        .log()?
        .sub(&target_variance.affine(1.0, 1e-5)?.sqrt()?.log()?)?
        .sqr()?
        .mean_all()?;
    mean_loss.add(&contrast_loss.affine(0.25, 0.0)?)
}

fn gradient_distribution_loss(image: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (_, _, h, w) = image.dims4()?;
    let image_dx = image
        .narrow(3, 1, w - 1)?
        .sub(&image.narrow(3, 0, w - 1)?)?;
    let target_dx = target
        .narrow(3, 1, w - 1)?
        .sub(&target.narrow(3, 0, w - 1)?)?;
    let image_dy = image
        .narrow(2, 1, h - 1)?
        .sub(&image.narrow(2, 0, h - 1)?)?;
    let target_dy = target
        .narrow(2, 1, h - 1)?
        .sub(&target.narrow(2, 0, h - 1)?)?;
    gradient_stat_distance(&image_dx, &target_dx)?
        .add(&gradient_stat_distance(&image_dy, &target_dy)?)?
        .affine(0.5, 0.0)
}

fn gradient_stat_distance(image: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (_, channels, h, w) = image.dims4()?;
    let statistics = |value: &Tensor| -> Result<(Tensor, Tensor)> {
        let flat = value.reshape((channels, h * w))?;
        let mean_abs = flat.abs()?.mean(1)?.affine(1.0, 1e-5)?.log()?;
        let rms = flat.sqr()?.mean(1)?.affine(1.0, 1e-5)?.sqrt()?.log()?;
        Ok((mean_abs, rms))
    };
    let (image_abs, image_rms) = statistics(image)?;
    let (target_abs, target_rms) = statistics(target)?;
    image_abs
        .sub(&target_abs)?
        .sqr()?
        .mean_all()?
        .add(&image_rms.sub(&target_rms)?.sqr()?.mean_all()?)?
        .affine(0.5, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn identical_images_have_near_zero_loss() -> Result<()> {
        let config = RunConfig {
            mode: TrainingMode::Texture,
            ..RunConfig::default()
        };
        let image = Tensor::ones((1, 3, 16, 16), DType::F32, &Device::Cpu)?.affine(0.4, 0.0)?;
        let rendered = RenderOutput {
            image: image.clone(),
            gamut_excess: Tensor::new(0.0f32, &Device::Cpu)?,
            grounded_image: image.clone(),
            emergent_visual: Tensor::zeros_like(&image)?,
            grounded_lab: Tensor::zeros_like(&image)?,
            emergent_lab: Tensor::zeros_like(&image)?,
            emergence_strength: 0.0,
            state_only_image: None,
            learned_only_image: None,
        };
        let state = Tensor::zeros((1, 12, 16, 16), DType::F32, &Device::Cpu)?;
        let memory = Tensor::zeros((1, 32), DType::F32, &Device::Cpu)?;
        let loss = visual_loss(
            &rendered,
            &image,
            &state,
            &state,
            &memory,
            &config,
            1.0,
            0.0,
            BoundaryMode::Natural,
        )?;
        assert!(loss.total.to_scalar::<f32>()? < 1e-7);
        Ok(())
    }

    #[test]
    fn stability_barrier_ignores_center_and_penalizes_excess() -> Result<()> {
        let values = Tensor::new(&[-2.0f32, -0.5, 0.0, 0.5, 2.0], &Device::Cpu)?;
        let loss = stability_barrier(&values, 1.0)?.to_scalar::<f32>()?;
        assert!((loss - 0.4).abs() < 1e-6);
        let centered = Tensor::new(&[-0.5f32, 0.0, 0.5], &Device::Cpu)?;
        assert_eq!(stability_barrier(&centered, 1.0)?.to_scalar::<f32>()?, 0.0);
        Ok(())
    }
}
