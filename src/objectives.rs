use crate::config::{RunConfig, TrainingMode};
use crate::render::{channel_moments, seam_energy, RenderOutput};
use crate::tensor_ops::periodic_shift;
use candle_core::{Result, Tensor};

pub struct LossOutput {
    pub total: Tensor,
    pub content: Tensor,
    pub palette: Tensor,
    pub structure: Tensor,
    pub seam: Tensor,
    pub gamut: Tensor,
    pub state: Tensor,
    pub memory: Tensor,
}

pub fn visual_loss(
    rendered: &RenderOutput,
    target: &Tensor,
    micro: &Tensor,
    macro_field: &Tensor,
    memory: &Tensor,
    config: &RunConfig,
) -> Result<LossOutput> {
    let image = &rendered.image;
    let target = target.detach();
    let content = match config.mode {
        TrainingMode::Single | TrainingMode::Family => image.sub(&target)?.abs()?.mean_all()?,
        TrainingMode::Texture => texture_loss(image, &target)?,
    };
    let palette = moment_loss(image, &target)?;
    let structure = gradient_distribution_loss(image, &target)?;
    let seam = seam_energy(image)?;
    let gamut = rendered.gamut_excess.clone();
    let state = stability_barrier(micro, config.state_soft_limit)?
        .add(&stability_barrier(macro_field, config.state_soft_limit)?.affine(0.5, 0.0)?)?;
    let memory = stability_barrier(memory, 0.5 * config.memory_limit)?;
    let total = content
        .affine(config.loss_content as f64, 0.0)?
        .add(&palette.affine(config.loss_palette as f64, 0.0)?)?
        .add(&structure.affine(config.loss_structure as f64, 0.0)?)?
        .add(&seam.affine(config.loss_seam as f64, 0.0)?)?
        .add(&gamut.affine(config.loss_gamut as f64, 0.0)?)?
        .add(&state.affine(config.loss_state as f64, 0.0)?)?
        .add(&memory.affine(config.loss_memory as f64, 0.0)?)?;
    Ok(LossOutput {
        total,
        content,
        palette,
        structure,
        seam,
        gamut,
        state,
        memory,
    })
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
        };
        let state = Tensor::zeros((1, 12, 16, 16), DType::F32, &Device::Cpu)?;
        let memory = Tensor::zeros((1, 32), DType::F32, &Device::Cpu)?;
        let loss = visual_loss(&rendered, &image, &state, &state, &memory, &config)?;
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
