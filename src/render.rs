use crate::config::{RunConfig, StylePreset};
use crate::tensor_ops::{
    broadcast_vector, coordinate_features, pixelwise_linear_mode, PeriodicUpsampler,
};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor, D};
use candle_nn::{Linear, VarBuilder};
use std::path::{Path, PathBuf};

pub struct ImplicitRenderer {
    input: Linear,
    blocks: Vec<Linear>,
    output: Linear,
    state_skip: f32,
    chroma: f32,
    gamma: f64,
    style: StylePreset,
}

pub struct RenderPlan {
    pub resolution: usize,
    micro: PeriodicUpsampler,
    macro_field: PeriodicUpsampler,
    coordinates: Tensor,
}

pub struct RenderOutput {
    pub image: Tensor,
    /// Mean excursion of pre-clipped linear RGB outside [0, 1]. This stays in
    /// the loss graph so saturated output heads retain a corrective gradient.
    pub gamut_excess: Tensor,
}

impl RenderPlan {
    pub fn new(config: &RunConfig, resolution: usize, device: &Device) -> Result<Self> {
        Ok(Self {
            resolution,
            micro: PeriodicUpsampler::new(
                config.micro_size,
                config.micro_size,
                resolution,
                resolution,
                device,
            )?,
            macro_field: PeriodicUpsampler::new(
                config.macro_size,
                config.macro_size,
                resolution,
                resolution,
                device,
            )?,
            coordinates: coordinate_features(resolution, config.coord_bands, device)?
                .affine(config.coord_gain as f64, 0.0)?,
        })
    }
}

impl ImplicitRenderer {
    pub fn new(config: &RunConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let input_features = config.channels * 2 + config.genome_dim + config.coord_bands * 4;
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
            output: candle_nn::linear(config.render_hidden, 3, vb.pp("oklab"))?,
            state_skip: config.state_skip,
            chroma: config.chroma,
            gamma: config.gamma as f64,
            style: config.style,
        })
    }

    pub fn render(
        &self,
        micro: &Tensor,
        macro_field: &Tensor,
        genome: &Tensor,
        plan: &RenderPlan,
        tracked: bool,
    ) -> Result<RenderOutput> {
        let micro_up = plan.micro.apply(micro)?;
        let macro_up = plan.macro_field.apply(macro_field)?;
        let genome_field = broadcast_vector(genome, plan.resolution, plan.resolution)?;
        let features = Tensor::cat(&[&micro_up, &macro_up, &plan.coordinates, &genome_field], 1)?;
        let mut hidden = swish(&pixelwise_linear_mode(&features, &self.input, tracked)?)?;
        for block in &self.blocks {
            let residual = swish(&pixelwise_linear_mode(&hidden, block, tracked)?)?;
            hidden = hidden.add(&residual.affine(0.5, 0.0)?)?;
        }
        let learned_lab = pixelwise_linear_mode(&hidden, &self.output, tracked)?;
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
        let lab_raw = learned_lab.add(&state_lab)?;
        let l = candle_nn::ops::sigmoid(&lab_raw.narrow(1, 0, 1)?)?.affine(0.84, 0.08)?;
        let a = lab_raw
            .narrow(1, 1, 1)?
            .tanh()?
            .affine(self.chroma as f64, 0.0)?;
        let b = lab_raw
            .narrow(1, 2, 1)?
            .tanh()?
            .affine(self.chroma as f64, 0.0)?;
        let rgb_linear = oklab_to_linear_rgb(&l, &a, &b)?;
        let clipped = rgb_linear.clamp(0.0f32, 1.0f32)?;
        let gamut_excess = rgb_linear.sub(&clipped)?.abs()?.mean_all()?;
        let image = clipped
            .affine(1.0, 1e-6)?
            .powf(1.0 / self.gamma)?
            .clamp(0.0f32, 1.0f32)?;
        Ok(RenderOutput {
            image,
            gamut_excess,
        })
    }
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
}
