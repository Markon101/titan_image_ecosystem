use crate::config::RunConfig;
use crate::tensor_ops::{laplacian_kernel, laplacian_with_kernel, splitmix64, zeros};
use anyhow::Result;
use candle_core::{Device, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

pub struct ScaleFields {
    fractal_target: Tensor,
    quasiperiodic_target: Tensor,
    zeros_before_fractal: Tensor,
    zeros_after_fractal: Tensor,
    zeros_before_quasi: Tensor,
    zeros_after_quasi: Tensor,
}

pub struct PhysicalOperators {
    pub micro_fields: ScaleFields,
    pub macro_fields: ScaleFields,
    laplacian_two: Tensor,
    laplacian_three: Tensor,
    config: RunConfig,
}

impl PhysicalOperators {
    pub fn new(config: &RunConfig, device: &Device) -> Result<Self> {
        Ok(Self {
            micro_fields: ScaleFields::new(config.micro_size, config, config.seed, device)?,
            macro_fields: ScaleFields::new(
                config.macro_size,
                config,
                config.seed ^ 0x510f_a11e,
                device,
            )?,
            laplacian_two: laplacian_kernel(2, device)?,
            laplacian_three: laplacian_kernel(3, device)?,
            config: config.clone(),
        })
    }

    pub fn reaction_diffusion(&self, field: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = field.dims4()?;
        let u = field.narrow(1, 0, 1)?;
        let v = field.narrow(1, 1, 1)?;
        let lap = laplacian_with_kernel(&Tensor::cat(&[&u, &v], 1)?, &self.laplacian_two)?;
        let lap_u = lap.narrow(1, 0, 1)?;
        let lap_v = lap.narrow(1, 1, 1)?;
        let uv2 = u.mul(&v.sqr()?)?;
        let feed = self.config.rd_feed as f64;
        let kill = self.config.rd_kill as f64;
        let du = lap_u
            .affine(self.config.rd_diffusion_u as f64, 0.0)?
            .sub(&uv2)?
            .add(&u.affine(-feed, feed)?)?;
        let dv = lap_v
            .affine(self.config.rd_diffusion_v as f64, 0.0)?
            .add(&uv2)?
            .sub(&v.affine(feed + kill, 0.0)?)?;
        let rest = zeros(self.config.channels - 2, h, w, field.device())?;
        Ok(Tensor::cat(&[&du, &dv, &rest], 1)?.affine(self.config.reaction_gain as f64, 0.0)?)
    }

    pub fn complex_phase(&self, field: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = field.dims4()?;
        let re = field.narrow(1, 2, 1)?;
        let im = field.narrow(1, 3, 1)?;
        let lap = laplacian_with_kernel(&Tensor::cat(&[&re, &im], 1)?, &self.laplacian_two)?;
        let lap_re = lap.narrow(1, 0, 1)?;
        let lap_im = lap.narrow(1, 1, 1)?;
        let magnitude_sq = re.sqr()?.add(&im.sqr()?)?;
        let nonlinear_re = magnitude_sq.mul(&re)?;
        let nonlinear_im = magnitude_sq.mul(&im)?;
        let growth = self.config.phase_growth as f64;
        let saturation = self.config.phase_saturation as f64;
        let frequency = self.config.phase_frequency as f64;
        let diffusion = self.config.phase_diffusion as f64;
        let dispersion = self.config.phase_dispersion as f64;
        let dre = re
            .affine(growth, 0.0)?
            .add(&lap_re.affine(diffusion, 0.0)?)?
            .sub(&lap_im.affine(dispersion, 0.0)?)?
            .sub(&nonlinear_re.affine(saturation, 0.0)?)?
            .add(&nonlinear_im.affine(frequency, 0.0)?)?;
        let dim = im
            .affine(growth, 0.0)?
            .add(&lap_im.affine(diffusion, 0.0)?)?
            .add(&lap_re.affine(dispersion, 0.0)?)?
            .sub(&nonlinear_im.affine(saturation, 0.0)?)?
            .sub(&nonlinear_re.affine(frequency, 0.0)?)?;
        let prefix = zeros(2, h, w, field.device())?;
        let rest = zeros(self.config.channels - 4, h, w, field.device())?;
        Ok(Tensor::cat(&[&prefix, &dre, &dim, &rest], 1)?
            .affine(self.config.phase_gain as f64, 0.0)?)
    }

    /// A damped cyclic three-field oscillator in channels 6..8. The learned
    /// NCA and the other operators continually excite it; diffusion organizes
    /// that rotation into traveling color/morphology fronts.
    pub fn cyclic_chemistry(&self, field: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = field.dims4()?;
        let species = field.narrow(1, 6, 3)?;
        let lap = laplacian_with_kernel(&species, &self.laplacian_three)?;
        let a = species.narrow(1, 0, 1)?;
        let b = species.narrow(1, 1, 1)?;
        let c = species.narrow(1, 2, 1)?;
        let lap_a = lap.narrow(1, 0, 1)?;
        let lap_b = lap.narrow(1, 1, 1)?;
        let lap_c = lap.narrow(1, 2, 1)?;
        let oscillator = |self_field: &Tensor,
                          positive: &Tensor,
                          negative: &Tensor,
                          lap_field: &Tensor|
         -> candle_core::Result<Tensor> {
            lap_field
                .affine(0.055, 0.0)?
                .add(&positive.affine(0.55, 0.0)?)?
                .sub(&negative.affine(0.55, 0.0)?)?
                .sub(&self_field.affine(0.08, 0.0)?)?
                .sub(&self_field.sqr()?.mul(self_field)?.affine(0.025, 0.0)?)
        };
        let da = oscillator(&a, &b, &c, &lap_a)?;
        let db = oscillator(&b, &c, &a, &lap_b)?;
        let dc = oscillator(&c, &a, &b, &lap_c)?;
        let prefix = zeros(6, h, w, field.device())?;
        let rest = zeros(self.config.channels - 9, h, w, field.device())?;
        Ok(Tensor::cat(&[&prefix, &da, &db, &dc, &rest], 1)?
            .affine(self.config.cyclic_gain as f64, 0.0)?)
    }

    pub fn forcing(&self, field: &Tensor, scale: &ScaleFields) -> Result<Tensor> {
        let mut delta = zeros(
            self.config.channels,
            field.dim(2)?,
            field.dim(3)?,
            field.device(),
        )?;
        if self.config.fractal_gain > 0.0 {
            delta = delta.add(&scale.attraction(field, 4, true, self.config.fractal_gain)?)?;
        }
        if self.config.quasiperiodic_gain > 0.0 {
            delta =
                delta.add(&scale.attraction(field, 5, false, self.config.quasiperiodic_gain)?)?;
        }
        Ok(delta)
    }
}

impl ScaleFields {
    fn new(size: usize, config: &RunConfig, seed: u64, device: &Device) -> Result<Self> {
        let fractal = contractive_ifs_field(size, seed);
        let quasi = quasiperiodic_field(size, seed);
        Ok(Self {
            fractal_target: Tensor::from_vec(fractal, (1, 1, size, size), device)?,
            quasiperiodic_target: Tensor::from_vec(quasi, (1, 1, size, size), device)?,
            zeros_before_fractal: zeros(4, size, size, device)?,
            zeros_after_fractal: zeros(config.channels - 5, size, size, device)?,
            zeros_before_quasi: zeros(5, size, size, device)?,
            zeros_after_quasi: zeros(config.channels - 6, size, size, device)?,
        })
    }

    fn attraction(
        &self,
        field: &Tensor,
        channel: usize,
        fractal: bool,
        gain: f32,
    ) -> Result<Tensor> {
        let target = if fractal {
            &self.fractal_target
        } else {
            &self.quasiperiodic_target
        };
        let before = if fractal {
            &self.zeros_before_fractal
        } else {
            &self.zeros_before_quasi
        };
        let after = if fractal {
            &self.zeros_after_fractal
        } else {
            &self.zeros_after_quasi
        };
        let signal = target
            .sub(&field.narrow(1, channel, 1)?)?
            .affine(gain as f64, 0.0)?;
        Ok(Tensor::cat(&[before, &signal, after], 1)?)
    }
}

fn contractive_ifs_field(size: usize, seed: u64) -> Vec<f32> {
    let maps = [
        (0.52f32, 0.15f32, -0.42f32, -0.18f32),
        (0.49, 2.12, 0.43, -0.12),
        (0.46, -1.93, 0.02, 0.48),
    ];
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x1f5_f1a9e);
    let rotation = ((splitmix64(seed ^ 0xf1a9_e5ca) >> 40) as f32 / (1u32 << 24) as f32)
        * std::f32::consts::TAU;
    let rotation_cos = rotation.cos();
    let rotation_sin = rotation.sin();
    let mut histogram = vec![0.0f32; size * size];
    let (mut x, mut y) = (0.0f32, 0.0f32);
    for iteration in 0..size * size * 24 {
        let (scale, angle, tx, ty) = maps[rng.gen_range(0..maps.len())];
        let cosine = angle.cos();
        let sine = angle.sin();
        let next_x = scale * (cosine * x - sine * y) + tx;
        let next_y = scale * (sine * x + cosine * y) + ty;
        x = next_x;
        y = next_y;
        if iteration > 32 {
            let display_x = rotation_cos * x - rotation_sin * y;
            let display_y = rotation_sin * x + rotation_cos * y;
            let px = (((display_x + 1.2) / 2.4) * size as f32).floor() as isize;
            let py = (((display_y + 1.2) / 2.4) * size as f32).floor() as isize;
            if px >= 0 && py >= 0 && px < size as isize && py < size as isize {
                histogram[py as usize * size + px as usize] += 1.0;
            }
        }
    }
    for _ in 0..3 {
        histogram = toroidal_blur(&histogram, size);
    }
    normalize_centered(&mut histogram);
    histogram
}

fn quasiperiodic_field(size: usize, seed: u64) -> Vec<f32> {
    let phase = ((splitmix64(seed) >> 40) as f32 / (1u32 << 24) as f32) * std::f32::consts::TAU;
    let mut values = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let nx = x as f32 / size as f32;
            let ny = y as f32 / size as f32;
            let a =
                std::f32::consts::TAU * (1.618_034 * nx + std::f32::consts::SQRT_2 * ny) + phase;
            let b = std::f32::consts::TAU * (2.414_213_7 * nx - 1.732_050_8 * ny) - phase;
            let c = std::f32::consts::TAU * (0.754_877_7 * nx + 2.236_068 * ny);
            values.push((a.sin() + 0.65 * b.cos() + 0.35 * c.sin()) / 2.0);
        }
    }
    normalize_centered(&mut values);
    values
}

fn toroidal_blur(values: &[f32], size: usize) -> Vec<f32> {
    let mut output = vec![0.0; values.len()];
    for y in 0..size {
        for x in 0..size {
            let mut sum = 0.0;
            for dy in [size - 1, 0, 1] {
                for dx in [size - 1, 0, 1] {
                    sum += values[((y + dy) % size) * size + ((x + dx) % size)];
                }
            }
            output[y * size + x] = sum / 9.0;
        }
    }
    output
}

fn normalize_centered(values: &mut [f32]) {
    let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len().max(1) as f32;
    let scale = variance.sqrt().max(1e-6);
    for value in values {
        *value = ((*value - mean) / (2.5 * scale)).clamp(-1.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forcing_fields_are_finite_and_centered() {
        let field = contractive_ifs_field(32, 1);
        assert!(field.iter().all(|value| value.is_finite()));
        let mean = field.iter().sum::<f32>() / field.len() as f32;
        assert!(mean.abs() < 0.1);
    }
}
