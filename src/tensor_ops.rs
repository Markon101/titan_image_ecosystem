use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Linear, Module};

pub const PERCEPTION_FILTERS_PER_RING: usize = 4;
pub const PERCEPTION_RINGS: usize = 2;
pub const PERCEPTION_FEATURES_PER_CHANNEL: usize = PERCEPTION_FILTERS_PER_RING * PERCEPTION_RINGS;

pub fn pad_toroidal(x: &Tensor, pad: usize) -> Result<Tensor> {
    if pad == 0 {
        return Ok(x.clone());
    }
    let w = x.dim(3)?;
    let left = x.narrow(3, w - pad, pad)?;
    let right = x.narrow(3, 0, pad)?;
    let xw = Tensor::cat(&[&left, x, &right], 3)?;
    let h = xw.dim(2)?;
    let top = xw.narrow(2, h - pad, pad)?;
    let bottom = xw.narrow(2, 0, pad)?;
    Tensor::cat(&[&top, &xw, &bottom], 2)
}

pub fn pixelwise_linear(x: &Tensor, linear: &Linear) -> Result<Tensor> {
    pixelwise_linear_mode(x, linear, true)
}

/// Pointwise linear projection with an inference path that detaches weights.
/// This prevents decoder-only windows from building and immediately discarding
/// a recurrent-core autograd graph.
pub fn pixelwise_linear_mode(x: &Tensor, linear: &Linear, tracked: bool) -> Result<Tensor> {
    let detached;
    let linear = if tracked {
        linear
    } else {
        detached = Linear::new(linear.weight().detach(), linear.bias().map(Tensor::detach));
        &detached
    };
    let (b, c, h, w) = x.dims4()?;
    let flat = x.reshape((b, c, h * w))?.transpose(1, 2)?.contiguous()?;
    let out = linear.forward(&flat)?;
    let out_channels = out.dim(2)?;
    out.transpose(1, 2)?
        .contiguous()?
        .reshape((b, out_channels, h, w))
}

pub fn perception_kernel(channels: usize, device: &Device) -> Result<Tensor> {
    let filters: [[f32; 9]; PERCEPTION_FILTERS_PER_RING] = [
        [0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [-1., 0., 1., -2., 0., 2., -1., 0., 1.],
        [-1., -2., -1., 0., 0., 0., 1., 2., 1.],
        [0., 1., 0., 1., -4., 1., 0., 1., 0.],
    ];
    let mut values = vec![0.0f32; channels * PERCEPTION_FILTERS_PER_RING * 9];
    for channel in 0..channels {
        for (filter_index, filter) in filters.iter().enumerate() {
            let offset = (channel * PERCEPTION_FILTERS_PER_RING + filter_index) * 9;
            for (index, value) in filter.iter().enumerate() {
                values[offset + index] = if filter_index == 0 {
                    *value
                } else {
                    *value / 8.0
                };
            }
        }
    }
    Tensor::from_vec(
        values,
        (channels * PERCEPTION_FILTERS_PER_RING, 1, 3, 3),
        device,
    )
}

pub fn perceive_multiscale(x: &Tensor, kernel: &Tensor, channels: usize) -> Result<Tensor> {
    let near = pad_toroidal(x, 1)?.conv2d(kernel, 0, 1, 1, channels)?;
    let far = pad_toroidal(x, 2)?.conv2d(kernel, 0, 1, 2, channels)?;
    Tensor::cat(&[&near, &far], 1)
}

pub fn laplacian_kernel(channels: usize, device: &Device) -> Result<Tensor> {
    let base = [0.0f32, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let mut values = Vec::with_capacity(channels * 9);
    for _ in 0..channels {
        values.extend_from_slice(&base);
    }
    Tensor::from_vec(values, (channels, 1, 3, 3), device)
}

pub fn laplacian_with_kernel(x: &Tensor, kernel: &Tensor) -> Result<Tensor> {
    let channels = x.dim(1)?;
    pad_toroidal(x, 1)?.conv2d(kernel, 0, 1, 1, channels)
}

pub fn laplacian(x: &Tensor) -> Result<Tensor> {
    let kernel = laplacian_kernel(x.dim(1)?, x.device())?;
    laplacian_with_kernel(x, &kernel)
}

pub fn deterministic_clock_mask(
    h: usize,
    w: usize,
    seed: u64,
    step: u64,
    probability: f32,
    device: &Device,
) -> Result<Tensor> {
    let mut values = Vec::<f32>::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            let key = seed
                ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ (y as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
                ^ (x as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
            let value = splitmix64(key);
            let unit = (value >> 40) as f32 / (1u32 << 24) as f32;
            values.push(if unit < probability { 1.0 } else { 0.0 });
        }
    }
    Tensor::from_vec(values, (1, 1, h, w), device)
}

/// Global octave Fourier coordinates. v4's 64-cycle cell coordinates visibly
/// stamped the decoder lattice into output; v9 uses only canvas-scale bands.
pub fn coordinate_features(resolution: usize, bands: usize, device: &Device) -> Result<Tensor> {
    coordinate_features_window(resolution, bands, 0.0, 0.0, 1.0, device)
}

pub fn coordinate_features_window(
    resolution: usize,
    bands: usize,
    x0: f32,
    y0: f32,
    size: f32,
    device: &Device,
) -> Result<Tensor> {
    let channels = bands * 4;
    let plane = resolution * resolution;
    let mut values = vec![0.0f32; channels * plane];
    for y in 0..resolution {
        for x in 0..resolution {
            let index = y * resolution + x;
            let nx = x0 + size * (x as f32 + 0.5) / resolution as f32;
            let ny = y0 + size * (y as f32 + 0.5) / resolution as f32;
            for band in 0..bands {
                let frequency = (1usize << band) as f32;
                let phase_x = std::f32::consts::TAU * frequency * nx;
                let phase_y = std::f32::consts::TAU * frequency * ny;
                let offset = band * 4 * plane;
                values[offset + index] = phase_x.sin();
                values[offset + plane + index] = phase_x.cos();
                values[offset + 2 * plane + index] = phase_y.sin();
                values[offset + 3 * plane + index] = phase_y.cos();
            }
        }
    }
    Tensor::from_vec(values, (1, channels, resolution, resolution), device)
}

#[derive(Clone)]
pub struct PeriodicUpsampler {
    input_h: usize,
    input_w: usize,
    output_h: usize,
    output_w: usize,
    x0: Tensor,
    x1: Tensor,
    wx0: Tensor,
    wx1: Tensor,
    y0: Tensor,
    y1: Tensor,
    wy0: Tensor,
    wy1: Tensor,
}

impl PeriodicUpsampler {
    pub fn new(
        input_h: usize,
        input_w: usize,
        output_h: usize,
        output_w: usize,
        device: &Device,
    ) -> Result<Self> {
        let (x0, x1, wx0, wx1) = interpolation_axis(input_w, output_w);
        let (y0, y1, wy0, wy1) = interpolation_axis(input_h, output_h);
        Ok(Self {
            input_h,
            input_w,
            output_h,
            output_w,
            x0: Tensor::from_vec(x0, output_w, device)?,
            x1: Tensor::from_vec(x1, output_w, device)?,
            wx0: Tensor::from_vec(wx0, (1, 1, 1, output_w), device)?,
            wx1: Tensor::from_vec(wx1, (1, 1, 1, output_w), device)?,
            y0: Tensor::from_vec(y0, output_h, device)?,
            y1: Tensor::from_vec(y1, output_h, device)?,
            wy0: Tensor::from_vec(wy0, (1, 1, output_h, 1), device)?,
            wy1: Tensor::from_vec(wy1, (1, 1, output_h, 1), device)?,
        })
    }

    pub fn new_bounded(
        input_h: usize,
        input_w: usize,
        output_h: usize,
        output_w: usize,
        device: &Device,
    ) -> Result<Self> {
        let (x0, x1, wx0, wx1) = Self::interpolation_axis_bounded(input_w, output_w);
        let (y0, y1, wy0, wy1) = Self::interpolation_axis_bounded(input_h, output_h);
        Ok(Self {
            input_h,
            input_w,
            output_h,
            output_w,
            x0: Tensor::from_vec(x0, output_w, device)?,
            x1: Tensor::from_vec(x1, output_w, device)?,
            wx0: Tensor::from_vec(wx0, (1, 1, 1, output_w), device)?,
            wx1: Tensor::from_vec(wx1, (1, 1, 1, output_w), device)?,
            y0: Tensor::from_vec(y0, output_h, device)?,
            y1: Tensor::from_vec(y1, output_h, device)?,
            wy0: Tensor::from_vec(wy0, (1, 1, output_h, 1), device)?,
            wy1: Tensor::from_vec(wy1, (1, 1, output_h, 1), device)?,
        })
    }
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, input_h, input_w) = x.dims4()?;
        if input_h != self.input_h || input_w != self.input_w {
            return Err(candle_core::Error::Msg(format!(
                "periodic upsampler expected {}x{}, got {input_h}x{input_w}",
                self.input_h, self.input_w
            )));
        }
        if self.input_h == self.output_h && self.input_w == self.output_w {
            return Ok(x.clone());
        }
        let horizontal = x
            .index_select(&self.x0, 3)?
            .broadcast_mul(&self.wx0)?
            .add(&x.index_select(&self.x1, 3)?.broadcast_mul(&self.wx1)?)?;
        horizontal
            .index_select(&self.y0, 2)?
            .broadcast_mul(&self.wy0)?
            .add(
                &horizontal
                    .index_select(&self.y1, 2)?
                    .broadcast_mul(&self.wy1)?,
            )
    }

    fn interpolation_axis_bounded(
        input: usize,
        output: usize,
    ) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
        let mut lower = Vec::with_capacity(output);
        let mut upper = Vec::with_capacity(output);
        let mut lower_weight = Vec::with_capacity(output);
        let mut upper_weight = Vec::with_capacity(output);
        for position in 0..output {
            let source = (position as f32 + 0.5) * input as f32 / output as f32 - 0.5;
            let floor = source.floor();
            let base = (floor as isize).clamp(0, input.saturating_sub(1) as isize) as usize;
            let next = (base + 1).min(input - 1);
            let fraction = if source < 0.0 || source >= (input - 1) as f32 {
                0.0
            } else {
                source - floor
            };
            lower.push(base as u32);
            upper.push(next as u32);
            lower_weight.push(1.0 - fraction);
            upper_weight.push(fraction);
        }
        (lower, upper, lower_weight, upper_weight)
    }
}

pub fn periodic_bilinear_upsample(x: &Tensor, output_h: usize, output_w: usize) -> Result<Tensor> {
    let (_, _, input_h, input_w) = x.dims4()?;
    PeriodicUpsampler::new(input_h, input_w, output_h, output_w, x.device())?.apply(x)
}

fn interpolation_axis(input: usize, output: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut lower = Vec::with_capacity(output);
    let mut upper = Vec::with_capacity(output);
    let mut lower_weight = Vec::with_capacity(output);
    let mut upper_weight = Vec::with_capacity(output);
    for position in 0..output {
        let source = (position as f32 + 0.5) * input as f32 / output as f32 - 0.5;
        let base_signed = source.floor() as isize;
        let base = base_signed.rem_euclid(input as isize) as usize;
        let fraction = source - source.floor();
        lower.push(base as u32);
        upper.push(((base + 1) % input) as u32);
        lower_weight.push(1.0 - fraction);
        upper_weight.push(fraction);
    }
    (lower, upper, lower_weight, upper_weight)
}

pub fn broadcast_vector(vector: &Tensor, h: usize, w: usize) -> Result<Tensor> {
    let channels = vector.elem_count();
    vector
        .reshape((1, channels, 1, 1))?
        .broadcast_as((1, channels, h, w))
}

pub fn zeros(channels: usize, h: usize, w: usize, device: &Device) -> Result<Tensor> {
    Tensor::zeros((1, channels, h, w), DType::F32, device)
}

pub fn periodic_shift(x: &Tensor, dy: usize, dx: usize) -> Result<Tensor> {
    roll_axis(&roll_axis(x, 2, dy)?, 3, dx)
}

fn roll_axis(x: &Tensor, dimension: usize, shift: usize) -> Result<Tensor> {
    let size = x.dim(dimension)?;
    let shift = shift % size;
    if shift == 0 {
        return Ok(x.clone());
    }
    let tail = x.narrow(dimension, size - shift, shift)?;
    let head = x.narrow(dimension, 0, size - shift)?;
    Tensor::cat(&[&tail, &head], dimension)
}

pub fn mean_abs(x: &Tensor) -> Result<f32> {
    x.abs()?.mean_all()?.to_scalar::<f32>()
}

pub fn variance(x: &Tensor) -> Result<Tensor> {
    let mean = x.mean_all()?;
    x.broadcast_sub(&mean)?.sqr()?.mean_all()
}

/// Smooth odd projection with asymptotes at +/- limit. Unlike a hard clamp,
/// its derivative remains nonzero for every finite input, while the quartic
/// shoulder stays close to identity through the useful center of the state.
pub fn smooth_limit(value: &Tensor, limit: f32) -> Result<Tensor> {
    let scaled = value.affine(1.0 / limit as f64, 0.0)?;
    let denominator = scaled.sqr()?.sqr()?.affine(1.0, 1.0)?.powf(0.25)?;
    value.broadcast_div(&denominator)
}

pub fn channel_mean(x: &Tensor) -> Result<Tensor> {
    let (_, c, h, w) = x.dims4()?;
    x.reshape((1, c, h * w))?.mean(D::Minus1)
}

pub fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{Init, VarBuilder, VarMap};

    #[test]
    fn clock_is_repeatable_and_nontrivial() -> Result<()> {
        let a = deterministic_clock_mask(8, 8, 7, 11, 0.7, &Device::Cpu)?;
        let b = deterministic_clock_mask(8, 8, 7, 11, 0.7, &Device::Cpu)?;
        assert_eq!(
            a.flatten_all()?.to_vec1::<f32>()?,
            b.flatten_all()?.to_vec1::<f32>()?
        );
        let sum = a.sum_all()?.to_scalar::<f32>()?;
        assert!(sum > 0.0 && sum < 64.0);
        Ok(())
    }

    #[test]
    fn toroidal_laplacian_annihilates_constants() -> Result<()> {
        let x = Tensor::ones((1, 3, 9, 9), DType::F32, &Device::Cpu)?;
        assert!(mean_abs(&laplacian(&x)?)? < 1e-6);
        Ok(())
    }

    #[test]
    fn periodic_bilinear_preserves_constants() -> Result<()> {
        let x = Tensor::ones((1, 2, 3, 5), DType::F32, &Device::Cpu)?;
        let plan = PeriodicUpsampler::new(3, 5, 11, 13, &Device::Cpu)?;
        let y = plan.apply(&x)?;
        assert_eq!(y.dims4()?, (1, 2, 11, 13));
        assert!(mean_abs(&y.affine(1.0, -1.0)?)? < 1e-6);
        Ok(())
    }

    #[test]
    fn equal_resolution_is_identity() -> Result<()> {
        let values: Vec<f32> = (0..35).map(|value| value as f32 / 35.0).collect();
        let x = Tensor::from_vec(values, (1, 1, 5, 7), &Device::Cpu)?;
        let y = PeriodicUpsampler::new(5, 7, 5, 7, &Device::Cpu)?.apply(&x)?;
        assert!(mean_abs(&x.sub(&y)?)? < 1e-7);
        Ok(())
    }

    #[test]
    fn periodic_shift_wraps() -> Result<()> {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 1, 4), &Device::Cpu)?;
        let y = periodic_shift(&x, 0, 1)?;
        assert_eq!(y.flatten_all()?.to_vec1::<f32>()?, vec![4., 1., 2., 3.]);
        Ok(())
    }

    #[test]
    fn smooth_limit_is_bounded_odd_and_near_identity_at_the_origin() -> Result<()> {
        let values = Tensor::new(&[-100.0f32, -0.1, 0.0, 0.1, 100.0], &Device::Cpu)?;
        let limited = smooth_limit(&values, 3.5)?.to_vec1::<f32>()?;
        assert!(limited.iter().all(|value| value.abs() < 3.5));
        assert!((limited[0] + limited[4]).abs() < 1e-5);
        assert!((limited[1] + 0.1).abs() < 1e-5);
        assert_eq!(limited[2], 0.0);
        assert!((limited[3] - 0.1).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn smooth_limit_retains_a_finite_gradient_outside_the_shoulder() -> Result<()> {
        let variables = VarMap::new();
        let builder = VarBuilder::from_varmap(&variables, DType::F32, &Device::Cpu);
        let value = builder.get_with_hints(1, "probe", Init::Const(10.0))?;
        let output = smooth_limit(&value, 3.5)?.sum_all()?;
        let gradients = output.backward()?;
        let gradient = gradients
            .get(&value)
            .expect("smooth limit input gradient")
            .to_vec1::<f32>()?[0];
        assert!(gradient.is_finite());
        assert!(gradient > 0.0);
        Ok(())
    }
}
