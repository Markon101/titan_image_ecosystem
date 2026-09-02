use crate::config::{ComputeBackend, RunConfig};
use crate::tensor_ops::{
    broadcast_vector, deterministic_clock_mask, linear_mode, perceive_multiscale,
    perception_kernel, splitmix64, PERCEPTION_FEATURES_PER_CHANNEL,
};
use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Init, Linear, VarBuilder};
#[cfg(feature = "opencl")]
use std::sync::Mutex;

const CLOCK_MASKS: usize = 8;

#[cfg(feature = "opencl")]
struct OpenClSlot {
    attempted: bool,
    backend: Option<crate::opencl::OpenClNca>,
    failure: Option<String>,
}

pub struct NeuralCa {
    perception: Tensor,
    input: Linear,
    hidden: Linear,
    output: Linear,
    clock_masks: Vec<Tensor>,
    clock_salt: u64,
    channels: usize,
    size: usize,
    gain: f32,
    compute_backend: ComputeBackend,
    #[cfg(feature = "opencl")]
    opencl: Mutex<OpenClSlot>,
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
            size,
            gain: config.nca_gain,
            compute_backend: config.compute_backend,
            #[cfg(feature = "opencl")]
            opencl: Mutex::new(OpenClSlot {
                attempted: false,
                backend: None,
                failure: None,
            }),
        })
    }

    pub fn prepare_inference_backend(&self) -> Result<bool> {
        if self.compute_backend == ComputeBackend::Cpu {
            return Ok(false);
        }
        #[cfg(not(feature = "opencl"))]
        {
            match self.compute_backend {
                ComputeBackend::OpenCl => {
                    bail!("OpenCL NCA requires cargo build --features opencl")
                }
                ComputeBackend::Auto | ComputeBackend::Cpu => Ok(false),
            }
        }
        #[cfg(feature = "opencl")]
        {
            let mut slot = self.opencl.lock().expect("OpenCL NCA mutex poisoned");
            self.initialize_opencl(&mut slot)?;
            Ok(slot.backend.is_some())
        }
    }

    pub fn refresh_opencl_weights(&self) -> Result<bool> {
        if self.compute_backend != ComputeBackend::OpenCl {
            return Ok(false);
        }
        #[cfg(not(feature = "opencl"))]
        {
            bail!("OpenCL NCA requires cargo build --features opencl");
        }
        #[cfg(feature = "opencl")]
        {
            let layers = vec![
                linear_data(&self.input)?,
                linear_data(&self.hidden)?,
                linear_data(&self.output)?,
            ];
            let mut slot = self.opencl.lock().expect("OpenCL NCA mutex poisoned");
            self.initialize_opencl(&mut slot)?;
            let backend = slot
                .backend
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("OpenCL NCA did not initialize"))?;
            backend.refresh_weights(&layers)?;
            Ok(true)
        }
    }

    pub fn delta(
        &self,
        field: &Tensor,
        macro_context: &Tensor,
        genome: &Tensor,
        seed: u64,
        step: u64,
        tracked: bool,
    ) -> Result<Tensor> {
        let clock = splitmix64(seed ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ self.clock_salt)
            as usize
            % self.clock_masks.len();
        if !tracked {
            if let Some(delta) = self.opencl_delta(field, macro_context, genome, clock)? {
                return Ok(delta);
            }
        }
        let genome_field = broadcast_vector(genome, self.size, self.size)?;
        let perceived = perceive_multiscale(field, &self.perception, self.channels)?;
        let features = Tensor::cat(&[&perceived, macro_context, &genome_field], 1)?;
        let (batch, feature_count, height, width) = features.dims4()?;
        let pixels = height * width;
        let features = features
            .reshape((batch, feature_count, pixels))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch * pixels, feature_count))?;
        let h1 = swish(&linear_mode(&features, &self.input, tracked)?)?;
        let residual = swish(&linear_mode(&h1, &self.hidden, tracked)?)?;
        let hidden = h1.add(&residual.affine(0.5, 0.0)?)?;
        let raw = linear_mode(&hidden, &self.output, tracked)?
            .tanh()?
            .reshape((batch, pixels, self.channels))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, self.channels, height, width))?;
        Ok(raw
            .broadcast_mul(&self.clock_masks[clock])?
            .affine(self.gain as f64, 0.0)?)
    }

    fn opencl_delta(
        &self,
        field: &Tensor,
        macro_context: &Tensor,
        genome: &Tensor,
        clock: usize,
    ) -> Result<Option<Tensor>> {
        if self.compute_backend == ComputeBackend::Cpu {
            return Ok(None);
        }
        #[cfg(not(feature = "opencl"))]
        {
            let _ = (field, macro_context, genome, clock);
            match self.compute_backend {
                ComputeBackend::OpenCl => {
                    bail!("OpenCL NCA requires cargo build --features opencl")
                }
                ComputeBackend::Auto | ComputeBackend::Cpu => Ok(None),
            }
        }
        #[cfg(feature = "opencl")]
        {
            let mut slot = self.opencl.lock().expect("OpenCL NCA mutex poisoned");
            self.initialize_opencl(&mut slot)?;
            let Some(backend) = slot.backend.as_mut() else {
                return Ok(None);
            };
            let field_values = field.flatten_all()?.to_vec1::<f32>()?;
            let context_values = macro_context.flatten_all()?.to_vec1::<f32>()?;
            let genome_values = genome.flatten_all()?.to_vec1::<f32>()?;
            let output = backend.run(&field_values, &context_values, &genome_values, clock)?;
            Ok(Some(Tensor::from_vec(
                output,
                (1, self.channels, self.size, self.size),
                field.device(),
            )?))
        }
    }

    #[cfg(feature = "opencl")]
    fn initialize_opencl(&self, slot: &mut OpenClSlot) -> Result<()> {
        if slot.attempted {
            if self.compute_backend == ComputeBackend::OpenCl {
                if let Some(message) = &slot.failure {
                    bail!("OpenCL NCA initialization failed: {message}");
                }
            }
            return Ok(());
        }
        slot.attempted = true;
        let result = self.opencl_payload().and_then(|(layers, masks)| {
            crate::opencl::OpenClNca::new(&layers, self.size, self.channels, self.gain, &masks)
        });
        match result {
            Ok(backend) => slot.backend = Some(backend),
            Err(error) if self.compute_backend == ComputeBackend::Auto => {
                let message = format!("{error:#}");
                eprintln!("OPENCL NCA fallback ({}px): {message}", self.size);
                slot.failure = Some(message);
            }
            Err(error) => {
                slot.failure = Some(format!("{error:#}"));
                return Err(error);
            }
        }
        Ok(())
    }

    #[cfg(feature = "opencl")]
    fn opencl_payload(&self) -> Result<(Vec<crate::opencl::LinearData>, Vec<Vec<f32>>)> {
        let layers = vec![
            linear_data(&self.input)?,
            linear_data(&self.hidden)?,
            linear_data(&self.output)?,
        ];
        let masks = self
            .clock_masks
            .iter()
            .map(|mask| Ok(mask.flatten_all()?.to_vec1::<f32>()?))
            .collect::<Result<Vec<_>>>()?;
        Ok((layers, masks))
    }
}

#[cfg(feature = "opencl")]
fn linear_data(linear: &Linear) -> Result<crate::opencl::LinearData> {
    let (output, input) = linear.weight().dims2()?;
    let weight = linear.weight().flatten_all()?.to_vec1::<f32>()?;
    let bias = linear
        .bias()
        .ok_or_else(|| anyhow::anyhow!("OpenCL NCA requires linear biases"))?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(crate::opencl::LinearData {
        input,
        output,
        weight,
        bias,
    })
}

fn swish(x: &Tensor) -> candle_core::Result<Tensor> {
    x.mul(&candle_nn::ops::sigmoid(x)?)
}

#[cfg(all(test, feature = "opencl"))]
mod tests {
    use super::*;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn opencl_nca_delta_matches_cpu() -> Result<()> {
        if std::env::var_os("TITAN_OPENCL_TEST").is_none() {
            return Ok(());
        }
        let cpu_config = RunConfig {
            micro_size: 16,
            macro_size: 8,
            channels: 12,
            genome_dim: 4,
            ca_hidden: 32,
            compute_backend: ComputeBackend::Cpu,
            ..RunConfig::default()
        };
        let vars = VarMap::new();
        let cpu = NeuralCa::new(
            &cpu_config,
            16,
            0xc10c_0001,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("nca"),
            &Device::Cpu,
        )?;
        let mut gpu_config = cpu_config.clone();
        gpu_config.compute_backend = ComputeBackend::OpenCl;
        let gpu = NeuralCa::new(
            &gpu_config,
            16,
            0xc10c_0001,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("nca"),
            &Device::Cpu,
        )?;
        {
            let data = vars.data().lock().expect("VarMap mutex poisoned");
            for (name, variable) in data.iter() {
                let values = (0..variable.elem_count())
                    .map(|index| ((index as f32 * 0.011 + name.len() as f32).sin()) * 0.035)
                    .collect::<Vec<_>>();
                variable.set(&Tensor::from_vec(
                    values,
                    variable.shape().clone(),
                    &Device::Cpu,
                )?)?;
            }
        }
        let field = Tensor::from_vec(
            (0..12 * 16 * 16)
                .map(|index| (index as f32 * 0.019).sin() * 0.4)
                .collect::<Vec<_>>(),
            (1, 12, 16, 16),
            &Device::Cpu,
        )?;
        let context = Tensor::from_vec(
            (0..12 * 16 * 16)
                .map(|index| (index as f32 * 0.023).cos() * 0.2)
                .collect::<Vec<_>>(),
            (1, 12, 16, 16),
            &Device::Cpu,
        )?;
        let genome = Tensor::new(&[0.1f32, -0.2, 0.3, -0.4], &Device::Cpu)?;
        let cpu = cpu.delta(&field, &context, &genome, 42, 17, false)?;
        let gpu = gpu.delta(&field, &context, &genome, 42, 17, false)?;
        let difference = gpu.sub(&cpu)?.abs()?;
        let max_abs = difference.max_all()?.to_scalar::<f32>()?;
        let mean_abs = difference.mean_all()?.to_scalar::<f32>()?;
        let rms = difference.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
        eprintln!(
            "OPENCL NCA delta parity | max_abs={max_abs:.8} mean_abs={mean_abs:.8} rms={rms:.8}"
        );
        assert!(max_abs <= 2e-4, "OpenCL NCA max abs drift {max_abs}");
        assert!(mean_abs <= 2e-5, "OpenCL NCA mean abs drift {mean_abs}");
        Ok(())
    }
}
