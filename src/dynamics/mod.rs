mod nca;
mod operators;

use crate::config::{Integrator, RunConfig};
use crate::interface::{InterfaceOutput, RecurrentInterface};
use crate::state::WorldState;
use crate::tensor_ops::{mean_abs, pixelwise_linear_mode, smooth_limit, zeros, PeriodicUpsampler};
use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Linear, VarBuilder};
use nca::NeuralCa;
use operators::{PhysicalOperators, ScaleFields};

pub struct DynamicsSystem {
    micro_ca: NeuralCa,
    macro_ca: NeuralCa,
    interface: RecurrentInterface,
    reference_micro_drive: Linear,
    reference_macro_drive: Linear,
    physical: PhysicalOperators,
    macro_to_micro: PeriodicUpsampler,
    macro_zero_context: Tensor,
    reference_micro_zero: Tensor,
    reference_macro_zero: Tensor,
    config: RunConfig,
    seed: u64,
}

pub struct StepOutput {
    pub world: WorldState,
    pub micro_movement: f32,
    pub macro_movement: f32,
    pub micro_reference_drive_rms: f32,
    pub macro_reference_drive_rms: f32,
    pub macro_updated: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DynamicsAblation {
    pub disable_interface: bool,
    pub freeze_micro: bool,
    pub freeze_macro: bool,
    pub disable_nca: bool,
    pub disable_reaction: bool,
    pub disable_phase: bool,
    pub disable_cyclic: bool,
    pub disable_forcing: bool,
}

#[derive(Clone, Copy)]
enum FieldScale {
    Micro,
    Macro,
}

impl DynamicsSystem {
    pub fn new(config: &RunConfig, vb: VarBuilder<'_>, device: &Device) -> Result<Self> {
        Ok(Self {
            micro_ca: NeuralCa::new(
                config,
                config.micro_size,
                0xc10c_0001,
                vb.pp("micro_ca"),
                device,
            )?,
            macro_ca: NeuralCa::new(
                config,
                config.macro_size,
                0xc10c_0002,
                vb.pp("macro_ca"),
                device,
            )?,
            reference_micro_drive: candle_nn::linear_no_bias(
                3,
                config.channels,
                vb.pp("reference_micro_drive"),
            )?,
            reference_macro_drive: candle_nn::linear_no_bias(
                3,
                config.channels,
                vb.pp("reference_macro_drive"),
            )?,
            interface: RecurrentInterface::new(config, vb.pp("interface"), device)?,
            physical: PhysicalOperators::new(config, device)?,
            macro_to_micro: PeriodicUpsampler::new(
                config.macro_size,
                config.macro_size,
                config.micro_size,
                config.micro_size,
                device,
            )?,
            macro_zero_context: zeros(
                config.channels,
                config.macro_size,
                config.macro_size,
                device,
            )?,
            reference_micro_zero: zeros(3, config.micro_size, config.micro_size, device)?,
            reference_macro_zero: zeros(3, config.macro_size, config.macro_size, device)?,
            config: config.clone(),
            seed: config.seed,
        })
    }

    pub fn prepare_inference_backend(&self) -> Result<Option<String>> {
        if self.config.compute_backend == crate::config::ComputeBackend::Cpu {
            return Ok(None);
        }
        let micro = self.micro_ca.prepare_inference_backend()?;
        let macro_field = self.macro_ca.prepare_inference_backend()?;
        if micro && macro_field {
            Ok(Some(format!(
                "NCA FP32 {}px + {}px",
                self.config.micro_size, self.config.macro_size
            )))
        } else {
            Ok(None)
        }
    }

    pub fn refresh_opencl_weights(&self) -> Result<bool> {
        let micro = self.micro_ca.refresh_opencl_weights()?;
        let macro_field = self.macro_ca.refresh_opencl_weights()?;
        Ok(micro && macro_field)
    }

    /// Advance one world step. The local NCA handles dense spatial refinement;
    /// a small recurrent token interface performs global read/reason/write.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &self,
        world: &WorldState,
        genome: &Tensor,
        reference_micro: Option<&Tensor>,
        reference_macro: Option<&Tensor>,
        reference_fidelity: f32,
        tracked: bool,
    ) -> Result<StepOutput> {
        self.step_with_local_reference(
            world,
            genome,
            reference_micro,
            reference_macro,
            reference_micro,
            reference_macro,
            reference_fidelity,
            tracked,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_with_local_reference(
        &self,
        world: &WorldState,
        genome: &Tensor,
        reference_micro: Option<&Tensor>,
        reference_macro: Option<&Tensor>,
        local_reference_micro: Option<&Tensor>,
        local_reference_macro: Option<&Tensor>,
        reference_fidelity: f32,
        tracked: bool,
    ) -> Result<StepOutput> {
        self.step_ablated(
            world,
            genome,
            reference_micro,
            reference_macro,
            local_reference_micro,
            local_reference_macro,
            reference_fidelity,
            tracked,
            &DynamicsAblation::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn step_ablated(
        &self,
        world: &WorldState,
        genome: &Tensor,
        reference_micro: Option<&Tensor>,
        reference_macro: Option<&Tensor>,
        local_reference_micro: Option<&Tensor>,
        local_reference_macro: Option<&Tensor>,
        reference_fidelity: f32,
        tracked: bool,
        ablation: &DynamicsAblation,
    ) -> Result<StepOutput> {
        let reference_micro = reference_micro.unwrap_or(&self.reference_micro_zero);
        let reference_macro = reference_macro.unwrap_or(&self.reference_macro_zero);
        let local_reference_micro = local_reference_micro.unwrap_or(reference_micro);
        let local_reference_macro = local_reference_macro.unwrap_or(reference_macro);
        let reference_gain = self.config.reconstruction.local_reference_gain * reference_fidelity;
        let micro_reference_drive =
            pixelwise_linear_mode(local_reference_micro, &self.reference_micro_drive, tracked)?
                .tanh()?
                .affine(reference_gain as f64, 0.0)?;
        let macro_reference_drive =
            pixelwise_linear_mode(local_reference_macro, &self.reference_macro_drive, tracked)?
                .tanh()?
                .affine(reference_gain as f64, 0.0)?;
        let micro_reference_drive_rms = micro_reference_drive
            .sqr()?
            .mean_all()?
            .sqrt()?
            .to_scalar::<f32>()?;
        let macro_reference_drive_rms = macro_reference_drive
            .sqr()?
            .mean_all()?
            .sqrt()?
            .to_scalar::<f32>()?;
        let age_phase =
            (world.age as f32 / self.config.episode_steps.max(1) as f32).clamp(0.0, 1.0);
        let interface = if ablation.disable_interface {
            InterfaceOutput {
                micro_bias: zeros(
                    self.config.channels,
                    self.config.micro_size,
                    self.config.micro_size,
                    world.micro.device(),
                )?,
                macro_bias: zeros(
                    self.config.channels,
                    self.config.macro_size,
                    self.config.macro_size,
                    world.micro.device(),
                )?,
                memory: world.memory.clone(),
            }
        } else {
            self.interface.forward(
                &world.micro,
                &world.macro_field,
                reference_micro,
                reference_macro,
                genome,
                &world.memory,
                reference_fidelity,
                age_phase,
                tracked,
                world.morph_active_depth,
            )?
        };

        let macro_updated = !ablation.freeze_macro
            && world
                .age
                .is_multiple_of(self.config.macro_update_every as u64);
        let next_macro = if macro_updated {
            self.integrate(
                &world.macro_field,
                &self.macro_zero_context,
                genome,
                &interface.macro_bias.add(&macro_reference_drive)?,
                world.step,
                FieldScale::Macro,
                tracked,
                ablation,
            )?
        } else {
            world.macro_field.clone()
        };
        let macro_context = self.macro_to_micro.apply(&next_macro)?;
        let next_micro = if ablation.freeze_micro {
            world.micro.clone()
        } else {
            self.integrate(
                &world.micro,
                &macro_context,
                genome,
                &interface.micro_bias.add(&micro_reference_drive)?,
                world.step,
                FieldScale::Micro,
                tracked,
                ablation,
            )?
        };
        let micro_movement = mean_abs(&next_micro.sub(&world.micro)?)?;
        let macro_movement = if macro_updated {
            mean_abs(&next_macro.sub(&world.macro_field)?)?
        } else {
            0.0
        };
        for (name, value) in [
            ("micro movement", micro_movement),
            ("macro movement", macro_movement),
            ("micro reference drive RMS", micro_reference_drive_rms),
            ("macro reference drive RMS", macro_reference_drive_rms),
        ] {
            if !value.is_finite() {
                bail!("non-finite {name}; recurrent step was rejected");
            }
        }
        Ok(StepOutput {
            world: WorldState {
                micro: next_micro,
                macro_field: next_macro,
                memory: interface.memory,
                step: world.step + 1,
                age: world.age + 1,
                episode: world.episode,
                target_index: world.target_index,
                morph_active_depth: world.morph_active_depth,
                morph_generation: world.morph_generation,
                morph_birth_generations: world.morph_birth_generations.clone(),
            },
            micro_movement,
            macro_movement,
            macro_updated,
            micro_reference_drive_rms,
            macro_reference_drive_rms,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn integrate(
        &self,
        field: &Tensor,
        macro_context: &Tensor,
        genome: &Tensor,
        interface_bias: &Tensor,
        step: u64,
        scale: FieldScale,
        tracked: bool,
        ablation: &DynamicsAblation,
    ) -> Result<Tensor> {
        let (ca, scale_fields): (&NeuralCa, &ScaleFields) = match scale {
            FieldScale::Micro => (&self.micro_ca, &self.physical.micro_fields),
            FieldScale::Macro => (&self.macro_ca, &self.physical.macro_fields),
        };
        let derivative = match self.config.integrator {
            Integrator::Euler => self.derivative(
                field,
                macro_context,
                genome,
                interface_bias,
                step,
                ca,
                scale_fields,
                tracked,
                ablation,
            )?,
            Integrator::Midpoint => {
                let k1 = self.derivative(
                    field,
                    macro_context,
                    genome,
                    interface_bias,
                    step,
                    ca,
                    scale_fields,
                    tracked,
                    ablation,
                )?;
                let midpoint = field.add(&k1.affine((0.5 * self.config.dt) as f64, 0.0)?)?;
                self.derivative(
                    &midpoint,
                    macro_context,
                    genome,
                    interface_bias,
                    step,
                    ca,
                    scale_fields,
                    tracked,
                    ablation,
                )?
            }
        };
        let proposed = field.add(&derivative.affine(self.config.dt as f64, 0.0)?)?;
        smooth_limit(&proposed, self.config.state_limit).map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    fn derivative(
        &self,
        field: &Tensor,
        macro_context: &Tensor,
        genome: &Tensor,
        interface_bias: &Tensor,
        step: u64,
        ca: &NeuralCa,
        scale_fields: &ScaleFields,
        tracked: bool,
        ablation: &DynamicsAblation,
    ) -> Result<Tensor> {
        let (_, channels, height, width) = field.dims4()?;
        let local = if ablation.disable_nca {
            zeros(channels, height, width, field.device())?
        } else {
            ca.delta(field, macro_context, genome, self.seed, step, tracked)?
        };
        let mut delta = local.add(interface_bias)?;
        if self.config.state_leak > 0.0 {
            delta = delta.add(&field.affine(-(self.config.state_leak as f64), 0.0)?)?;
        }
        if self.config.reaction_gain > 0.0 && !ablation.disable_reaction {
            delta = delta.add(&self.physical.reaction_diffusion(field)?)?;
        }
        if self.config.phase_gain > 0.0 && !ablation.disable_phase {
            delta = delta.add(&self.physical.complex_phase(field)?)?;
        }
        if self.config.cyclic_gain > 0.0 && !ablation.disable_cyclic {
            delta = delta.add(&self.physical.cyclic_chemistry(field)?)?;
        }
        if (self.config.fractal_gain > 0.0 || self.config.quasiperiodic_gain > 0.0)
            && !ablation.disable_forcing
        {
            delta = delta.add(&self.physical.forcing(field, scale_fields)?)?;
        }
        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "opencl")]
    use anyhow::Context;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};

    #[test]
    fn multirate_step_preserves_runtime_shapes_and_finiteness() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig {
            micro_size: 32,
            macro_size: 16,
            channels: 12,
            genome_dim: 4,
            ca_hidden: 32,
            interface_grid: 4,
            interface_width: 32,
            interface_loops: 2,
            morph_layers: 3,
            morph_depth: 2,
            train_resolution: 64,
            output_resolution: 64,
            ..RunConfig::default()
        };
        let vars = VarMap::new();
        let system = DynamicsSystem::new(
            &config,
            VarBuilder::from_varmap(&vars, DType::F32, &device),
            &device,
        )?;
        let parameter_names: Vec<String> = vars
            .data()
            .lock()
            .expect("VarMap mutex poisoned")
            .keys()
            .cloned()
            .collect();
        assert!(!parameter_names
            .iter()
            .any(|name| name.contains("reference_micro_drive.bias")));
        assert!(!parameter_names
            .iter()
            .any(|name| name.contains("reference_macro_drive.bias")));
        let world = WorldState::fresh(&config, 42, &device)?;
        let genome = Tensor::new(&[0.0f32, 0.25, -0.5, 1.0], &device)?;
        let reference_micro = Tensor::zeros((1, 3, 32, 32), DType::F32, &device)?;
        let reference_macro = Tensor::zeros((1, 3, 16, 16), DType::F32, &device)?;
        let output = system.step(
            &world,
            &genome,
            Some(&reference_micro),
            Some(&reference_macro),
            0.5,
            true,
        )?;
        assert_eq!(output.world.micro.dims4()?, (1, 12, 32, 32));
        assert_eq!(output.world.macro_field.dims4()?, (1, 12, 16, 16));
        assert_eq!(output.world.memory.dims2()?, (1, 32));
        assert!(output.micro_movement.is_finite());
        assert!(output.macro_movement.is_finite());
        Ok(())
    }

    fn reference_test_config() -> RunConfig {
        RunConfig {
            micro_size: 24,
            macro_size: 12,
            channels: 12,
            genome_dim: 4,
            ca_hidden: 32,
            interface_grid: 3,
            interface_width: 32,
            interface_loops: 1,
            morph_layers: 2,
            morph_depth: 1,
            morph_growth: crate::config::MorphGrowthConfig {
                min_depth: 1,
                max_depth: 2,
                ..RunConfig::default().morph_growth
            },
            reaction_gain: 0.0,
            phase_gain: 0.0,
            fractal_gain: 0.0,
            quasiperiodic_gain: 0.0,
            cyclic_gain: 0.0,
            train_resolution: 24,
            output_resolution: 24,
            ..RunConfig::default()
        }
    }

    #[test]
    fn local_reference_fidelity_zero_has_exactly_no_influence() -> Result<()> {
        let device = Device::Cpu;
        let config = reference_test_config();
        let vars = VarMap::new();
        let system = DynamicsSystem::new(
            &config,
            VarBuilder::from_varmap(&vars, DType::F32, &device),
            &device,
        )?;
        let world = WorldState::fresh(&config, 42, &device)?;
        let genome = Tensor::zeros(4, DType::F32, &device)?;
        let micro_zero = Tensor::zeros((1, 3, 24, 24), DType::F32, &device)?;
        let macro_zero = Tensor::zeros((1, 3, 12, 12), DType::F32, &device)?;
        let micro_one = Tensor::ones((1, 3, 24, 24), DType::F32, &device)?;
        let macro_one = Tensor::ones((1, 3, 12, 12), DType::F32, &device)?;
        let zero = system.step(
            &world,
            &genome,
            Some(&micro_zero),
            Some(&macro_zero),
            0.0,
            false,
        )?;
        let one = system.step(
            &world,
            &genome,
            Some(&micro_one),
            Some(&macro_one),
            0.0,
            false,
        )?;
        assert_eq!(
            zero.world.micro.flatten_all()?.to_vec1::<f32>()?,
            one.world.micro.flatten_all()?.to_vec1::<f32>()?
        );
        assert_eq!(one.micro_reference_drive_rms, 0.0);
        assert_eq!(one.macro_reference_drive_rms, 0.0);
        Ok(())
    }

    #[test]
    fn local_reference_fidelity_one_is_deterministic_and_observable() -> Result<()> {
        let device = Device::Cpu;
        let config = reference_test_config();
        let vars = VarMap::new();
        let system = DynamicsSystem::new(
            &config,
            VarBuilder::from_varmap(&vars, DType::F32, &device),
            &device,
        )?;
        let world = WorldState::fresh(&config, 42, &device)?;
        let genome = Tensor::zeros(4, DType::F32, &device)?;
        let reference_micro = Tensor::ones((1, 3, 24, 24), DType::F32, &device)?;
        let reference_macro = Tensor::ones((1, 3, 12, 12), DType::F32, &device)?;
        let first = system.step(
            &world,
            &genome,
            Some(&reference_micro),
            Some(&reference_macro),
            1.0,
            false,
        )?;
        let second = system.step(
            &world,
            &genome,
            Some(&reference_micro),
            Some(&reference_macro),
            1.0,
            false,
        )?;
        assert!(first.micro_reference_drive_rms > 0.0);
        assert!(first.macro_reference_drive_rms > 0.0);
        assert_eq!(
            first.world.micro.flatten_all()?.to_vec1::<f32>()?,
            second.world.micro.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_pure_nca_rollout_matches_cpu() -> Result<()> {
        if std::env::var_os("TITAN_OPENCL_TEST").is_none() {
            return Ok(());
        }
        let mut cpu_config = reference_test_config();
        cpu_config.compute_backend = crate::config::ComputeBackend::Cpu;
        cpu_config.macro_update_every = 2;
        cpu_config.episode_steps = 64;
        let vars = VarMap::new();
        let cpu = DynamicsSystem::new(
            &cpu_config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("dynamics"),
            &Device::Cpu,
        )?;
        let mut gpu_config = cpu_config.clone();
        gpu_config.compute_backend = crate::config::ComputeBackend::OpenCl;
        let gpu = DynamicsSystem::new(
            &gpu_config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("dynamics"),
            &Device::Cpu,
        )?;
        {
            let data = vars.data().lock().expect("VarMap mutex poisoned");
            for (name, variable) in data.iter() {
                let values = (0..variable.elem_count())
                    .map(|index| {
                        let wave = (index as f32 * 0.009 + name.len() as f32).sin();
                        if name.contains("norm.weight") {
                            1.0 + wave * 0.01
                        } else if name.ends_with(".weight") {
                            wave * 0.025
                        } else {
                            wave * 0.005
                        }
                    })
                    .collect::<Vec<_>>();
                variable.set(&Tensor::from_vec(
                    values,
                    variable.shape().clone(),
                    &Device::Cpu,
                )?)?;
            }
        }
        let initial = WorldState::fresh(&cpu_config, 0x5151, &Device::Cpu)?;
        let genome = Tensor::new(&[0.2f32, -0.1, 0.35, -0.25], &Device::Cpu)?;
        let reference_micro = Tensor::from_vec(
            (0..3 * 24 * 24)
                .map(|index| (index as f32 * 0.007).sin() * 0.4)
                .collect::<Vec<_>>(),
            (1, 3, 24, 24),
            &Device::Cpu,
        )?;
        let reference_macro = Tensor::from_vec(
            (0..3 * 12 * 12)
                .map(|index| (index as f32 * 0.013).cos() * 0.3)
                .collect::<Vec<_>>(),
            (1, 3, 12, 12),
            &Device::Cpu,
        )?;
        let mut cpu_world = initial.clone();
        let mut gpu_world = initial;
        let cpu_started = std::time::Instant::now();
        let mut cpu_worlds = Vec::new();
        for age in 1..=32 {
            cpu_world = cpu
                .step(
                    &cpu_world,
                    &genome,
                    Some(&reference_micro),
                    Some(&reference_macro),
                    0.65,
                    false,
                )?
                .world;
            if [1, 8, 32].contains(&age) {
                cpu_worlds.push(cpu_world.clone());
            }
        }
        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
        let gpu_started = std::time::Instant::now();
        let mut comparison_index = 0;
        for age in 1..=32 {
            gpu_world = gpu
                .step(
                    &gpu_world,
                    &genome,
                    Some(&reference_micro),
                    Some(&reference_macro),
                    0.65,
                    false,
                )?
                .world;
            if ![1, 8, 32].contains(&age) {
                continue;
            }
            let cpu_world = &cpu_worlds[comparison_index];
            comparison_index += 1;
            let micro = gpu_world.micro.sub(&cpu_world.micro)?.abs()?;
            let macro_field = gpu_world.macro_field.sub(&cpu_world.macro_field)?.abs()?;
            let memory = gpu_world.memory.sub(&cpu_world.memory)?.abs()?;
            let max_abs = micro
                .max_all()?
                .to_scalar::<f32>()?
                .max(macro_field.max_all()?.to_scalar::<f32>()?)
                .max(memory.max_all()?.to_scalar::<f32>()?);
            let mean_abs = (micro.mean_all()?.to_scalar::<f32>()?
                + macro_field.mean_all()?.to_scalar::<f32>()?
                + memory.mean_all()?.to_scalar::<f32>()?)
                / 3.0;
            eprintln!(
                "OPENCL dynamics parity | age={age} max_abs={max_abs:.8} mean_abs={mean_abs:.8}"
            );
            let max_tolerance = if age == 32 { 2e-3 } else { 5e-4 };
            assert!(max_abs <= max_tolerance, "age {age} drift {max_abs}");
            assert!(mean_abs <= 2e-4, "age {age} mean drift {mean_abs}");
        }
        let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "OPENCL dynamics benchmark small 32-step | cpu_ms={cpu_ms:.3} gpu_ms={gpu_ms:.3} speedup={:.3}x",
            cpu_ms / gpu_ms
        );
        Ok(())
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn opencl_checkpoint_pure_nca_rollout_and_render() -> Result<()> {
        let Some(metadata_path) = std::env::var_os("TITAN_OPENCL_DYNAMICS_METADATA") else {
            return Ok(());
        };
        let model_path = std::env::var_os("TITAN_OPENCL_DYNAMICS_MODEL")
            .context("TITAN_OPENCL_DYNAMICS_MODEL is required")?;
        let world_path = std::env::var_os("TITAN_OPENCL_DYNAMICS_WORLD")
            .context("TITAN_OPENCL_DYNAMICS_WORLD is required")?;
        let metadata: serde_json::Value = serde_json::from_slice(&std::fs::read(metadata_path)?)?;
        let mut cpu_config: RunConfig = serde_json::from_value(metadata["config"].clone())?;
        cpu_config.compute_backend = crate::config::ComputeBackend::Cpu;
        cpu_config.analysis.emergence_gallery = false;
        let vars = VarMap::new();
        let cpu = DynamicsSystem::new(
            &cpu_config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("dynamics"),
            &Device::Cpu,
        )?;
        let cpu_renderer = crate::render::ImplicitRenderer::new(
            &cpu_config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("renderer"),
        )?;
        let mut gpu_config = cpu_config.clone();
        gpu_config.compute_backend = crate::config::ComputeBackend::OpenCl;
        let gpu = DynamicsSystem::new(
            &gpu_config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("dynamics"),
            &Device::Cpu,
        )?;
        let gpu_renderer = crate::render::ImplicitRenderer::new(
            &gpu_config,
            VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu).pp("renderer"),
        )?;
        let saved = candle_core::safetensors::load(model_path, &Device::Cpu)?;
        {
            let data = vars.data().lock().expect("VarMap mutex poisoned");
            for (name, variable) in data.iter() {
                variable.set(
                    saved
                        .get(name)
                        .with_context(|| format!("checkpoint missing {name}"))?,
                )?;
            }
        }
        let saved_world = candle_core::safetensors::load(world_path, &Device::Cpu)?;
        let scalar = |name: &str| -> Result<u64> {
            Ok(saved_world
                .get(name)
                .with_context(|| format!("world missing {name}"))?
                .to_scalar::<i64>()? as u64)
        };
        let birth_generations = saved_world
            .get("world.morph_birth_generations")
            .context("world missing morph birth generations")?
            .to_vec1::<i64>()?
            .into_iter()
            .map(|value| value as u64)
            .collect();
        let initial = WorldState {
            micro: saved_world
                .get("world.micro")
                .context("world missing micro")?
                .clone(),
            macro_field: saved_world
                .get("world.macro")
                .context("world missing macro")?
                .clone(),
            memory: saved_world
                .get("world.memory")
                .context("world missing memory")?
                .clone(),
            step: scalar("world.step")?,
            age: scalar("world.age")?,
            episode: scalar("world.episode")?,
            target_index: scalar("world.target_index")? as usize,
            morph_active_depth: scalar("world.morph_active_depth")? as usize,
            morph_generation: scalar("world.morph_generation")?,
            morph_birth_generations: birth_generations,
        };
        let mut corpus = crate::corpus::ImageCorpus::new(&cpu_config, &Device::Cpu)?;
        let sample = corpus.sample_index(initial.target_index, &Device::Cpu)?;
        let fidelity = 0.65;
        let mut warm_world = initial.clone();
        warm_world.age -= warm_world.age % cpu_config.macro_update_every as u64;
        let _ = cpu.step(
            &warm_world,
            &sample.genome_tensor,
            Some(&sample.reference_micro),
            Some(&sample.reference_macro),
            fidelity,
            false,
        )?;
        let _ = gpu.step(
            &warm_world,
            &sample.genome_tensor,
            Some(&sample.reference_micro),
            Some(&sample.reference_macro),
            fidelity,
            false,
        )?;
        gpu_renderer
            .prepare_inference_backend()?
            .context("OpenCL renderer did not initialize")?;

        let mut cpu_world = initial.clone();
        let cpu_started = std::time::Instant::now();
        let mut cpu_worlds = Vec::new();
        for relative_age in 1..=32 {
            cpu_world = cpu
                .step(
                    &cpu_world,
                    &sample.genome_tensor,
                    Some(&sample.reference_micro),
                    Some(&sample.reference_macro),
                    fidelity,
                    false,
                )?
                .world;
            if [1, 8, 32].contains(&relative_age) {
                cpu_worlds.push(cpu_world.clone());
            }
        }
        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
        let mut gpu_world = initial;
        let gpu_started = std::time::Instant::now();
        let mut comparison_index = 0;
        for relative_age in 1..=32 {
            gpu_world = gpu
                .step(
                    &gpu_world,
                    &sample.genome_tensor,
                    Some(&sample.reference_micro),
                    Some(&sample.reference_macro),
                    fidelity,
                    false,
                )?
                .world;
            if ![1, 8, 32].contains(&relative_age) {
                continue;
            }
            let cpu_world = &cpu_worlds[comparison_index];
            comparison_index += 1;
            let micro = gpu_world.micro.sub(&cpu_world.micro)?.abs()?;
            let macro_field = gpu_world.macro_field.sub(&cpu_world.macro_field)?.abs()?;
            let memory = gpu_world.memory.sub(&cpu_world.memory)?.abs()?;
            let max_abs = micro
                .max_all()?
                .to_scalar::<f32>()?
                .max(macro_field.max_all()?.to_scalar::<f32>()?)
                .max(memory.max_all()?.to_scalar::<f32>()?);
            let mean_abs = (micro.mean_all()?.to_scalar::<f32>()?
                + macro_field.mean_all()?.to_scalar::<f32>()?
                + memory.mean_all()?.to_scalar::<f32>()?)
                / 3.0;
            eprintln!("OPENCL checkpoint dynamics | age=+{relative_age} max_abs={max_abs:.8} mean_abs={mean_abs:.8}");
            assert!(max_abs <= 5e-3, "age +{relative_age} drift {max_abs}");
            assert!(
                mean_abs <= 5e-4,
                "age +{relative_age} mean drift {mean_abs}"
            );
        }
        let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1000.0;
        let plan = crate::render::RenderPlan::new(&cpu_config, 192, &Device::Cpu)?;
        let cpu_render = cpu_renderer.render(
            &cpu_world.micro,
            &cpu_world.macro_field,
            &sample.genome_tensor,
            &plan,
            false,
        )?;
        let gpu_render = gpu_renderer.render(
            &gpu_world.micro,
            &gpu_world.macro_field,
            &sample.genome_tensor,
            &plan,
            false,
        )?;
        let render_difference = gpu_render.image.sub(&cpu_render.image)?.abs()?;
        let render_max = render_difference.max_all()?.to_scalar::<f32>()?;
        let render_mean = render_difference.mean_all()?.to_scalar::<f32>()?;
        let render_rms = render_difference
            .sqr()?
            .mean_all()?
            .sqrt()?
            .to_scalar::<f32>()?;
        eprintln!("OPENCL checkpoint dynamics benchmark | steps=32 cpu_ms={cpu_ms:.3} gpu_ms={gpu_ms:.3} speedup={:.3}x render_max={render_max:.8} render_mean={render_mean:.8} render_rms={render_rms:.8}", cpu_ms / gpu_ms);
        assert!(render_max <= 5e-3, "render drift {render_max}");
        assert!(render_mean <= 5e-4, "render mean drift {render_mean}");
        Ok(())
    }
}
