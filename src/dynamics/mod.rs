mod nca;
mod operators;

use crate::config::{Integrator, RunConfig};
use crate::state::WorldState;
use crate::tensor_ops::{broadcast_vector, mean_abs, zeros, PeriodicUpsampler};
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use nca::NeuralCa;
use operators::{PhysicalOperators, ScaleFields};

pub struct DynamicsSystem {
    micro_ca: NeuralCa,
    macro_ca: NeuralCa,
    physical: PhysicalOperators,
    macro_to_micro: PeriodicUpsampler,
    macro_zero_context: Tensor,
    config: RunConfig,
    seed: u64,
}

pub struct StepOutput {
    pub world: WorldState,
    pub micro_movement: f32,
    pub macro_movement: f32,
    pub macro_updated: bool,
}

#[derive(Clone, Copy)]
enum FieldScale {
    Micro,
    Macro,
}

impl DynamicsSystem {
    pub fn new(config: &RunConfig, vb: VarBuilder<'_>, device: &Device) -> Result<Self> {
        Ok(Self {
            micro_ca: NeuralCa::new(config, vb.pp("micro_ca"), device)?,
            macro_ca: NeuralCa::new(config, vb.pp("macro_ca"), device)?,
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
            config: config.clone(),
            seed: config.seed,
        })
    }

    /// Advance one world step. `tracked=false` detaches the NCA parameters at
    /// their pointwise projections, producing a genuinely tape-free core pass.
    pub fn step(&self, world: &WorldState, genome: &Tensor, tracked: bool) -> Result<StepOutput> {
        let macro_updated = world
            .age
            .is_multiple_of(self.config.macro_update_every as u64);
        let next_macro = if macro_updated {
            self.integrate(
                &world.macro_field,
                &self.macro_zero_context,
                genome,
                world.step,
                FieldScale::Macro,
                tracked,
            )?
        } else {
            world.macro_field.clone()
        };
        let macro_context = self.macro_to_micro.apply(&next_macro)?;
        let next_micro = self.integrate(
            &world.micro,
            &macro_context,
            genome,
            world.step,
            FieldScale::Micro,
            tracked,
        )?;
        let micro_movement = mean_abs(&next_micro.sub(&world.micro)?)?;
        let macro_movement = if macro_updated {
            mean_abs(&next_macro.sub(&world.macro_field)?)?
        } else {
            0.0
        };
        Ok(StepOutput {
            world: WorldState {
                micro: next_micro,
                macro_field: next_macro,
                step: world.step + 1,
                age: world.age + 1,
                episode: world.episode,
                target_index: world.target_index,
            },
            micro_movement,
            macro_movement,
            macro_updated,
        })
    }

    fn integrate(
        &self,
        field: &Tensor,
        macro_context: &Tensor,
        genome: &Tensor,
        step: u64,
        scale: FieldScale,
        tracked: bool,
    ) -> Result<Tensor> {
        let (ca, scale_fields): (&NeuralCa, &ScaleFields) = match scale {
            FieldScale::Micro => (&self.micro_ca, &self.physical.micro_fields),
            FieldScale::Macro => (&self.macro_ca, &self.physical.macro_fields),
        };
        let (_, _, h, w) = field.dims4()?;
        let genome_field = broadcast_vector(genome, h, w)?;
        let derivative = match self.config.integrator {
            Integrator::Euler => self.derivative(
                field,
                macro_context,
                &genome_field,
                step,
                ca,
                scale_fields,
                tracked,
            )?,
            Integrator::Midpoint => {
                let k1 = self.derivative(
                    field,
                    macro_context,
                    &genome_field,
                    step,
                    ca,
                    scale_fields,
                    tracked,
                )?;
                let midpoint = field.add(&k1.affine((0.5 * self.config.dt) as f64, 0.0)?)?;
                self.derivative(
                    &midpoint,
                    macro_context,
                    &genome_field,
                    step,
                    ca,
                    scale_fields,
                    tracked,
                )?
            }
        };
        field
            .add(&derivative.affine(self.config.dt as f64, 0.0)?)?
            .clamp(-self.config.state_limit, self.config.state_limit)
            .map_err(Into::into)
    }

    #[allow(clippy::too_many_arguments)]
    fn derivative(
        &self,
        field: &Tensor,
        macro_context: &Tensor,
        genome_field: &Tensor,
        step: u64,
        ca: &NeuralCa,
        scale_fields: &ScaleFields,
        tracked: bool,
    ) -> Result<Tensor> {
        let mut delta = ca.delta(field, macro_context, genome_field, self.seed, step, tracked)?;
        if self.config.reaction_gain > 0.0 {
            delta = delta.add(&self.physical.reaction_diffusion(field)?)?;
        }
        if self.config.phase_gain > 0.0 {
            delta = delta.add(&self.physical.complex_phase(field)?)?;
        }
        if self.config.cyclic_gain > 0.0 {
            delta = delta.add(&self.physical.cyclic_chemistry(field)?)?;
        }
        if self.config.fractal_gain > 0.0 || self.config.quasiperiodic_gain > 0.0 {
            delta = delta.add(&self.physical.forcing(field, scale_fields)?)?;
        }
        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let world = WorldState::fresh(&config, 42, &device)?;
        let genome = Tensor::new(&[0.0f32, 0.25, -0.5, 1.0], &device)?;
        let output = system.step(&world, &genome, true)?;
        assert_eq!(output.world.micro.dims4()?, (1, 12, 32, 32));
        assert_eq!(output.world.macro_field.dims4()?, (1, 12, 16, 16));
        assert!(output.micro_movement.is_finite());
        assert!(output.macro_movement.is_finite());
        Ok(())
    }
}
