use crate::config::RunConfig;
use crate::tensor_ops::splitmix64;
use candle_core::{DType, Device, Result, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Clone)]
pub struct WorldState {
    pub micro: Tensor,
    pub macro_field: Tensor,
    pub memory: Tensor,
    /// Global completed development steps. This never resets between episodes.
    pub step: u64,
    /// Development age of the current seeded organism.
    pub age: u64,
    pub episode: u64,
    pub target_index: usize,
}

impl WorldState {
    pub fn fresh(config: &RunConfig, seed: u64, device: &Device) -> Result<Self> {
        Ok(Self {
            micro: seeded_field(config.micro_size, config.channels, seed, device)?,
            macro_field: seeded_field(
                config.macro_size,
                config.channels,
                seed ^ 0xa11e_9a2d,
                device,
            )?,
            memory: Tensor::zeros((1, config.interface_width), DType::F32, device)?,
            step: 0,
            age: 0,
            episode: 0,
            target_index: 0,
        })
    }

    pub fn reseed_for_episode(
        &self,
        config: &RunConfig,
        seed: u64,
        episode: u64,
        target_index: usize,
    ) -> Result<Self> {
        if config.episode_reset <= 0.0 {
            return Ok(Self {
                micro: self.micro.detach(),
                macro_field: self.macro_field.detach(),
                memory: self.memory.detach(),
                step: self.step,
                age: self.age,
                episode,
                target_index,
            });
        }
        let seeded = Self::fresh(config, seed, self.micro.device())?;
        let reset = config.episode_reset as f64;
        let keep = 1.0 - reset;
        Ok(Self {
            micro: self
                .micro
                .detach()
                .affine(keep, 0.0)?
                .add(&seeded.micro.affine(reset, 0.0)?)?,
            macro_field: self
                .macro_field
                .detach()
                .affine(keep, 0.0)?
                .add(&seeded.macro_field.affine(reset, 0.0)?)?,
            memory: self.memory.detach().affine(keep, 0.0)?,
            step: self.step,
            age: 0,
            episode,
            target_index,
        })
    }

    pub fn detached(&self) -> Self {
        Self {
            micro: self.micro.detach(),
            macro_field: self.macro_field.detach(),
            memory: self.memory.detach(),
            step: self.step,
            age: self.age,
            episode: self.episode,
            target_index: self.target_index,
        }
    }
}

fn seeded_field(size: usize, channels: usize, seed: u64, device: &Device) -> Result<Tensor> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let plane = size * size;
    let mut values = vec![0.0f32; channels * plane];
    let tau = std::f32::consts::TAU;
    let seed_phase = ((splitmix64(seed) >> 40) as f32 / (1u32 << 24) as f32) * tau;
    for y in 0..size {
        for x in 0..size {
            let index = y * size + x;
            let nx = (x as f32 + 0.5) / size as f32;
            let ny = (y as f32 + 0.5) / size as f32;
            let dx = nx - 0.5;
            let dy = ny - 0.5;
            let seed_blob = (-90.0 * (dx * dx + dy * dy)).exp();
            values[index] = 0.96 + rng.gen_range(-0.02..0.02);
            values[plane + index] = 0.20 * seed_blob + rng.gen_range(0.0..0.015);
            let phase = tau * (1.618_034 * nx + std::f32::consts::SQRT_2 * ny) + seed_phase;
            values[2 * plane + index] = 0.25 * phase.cos() + rng.gen_range(-0.03..0.03);
            values[3 * plane + index] = 0.25 * phase.sin() + rng.gen_range(-0.03..0.03);
            // Three out-of-phase seeds give the cyclic subsystem something
            // spatial to rotate without privileging a single species.
            values[6 * plane + index] = 0.08 * (phase + 0.0).sin();
            values[7 * plane + index] = 0.08 * (phase + tau / 3.0).sin();
            values[8 * plane + index] = 0.08 * (phase + 2.0 * tau / 3.0).sin();
            for channel in 4..channels {
                if !(6..=8).contains(&channel) {
                    values[channel * plane + index] = rng.gen_range(-0.04..0.04);
                }
            }
        }
    }
    Tensor::from_vec(values, (1, channels, size, size), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_world_is_seed_deterministic_and_runtime_sized() -> Result<()> {
        let config = RunConfig {
            micro_size: 32,
            macro_size: 16,
            channels: 12,
            train_resolution: 64,
            output_resolution: 64,
            ..RunConfig::default()
        };
        let a = WorldState::fresh(&config, 9, &Device::Cpu)?;
        let b = WorldState::fresh(&config, 9, &Device::Cpu)?;
        assert_eq!(a.micro.dims4()?, (1, 12, 32, 32));
        assert_eq!(
            a.micro.flatten_all()?.to_vec1::<f32>()?,
            b.micro.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
