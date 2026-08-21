use crate::config::{OptimizerKind, RunConfig};
use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, Device, Tensor, Var};
use candle_nn::{ParamsAdamW, VarMap};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

const MUON_A: f64 = 3.4445;
const MUON_B: f64 = -4.7750;
const MUON_C: f64 = 2.0315;
const MUON_EPSILON: f64 = 1e-7;

struct AdamVariable {
    name: String,
    variable: Var,
    first: Var,
    second: Var,
    use_muon: bool,
}

pub struct PersistentAdamW {
    variables: Vec<AdamVariable>,
    updates: u64,
    params: ParamsAdamW,
    peak_learning_rate: f64,
    grad_clip: f64,
    warmup_updates: usize,
    optimizer_kind: OptimizerKind,
    muon_momentum: f64,
    muon_ns_steps: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct OptimizerStats {
    pub gradient_norm: f32,
    pub gradient_rms: f32,
    pub clip_scale: f32,
    pub effective_learning_rate: f64,
    pub updated_variables: usize,
    pub updated_parameters: usize,
    pub muon_variables: usize,
    pub backward_seconds: f64,
    pub step_seconds: f64,
}

impl PersistentAdamW {
    pub fn new(varmap: &VarMap, config: &RunConfig) -> candle_core::Result<Self> {
        let data = varmap.data().lock().expect("VarMap mutex poisoned");
        let mut names: Vec<String> = data.keys().cloned().collect();
        names.sort();
        let mut variables = Vec::with_capacity(names.len());
        for name in names {
            let variable = data[&name].clone();
            if !variable.dtype().is_float() {
                continue;
            }
            variables.push(AdamVariable {
                use_muon: config.optimizer == OptimizerKind::HybridMuon
                    && muon_candidate(&name, variable.as_tensor()),
                name,
                first: Var::zeros(variable.shape(), variable.dtype(), variable.device())?,
                // A stable, versioned state layout keeps optimizer migration and
                // exact resume simple. Muon matrices leave this tensor at zero.
                second: Var::zeros(variable.shape(), variable.dtype(), variable.device())?,
                variable,
            });
        }
        Ok(Self {
            variables,
            updates: 0,
            params: ParamsAdamW {
                lr: config.learning_rate,
                beta1: config.beta1,
                beta2: config.beta2,
                eps: config.adam_epsilon,
                weight_decay: config.weight_decay,
            },
            peak_learning_rate: config.learning_rate,
            grad_clip: config.grad_clip,
            warmup_updates: config.warmup_updates,
            optimizer_kind: config.optimizer,
            muon_momentum: config.muon_momentum,
            muon_ns_steps: config.muon_ns_steps,
        })
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<OptimizerStats> {
        let backward_started = Instant::now();
        let gradients = loss.backward()?;
        let backward_seconds = backward_started.elapsed().as_secs_f64();
        let step_started = Instant::now();
        let mut stats = self.step(&gradients)?;
        stats.backward_seconds = backward_seconds;
        stats.step_seconds = step_started.elapsed().as_secs_f64();
        Ok(stats)
    }

    pub fn updates(&self) -> u64 {
        self.updates
    }

    pub fn kind(&self) -> OptimizerKind {
        self.optimizer_kind
    }

    fn step(&mut self, gradients: &GradStore) -> Result<OptimizerStats> {
        let mut squared_norm = 0.0f64;
        let mut updated_variables = 0usize;
        let mut updated_parameters = 0usize;
        let mut muon_variables = 0usize;
        for state in &self.variables {
            if let Some(gradient) = gradients.get(state.variable.as_tensor()) {
                squared_norm += gradient.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                updated_variables += 1;
                updated_parameters += gradient.elem_count();
                muon_variables += usize::from(state.use_muon);
            }
        }
        let gradient_norm = squared_norm.sqrt();
        if !gradient_norm.is_finite() {
            bail!("non-finite gradient norm; optimizer update was not applied");
        }
        if updated_variables == 0 {
            bail!("loss produced no trainable gradients");
        }
        let clip_scale = if self.grad_clip > 0.0 && gradient_norm > self.grad_clip {
            self.grad_clip / gradient_norm.max(1e-12)
        } else {
            1.0
        };
        let next_update = self.updates + 1;
        let warmup_gain = if self.warmup_updates == 0 {
            1.0
        } else {
            (next_update as f64 / self.warmup_updates as f64).min(1.0)
        };
        let learning_rate = self.peak_learning_rate * warmup_gain;
        let params = &self.params;
        let first_bias = 1.0 / (1.0 - params.beta1.powi(next_update as i32));
        let second_bias = 1.0 / (1.0 - params.beta2.powi(next_update as i32));
        for state in &self.variables {
            let Some(raw_gradient) = gradients.get(state.variable.as_tensor()) else {
                continue;
            };
            let gradient = raw_gradient.affine(clip_scale, 0.0)?;
            let decayed = state
                .variable
                .as_tensor()
                .affine(1.0 - learning_rate * params.weight_decay, 0.0)?;
            if state.use_muon {
                let next_momentum = state
                    .first
                    .as_tensor()
                    .affine(self.muon_momentum, 0.0)?
                    .add(&gradient.affine(1.0 - self.muon_momentum, 0.0)?)?;
                let nesterov = next_momentum
                    .affine(self.muon_momentum, 0.0)?
                    .add(&gradient.affine(1.0 - self.muon_momentum, 0.0)?)?;
                let orthogonal = zeropower_newton_schulz(&nesterov, self.muon_ns_steps)?;
                let (rows, columns) = gradient.dims2()?;
                // Match the RMS convention used by current hybrid-Muon
                // implementations so the AdamW learning-rate scale remains a
                // useful starting point for an ablation.
                let adjustment = 0.2 * (rows.max(columns) as f64).sqrt();
                state
                    .variable
                    .set(&decayed.sub(&orthogonal.affine(learning_rate * adjustment, 0.0)?)?)?;
                state.first.set(&next_momentum)?;
            } else {
                let next_first = state
                    .first
                    .as_tensor()
                    .affine(params.beta1, 0.0)?
                    .add(&gradient.affine(1.0 - params.beta1, 0.0)?)?;
                let next_second = state
                    .second
                    .as_tensor()
                    .affine(params.beta2, 0.0)?
                    .add(&gradient.sqr()?.affine(1.0 - params.beta2, 0.0)?)?;
                let adjusted = next_first.affine(first_bias, 0.0)?.div(
                    &next_second
                        .affine(second_bias, 0.0)?
                        .sqrt()?
                        .affine(1.0, params.eps)?,
                )?;
                state
                    .variable
                    .set(&decayed.sub(&adjusted.affine(learning_rate, 0.0)?)?)?;
                state.first.set(&next_first)?;
                state.second.set(&next_second)?;
            }
        }
        self.updates = next_update;
        Ok(OptimizerStats {
            gradient_norm: gradient_norm as f32,
            gradient_rms: (gradient_norm / (updated_parameters as f64).sqrt()) as f32,
            clip_scale: clip_scale as f32,
            effective_learning_rate: learning_rate,
            updated_variables,
            updated_parameters,
            muon_variables,
            backward_seconds: 0.0,
            step_seconds: 0.0,
        })
    }

    pub fn save(&self, path: &Path, world_step: u64, device: &Device) -> Result<()> {
        let mut tensors = HashMap::with_capacity(self.variables.len() * 2 + 3);
        tensors.insert(
            "optimizer.updates".to_owned(),
            Tensor::new(self.updates as i64, device)?,
        );
        tensors.insert(
            "optimizer.world_step".to_owned(),
            Tensor::new(world_step as i64, device)?,
        );
        tensors.insert(
            "optimizer.kind".to_owned(),
            Tensor::new(self.optimizer_kind as i64, device)?,
        );
        for state in &self.variables {
            tensors.insert(
                format!("optimizer.m.{}", state.name),
                state.first.as_tensor().clone(),
            );
            tensors.insert(
                format!("optimizer.v.{}", state.name),
                state.second.as_tensor().clone(),
            );
        }
        atomic_safetensors(&tensors, path)
    }

    pub fn load(
        &mut self,
        path: &Path,
        expected_world_step: u64,
        device: &Device,
    ) -> Result<usize> {
        let tensors = candle_core::safetensors::load(path, device)
            .with_context(|| format!("cannot load optimizer {}", path.display()))?;
        let world_step = tensors
            .get("optimizer.world_step")
            .context("optimizer has no world step")?
            .to_scalar::<i64>()? as u64;
        if world_step != expected_world_step {
            bail!("optimizer/world mismatch: optimizer {world_step}, world {expected_world_step}");
        }
        let saved_kind = tensors
            .get("optimizer.kind")
            .context("optimizer has no optimizer-kind marker")?
            .to_scalar::<i64>()?;
        if saved_kind != self.optimizer_kind as i64 {
            bail!("optimizer kind does not match the requested v7 configuration");
        }
        let updates = tensors
            .get("optimizer.updates")
            .context("optimizer has no update counter")?
            .to_scalar::<i64>()? as u64;
        let mut loaded = 0;
        for state in &self.variables {
            let first = tensors
                .get(&format!("optimizer.m.{}", state.name))
                .with_context(|| format!("missing first moment for {}", state.name))?;
            let second = tensors
                .get(&format!("optimizer.v.{}", state.name))
                .with_context(|| format!("missing second moment for {}", state.name))?;
            if first.dims() != state.variable.dims() || second.dims() != state.variable.dims() {
                bail!("optimizer moment shape mismatch for {}", state.name);
            }
            state.first.set(first)?;
            state.second.set(second)?;
            loaded += 1;
        }
        self.updates = updates;
        Ok(loaded)
    }
}

fn muon_candidate(name: &str, tensor: &Tensor) -> bool {
    if tensor.dims().len() != 2 || !name.ends_with(".weight") {
        return false;
    }
    let matrix_role = name.contains(".query.")
        || name.contains(".key.")
        || name.contains(".value.")
        || name.contains(".attention_output.")
        || name.contains(".feedforward_")
        || name.contains(".morphic_");
    matrix_role && !name.contains(".norm.") && !name.contains(".write.")
}

fn zeropower_newton_schulz(gradient: &Tensor, steps: usize) -> Result<Tensor> {
    let (rows, columns) = gradient.dims2()?;
    let transposed = rows > columns;
    let mut value = if transposed {
        gradient.t()?.contiguous()?
    } else {
        gradient.clone()
    };
    let norm = value
        .sqr()?
        .sum_all()?
        .sqrt()?
        .to_scalar::<f32>()?
        .max(MUON_EPSILON as f32) as f64;
    value = value.affine(1.0 / norm, 0.0)?;
    for _ in 0..steps {
        let gram = value.matmul(&value.t()?)?;
        let polynomial = gram
            .affine(MUON_B, 0.0)?
            .add(&gram.matmul(&gram)?.affine(MUON_C, 0.0)?)?;
        value = value
            .affine(MUON_A, 0.0)?
            .add(&polynomial.matmul(&value)?)?;
    }
    if transposed {
        value.t()?.contiguous().map_err(Into::into)
    } else {
        Ok(value)
    }
}

fn atomic_safetensors(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    let temporary = path.with_extension("safetensors.tmp");
    candle_core::safetensors::save(tensors, &temporary)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    use candle_nn::{Module, VarBuilder, VarMap};
    #[test]
    fn newton_schulz_preserves_matrix_shape_and_finiteness() -> Result<()> {
        let matrix = Tensor::arange(0f32, 48f32, &Device::Cpu)?.reshape((6, 8))?;
        let orthogonal = zeropower_newton_schulz(&matrix, 5)?;
        assert_eq!(orthogonal.dims2()?, (6, 8));
        assert!(orthogonal.abs()?.max_all()?.to_scalar::<f32>()?.is_finite());
        assert_eq!(orthogonal.dtype(), DType::F32);
        Ok(())
    }

    #[test]
    fn hybrid_muon_routes_interface_matrix_and_updates_it() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig {
            optimizer: OptimizerKind::HybridMuon,
            interface_width: 32,
            ..RunConfig::default()
        };
        let variables = VarMap::new();
        let builder = VarBuilder::from_varmap(&variables, DType::F32, &device);
        let linear = candle_nn::linear(32, 32, builder.pp("dynamics").pp("interface").pp("query"))?;
        let mut optimizer = PersistentAdamW::new(&variables, &config)?;
        let input = Tensor::ones((4, 32), DType::F32, &device)?;
        let loss = linear.forward(&input)?.sqr()?.mean_all()?;
        let stats = optimizer.backward_step(&loss)?;
        assert_eq!(stats.muon_variables, 1);
        Ok(())
    }
}
