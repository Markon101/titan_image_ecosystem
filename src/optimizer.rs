use crate::config::RunConfig;
use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, Device, Tensor, Var};
use candle_nn::{ParamsAdamW, VarMap};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

struct AdamVariable {
    name: String,
    variable: Var,
    first: Var,
    second: Var,
}

pub struct PersistentAdamW {
    variables: Vec<AdamVariable>,
    updates: u64,
    params: ParamsAdamW,
    peak_learning_rate: f64,
    grad_clip: f64,
    warmup_updates: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct OptimizerStats {
    pub gradient_norm: f32,
    pub clip_scale: f32,
    pub effective_learning_rate: f64,
    pub updated_variables: usize,
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
                name,
                first: Var::zeros(variable.shape(), variable.dtype(), variable.device())?,
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

    fn step(&mut self, gradients: &GradStore) -> Result<OptimizerStats> {
        let mut squared_norm = 0.0f64;
        let mut updated_variables = 0usize;
        for state in &self.variables {
            if let Some(gradient) = gradients.get(state.variable.as_tensor()) {
                squared_norm += gradient.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                updated_variables += 1;
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
            let next_value = state
                .variable
                .as_tensor()
                .affine(1.0 - learning_rate * params.weight_decay, 0.0)?
                .sub(&adjusted.affine(learning_rate, 0.0)?)?;
            state.first.set(&next_first)?;
            state.second.set(&next_second)?;
            state.variable.set(&next_value)?;
        }
        self.updates = next_update;
        Ok(OptimizerStats {
            gradient_norm: gradient_norm as f32,
            clip_scale: clip_scale as f32,
            effective_learning_rate: learning_rate,
            updated_variables,
            backward_seconds: 0.0,
            step_seconds: 0.0,
        })
    }

    pub fn save(&self, path: &Path, world_step: u64, device: &Device) -> Result<()> {
        let mut tensors = HashMap::with_capacity(self.variables.len() * 2 + 2);
        tensors.insert(
            "optimizer.updates".to_owned(),
            Tensor::new(self.updates as i64, device)?,
        );
        tensors.insert(
            "optimizer.world_step".to_owned(),
            Tensor::new(world_step as i64, device)?,
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

fn atomic_safetensors(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    let temporary = path.with_extension("safetensors.tmp");
    candle_core::safetensors::save(tensors, &temporary)?;
    std::fs::rename(&temporary, path)?;
    Ok(())
}
