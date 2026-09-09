use crate::config::{OptimizerKind, RunConfig};
use anyhow::{bail, Context, Result};
use candle_core::{backprop::GradStore, Device, Tensor, Var};
use candle_nn::{ParamsAdamW, VarMap};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

const MUON_A: f64 = 3.4445;
const MUON_B: f64 = -4.7750;
const MUON_C: f64 = 2.0315;
const MUON_EPSILON: f64 = 1e-7;
pub const OPTIMIZER_LAYOUT_VERSION: u32 = 1;

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
    diagnostics: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct OptimizerStats {
    pub groups: std::collections::BTreeMap<String, crate::gradient_diagnostics::GroupStats>,
    pub gradient_norm: f32,
    pub gradient_rms: f32,
    pub clip_scale: f32,
    pub core_gradient_rms: f32,
    pub decoder_gradient_rms: f32,
    pub core_updated_parameters: usize,
    pub decoder_updated_parameters: usize,
    pub grounded_gradient_rms: f32,
    pub emergent_gradient_rms: f32,
    pub flow_gradient_rms: f32,
    pub grounded_update_rms: f32,
    pub emergent_update_rms: f32,
    pub grounded_update_weight_ratio: f32,
    pub emergent_update_weight_ratio: f32,
    pub grounded_updated_parameters: usize,
    pub emergent_updated_parameters: usize,
    pub effective_learning_rate: f64,
    pub updated_variables: usize,
    pub updated_parameters: usize,
    pub muon_variables: usize,
    pub backward_seconds: f64,
    pub step_seconds: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct OptimizerMigrationReport {
    pub preserved_moment_pairs: usize,
    pub new_moment_pairs: usize,
    pub preserved_moment_elements: usize,
    pub new_moment_elements: usize,
    pub updates_preserved: u64,
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
            diagnostics: config.experiment.optimizer_diagnostics,
        })
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<OptimizerStats> {
        self.backward_step_with_merged(loss, |_| Ok(None))
    }

    pub fn backward_step_with<F>(
        &mut self,
        loss: &Tensor,
        augment_gradients: F,
    ) -> Result<OptimizerStats>
    where
        F: FnOnce(&mut GradStore) -> Result<()>,
    {
        self.backward_step_with_merged(loss, |gradients| {
            augment_gradients(gradients)?;
            Ok(None)
        })
    }

    pub fn backward_step_with_merged<F>(
        &mut self,
        loss: &Tensor,
        augment_gradients: F,
    ) -> Result<OptimizerStats>
    where
        F: FnOnce(&mut GradStore) -> Result<Option<GradStore>>,
    {
        let backward_started = Instant::now();
        let mut gradients = loss.backward()?;
        if let Some(additional) = augment_gradients(&mut gradients)? {
            for state in &self.variables {
                let Some(extra) = additional.get(state.variable.as_tensor()) else {
                    continue;
                };
                let merged = if let Some(existing) = gradients.get(state.variable.as_tensor()) {
                    existing.add(extra)?
                } else {
                    extra.clone()
                };
                gradients.insert(state.variable.as_tensor(), merged);
            }
        }
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
        let mut groups =
            std::collections::BTreeMap::<String, crate::gradient_diagnostics::GroupStats>::new();
        if self.diagnostics {
            for state in &self.variables {
                groups
                    .entry(crate::gradient_diagnostics::group(&state.name).to_owned())
                    .or_default()
                    .observe(
                        state.variable.as_tensor(),
                        gradients.get(state.variable.as_tensor()),
                    )?;
            }
        }
        let mut squared_norm = 0.0f64;
        let mut updated_variables = 0usize;
        let mut core_squared_norm = 0.0f64;
        let mut decoder_squared_norm = 0.0f64;
        let mut core_updated_parameters = 0usize;
        let mut decoder_updated_parameters = 0usize;
        let mut updated_parameters = 0usize;
        let mut grounded_squared_norm = 0.0f64;
        let mut emergent_squared_norm = 0.0f64;
        let mut flow_squared_norm = 0.0f64;
        let mut grounded_parameters = 0usize;
        let mut emergent_parameters = 0usize;
        let mut flow_parameters = 0usize;
        let mut muon_variables = 0usize;
        for state in &self.variables {
            if let Some(gradient) = gradients.get(state.variable.as_tensor()) {
                let gradient_energy = gradient.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
                squared_norm += gradient_energy;
                if state.name.starts_with("dynamics.") {
                    core_squared_norm += gradient_energy;
                    core_updated_parameters += gradient.elem_count();
                } else if state.name.starts_with("renderer.") {
                    decoder_squared_norm += gradient_energy;
                    decoder_updated_parameters += gradient.elem_count();
                }
                updated_variables += 1;
                updated_parameters += gradient.elem_count();
                if state.name.starts_with("renderer.grounded.") {
                    grounded_squared_norm += gradient_energy;
                    grounded_parameters += gradient.elem_count();
                } else if state.name.starts_with("renderer.emergent.") {
                    emergent_squared_norm += gradient_energy;
                    emergent_parameters += gradient.elem_count();
                } else if state.name.starts_with("flow.") {
                    flow_squared_norm += gradient_energy;
                    flow_parameters += gradient.elem_count();
                }
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
        let mut grounded_update_energy = 0.0f64;
        let mut emergent_update_energy = 0.0f64;
        let mut grounded_weight_energy = 0.0f64;
        let mut emergent_weight_energy = 0.0f64;
        for state in &self.variables {
            let Some(raw_gradient) = gradients.get(state.variable.as_tensor()) else {
                continue;
            };
            let gradient = raw_gradient.affine(clip_scale, 0.0)?;
            let decayed = state
                .variable
                .as_tensor()
                .affine(1.0 - learning_rate * params.weight_decay, 0.0)?;
            let next_value = if state.use_muon {
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
                let adjustment = 0.2 * (rows.max(columns) as f64).sqrt();
                state.first.set(&next_momentum)?;
                decayed.sub(&orthogonal.affine(learning_rate * adjustment, 0.0)?)?
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
                state.first.set(&next_first)?;
                state.second.set(&next_second)?;
                decayed.sub(&adjusted.affine(learning_rate, 0.0)?)?
            };
            if state.name.starts_with("renderer.grounded.") {
                grounded_update_energy += next_value
                    .sub(state.variable.as_tensor())?
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()? as f64;
                grounded_weight_energy += state
                    .variable
                    .as_tensor()
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()? as f64;
            } else if state.name.starts_with("renderer.emergent.") {
                emergent_update_energy += next_value
                    .sub(state.variable.as_tensor())?
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()? as f64;
                emergent_weight_energy += state
                    .variable
                    .as_tensor()
                    .sqr()?
                    .sum_all()?
                    .to_scalar::<f32>()? as f64;
            }
            if self.diagnostics {
                groups
                    .get_mut(crate::gradient_diagnostics::group(&state.name))
                    .unwrap()
                    .update_energy += crate::gradient_diagnostics::energy(
                    &next_value.sub(state.variable.as_tensor())?,
                )?;
            }
            state.variable.set(&next_value)?;
        }
        for group in groups.values_mut() {
            group.finish(squared_norm);
        }
        self.updates = next_update;
        Ok(OptimizerStats {
            groups,
            gradient_norm: gradient_norm as f32,
            gradient_rms: (gradient_norm / (updated_parameters as f64).sqrt()) as f32,
            core_gradient_rms: if core_updated_parameters > 0 {
                (core_squared_norm / core_updated_parameters as f64).sqrt() as f32
            } else {
                0.0
            },
            decoder_gradient_rms: if decoder_updated_parameters > 0 {
                (decoder_squared_norm / decoder_updated_parameters as f64).sqrt() as f32
            } else {
                0.0
            },
            grounded_gradient_rms: if grounded_parameters > 0 {
                (grounded_squared_norm / grounded_parameters as f64).sqrt() as f32
            } else {
                0.0
            },
            emergent_gradient_rms: if emergent_parameters > 0 {
                (emergent_squared_norm / emergent_parameters as f64).sqrt() as f32
            } else {
                0.0
            },
            flow_gradient_rms: if flow_parameters > 0 {
                (flow_squared_norm / flow_parameters as f64).sqrt() as f32
            } else {
                0.0
            },
            grounded_update_rms: if grounded_parameters > 0 {
                (grounded_update_energy / grounded_parameters as f64).sqrt() as f32
            } else {
                0.0
            },
            emergent_update_rms: if emergent_parameters > 0 {
                (emergent_update_energy / emergent_parameters as f64).sqrt() as f32
            } else {
                0.0
            },
            grounded_update_weight_ratio: (grounded_update_energy
                / grounded_weight_energy.max(1e-20))
            .sqrt() as f32,
            emergent_update_weight_ratio: (emergent_update_energy
                / emergent_weight_energy.max(1e-20))
            .sqrt() as f32,
            grounded_updated_parameters: grounded_parameters,
            emergent_updated_parameters: emergent_parameters,
            clip_scale: clip_scale as f32,
            effective_learning_rate: learning_rate,
            updated_variables,
            updated_parameters,
            core_updated_parameters,
            decoder_updated_parameters,
            muon_variables,
            backward_seconds: 0.0,
            step_seconds: 0.0,
        })
    }

    pub fn moment_rms(&self) -> Result<HashMap<String, (f32, f32)>> {
        let mut output = HashMap::with_capacity(self.variables.len());
        for state in &self.variables {
            let first = state
                .first
                .as_tensor()
                .sqr()?
                .mean_all()?
                .sqrt()?
                .to_scalar::<f32>()?;
            let second = state
                .second
                .as_tensor()
                .sqr()?
                .mean_all()?
                .sqrt()?
                .to_scalar::<f32>()?;
            output.insert(state.name.clone(), (first, second));
        }
        Ok(output)
    }
    pub fn save(
        &self,
        path: &Path,
        world_step: u64,
        checkpoint_id: u64,
        device: &Device,
    ) -> Result<()> {
        let mut tensors = HashMap::with_capacity(self.variables.len() * 2 + 5);
        tensors.insert(
            "optimizer.updates".to_owned(),
            Tensor::new(self.updates as i64, device)?,
        );
        tensors.insert(
            "optimizer.world_step".to_owned(),
            Tensor::new(world_step as i64, device)?,
        );
        tensors.insert(
            "optimizer.checkpoint_id".to_owned(),
            Tensor::new(checkpoint_id as i64, device)?,
        );
        tensors.insert(
            "optimizer.layout_version".to_owned(),
            Tensor::new(OPTIMIZER_LAYOUT_VERSION as i64, device)?,
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

    pub fn load_migrating(
        &mut self,
        path: &Path,
        expected_world_step: u64,
        expected_checkpoint_id: u64,
        new_parameter_names: &[String],
        device: &Device,
    ) -> Result<OptimizerMigrationReport> {
        let tensors = candle_core::safetensors::load(path, device)
            .with_context(|| format!("cannot load optimizer {}", path.display()))?;
        let scalar = |name: &str| -> Result<u64> {
            Ok(tensors
                .get(name)
                .with_context(|| format!("optimizer missing {name}"))?
                .to_scalar::<i64>()? as u64)
        };
        if scalar("optimizer.world_step")? != expected_world_step {
            bail!("optimizer/world step mismatch");
        }
        if scalar("optimizer.checkpoint_id")? != expected_checkpoint_id {
            bail!("optimizer checkpoint transaction does not match manifest");
        }
        if scalar("optimizer.layout_version")? != OPTIMIZER_LAYOUT_VERSION as u64 {
            bail!("optimizer routing/layout version is incompatible");
        }
        if scalar("optimizer.kind")? != self.optimizer_kind as u64 {
            bail!("optimizer kind does not match the requested v9 configuration");
        }
        let updates = scalar("optimizer.updates")?;
        let current_names: HashSet<&str> = self
            .variables
            .iter()
            .map(|state| state.name.as_str())
            .collect();
        let allowed_new: HashSet<&str> = new_parameter_names.iter().map(String::as_str).collect();
        let saved_names: HashSet<&str> = tensors
            .keys()
            .filter_map(|name| name.strip_prefix("optimizer.m."))
            .collect();
        for saved in &saved_names {
            if !current_names.contains(saved) {
                bail!("saved optimizer moment refers to removed or unknown parameter {saved}");
            }
            if !tensors.contains_key(&format!("optimizer.v.{saved}")) {
                bail!("optimizer has only one moment tensor for {saved}");
            }
        }
        for name in tensors
            .keys()
            .filter_map(|name| name.strip_prefix("optimizer.v."))
        {
            if !saved_names.contains(name) {
                bail!("optimizer has a second moment without a first moment for {name}");
            }
        }

        let mut report = OptimizerMigrationReport {
            updates_preserved: updates,
            ..OptimizerMigrationReport::default()
        };
        for state in &self.variables {
            let first_name = format!("optimizer.m.{}", state.name);
            let second_name = format!("optimizer.v.{}", state.name);
            match (tensors.get(&first_name), tensors.get(&second_name)) {
                (Some(_), Some(_)) if allowed_new.contains(state.name.as_str()) => {
                    bail!(
                        "optimizer unexpectedly contains moments for newly grafted parameter {}",
                        state.name
                    );
                }
                (Some(first), Some(second)) => {
                    if first.dims() != state.variable.dims()
                        || second.dims() != state.variable.dims()
                        || first.dtype() != state.variable.dtype()
                        || second.dtype() != state.variable.dtype()
                    {
                        bail!("optimizer moment shape/dtype mismatch for {}", state.name);
                    }
                    ensure_finite_tensor(first, "first optimizer moment", &state.name)?;
                    ensure_finite_tensor(second, "second optimizer moment", &state.name)?;
                    state.first.set(first)?;
                    state.second.set(second)?;
                    report.preserved_moment_pairs += 1;
                    report.preserved_moment_elements += 2 * first.elem_count();
                }
                (None, None) if allowed_new.contains(state.name.as_str()) => {
                    let first_max = state
                        .first
                        .as_tensor()
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?;
                    let second_max = state
                        .second
                        .as_tensor()
                        .abs()?
                        .max_all()?
                        .to_scalar::<f32>()?;
                    if first_max != 0.0 || second_max != 0.0 {
                        bail!("new optimizer moments were not zero for {}", state.name);
                    }
                    report.new_moment_pairs += 1;
                    report.new_moment_elements += 2 * state.variable.elem_count();
                }
                (None, None) => {
                    bail!(
                        "missing optimizer moments for copied parameter {}",
                        state.name
                    );
                }
                _ => bail!("optimizer has an incomplete moment pair for {}", state.name),
            }
        }
        self.updates = updates;
        Ok(report)
    }
}

fn ensure_finite_tensor(tensor: &Tensor, role: &str, name: &str) -> Result<()> {
    let rms = tensor.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    let maximum = tensor.abs()?.max_all()?.to_scalar::<f32>()?;
    if !rms.is_finite() || !maximum.is_finite() {
        bail!("non-finite {role} for {name}");
    }
    Ok(())
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
    let raw_norm = value.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()? as f64;
    if !raw_norm.is_finite() {
        bail!("non-finite Muon direction norm; optimizer update was not applied");
    }
    let norm = raw_norm.max(MUON_EPSILON);
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
    let output = if transposed {
        value.t()?.contiguous()?
    } else {
        value
    };
    let output_rms = output.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
    if !output_rms.is_finite() {
        bail!("non-finite Muon direction after Newton-Schulz; optimizer update was not applied");
    }
    Ok(output)
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

    fn optimizer_test_path(label: &str) -> std::path::PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "titan-image-v9-optimizer-{label}-{}-{nonce}.safetensors",
            std::process::id()
        ))
    }

    fn add_test_linear(vars: &VarMap, name: &str, device: &Device) -> Result<()> {
        let builder = VarBuilder::from_varmap(vars, DType::F32, device).pp(name);
        let _ = candle_nn::linear(2, 2, builder)?;
        Ok(())
    }

    #[test]
    fn optimizer_migration_preserves_old_moments_and_zeros_new_moments() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig::default();
        let old_vars = VarMap::new();
        add_test_linear(&old_vars, "kept", &device)?;
        let mut old = PersistentAdamW::new(&old_vars, &config)?;
        let mut expected = HashMap::new();
        for (index, state) in old.variables.iter().enumerate() {
            let first_value = index as f64 + 1.25;
            let second_value = index as f64 + 11.5;
            let first = Tensor::ones(
                state.variable.shape().clone(),
                state.variable.dtype(),
                &device,
            )?
            .affine(first_value, 0.0)?;
            let second = Tensor::ones(
                state.variable.shape().clone(),
                state.variable.dtype(),
                &device,
            )?
            .affine(second_value, 0.0)?;
            state.first.set(&first)?;
            state.second.set(&second)?;
            expected.insert(
                state.name.clone(),
                (
                    first.flatten_all()?.to_vec1::<f32>()?,
                    second.flatten_all()?.to_vec1::<f32>()?,
                ),
            );
        }
        old.updates = 37;
        let path = optimizer_test_path("roundtrip");
        old.save(&path, 91, 0x1234, &device)?;

        let current_vars = VarMap::new();
        add_test_linear(&current_vars, "kept", &device)?;
        add_test_linear(&current_vars, "grafted", &device)?;
        let mut current = PersistentAdamW::new(&current_vars, &config)?;
        let new_names: Vec<String> = current
            .variables
            .iter()
            .filter(|state| state.name.starts_with("grafted."))
            .map(|state| state.name.clone())
            .collect();
        let report = current.load_migrating(&path, 91, 0x1234, &new_names, &device)?;
        assert_eq!(report.updates_preserved, 37);
        assert_eq!(report.preserved_moment_pairs, expected.len());
        assert_eq!(report.new_moment_pairs, new_names.len());
        for state in &current.variables {
            let first = state.first.as_tensor().flatten_all()?.to_vec1::<f32>()?;
            let second = state.second.as_tensor().flatten_all()?.to_vec1::<f32>()?;
            if let Some((expected_first, expected_second)) = expected.get(&state.name) {
                assert_eq!(
                    &first, expected_first,
                    "first moment changed for {}",
                    state.name
                );
                assert_eq!(
                    &second, expected_second,
                    "second moment changed for {}",
                    state.name
                );
            } else {
                assert!(first.iter().all(|value| *value == 0.0));
                assert!(second.iter().all(|value| *value == 0.0));
            }
        }
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn optimizer_migration_rejects_unknown_and_missing_moments() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig::default();

        let saved_vars = VarMap::new();
        add_test_linear(&saved_vars, "kept", &device)?;
        add_test_linear(&saved_vars, "retired", &device)?;
        let saved = PersistentAdamW::new(&saved_vars, &config)?;
        let unknown_path = optimizer_test_path("unknown");
        saved.save(&unknown_path, 7, 11, &device)?;
        let current_vars = VarMap::new();
        add_test_linear(&current_vars, "kept", &device)?;
        let mut current = PersistentAdamW::new(&current_vars, &config)?;
        let error = current
            .load_migrating(&unknown_path, 7, 11, &[], &device)
            .unwrap_err()
            .to_string();
        assert!(error.contains("removed or unknown parameter"), "{error}");
        std::fs::remove_file(unknown_path)?;

        let complete = PersistentAdamW::new(&current_vars, &config)?;
        let missing_path = optimizer_test_path("missing");
        complete.save(&missing_path, 8, 12, &device)?;
        let mut tensors = candle_core::safetensors::load(&missing_path, &device)?;
        tensors.remove("optimizer.v.kept.bias");
        atomic_safetensors(&tensors, &missing_path)?;
        let mut current = PersistentAdamW::new(&current_vars, &config)?;
        let error = current
            .load_migrating(&missing_path, 8, 12, &[], &device)
            .unwrap_err()
            .to_string();
        assert!(error.contains("only one moment tensor"), "{error}");
        std::fs::remove_file(missing_path)?;
        Ok(())
    }

    #[test]
    fn optimizer_migration_rejects_transaction_mismatch_and_saved_new_moments() -> Result<()> {
        let device = Device::Cpu;
        let config = RunConfig::default();
        let vars = VarMap::new();
        add_test_linear(&vars, "kept", &device)?;
        add_test_linear(&vars, "grafted", &device)?;
        let saved = PersistentAdamW::new(&vars, &config)?;
        let path = optimizer_test_path("transaction");
        saved.save(&path, 13, 21, &device)?;

        let mut current = PersistentAdamW::new(&vars, &config)?;
        let error = current
            .load_migrating(&path, 13, 22, &[], &device)
            .unwrap_err()
            .to_string();
        assert!(error.contains("transaction"), "{error}");

        let declared_new = current
            .variables
            .iter()
            .filter(|state| state.name.starts_with("grafted."))
            .map(|state| state.name.clone())
            .collect::<Vec<_>>();
        let error = current
            .load_migrating(&path, 13, 21, &declared_new, &device)
            .unwrap_err()
            .to_string();
        assert!(error.contains("newly grafted parameter"), "{error}");
        std::fs::remove_file(path)?;
        Ok(())
    }
}
