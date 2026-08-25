use crate::metrics::{ImageDiagnostics, MetricRecord, StateDiagnostics};
use crate::optimizer::PersistentAdamW;
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarMap;
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

#[derive(Clone, Debug, Serialize)]
pub struct AnatomySnapshot {
    pub physical_layers: usize,
    pub active_depth: usize,
    pub graft_generation: u64,
    pub block_birth_generations: Vec<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct GraftEvent {
    pub event_type: &'static str,
    pub schema_version: u32,
    pub event_id: String,
    pub world_step: u64,
    pub checkpoint_id_before: u64,
    pub checkpoint_id_after: u64,
    pub old_anatomy: AnatomySnapshot,
    pub new_anatomy: AnatomySnapshot,
    pub copied_tensor_count: usize,
    pub copied_parameter_count: usize,
    pub new_tensor_count: usize,
    pub new_parameter_count: usize,
    pub resized_tensors: Vec<String>,
    pub skipped_tensors: Vec<String>,
    pub optimizer_moments_preserved: usize,
    pub optimizer_moments_new: usize,
    pub optimizer_updates_preserved: u64,
    pub pre_graft_loss_total: f32,
    pub pre_graft_loss_grounding: f32,
    pub pre_graft_micro: StateSnapshot,
    pub pre_graft_macro: StateSnapshot,
    pub pre_graft_memory_rms: f32,
    pub pre_graft_output_fingerprint: String,
    pub pre_graft_image: ImageSnapshot,
    pub immediate_output_l1: f32,
    pub immediate_memory_rms_delta: f32,
    pub committed: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct MorphActivationEvent {
    pub event_type: &'static str,
    pub schema_version: u32,
    pub world_step: u64,
    pub old_active_depth: usize,
    pub new_active_depth: usize,
    pub physical_layers: usize,
    pub plateau_improvement: f32,
    pub seam_before: f32,
    pub function_preservation_l1: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct StateSnapshot {
    pub rms: f32,
    pub mean_abs: f32,
    pub near_bound_fraction: f32,
    pub channel_rms_min: f32,
    pub channel_rms_max: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ImageSnapshot {
    pub spatial_variance: f32,
    pub seam: f32,
    pub edge: f32,
    pub means: [f32; 3],
    pub variances: [f32; 3],
}

#[derive(Clone, Debug, Default)]
struct TargetAccumulator {
    name: String,
    windows: usize,
    flow_windows: usize,
    clipped_windows: usize,
    global_windows: usize,
    detail_windows: usize,
    late_windows: usize,
    total: f64,
    grounding: f64,
    content: f64,
    structure: f64,
    palette: f64,
    flow: f64,
    gradient: f64,
    movement: f64,
    memory: f64,
    global_grounding: f64,
    detail_grounding: f64,
    late_total: f64,
    late_grounding: f64,
    late_movement: f64,
    first_grounding: Option<f32>,
    last_grounding: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct TargetStatistics {
    pub target_index: usize,
    pub target_name: String,
    pub windows: usize,
    pub global_windows: usize,
    pub detail_windows: usize,
    pub mean_total_loss: f32,
    pub mean_grounding_loss: f32,
    pub mean_content_loss: f32,
    pub mean_structure_loss: f32,
    pub mean_palette_loss: f32,
    pub mean_flow_loss: Option<f32>,
    pub mean_gradient_demand: f32,
    pub clip_rate: f32,
    pub grounding_adaptation_per_window: f32,
    pub mean_movement: f32,
    pub mean_memory_rms: f32,
    pub mean_global_grounding: Option<f32>,
    pub mean_detail_grounding: Option<f32>,
    pub late_age_windows: usize,
    pub late_age_total_loss: Option<f32>,
    pub late_age_grounding_loss: Option<f32>,
    pub late_age_movement: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct TargetStatisticsReport {
    pub schema_version: u32,
    pub invocation_start_step: u64,
    pub world_step: u64,
    pub targets: Vec<TargetStatistics>,
}

#[derive(Default)]
pub struct TargetTelemetry {
    invocation_start_step: u64,
    targets: BTreeMap<usize, TargetAccumulator>,
}

impl TargetTelemetry {
    pub fn new(invocation_start_step: u64) -> Self {
        Self {
            invocation_start_step,
            targets: BTreeMap::new(),
        }
    }

    pub fn observe(&mut self, target_name: &str, episode_steps: usize, record: &MetricRecord) {
        let target = self.targets.entry(record.target_index).or_default();
        if target.name.is_empty() {
            target.name = target_name.to_owned();
        }
        target.windows += 1;
        target.total += record.loss_total as f64;
        target.grounding += record.loss_grounding as f64;
        target.content += record.loss_content as f64;
        target.structure += record.loss_structure as f64;
        target.palette += record.loss_palette as f64;
        target.gradient += record.gradient_rms as f64;
        target.movement += (record.micro_movement_mean + record.macro_movement_mean) as f64;
        target.memory += record.interface_memory_rms as f64;
        target.clipped_windows += usize::from(record.gradient_clip_scale < 0.999_999);
        if record.flow_active {
            target.flow_windows += 1;
            target.flow += record.flow_loss as f64;
        }
        if record.supervision == "crop" {
            target.detail_windows += 1;
            target.detail_grounding += record.loss_grounding as f64;
        } else {
            target.global_windows += 1;
            target.global_grounding += record.loss_grounding as f64;
        }
        if record.age as usize >= episode_steps.saturating_mul(3) / 4 {
            target.late_windows += 1;
            target.late_total += record.loss_total as f64;
            target.late_grounding += record.loss_grounding as f64;
            target.late_movement +=
                (record.micro_movement_mean + record.macro_movement_mean) as f64;
        }
        target.first_grounding.get_or_insert(record.loss_grounding);
        target.last_grounding = record.loss_grounding;
    }

    pub fn report(&self, world_step: u64) -> TargetStatisticsReport {
        let targets = self
            .targets
            .iter()
            .map(|(target_index, target)| {
                let divisor = target.windows.max(1) as f64;
                let optional_mean = |sum: f64, count: usize| {
                    (count > 0).then_some((sum / count.max(1) as f64) as f32)
                };
                TargetStatistics {
                    target_index: *target_index,
                    target_name: target.name.clone(),
                    windows: target.windows,
                    global_windows: target.global_windows,
                    detail_windows: target.detail_windows,
                    mean_total_loss: (target.total / divisor) as f32,
                    mean_grounding_loss: (target.grounding / divisor) as f32,
                    mean_content_loss: (target.content / divisor) as f32,
                    mean_structure_loss: (target.structure / divisor) as f32,
                    mean_palette_loss: (target.palette / divisor) as f32,
                    mean_flow_loss: optional_mean(target.flow, target.flow_windows),
                    mean_gradient_demand: (target.gradient / divisor) as f32,
                    clip_rate: target.clipped_windows as f32 / target.windows.max(1) as f32,
                    grounding_adaptation_per_window: target.first_grounding.map_or(0.0, |first| {
                        (first - target.last_grounding) / target.windows.max(1) as f32
                    }),
                    mean_movement: (target.movement / divisor) as f32,
                    mean_memory_rms: (target.memory / divisor) as f32,
                    mean_global_grounding: optional_mean(
                        target.global_grounding,
                        target.global_windows,
                    ),
                    mean_detail_grounding: optional_mean(
                        target.detail_grounding,
                        target.detail_windows,
                    ),
                    late_age_windows: target.late_windows,
                    late_age_total_loss: optional_mean(target.late_total, target.late_windows),
                    late_age_grounding_loss: optional_mean(
                        target.late_grounding,
                        target.late_windows,
                    ),
                    late_age_movement: optional_mean(target.late_movement, target.late_windows),
                }
            })
            .collect();
        TargetStatisticsReport {
            schema_version: crate::config::SCHEMA_VERSION,
            invocation_start_step: self.invocation_start_step,
            world_step,
            targets,
        }
    }
}

impl From<&StateDiagnostics> for StateSnapshot {
    fn from(value: &StateDiagnostics) -> Self {
        Self {
            rms: value.rms,
            mean_abs: value.mean_abs,
            near_bound_fraction: value.clamp_fraction,
            channel_rms_min: value.channel_rms_min,
            channel_rms_max: value.channel_rms_max,
        }
    }
}

impl From<&ImageDiagnostics> for ImageSnapshot {
    fn from(value: &ImageDiagnostics) -> Self {
        Self {
            spatial_variance: value.variance,
            seam: value.seam,
            edge: value.edge,
            means: value.means,
            variances: value.variances,
        }
    }
}

pub fn append_jsonl<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value)?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}

pub fn tensor_fingerprint(tensor: &Tensor) -> Result<String> {
    let values = tensor.detach().flatten_all()?.to_vec1::<f32>()?;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let stride = (values.len() / 4096).max(1);
    for value in values.iter().step_by(stride) {
        let quantized = (value.clamp(-16.0, 16.0) * 4096.0).round() as i32;
        for byte in quantized.to_le_bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

#[derive(Clone, Debug, Serialize)]
pub struct ParameterTensorStatistics {
    pub name: String,
    pub subsystem: String,
    pub parameter_count: usize,
    pub active: bool,
    pub birth_generation: u64,
    pub weight_rms: f32,
    pub weight_max_abs: f32,
    pub zero_fraction: f32,
    pub row_energy_participation_ratio: Option<f32>,
    pub first_moment_rms: f32,
    pub second_moment_rms: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct SubsystemStatistics {
    pub subsystem: String,
    pub allocated_parameters: usize,
    pub active_parameters: usize,
    pub tensor_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelStatistics {
    pub world_step: u64,
    pub total_parameters: usize,
    pub active_parameters: usize,
    pub inactive_reserve_parameters: usize,
    pub physical_morph_layers: usize,
    pub active_morph_depth: usize,
    pub tensors: Vec<ParameterTensorStatistics>,
    pub subsystems: Vec<SubsystemStatistics>,
}

pub fn collect_model_statistics(
    varmap: &VarMap,
    optimizer: &PersistentAdamW,
    world_step: u64,
    active_morph_depth: usize,
    morph_birth_generations: &[u64],
    flow_active: bool,
) -> Result<ModelStatistics> {
    let moments = optimizer.moment_rms()?;
    let data = varmap.data().lock().expect("VarMap mutex poisoned");
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    let mut tensors = Vec::with_capacity(names.len());
    let mut subsystem_counts: HashMap<String, (usize, usize, usize)> = HashMap::new();
    let mut total_parameters = 0usize;
    let mut active_parameters = 0usize;
    for name in names {
        let variable = &data[name];
        let tensor = variable.as_tensor();
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        let count = values.len();
        let zero_fraction =
            values.iter().filter(|value| **value == 0.0).count() as f32 / count.max(1) as f32;
        let weight_rms = tensor.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
        let weight_max_abs = tensor.abs()?.max_all()?.to_scalar::<f32>()?;
        let morph_index = morphic_index(name);
        let active = morph_index.is_none_or(|index| index < active_morph_depth)
            && (flow_active || !name.starts_with("flow."));
        let birth_generation = morph_index
            .and_then(|index| morph_birth_generations.get(index).copied())
            .unwrap_or(0);
        let participation = if tensor.dims().len() == 2 {
            let energies = tensor.sqr()?.sum(1)?.to_vec1::<f32>()?;
            let sum = energies.iter().sum::<f32>();
            let squared = energies.iter().map(|value| value * value).sum::<f32>();
            Some(sum * sum / squared.max(1e-20))
        } else {
            None
        };
        let subsystem = parameter_subsystem(name);
        let entry = subsystem_counts
            .entry(subsystem.clone())
            .or_insert((0, 0, 0));
        entry.0 += count;
        entry.1 += usize::from(active) * count;
        entry.2 += 1;
        total_parameters += count;
        active_parameters += usize::from(active) * count;
        let (first_moment_rms, second_moment_rms) =
            moments.get(name).copied().unwrap_or((0.0, 0.0));
        tensors.push(ParameterTensorStatistics {
            name: name.clone(),
            subsystem,
            parameter_count: count,
            active,
            birth_generation,
            weight_rms,
            weight_max_abs,
            zero_fraction,
            row_energy_participation_ratio: participation,
            first_moment_rms,
            second_moment_rms,
        });
    }
    let mut subsystems: Vec<SubsystemStatistics> = subsystem_counts
        .into_iter()
        .map(
            |(subsystem, (allocated_parameters, active_parameters, tensor_count))| {
                SubsystemStatistics {
                    subsystem,
                    allocated_parameters,
                    active_parameters,
                    tensor_count,
                }
            },
        )
        .collect();
    subsystems.sort_by(|a, b| a.subsystem.cmp(&b.subsystem));
    Ok(ModelStatistics {
        world_step,
        total_parameters,
        active_parameters,
        inactive_reserve_parameters: total_parameters - active_parameters,
        physical_morph_layers: morph_birth_generations.len(),
        active_morph_depth,
        tensors,
        subsystems,
    })
}

fn parameter_subsystem(name: &str) -> String {
    if let Some(index) = morphic_index(name) {
        return format!("morph_block_{index:03}");
    }
    for (needle, label) in [
        ("dynamics.micro_ca.", "micro_nca"),
        ("dynamics.macro_ca.", "macro_nca"),
        ("dynamics.reference_micro_drive.", "reference_micro_drive"),
        ("dynamics.reference_macro_drive.", "reference_macro_drive"),
        ("dynamics.interface.gru.", "interface_gru"),
        ("dynamics.interface.", "recurrent_interface"),
        ("renderer.grounded.", "grounded_head"),
        ("renderer.emergent.", "emergent_head"),
        ("renderer.", "renderer_shared"),
        ("flow.", "flow_renderer"),
    ] {
        if name.starts_with(needle) {
            return label.to_owned();
        }
    }
    "other".to_owned()
}

fn morphic_index(name: &str) -> Option<usize> {
    let marker = ".morphic_";
    let start = name.find(marker)? + marker.len();
    name[start..].split('.').next()?.parse().ok()
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn output_fingerprint_is_deterministic_and_sensitive() -> Result<()> {
        let a = Tensor::new(&[0.0f32, 1.0, 2.0], &Device::Cpu)?;
        let b = Tensor::new(&[0.0f32, 1.0, 2.1], &Device::Cpu)?;
        assert_eq!(tensor_fingerprint(&a)?, tensor_fingerprint(&a)?);
        assert_ne!(tensor_fingerprint(&a)?, tensor_fingerprint(&b)?);
        Ok(())
    }
}
