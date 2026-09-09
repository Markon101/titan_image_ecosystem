//! Scalar diagnostics use f64 host accumulation and do not alter gradients.
use anyhow::Result;
use candle_core::Tensor;
use serde::Serialize;

pub fn group(name: &str) -> &'static str {
    if name.contains("norm.weight") {
        "normalization_scales"
    } else if name.contains(".micro_ca.") {
        "micro_nca"
    } else if name.contains(".macro_ca.") {
        "macro_nca"
    } else if name.contains("reference_") && name.contains("_drive") {
        "reference_drives"
    } else if name.contains(".morphic_") {
        "morphic_blocks"
    } else if name.contains(".gru.") {
        "interface_gru"
    } else if name.starts_with("dynamics.interface.") {
        "interface_attention_feedforward_projections"
    } else if name.starts_with("renderer.grounded.") {
        "grounded_head"
    } else if name.starts_with("renderer.emergent.") {
        "emergent_head"
    } else if name.starts_with("renderer.") {
        "renderer_shared"
    } else {
        "flow_or_other"
    }
}
#[derive(Clone, Debug, Default, Serialize)]
pub struct GroupStats {
    pub parameters: usize,
    pub gradient_reachable_parameters: usize,
    pub nonzero_gradient_parameters: usize,
    pub gradient_l2: f64,
    pub gradient_rms: f64,
    pub update_rms: f64,
    pub parameter_rms: f64,
    pub update_parameter_ratio: Option<f64>,
    pub clip_energy_fraction: f64,
    pub nonzero_gradient_fraction: f64,
    #[serde(skip)]
    pub gradient_energy: f64,
    #[serde(skip)]
    pub parameter_energy: f64,
    #[serde(skip)]
    pub update_energy: f64,
}
impl GroupStats {
    pub fn observe(&mut self, parameter: &Tensor, gradient: Option<&Tensor>) -> Result<()> {
        self.parameters += parameter.elem_count();
        self.parameter_energy += energy(parameter)?;
        if let Some(g) = gradient {
            let v = g.flatten_all()?.to_vec1::<f32>()?;
            self.gradient_reachable_parameters += v.len();
            self.nonzero_gradient_parameters += v.iter().filter(|x| **x != 0.0).count();
            self.gradient_energy += v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
        }
        Ok(())
    }
    pub fn finish(&mut self, total_gradient_energy: f64) {
        let n = self.parameters.max(1) as f64;
        self.gradient_l2 = self.gradient_energy.sqrt();
        self.gradient_rms = (self.gradient_energy / n).sqrt();
        self.parameter_rms = (self.parameter_energy / n).sqrt();
        self.update_rms = (self.update_energy / n).sqrt();
        self.update_parameter_ratio = (self.parameter_energy > 0.0)
            .then(|| (self.update_energy / self.parameter_energy).sqrt());
        self.clip_energy_fraction = self.gradient_energy / total_gradient_energy.max(1e-30);
        self.nonzero_gradient_fraction = self.nonzero_gradient_parameters as f64 / n;
    }
}
pub fn energy(t: &Tensor) -> Result<f64> {
    Ok(t.flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|x| f64::from(*x).powi(2))
        .sum())
}
