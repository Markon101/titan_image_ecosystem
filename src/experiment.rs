//! Opt-in training semantics. Defaults preserve the historical v9 signature.
use crate::config::RunConfig;
use crate::tensor_ops::splitmix64;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormTraining {
    #[default]
    Legacy,
    Differentiable,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Withdrawal {
    pub observe: usize,
    pub taper: usize,
    pub autonomous_tail: usize,
    pub min_fidelity: f32,
    pub max_fidelity: f32,
    pub guided_anchor_probability: f32,
}

impl Withdrawal {
    pub fn fidelity(&self, age: u64, episode: u64, seed: u64) -> f32 {
        let anchor =
            (splitmix64(seed ^ episode ^ 0x7769_7468_6472_6177) >> 40) as f32 / (1u32 << 24) as f32;
        if anchor < self.guided_anchor_probability || age < self.observe as u64 {
            return self.max_fidelity;
        }
        let elapsed = age - self.observe as u64;
        if elapsed >= self.taper as u64 {
            return 0.0; // min_fidelity applies to taper, NEVER the autonomous tail.
        }
        self.max_fidelity
            + (self.min_fidelity - self.max_fidelity) * (elapsed + 1) as f32 / self.taper as f32
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TrainingExperiment {
    pub norm: NormTraining,
    pub withdrawal: Option<Withdrawal>,
    pub saturation_penalty: Option<SaturationPenalty>,
    /// Output-only diagnostics; excluded from the learned-evolution signature.
    pub optimizer_diagnostics: bool,
    pub panel: Option<PanelConfig>,
}

impl TrainingExperiment {
    pub fn signature(&self, legacy: u64) -> u64 {
        let base = self.base_signature(legacy);
        let Some(penalty) = self.active_saturation_penalty() else {
            return base;
        };
        let bytes = serde_json::to_vec(&("write-saturation-v1", penalty))
            .expect("finite validated saturation configuration");
        bytes.into_iter().fold(base, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        })
    }

    pub fn active_saturation_penalty(&self) -> Option<&SaturationPenalty> {
        self.saturation_penalty.as_ref().filter(|p| p.weight > 0.0)
    }

    fn base_signature(&self, legacy: u64) -> u64 {
        if self.norm == NormTraining::Legacy && self.withdrawal.is_none() {
            return legacy;
        }
        let bytes = serde_json::to_vec(&("training-experiment-v1", self.norm, &self.withdrawal))
            .expect("finite validated experiment configuration");
        bytes.into_iter().fold(legacy, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
        })
    }

    pub fn validate(&self, config: &RunConfig) -> Result<()> {
        if let Some(p) = &self.saturation_penalty {
            ensure!(
                p.weight.is_finite()
                    && p.weight >= 0.0
                    && p.threshold.is_finite()
                    && p.threshold > 0.0,
                "saturation penalty needs finite nonnegative weight and positive threshold"
            );
        }
        if let Some(p) = &self.panel {
            ensure!(
                p.recovery_horizon <= 2048,
                "recovery horizon exceeds 2048 steps"
            );
            ensure!(
                p.recovery_cases
                    .iter()
                    .enumerate()
                    .all(|(i, c)| !p.recovery_cases[..i].contains(c)),
                "duplicate recovery case"
            );
            ensure!(
                !p.targets.is_empty() && !p.seeds.is_empty() && p.stride > 0 && p.burn_in > 0,
                "panel needs targets, seeds, stride and true burn-in"
            );
            ensure!(
                !p.horizons.is_empty() && p.horizons.iter().all(|h| *h > 0 && *h <= 2048),
                "panel horizons must be in 1..=2048"
            );
            ensure!(
                p.history_capacity > 0
                    && p.history_capacity <= 32
                    && p.targets.len() <= 16
                    && p.seeds.len() <= 8,
                "panel exceeds bounded history/target/seed capacity"
            );
            ensure!(
                p.diagnostic_ages.iter().all(|a| *a <= p.burn_in),
                "diagnostic ages exceed burn-in"
            );
        }
        if let Some(w) = &self.withdrawal {
            ensure!(
                config.episode_reset > 0.0 && !config.variable_age_enabled(),
                "withdrawal v1 requires fixed episodes with resetting age"
            );
            ensure!(
                w.observe
                    .checked_add(w.taper)
                    .and_then(|n| n.checked_add(w.autonomous_tail))
                    == Some(config.episode_steps),
                "withdrawal phase durations must sum to episode_steps"
            );
            ensure!(
                w.autonomous_tail > 0,
                "withdrawal needs a nonempty autonomous tail"
            );
            ensure!(
                (0.0..=1.0).contains(&w.min_fidelity)
                    && (w.min_fidelity..=1.0).contains(&w.max_fidelity)
                    && (0.0..=1.0).contains(&w.guided_anchor_probability),
                "invalid withdrawal fidelity/probability"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn withdrawal_has_real_zero_tail_and_deterministic_anchors() {
        let mut w = Withdrawal {
            observe: 2,
            taper: 2,
            autonomous_tail: 4,
            min_fidelity: 0.2,
            max_fidelity: 1.0,
            guided_anchor_probability: 0.0,
        };
        let values: Vec<_> = (0..8).map(|age| w.fidelity(age, 7, 42)).collect();
        assert_eq!(&values[..2], &[1.0, 1.0]);
        assert!((values[2] - 0.6).abs() < 1e-6);
        assert!((values[3] - 0.2).abs() < 1e-6);
        assert_eq!(&values[4..], &[0.0; 4]);
        w.guided_anchor_probability = 1.0;
        assert_eq!(w.fidelity(7, 7, 42), 1.0);
    }
}

/// Frozen evaluation controls, excluded from training compatibility signatures.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PanelConfig {
    pub targets: Vec<usize>,
    pub seeds: Vec<u64>,
    pub burn_in: usize,
    pub diagnostic_ages: Vec<usize>,
    pub horizons: Vec<usize>,
    pub stride: usize,
    pub recovery_horizon: usize,
    #[serde(default = "default_recovery_cases")]
    pub recovery_cases: Vec<RecoveryCase>,
    pub history_capacity: usize,
    pub clock_robustness: bool,
}

/// Both heads are regularized at each tracked step, including off-cadence macro writes.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SaturationPenalty {
    pub weight: f32,
    pub threshold: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryCase {
    MicroNoise,
    MacroNoise,
    MicroPatch,
    MacroPatch,
    MemoryNoise,
    Combined,
}
impl RecoveryCase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MicroNoise => "micro_noise",
            Self::MacroNoise => "macro_noise",
            Self::MicroPatch => "micro_patch",
            Self::MacroPatch => "macro_patch",
            Self::MemoryNoise => "memory_noise",
            Self::Combined => "combined",
        }
    }
}
pub fn default_recovery_cases() -> Vec<RecoveryCase> {
    use RecoveryCase::*;
    vec![
        MicroNoise,
        MacroNoise,
        MicroPatch,
        MacroPatch,
        MemoryNoise,
        Combined,
    ]
}

#[cfg(test)]
mod compatibility_tests {
    use super::*;
    #[test]
    fn saturation_signature_preserves_v1_and_rejects_invalid_settings() -> Result<()> {
        let mut c = RunConfig::default();
        for norm in [NormTraining::Legacy, NormTraining::Differentiable] {
            c.experiment.norm = norm;
            c.experiment.saturation_penalty = None;
            let old = c.checkpoint_signature();
            let mut expected = 123u64;
            if norm != NormTraining::Legacy {
                for b in serde_json::to_vec(&(
                    "training-experiment-v1",
                    norm,
                    Option::<Withdrawal>::None,
                ))? {
                    expected = (expected ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
                }
            }
            assert_eq!(c.experiment.signature(123), expected);
            c.experiment.saturation_penalty = Some(SaturationPenalty {
                weight: 0.,
                threshold: 2.,
            });
            assert_eq!(old, c.checkpoint_signature());
            c.experiment.saturation_penalty.as_mut().unwrap().weight = 0.01;
            c.validate()?;
            assert_ne!(old, c.checkpoint_signature());
            let active = c.checkpoint_signature();
            c.experiment.saturation_penalty.as_mut().unwrap().threshold = 3.;
            assert_ne!(active, c.checkpoint_signature());
        }
        for weight in [-1., f32::NAN, f32::INFINITY] {
            c.experiment.saturation_penalty = Some(SaturationPenalty {
                weight,
                threshold: 2.,
            });
            assert!(c.validate().is_err());
        }
        for threshold in [0., -1., f32::NAN, f32::INFINITY] {
            c.experiment.saturation_penalty = Some(SaturationPenalty {
                weight: 0.,
                threshold,
            });
            assert!(c.validate().is_err());
        }
        Ok(())
    }
    #[test]
    fn recovery_selection_is_backward_compatible_and_validated() -> Result<()> {
        let old = serde_json::json!({"targets":[0],"seeds":[42],"burn_in":4,
            "diagnostic_ages":[0],"horizons":[4],"stride":1,"recovery_horizon":4,
            "history_capacity":3,"clock_robustness":false});
        let panel: PanelConfig = serde_json::from_value(old.clone())?;
        assert_eq!(panel.recovery_cases, default_recovery_cases());
        let mut invalid = old;
        invalid["recovery_cases"] = serde_json::json!(["unknown"]);
        assert!(serde_json::from_value::<PanelConfig>(invalid).is_err());
        let mut c = RunConfig::default();
        let signature = c.checkpoint_signature();
        c.experiment.panel = Some(panel);
        c.experiment.panel.as_mut().unwrap().recovery_cases = vec![RecoveryCase::MacroPatch];
        c.validate()?;
        assert_eq!(signature, c.checkpoint_signature());
        c.experiment
            .panel
            .as_mut()
            .unwrap()
            .recovery_cases
            .push(RecoveryCase::MacroPatch);
        assert!(c.validate().is_err());
        c.experiment.panel.as_mut().unwrap().recovery_cases.clear();
        c.validate()?;
        c.experiment.panel.as_mut().unwrap().recovery_horizon = 2049;
        assert!(c.validate().is_err());
        Ok(())
    }
}
