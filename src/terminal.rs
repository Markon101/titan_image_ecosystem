use crate::config::{ResearchPreset, TerminalMode};
use crate::metrics::MetricRecord;
use std::collections::VecDeque;
use std::io::IsTerminal;

pub struct TerminalReporter {
    mode: TerminalMode,
    ansi: bool,
    preset: ResearchPreset,
    grounding: VecDeque<f32>,
    emergence: VecDeque<f32>,
    movement: VecDeque<f32>,
    stability: VecDeque<f32>,
}

impl TerminalReporter {
    pub fn new(mode: TerminalMode, preset: ResearchPreset) -> Self {
        Self {
            mode,
            ansi: std::io::stdout().is_terminal(),
            preset,
            grounding: VecDeque::with_capacity(32),
            emergence: VecDeque::with_capacity(32),
            movement: VecDeque::with_capacity(32),
            stability: VecDeque::with_capacity(32),
        }
    }

    pub fn event(&self, kind: &str, message: impl AsRef<str>) {
        if self.mode == TerminalMode::Quiet {
            return;
        }
        if self.ansi {
            println!("\x1b[1;36m[{kind}]\x1b[0m {}", message.as_ref());
        } else {
            println!("[{kind}] {}", message.as_ref());
        }
    }

    pub fn training(&mut self, record: &MetricRecord) {
        if self.mode == TerminalMode::Quiet {
            return;
        }
        push(&mut self.grounding, record.loss_grounding);
        push(&mut self.emergence, record.emergent_contribution_rms);
        push(
            &mut self.movement,
            record.micro_movement_mean + record.macro_movement_mean,
        );
        push(
            &mut self.stability,
            1.0 - record.micro_clamp_fraction.max(record.macro_clamp_fraction),
        );
        let update = if record.core_trained {
            "CORE"
        } else {
            "DECODE"
        };
        let view = if record.supervision == "crop" {
            format!("CROP {:.1}x", record.detail_zoom)
        } else {
            "GLOBAL".to_owned()
        };
        match self.mode {
            TerminalMode::Compact => println!(
                "{:06} | ep{:03} a{:02} t{:02} | {:6} | {} | morph {}/{} | L {:.4} G {:.4} E {:.4} S {:.4} | move {:.4}/{:.4} mem {:.3} | clip {:.3} | {:.2} step/s",
                record.step,
                record.episode,
                record.age,
                record.target_index,
                update,
                view,
                record.active_morph_depth,
                record.physical_morph_layers,
                record.loss_total,
                record.loss_grounding,
                record.emergent_contribution_rms,
                record.loss_structure,
                record.micro_movement_mean,
                record.macro_movement_mean,
                record.interface_memory_rms,
                record.gradient_clip_scale,
                record.development_steps_per_second,
            ),
            TerminalMode::Rich => {
                println!(
                    "{:06} | {:?} | ep{:03} age{:02} target{:02} | {:6} {} | morph {}/{} g{} | total {:.5} endpoint {:.5} ground {:.5} emerg {:.5} | ref {:.3}/{:.3} state {:.3}/{:.3} mem {:.3} | grad c/d/f {:.5}/{:.5}/{:.5} clip {:.3} lr {:.2e} | {:.2} step/s",
                    record.step,
                    self.preset,
                    record.episode,
                    record.age,
                    record.target_index,
                    update,
                    view,
                    record.active_morph_depth,
                    record.physical_morph_layers,
                    record.morph_generation,
                    record.loss_total,
                    record.loss_endpoint,
                    record.loss_grounding,
                    record.emergent_contribution_rms,
                    record.micro_reference_drive_rms,
                    record.macro_reference_drive_rms,
                    record.micro_state_rms,
                    record.macro_state_rms,
                    record.interface_memory_rms,
                    record.core_gradient_rms,
                    record.decoder_gradient_rms,
                    record.flow_gradient_rms,
                    record.gradient_clip_scale,
                    record.effective_learning_rate,
                    record.development_steps_per_second,
                );
                println!(
                    " tape G:{} E:{} M:{} stable:{}",
                    sparkline(&self.grounding),
                    sparkline(&self.emergence),
                    sparkline(&self.movement),
                    sparkline(&self.stability),
                );
            }
            TerminalMode::Quiet => {}
        }
    }
}

fn push(values: &mut VecDeque<f32>, value: f32) {
    if values.len() == 32 {
        values.pop_front();
    }
    values.push_back(value);
}

fn sparkline(values: &VecDeque<f32>) -> String {
    const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() {
        return "-".to_owned();
    }
    let minimum = values.iter().copied().fold(f32::INFINITY, f32::min);
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (maximum - minimum).max(1e-8);
    values
        .iter()
        .map(|value| {
            let index = (((value - minimum) / range) * 7.0).round() as usize;
            BARS[index.min(7)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkline_is_bounded_and_deterministic() {
        let values = VecDeque::from(vec![0.0f32, 0.5, 1.0]);
        assert_eq!(sparkline(&values), sparkline(&values));
        assert_eq!(sparkline(&values).chars().count(), 3);
    }
}
