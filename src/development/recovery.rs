//! Small paired frozen recovery panel. Both arms start at the exact saved state.
use super::{advance, create, line, metrics, operators::Operators, Options};
use anyhow::{ensure, Result};
use candle_core::Tensor;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{json, Value};
use std::io::Write;
use titan_image::{
    dynamics::DynamicsSystem,
    render::{save_contact_sheet_resized, save_png, ImplicitRenderer, RenderPlan},
    state::WorldState,
    RunConfig,
};

fn damage(world: &WorldState, seed: u64, patch: bool) -> Result<WorldState> {
    let mut out = world.detached();
    let (_, channels, h, w) = world.macro_field.dims4()?;
    let mut data = metrics::values(&world.macro_field)?;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for c in 0..channels {
        for y in 0..h {
            for x in 0..w {
                let v = &mut data[c * h * w + y * w + x];
                if patch {
                    if (h / 3..(2 * h / 3).max(h / 3 + 1)).contains(&y)
                        && (w / 3..(2 * w / 3).max(w / 3 + 1)).contains(&x)
                    {
                        *v = 0.;
                    }
                } else {
                    *v += rng.gen_range(-0.03..0.03);
                }
            }
        }
    }
    out.macro_field = Tensor::from_vec(
        data.into_iter().map(|x| x as f32).collect::<Vec<_>>(),
        world.macro_field.shape(),
        world.macro_field.device(),
    )?;
    Ok(out)
}

fn distance(a: &WorldState, b: &WorldState) -> Result<f64> {
    let mut sum = 0.;
    for (a, b) in [
        (&a.micro, &b.micro),
        (&a.macro_field, &b.macro_field),
        (&a.memory, &b.memory),
    ] {
        let a = metrics::values(a)?;
        sum += metrics::distance(&a, &metrics::values(b)?) / (a.len() as f64).sqrt();
    }
    Ok(sum)
}

fn image_l1(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f64>() / a.len() as f64
}

fn image_stats(data: &[f64], resolution: usize) -> Value {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let mut edges = 0.;
    let mut count = 0;
    for plane in data.chunks_exact(resolution * resolution) {
        for y in 0..resolution {
            for x in 0..resolution {
                let v = plane[y * resolution + x];
                if x + 1 < resolution {
                    edges += (v - plane[y * resolution + x + 1]).powi(2);
                    count += 1;
                }
                if y + 1 < resolution {
                    edges += (v - plane[(y + 1) * resolution + x]).powi(2);
                    count += 1;
                }
            }
        }
    }
    json!({"mean":mean,"variance":variance,"edge_rms":(edges/count as f64).sqrt()})
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    o: &Options,
    config: &RunConfig,
    dynamics: &DynamicsSystem,
    renderer: &ImplicitRenderer,
    initial: &WorldState,
    genome: &Tensor,
) -> Result<Value> {
    let resolution = 96;
    let plan = RenderPlan::new(config, resolution, initial.micro.device())?;
    let noise = damage(initial, o.seed, false)?;
    let patch = damage(initial, o.seed, true)?;
    let initial_distances = [distance(initial, &noise)?, distance(initial, &patch)?];
    ensure!(
        initial_distances.iter().all(|d| *d > 1e-12),
        "damage has zero initial distance"
    );
    let mut arms = [
        (
            "off",
            Operators::default(),
            vec![initial.detached(), noise.detached(), patch.detached()],
        ),
        (
            "fixed",
            Operators {
                diffusion_max: 0.02,
                ..Default::default()
            },
            vec![initial.detached(), noise, patch],
        ),
    ];
    let mut file = create(&o.output.join("recovery.jsonl"))?;
    let stride = (o.steps / 8).max(1);
    let mut maxima = [[0f64; 2]; 2];
    let mut tail_maxima = [[0f64; 2]; 2];
    let mut endpoints = Vec::new();
    let mut frames = Vec::new();
    let mut reference_free_steps = 0;
    for offset in 0..=o.steps {
        let sample = offset % stride == 0 || offset == o.steps;
        let mut off_image = None;
        for (arm_index, (name, operator, states)) in arms.iter_mut().enumerate() {
            let mut ratios = Vec::new();
            for case in 0..2 {
                let ratio = distance(&states[0], &states[case + 1])? / initial_distances[case];
                maxima[arm_index][case] = maxima[arm_index][case].max(ratio);
                if offset >= o.steps * 3 / 4 {
                    tail_maxima[arm_index][case] = tail_maxima[arm_index][case].max(ratio);
                }
                ratios.push(ratio);
            }
            if sample {
                let mut residuals = Vec::new();
                for case in 0..2 {
                    let residual = super::residual::measure(
                        &states[0],
                        &states[case + 1],
                        initial_distances[case],
                        o.low_cutoff,
                        o.mid_cutoff,
                    )?;
                    let measured = residual["state_distance_ratio"].as_f64().unwrap();
                    ensure!(
                        (measured - ratios[case]).abs() <= 1e-12 * ratios[case].abs().max(1.),
                        "residual components disagree with recovery distance"
                    );
                    residuals.push(residual);
                }
                let mut images = Vec::new();
                for (state, case) in states.iter().zip(["control", "macro_noise", "macro_patch"]) {
                    let (_, emergence) = config.developmental_schedule(
                        (state.age as f32 / config.developmental_horizon() as f32).min(1.),
                    );
                    let image = renderer
                        .render_with_emergence(
                            &state.micro,
                            &state.macro_field,
                            genome,
                            &plan,
                            emergence,
                            false,
                        )?
                        .image;
                    let path = o.output.join(format!("{name}_{case}_{offset:05}.png"));
                    save_png(&image, &path)?;
                    if offset == o.steps {
                        frames.push(path);
                    }
                    images.push(metrics::values(&image)?);
                }
                let row = json!({"offset":offset,"age":states[0].age,"world_step":states[0].step,
                    "arm":name,"seed":o.seed,"runtime_references_present":false,"reference_fidelity":0.,
                    "same_clock_sequence":true,"state_distance_ratio":ratios,"residuals":residuals,
                    "render_l1_to_paired_control":[image_l1(&images[0],&images[1]),image_l1(&images[0],&images[2])],
                    "control_image":image_stats(&images[0],resolution),
                    "control_micro":metrics::field(&states[0].micro,o.low_cutoff,o.mid_cutoff)?,
                    "control_render_l1_to_off":off_image.as_ref().map(|x: &Vec<f64>| image_l1(&images[0],x))});
                line(&mut file, &row)?;
                if offset == o.steps {
                    endpoints.push(row);
                }
                if arm_index == 0 {
                    off_image = Some(images.remove(0));
                }
            }
            if offset < o.steps {
                for state in states.iter_mut() {
                    ensure!(
                        state.age == initial.age + offset as u64
                            && state.step == initial.step + offset as u64,
                        "paired recovery clock drift"
                    );
                    *state = advance(dynamics, state, genome, operator)?.0;
                    reference_free_steps += 1;
                }
            }
        }
        if offset > 0 && offset % 32 == 0 {
            eprintln!("paired recovery {offset}/{}", o.steps);
        }
    }
    file.flush()?;
    save_contact_sheet_resized(&frames, &o.output.join("final_contact_sheet.png"), 192)?;
    Ok(
        json!({"mode":"paired_frozen_recovery","steps":o.steps,"seed":o.seed,
        "cases":["macro_noise","macro_patch"],"arms":["off","fixed"],
        "initial_state_distances":initial_distances,"maximum_ratios":maxima,
        "last_quarter_maximum_ratios":tail_maxima,"endpoints":endpoints,
        "reference_free_step_calls":reference_free_steps,"optimizer_updates":0,
        "protocol":{"initialization":"exact saved state, no burn-in; fixed saved genome",
            "noise":"ChaCha8 uniform[-.03,.03), macro only, additive without new clamping",
            "patch":"erase central one-third in each dimension, all macro channels; seed independent",
            "diffusion":"fixed nu=.01; max=.02 and zero logits; same frozen sidecar as stage two",
            "distance":"sum of micro, macro, memory RMS distances, normalized by initial damage distance",
            "tail":"maximum at EVERY step in the final quarter; ratios below .5 indicate sustained paired contraction",
            "render":"96px unmastered RGB, frozen renderer with age-dependent emergence schedule",
            "interpretation":"paired robustness and smoothing diagnostics, not target reconstruction, useful repair, or generalization"}}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    #[test]
    fn damage_is_reproducible_macro_only_and_preserves_clocks() -> Result<()> {
        let c = RunConfig::default();
        let x = WorldState::fresh(&c, 42, &Device::Cpu)?;
        for patch in [false, true] {
            let a = damage(&x, 42, patch)?;
            let b = damage(&x, 42, patch)?;
            assert_eq!(metrics::full(&a)?, metrics::full(&b)?);
            assert_eq!(metrics::values(&a.micro)?, metrics::values(&x.micro)?);
            assert_eq!(metrics::values(&a.memory)?, metrics::values(&x.memory)?);
            assert_eq!((a.age, a.step, a.episode), (x.age, x.step, x.episode));
            assert!(distance(&a, &x)? > 0.);
            assert_eq!(distance(&a, &a)?, 0.);
        }
        assert_ne!(
            metrics::full(&damage(&x, 42, false)?)?,
            metrics::full(&damage(&x, 137, false)?)?
        );
        Ok(())
    }
    #[test]
    fn image_metrics_distinguish_flattening_from_identity() {
        let flat = vec![0.5; 3 * 4 * 4];
        assert_eq!(image_l1(&flat, &flat), 0.);
        assert_eq!(image_stats(&flat, 4)["edge_rms"], 0.);
        let checker: Vec<_> = (0..3 * 4 * 4)
            .map(|i| ((i / 4 + i % 4) % 2) as f64)
            .collect();
        assert!(image_stats(&checker, 4)["edge_rms"].as_f64().unwrap() > 0.);
        assert!(image_l1(&checker, &flat) > 0.);
    }
}
