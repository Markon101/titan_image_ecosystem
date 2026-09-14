//! Frozen sidecar projections: no new checkpoint parameters or optimizer state.
use super::metrics::{cancellation, difference, full, norm, values};
use anyhow::{ensure, Result};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use titan_image::state::WorldState;

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Projection {
    pub bias: f64,
    /// Sparse (input_channel, weight) coefficients shared over space and both grids.
    pub terms: Vec<(usize, f64)>,
}
impl Projection {
    fn validate(&self, channels: usize) -> Result<()> {
        ensure!(
            self.bias.is_finite()
                && self
                    .terms
                    .iter()
                    .all(|(c, w)| *c < channels && w.is_finite()),
            "invalid projection coefficient/channel"
        );
        Ok(())
    }
    fn at(&self, x: &[f64], plane: usize, i: usize) -> Result<f64> {
        let v = self.bias
            + self
                .terms
                .iter()
                .map(|(c, w)| x[c * plane + i] * w)
                .sum::<f64>();
        ensure!(v.is_finite(), "non-finite operator projection");
        Ok(v)
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffusionScope {
    #[default]
    Both,
    Micro,
    Macro,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Operators {
    pub transport_max: f64,
    pub diffusion_max: f64,
    pub diffusion_scope: DiffusionScope,
    pub velocity_x: Projection,
    pub velocity_y: Projection,
    pub diffusivity: Projection,
}
impl Operators {
    pub fn validate(&self, channels: usize) -> Result<()> {
        ensure!(
            self.transport_max.is_finite()
                && self.diffusion_max.is_finite()
                && self.transport_max >= 0.
                && self.diffusion_max >= 0.,
            "operator bounds must be nonnegative and finite"
        );
        ensure!(
            2. * self.transport_max + 4. * self.diffusion_max <= 1.,
            "unit-step stencil CFL requires 2*transport_max + 4*diffusion_max <= 1"
        );
        for p in [&self.velocity_x, &self.velocity_y, &self.diffusivity] {
            p.validate(channels)?;
        }
        Ok(())
    }
    pub fn enabled(&self) -> Vec<&'static str> {
        let mut v = vec!["legacy_G"];
        if self.transport_max > 0. {
            v.push("transport");
        }
        if self.diffusion_max > 0. {
            v.push("diffusion");
        }
        v
    }
    fn terms(
        &self,
        x: &[f64],
        c: usize,
        h: usize,
        w: usize,
        diffuse: bool,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = h * w;
        let mut t = vec![0.; x.len()];
        let mut d = t.clone();
        for y in 0..h {
            for a in 0..w {
                let i = y * w + a;
                let ax = self.transport_max * self.velocity_x.at(x, n, i)?.tanh();
                let ay = self.transport_max * self.velocity_y.at(x, n, i)?.tanh();
                let nu = if diffuse {
                    self.diffusion_max * (0.5 + 0.5 * (self.diffusivity.at(x, n, i)? / 2.).tanh())
                } else {
                    0.
                };
                for k in 0..c {
                    let at = |y: usize, a: usize| x[k * n + y * w + a];
                    let center = at(y, a);
                    let (left, right, up, down) = (
                        at(y, (a + w - 1) % w),
                        at(y, (a + 1) % w),
                        at((y + h - 1) % h, a),
                        at((y + 1) % h, a),
                    );
                    let dx = if ax >= 0. {
                        center - left
                    } else {
                        right - center
                    };
                    let dy = if ay >= 0. { center - up } else { down - center };
                    t[k * n + i] = -ax * dx - ay * dy;
                    d[k * n + i] = nu * (left + right + up + down - 4. * center);
                }
            }
        }
        Ok((t, d))
    }
    /// The sidecar correction is evaluated at x and added after the legacy map.
    pub fn apply(
        &self,
        x: &WorldState,
        mut next: WorldState,
        macro_updated: bool,
    ) -> Result<(WorldState, Value)> {
        let before = full(x)?;
        let legacy = full(&next)?;
        let r = difference(&legacy, &before);
        let mut t = vec![0.; before.len()];
        let mut d = t.clone();
        if self.transport_max > 0. || self.diffusion_max > 0. {
            let mut cursor = 0;
            for (source, dest, active, diffuse) in [
                (
                    &x.micro,
                    &mut next.micro,
                    true,
                    self.diffusion_scope != DiffusionScope::Macro,
                ),
                (
                    &x.macro_field,
                    &mut next.macro_field,
                    macro_updated,
                    self.diffusion_scope != DiffusionScope::Micro,
                ),
            ] {
                let (_, c, h, w) = source.dims4()?;
                if active && (self.transport_max > 0. || (diffuse && self.diffusion_max > 0.)) {
                    let (ft, fd) = self.terms(&values(source)?, c, h, w, diffuse)?;
                    let base = values(dest)?;
                    let v: Vec<f32> = (0..base.len())
                        .map(|i| (base[i] + ft[i] + fd[i]) as f32)
                        .collect();
                    ensure!(
                        v.iter().all(|v| v.is_finite()),
                        "non-finite corrected state; stopped without clamping"
                    );
                    *dest = Tensor::from_vec(v, source.shape(), source.device())?;
                    t[cursor..cursor + ft.len()].copy_from_slice(&ft);
                    d[cursor..cursor + fd.len()].copy_from_slice(&fd);
                }
                cursor += source.elem_count();
            }
        }
        let mut report = cancellation(&r, &t, &d, &vec![0.; r.len()]);
        let actual = difference(&full(&next)?, &before);
        let residual: Vec<_> = (0..r.len())
            .map(|i| actual[i] - r[i] - t[i] - d[i])
            .collect();
        report["f32_composition_residual_l2"] = json!(norm(&residual));
        report["actual_update_l2"] = json!(norm(&actual));
        Ok((next, report))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use titan_image::RunConfig;
    #[test]
    fn constant_fields_and_diffusion_checkerboard() -> Result<()> {
        let o = Operators {
            diffusion_max: 0.2,
            transport_max: 0.1,
            velocity_x: Projection {
                bias: 1.,
                terms: vec![],
            },
            ..Default::default()
        };
        o.validate(1)?;
        let (t, d) = o.terms(&[3.; 16], 1, 4, 4, true)?;
        assert_eq!(norm(&t) + norm(&d), 0.);
        let x: Vec<_> = (0..16)
            .map(|i| if (i / 4 + i % 4) % 2 == 0 { 1. } else { -1. })
            .collect();
        let diffusion = Operators {
            diffusion_max: 0.2,
            ..Default::default()
        };
        let (_, d) = diffusion.terms(&x, 1, 4, 4, true)?;
        let y: Vec<_> = x.iter().zip(&d).map(|(x, d)| x + d).collect();
        assert!((norm(&y) / norm(&x) - 0.2).abs() < 1e-12);
        assert!(d.iter().sum::<f64>().abs() < 1e-12);
        Ok(())
    }
    #[test]
    fn upwind_preserves_bounds_and_disabled_path_preserves_bits() -> Result<()> {
        let o = Operators {
            transport_max: 0.5,
            velocity_x: Projection {
                bias: 1.,
                terms: vec![],
            },
            ..Default::default()
        };
        let x = [0., 1., 0., 0.];
        let (t, _) = o.terms(&x, 1, 1, 4, true)?;
        let y: Vec<_> = x.iter().zip(&t).map(|(x, t)| x + t).collect();
        assert!(y.iter().all(|v| *v >= 0. && *v <= 1.));
        assert!((y.iter().sum::<f64>() - 1.).abs() < 1e-12);
        let config = RunConfig::default();
        let x = WorldState::fresh(&config, 42, &Device::Cpu)?;
        let mut next = x.detached();
        next.micro = next.micro.affine(1., 0.01)?;
        let original = full(&next)?;
        let (off, report) = Operators::default().apply(&x, next, false)?;
        assert_eq!(full(&off)?, original);
        assert_eq!(report["f32_composition_residual_l2"], 0.);
        Ok(())
    }
    #[test]
    fn grid_selection_preserves_excluded_fields_and_macro_cadence() -> Result<()> {
        let mut x = WorldState::fresh(&RunConfig::default(), 42, &Device::Cpu)?;
        let checker: Vec<f32> = (0..16)
            .map(|i| if (i / 4 + i % 4) % 2 == 0 { 1. } else { -1. })
            .collect();
        x.micro = Tensor::from_vec(checker.clone(), (1, 1, 4, 4), &Device::Cpu)?;
        x.macro_field = x.micro.clone();
        for scope in [
            DiffusionScope::Both,
            DiffusionScope::Micro,
            DiffusionScope::Macro,
        ] {
            let o = Operators {
                diffusion_max: 0.02,
                diffusion_scope: scope,
                ..Default::default()
            };
            for macro_updated in [false, true] {
                let mut next = x.detached();
                // Signed zero catches unintended re-encoding of excluded fields.
                next.micro = Tensor::from_vec(vec![-0.0f32; 16], (1, 1, 4, 4), &Device::Cpu)?;
                next.macro_field = next.micro.clone();
                let (out, _) = o.apply(&x, next, macro_updated)?;
                for (field, enabled) in [
                    (&out.micro, scope != DiffusionScope::Macro),
                    (
                        &out.macro_field,
                        macro_updated && scope != DiffusionScope::Micro,
                    ),
                ] {
                    for (v, original) in field.flatten_all()?.to_vec1::<f32>()?.iter().zip(&checker)
                    {
                        if enabled {
                            assert!((*v + 0.08 * original).abs() < 1e-7);
                        } else {
                            assert_eq!(v.to_bits(), (-0.0f32).to_bits());
                        }
                    }
                }
                assert_eq!(values(&out.memory)?, values(&x.memory)?);
                assert_eq!((out.age, out.step), (x.age, x.step));
            }
        }
        Ok(())
    }

    #[test]
    fn scope_defaults_are_compatible_and_do_not_scope_transport() -> Result<()> {
        let legacy: Operators = serde_json::from_str(r#"{"diffusion_max":0.02}"#)?;
        let explicit: Operators =
            serde_json::from_str(r#"{"diffusion_max":0.02,"diffusion_scope":"both"}"#)?;
        assert_eq!(legacy.diffusion_scope, DiffusionScope::Both);
        let x = WorldState::fresh(&RunConfig::default(), 42, &Device::Cpu)?;
        assert_eq!(
            full(&legacy.apply(&x, x.detached(), true)?.0)?,
            full(&explicit.apply(&x, x.detached(), true)?.0)?
        );
        let transport = Operators {
            transport_max: 0.1,
            velocity_x: Projection {
                bias: 1.,
                terms: vec![],
            },
            ..Default::default()
        };
        let baseline = full(&transport.apply(&x, x.detached(), true)?.0)?;
        for scope in [DiffusionScope::Micro, DiffusionScope::Macro] {
            let scoped = Operators {
                diffusion_scope: scope,
                ..transport.clone()
            };
            assert_eq!(full(&scoped.apply(&x, x.detached(), true)?.0)?, baseline);
        }
        assert!(serde_json::from_str::<Operators>(r#"{"diffusion_scope":"unknown"}"#).is_err());
        Ok(())
    }

    #[test]
    fn invalid_coefficients_and_cfl_rejected() {
        assert!(Operators {
            transport_max: 0.6,
            ..Default::default()
        }
        .validate(2)
        .is_err());
        assert!(Operators {
            diffusion_max: -1.,
            ..Default::default()
        }
        .validate(2)
        .is_err());
        assert!(Projection {
            bias: 0.,
            terms: vec![(2, 1.)]
        }
        .validate(2)
        .is_err());
        assert!(serde_json::from_str::<Operators>("{\"difusion_max\":0.1}").is_err());
    }
}
