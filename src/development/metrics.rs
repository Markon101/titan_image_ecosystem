use super::spectral::Spectrum;
use anyhow::{ensure, Result};
use candle_core::Tensor;
use serde_json::{json, Value};
use titan_image::state::WorldState;

pub fn values(t: &Tensor) -> Result<Vec<f64>> {
    let v: Vec<_> = t
        .flatten_all()?
        .to_vec1::<f32>()?
        .into_iter()
        .map(f64::from)
        .collect();
    ensure!(
        v.iter().all(|x| x.is_finite()),
        "non-finite state: stopping without clamping"
    );
    Ok(v)
}
pub fn full(w: &WorldState) -> Result<Vec<f64>> {
    let mut x = values(&w.micro)?;
    x.extend(values(&w.macro_field)?);
    x.extend(values(&w.memory)?);
    Ok(x)
}
pub fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
pub fn distance(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}
pub fn difference(x: &[f64], y: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), y.len());
    x.iter().zip(y).map(|(a, b)| a - b).collect()
}
fn variance(x: &[f64]) -> f64 {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64
}
pub fn field(t: &Tensor, low: f64, mid: f64) -> Result<Value> {
    let (b, c, h, w) = t.dims4()?;
    ensure!(b == 1, "expected batch one");
    let x = values(t)?;
    let e = norm(&x).powi(2);
    let mut spatial_gradient_energy = 0.;
    for k in 0..c {
        for y in 0..h {
            for a in 0..w {
                let at = |y: usize, a: usize| x[k * h * w + y * w + a];
                spatial_gradient_energy += ((at(y, (a + 1) % w) - at(y, (a + w - 1) % w)) * 0.5)
                    .powi(2)
                    + ((at((y + 1) % h, a) - at((y + h - 1) % h, a)) * 0.5).powi(2);
            }
        }
    }
    Ok(
        json!({"shape":[b,c,h,w],"energy_per_site":e/(2.*(h*w) as f64),
        "l2":e.sqrt(),"variance":variance(&x),
        "channel_variance":x.chunks(h*w).map(variance).collect::<Vec<_>>(),
        "spectral_energy":Spectrum::new(&x,h,w)?.energies(low,mid),
        "spatial_gradient_l2":spatial_gradient_energy.sqrt()}),
    )
}
pub fn cancellation(r: &[f64], t: &[f64], d: &[f64], s: &[f64]) -> Value {
    let total: Vec<_> = (0..r.len()).map(|i| r[i] + t[i] + d[i] + s[i]).collect();
    let ns = [norm(r), norm(t), norm(d), norm(s)];
    json!({"reaction_l2":ns[0],"transport_l2":ns[1],"diffusion_l2":ns[2],"stress_l2":ns[3],
        "sum_l2":norm(&total),"ratio":ns.iter().sum::<f64>()/(norm(&total)+1e-12)})
}
/// Descriptive energy-series outputs; clocks and scalar aliasing prevent attractor claims.
pub fn temporal(x: &[f64]) -> Value {
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let centered: Vec<_> = x.iter().map(|v| v - mean).collect();
    let ss = norm(&centered).powi(2);
    let ac: Vec<_> = (0..n.min(257))
        .map(|lag| {
            if ss > 1e-24 {
                Some(
                    centered[..n - lag]
                        .iter()
                        .zip(&centered[lag..])
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        / ss,
                )
            } else {
                None
            }
        })
        .collect();
    // Bound the O(n^2) temporal DFT independently of the rollout length.
    let tail = &centered[n.saturating_sub(1024)..];
    let spectrum:Vec<_>=(1..=tail.len()/2).map(|k| {
        let (mut re,mut im)=(0.,0.);
        for (i,v) in tail.iter().enumerate() {let a=std::f64::consts::TAU*(k*i) as f64/tail.len() as f64;re+=v*a.cos();im+=v*a.sin();}
        json!({"cycles_per_step":k as f64/tail.len() as f64,"power":(re*re+im*im)/tail.len() as f64})
    }).collect();
    json!({"observable":"full_state_energy_per_scalar","mean":mean,"autocorrelation":ac,
        "autocorrelation_normalization":"biased, fixed lag-zero denominator; null for constant series",
        "spectrum_tail_steps":tail.len(),"spectrum":spectrum,"attractor_classification":null})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cancellation_distinguishes_opposition() {
        let v = cancellation(&[2.], &[-2.], &[0.], &[0.]);
        assert_eq!(v["sum_l2"], 0.);
        assert_eq!(v["ratio"], 4e12);
        assert_eq!(cancellation(&[0.], &[0.], &[0.], &[0.])["ratio"], 0.);
    }
    #[test]
    fn constant_and_periodic_temporal_series() {
        assert!(temporal(&[1.; 16])["autocorrelation"][1].is_null());
        let v = temporal(
            &(0..64)
                .map(|i| (std::f64::consts::TAU * i as f64 / 8.).sin())
                .collect::<Vec<_>>(),
        );
        assert!((v["autocorrelation"][8].as_f64().unwrap() - 0.875).abs() < 1e-12);
        assert!(v["spectrum"][7]["power"].as_f64().unwrap() > 15.9);
    }
}
