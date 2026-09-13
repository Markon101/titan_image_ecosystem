//! Passive decomposition of damaged-minus-control residuals; all arithmetic is f64.
use super::{metrics, spectral::Spectrum};
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use titan_image::state::WorldState;

fn spatial(x: &[f64], h: usize, w: usize, low: f64, mid: f64) -> Result<Value> {
    let plane = h * w;
    ensure!(
        plane > 0 && !x.is_empty() && x.len().is_multiple_of(plane),
        "invalid residual shape"
    );
    let energy = x.iter().map(|v| v * v).sum::<f64>();
    let means: Vec<_> = x
        .chunks_exact(plane)
        .map(|c| c.iter().sum::<f64>() / plane as f64)
        .collect();
    let dc = means.iter().map(|m| m * m * plane as f64).sum::<f64>();
    let centered: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, v)| v - means[i / plane])
        .collect();
    let bands = if energy == 0. {
        [0.; 3]
    } else {
        Spectrum::new(&centered, h, w)?.energies(low, mid)
    };
    let error = (energy - dc - bands.iter().sum::<f64>()).abs();
    let relative_error = if energy > 0. { error / energy } else { 0. };
    ensure!(
        relative_error < 1e-9,
        "residual energy partition does not close: {relative_error}"
    );
    Ok(json!({"channel_means":means,"channel_mean_energy":dc,
        "spatial_band_energy_excluding_dc":bands,"partition_relative_error":relative_error,
        "energy_fractions_dc_low_mid_high":if energy>0. {Some([dc/energy,bands[0]/energy,bands[1]/energy,bands[2]/energy])} else {None}}))
}

pub(super) fn measure(
    control: &WorldState,
    damaged: &WorldState,
    initial_distance: f64,
    low: f64,
    mid: f64,
) -> Result<Value> {
    ensure!(
        initial_distance.is_finite() && initial_distance > 0.,
        "invalid residual normalization"
    );
    let mut fields = serde_json::Map::new();
    let mut sum_rms = 0.;
    for (name, a, b, spatial_field) in [
        ("micro", &control.micro, &damaged.micro, true),
        ("macro", &control.macro_field, &damaged.macro_field, true),
        ("memory", &control.memory, &damaged.memory, false),
    ] {
        ensure!(a.dims() == b.dims(), "residual field shapes differ");
        let delta = metrics::difference(&metrics::values(b)?, &metrics::values(a)?);
        let energy = delta.iter().map(|v| v * v).sum::<f64>();
        let rms = (energy / delta.len() as f64).sqrt();
        sum_rms += rms;
        let mut field = json!({"rms":rms,"l2_energy":energy,"scalar_count":delta.len(),
            "initial_distance_normalized_rms":rms/initial_distance});
        if spatial_field {
            let (_, _, h, w) = a.dims4()?;
            field["spatial"] = spatial(&delta, h, w, low, mid)?;
        }
        fields.insert(name.into(), field);
    }
    for field in fields.values_mut() {
        field["fraction_of_sum_rms"] = json!(if sum_rms > 0. {
            Some(field["rms"].as_f64().unwrap() / sum_rms)
        } else {
            None
        });
    }
    Ok(
        json!({"fields":fields,"sum_rms":sum_rms,"state_distance_ratio":sum_rms/initial_distance,
        "sign":"damaged minus paired undamaged control",
        "fractions":"field shares use sum of RMS; spatial fractions partition each field's squared L2 energy; zero-energy fractions are null"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn separates_known_dc_low_and_high_energy() -> Result<()> {
        let x: Vec<_> = (0..64)
            .map(|i| {
                2. + (std::f64::consts::TAU * (i % 8) as f64 / 8.).cos()
                    + if (i / 8 + i % 8) % 2 == 0 { 0.5 } else { -0.5 }
            })
            .collect();
        let r = spatial(&x, 8, 8, 0.125, 0.25)?;
        assert!((r["channel_mean_energy"].as_f64().unwrap() - 256.).abs() < 1e-9);
        for (v, expected) in r["spatial_band_energy_excluding_dc"]
            .as_array()
            .unwrap()
            .iter()
            .zip([32., 0., 16.])
        {
            assert!((v.as_f64().unwrap() - expected).abs() < 1e-9);
        }
        let zero = spatial(&[0.; 64], 8, 8, 0.125, 0.25)?;
        assert!(zero["energy_fractions_dc_low_mid_high"].is_null());
        Ok(())
    }
    #[test]
    fn memory_only_residual_has_all_distance_share_and_does_not_mutate_input() -> Result<()> {
        let c = titan_image::RunConfig::default();
        let a = WorldState::fresh(&c, 42, &candle_core::Device::Cpu)?;
        let before = metrics::full(&a)?;
        let mut b = a.detached();
        b.memory = b.memory.affine(1., 0.125)?;
        let r = measure(&a, &b, 0.25, 0.125, 0.25)?;
        assert_eq!(r["fields"]["micro"]["rms"], 0.);
        assert_eq!(r["fields"]["macro"]["rms"], 0.);
        assert_eq!(r["fields"]["memory"]["fraction_of_sum_rms"], 1.);
        assert!((r["state_distance_ratio"].as_f64().unwrap() - 0.5).abs() < 1e-7);
        assert_eq!(metrics::full(&a)?, before);
        assert!(measure(&a, &b, 0., 0.125, 0.25).is_err());
        Ok(())
    }
}
