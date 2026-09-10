//! Compact observables, per-band work and bounded spatial event fields.
use super::*;
use titan_image::tensor_ops::splitmix64;

pub fn row(world: &WorldState, previous: Option<&[f64]>, o: &Options) -> Result<Value> {
    let x = full(world)?;
    let mut projection = vec![0.; 16];
    let scale = (x.len() as f64).sqrt();
    for (j, v) in projection.iter_mut().enumerate() {
        *v = x
            .iter()
            .enumerate()
            .map(|(i, x)| {
                if splitmix64(o.seed ^ ((j as u64) << 32) ^ i as u64) & 1 == 0 {
                    *x
                } else {
                    -x
                }
            })
            .sum::<f64>()
            / scale;
    }
    let mut cursor = 0;
    let mut transfers = Vec::new();
    let mut channel_means = Vec::new();
    for field in [&world.micro, &world.macro_field] {
        let (_, _, h, w) = field.dims4()?;
        let v = values(field)?;
        channel_means.push(
            v.chunks(h * w)
                .map(|c| c.iter().sum::<f64>() / (h * w) as f64)
                .collect::<Vec<_>>(),
        );
        if let Some(old) = previous {
            let old = &old[cursor..cursor + v.len()];
            let delta = difference(&v, old);
            let before = Spectrum::new(old, h, w)?;
            let update = Spectrum::new(&delta, h, w)?;
            let after = Spectrum::new(&v, h, w)?;
            let t = before.cross(&update, o.low_cutoff, o.mid_cutoff);
            let de = update.energies(o.low_cutoff, o.mid_cutoff);
            let eb = before.energies(o.low_cutoff, o.mid_cutoff);
            let ea = after.energies(o.low_cutoff, o.mid_cutoff);
            let residual: Vec<_> = (0..3).map(|i| ea[i] - eb[i] - 2. * t[i] - de[i]).collect();
            transfers.push(json!({"work":t,"update_band_energy":de,"energy_change":(0..3).map(|i|ea[i]-eb[i]).collect::<Vec<_>>(),"budget_residual":residual}));
        }
        cursor += v.len();
    }
    // Fingerprint exact f32 state bytes; useful for parity, not a cryptographic identity.
    let mut fingerprint = 0xcbf29ce484222325u64;
    for v in &x {
        for b in (*v as f32).to_le_bytes() {
            fingerprint ^= b as u64;
            fingerprint = fingerprint.wrapping_mul(0x100000001b3);
        }
    }
    Ok(
        json!({"developmental_age":world.age,"state_fnv1a64":format!("{fingerprint:016x}"),"random_projection":projection,
        "channel_means":channel_means,"memory":values(&world.memory)?,"transfer_from_previous":transfers}),
    )
}
