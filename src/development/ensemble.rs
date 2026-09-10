//! Finite-amplitude central-difference QR ensemble, identical clocks and genome.
use super::*;

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
/// Twice-reorthogonalized modified Gram-Schmidt; columns and positive R diagonal.
pub fn qr(mut columns: Vec<Vec<f64>>) -> Result<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut diagonal = Vec::new();
    for i in 0..columns.len() {
        for _ in 0..2 {
            for j in 0..i {
                let projection = dot(&columns[i], &columns[j]);
                for k in 0..columns[i].len() {
                    columns[i][k] -= projection * columns[j][k];
                }
            }
        }
        let d = norm(&columns[i]);
        ensure!(
            d > 1e-14 && d.is_finite(),
            "QR rank loss or unresolved perturbation; no exponent flooring"
        );
        for v in &mut columns[i] {
            *v /= d;
        }
        diagonal.push(d);
    }
    Ok((columns, diagonal))
}
pub fn displace_full(x: &WorldState, direction: &[f64], epsilon: f64) -> Result<WorldState> {
    let mut y = x.detached();
    let mut cursor = 0;
    for t in [&mut y.micro, &mut y.macro_field, &mut y.memory] {
        let original = values(t)?;
        let v: Vec<f32> = original
            .iter()
            .zip(&direction[cursor..cursor + original.len()])
            .map(|(x, p)| (x + epsilon * p) as f32)
            .collect();
        ensure!(v.iter().all(|x| x.is_finite()), "nonfinite displaced state");
        cursor += original.len();
        *t = Tensor::from_vec(v, t.shape(), t.device())?;
    }
    ensure!(
        cursor == direction.len(),
        "full perturbation shape mismatch"
    );
    Ok(y)
}
pub struct Ensemble {
    plus: Vec<WorldState>,
    minus: Vec<WorldState>,
    sums: Vec<f64>,
    epsilon: f64,
    pub reset_relative_error: f64,
}
impl Ensemble {
    pub fn new(x: &WorldState, count: usize, epsilon: f64, seed: u64) -> Result<Self> {
        let n = full(x)?.len();
        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0x5152);
        let columns = (0..count)
            .map(|_| (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        let (q, _) = qr(columns)?;
        let mut e = Self {
            plus: vec![],
            minus: vec![],
            sums: vec![0.; count],
            epsilon,
            reset_relative_error: 0.,
        };
        e.reset(x, &q)?;
        Ok(e)
    }
    fn reset(&mut self, x: &WorldState, q: &[Vec<f64>]) -> Result<()> {
        self.plus.clear();
        self.minus.clear();
        self.reset_relative_error = 0.;
        for direction in q {
            let p = displace_full(x, direction, self.epsilon)?;
            let m = displace_full(x, direction, -self.epsilon)?;
            let actual: Vec<_> = difference(&full(&p)?, &full(&m)?)
                .iter()
                .map(|v| v / (2. * self.epsilon))
                .collect();
            self.reset_relative_error = self.reset_relative_error.max(distance(&actual, direction));
            self.plus.push(p);
            self.minus.push(m);
        }
        ensure!(
            self.reset_relative_error < 0.01,
            "ensemble reset distorted >1%; increase epsilon"
        );
        Ok(())
    }
    pub fn advance(
        &mut self,
        dynamics: &DynamicsSystem,
        genome: &Tensor,
        operators: &operators::Operators,
    ) -> Result<()> {
        for x in self.plus.iter_mut().chain(&mut self.minus) {
            *x = super::advance(dynamics, x, genome, operators)?.0;
        }
        Ok(())
    }
    pub fn measure(&mut self, base: &WorldState, elapsed: usize) -> Result<Value> {
        let mut columns = Vec::new();
        let b = full(base)?;
        let mut curvature = Vec::new();
        for (p, m) in self.plus.iter().zip(&self.minus) {
            let p = full(p)?;
            let m = full(m)?;
            columns.push(
                p.iter()
                    .zip(&m)
                    .map(|(p, m)| (p - m) / (2. * self.epsilon))
                    .collect(),
            );
            curvature.push(
                norm(
                    &p.iter()
                        .zip(&m)
                        .zip(&b)
                        .map(|((p, m), b)| (p + m) * 0.5 - b)
                        .collect::<Vec<_>>(),
                ) / self.epsilon,
            );
        }
        let (q, diag) = qr(columns)?;
        for (s, d) in self.sums.iter_mut().zip(&diag) {
            *s += d.ln();
        }
        let exponents: Vec<_> = self.sums.iter().map(|s| s / elapsed as f64).collect();
        let mut orthogonality: f64 = 0.;
        for i in 0..q.len() {
            for j in 0..q.len() {
                orthogonality =
                    orthogonality.max((dot(&q[i], &q[j]) - if i == j { 1. } else { 0. }).abs());
            }
        }
        let row = json!({"offset":elapsed,"developmental_age":base.age,"clock_step":base.step,"epsilon":self.epsilon,
            "qr_stretch":diag,"finite_time_exponents":exponents,"orthogonality_max_error":orthogonality,
            "input_reset_relative_error":self.reset_relative_error,"central_pair_midpoint_drift_over_epsilon":curvature});
        self.reset(base, &q)?;
        Ok(row)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn qr_known_linear_exponents_and_rank_loss() -> Result<()> {
        let (q, d) = qr(vec![vec![2., 0., 0.], vec![1., 0.5, 0.], vec![3., 2., 1.]])?;
        assert_eq!(d, vec![2., 0.5, 1.]);
        assert!((d[0].ln() - 2f64.ln()).abs() < 1e-15);
        assert_eq!(dot(&q[0], &q[1]), 0.);
        assert!(qr(vec![vec![1., 0.], vec![1., 0.]]).is_err());
        Ok(())
    }
    #[test]
    fn full_displacement_preserves_clocks() -> Result<()> {
        let x = WorldState::fresh(&RunConfig::default(), 42, &Device::Cpu)?;
        let mut p = vec![0.; full(&x)?.len()];
        *p.last_mut().unwrap() = 1.;
        let y = displace_full(&x, &p, 0.25)?;
        assert_eq!(x.step, y.step);
        assert_eq!(x.age, y.age);
        assert_eq!(values(&x.micro)?, values(&y.micro)?);
        assert!((distance(&full(&x)?, &full(&y)?) - 0.25).abs() < 1e-7);
        Ok(())
    }
}
