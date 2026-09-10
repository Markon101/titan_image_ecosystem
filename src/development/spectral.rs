//! Separable f64 DFT on the original grid. No resizing or implicit windowing.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Band {
    Low,
    Mid,
    High,
}

pub struct Spectrum {
    h: usize,
    w: usize,
    re: Vec<f64>,
    im: Vec<f64>,
}

fn table(n: usize) -> Vec<(f64, f64)> {
    (0..n * n)
        .map(|i| {
            let angle = std::f64::consts::TAU * (i / n * (i % n)) as f64 / n as f64;
            (angle.cos(), -angle.sin())
        })
        .collect()
}

fn transform(re: &[f64], im: &[f64], h: usize, w: usize, inverse: bool) -> (Vec<f64>, Vec<f64>) {
    let tw = table(w);
    let th = table(h);
    let mut ar = vec![0.; re.len()];
    let mut ai = ar.clone();
    let mut br = ar.clone();
    let mut bi = ar.clone();
    let sign = if inverse { -1. } else { 1. };
    for channel in 0..re.len() / (h * w) {
        let base = channel * h * w;
        for y in 0..h {
            for k in 0..w {
                let out = base + y * w + k;
                for x in 0..w {
                    let i = base + y * w + x;
                    let (c, s) = tw[k * w + x];
                    let s = s * sign;
                    ar[out] += re[i] * c - im[i] * s;
                    ai[out] += re[i] * s + im[i] * c;
                }
            }
        }
        for k in 0..h {
            for x in 0..w {
                let out = base + k * w + x;
                for y in 0..h {
                    let i = base + y * w + x;
                    let (c, s) = th[k * h + y];
                    let s = s * sign;
                    br[out] += ar[i] * c - ai[i] * s;
                    bi[out] += ar[i] * s + ai[i] * c;
                }
            }
        }
    }
    if inverse {
        for x in br.iter_mut().chain(bi.iter_mut()) {
            *x /= (h * w) as f64;
        }
    }
    (br, bi)
}

impl Spectrum {
    pub fn new(x: &[f64], h: usize, w: usize) -> Result<Self> {
        ensure!(
            h > 0 && w > 0 && x.len().is_multiple_of(h * w),
            "invalid spatial shape"
        );
        let (re, im) = transform(x, &vec![0.; x.len()], h, w, false);
        Ok(Self { h, w, re, im })
    }
    fn band(&self, i: usize, low: f64, mid: f64) -> usize {
        let y = i / self.w % self.h;
        let x = i % self.w;
        let f = ((x.min(self.w - x) as f64 / self.w as f64).powi(2)
            + (y.min(self.h - y) as f64 / self.h as f64).powi(2))
        .sqrt();
        if f <= low {
            0
        } else if f <= mid {
            1
        } else {
            2
        }
    }
    pub fn energies(&self, low: f64, mid: f64) -> [f64; 3] {
        let mut e = [0.; 3];
        for i in 0..self.re.len() {
            e[self.band(i, low, mid)] +=
                (self.re[i].powi(2) + self.im[i].powi(2)) / (self.h * self.w) as f64;
        }
        e
    }
    pub fn project(&self, band: Band, low: f64, mid: f64, remove_dc: bool) -> Vec<f64> {
        let b = match band {
            Band::Low => 0,
            Band::Mid => 1,
            Band::High => 2,
        };
        let mut re = self.re.clone();
        let mut im = self.im.clone();
        for i in 0..re.len() {
            if self.band(i, low, mid) != b || (remove_dc && i % (self.h * self.w) == 0) {
                re[i] = 0.;
                im[i] = 0.;
            }
        }
        transform(&re, &im, self.h, self.w, true).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parseval_bands_and_projection_on_rectangular_grid() -> Result<()> {
        let (h, w) = (6, 8);
        let x: Vec<_> = (0..h * w)
            .map(|i| {
                2. + (std::f64::consts::TAU * (i % w) as f64 / w as f64).cos()
                    + if i % 2 == 0 { 0.5 } else { -0.5 }
            })
            .collect();
        let s = Spectrum::new(&x, h, w)?;
        let e = s.energies(0.13, 0.3);
        assert!((e.iter().sum::<f64>() - x.iter().map(|v| v * v).sum::<f64>()).abs() < 1e-9);
        assert!((e[0] - 216.).abs() < 1e-9);
        assert!(e[1] < 1e-20);
        assert!((e[2] - 12.).abs() < 1e-9);
        let p = s.project(Band::Low, 0.13, 0.3, true);
        assert!(p.iter().sum::<f64>().abs() < 1e-10);
        for (i, v) in p.iter().enumerate() {
            assert!((v - (std::f64::consts::TAU * (i % w) as f64 / w as f64).cos()).abs() < 1e-10);
        }
        Ok(())
    }
}
