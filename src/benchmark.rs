//! Small deterministic reconstruction targets and reference metrics.
//!
//! The suite deliberately favors asymmetric spatial arrangements. It exposes
//! a model which matches global statistics while reusing one phenotype for
//! every target. Creation has no filesystem or random-number dependency.

use candle_core::{bail, Device, Result, Tensor};
use serde::Serialize;

pub const BENCHMARK_VERSION: u32 = 1;
pub const MIN_BENCHMARK_RESOLUTION: usize = 16;
pub const MAX_BENCHMARK_RESOLUTION: usize = 2048;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SyntheticTargetKind {
    OffsetCircle,
    OffsetSquare,
    Diagonal,
    BrokenGrid,
    NestedShapes,
    AsymmetricBlobs,
    Branching,
}

pub const SYNTHETIC_TARGETS: [SyntheticTargetKind; 7] = [
    SyntheticTargetKind::OffsetCircle,
    SyntheticTargetKind::OffsetSquare,
    SyntheticTargetKind::Diagonal,
    SyntheticTargetKind::BrokenGrid,
    SyntheticTargetKind::NestedShapes,
    SyntheticTargetKind::AsymmetricBlobs,
    SyntheticTargetKind::Branching,
];

impl SyntheticTargetKind {
    pub const fn id(self) -> &'static str {
        match self {
            Self::OffsetCircle => "offset-circle",
            Self::OffsetSquare => "offset-square",
            Self::Diagonal => "diagonal",
            Self::BrokenGrid => "broken-grid",
            Self::NestedShapes => "nested-shapes",
            Self::AsymmetricBlobs => "asymmetric-blobs",
            Self::Branching => "branching",
        }
    }

    pub const fn description(self) -> &'static str {
        match self {
            Self::OffsetCircle => "off-center circle with a small satellite",
            Self::OffsetSquare => "upper-right square with an asymmetric notch",
            Self::Diagonal => "rising diagonal with unequal endpoints",
            Self::BrokenGrid => "offset checker grid with missing cells",
            Self::NestedShapes => "non-concentric nested rectangle and ellipse",
            Self::AsymmetricBlobs => "unequal connected organic lobes",
            Self::Branching => "one-sided deterministic branching structure",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BenchmarkTargetMetadata {
    pub benchmark_version: u32,
    pub index: usize,
    pub id: &'static str,
    pub description: &'static str,
    pub asymmetric_by_design: bool,
    pub width: usize,
    pub height: usize,
}

pub struct BenchmarkTarget {
    pub kind: SyntheticTargetKind,
    pub metadata: BenchmarkTargetMetadata,
    /// Planar RGB in `[0, 1]`, shaped `(1, 3, height, width)`, on CPU.
    pub image: Tensor,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TargetReferenceMetrics {
    pub id: String,
    pub width: usize,
    pub height: usize,
    pub mean_rgb: [f32; 3],
    pub variance_rgb: [f32; 3],
    /// Luminance-weighted center of mass in normalized global coordinates.
    pub luminance_centroid: [f32; 2],
    pub luminance_rms: f32,
    pub edge_energy: f32,
    /// Mean absolute distance from a horizontal mirror of the target.
    pub horizontal_asymmetry: f32,
    /// Stable hash of quantized planar RGB values, useful in run metadata.
    pub fingerprint: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReconstructionReferenceMetrics {
    pub raw_l1: f32,
    pub raw_l2: f32,
    /// Spatially registered L1 after area-averaging to an 8x8 grid.
    pub coarse_spatial_l1: f32,
    /// L1 between registered luminance edge-magnitude fields.
    pub edge_l1: f32,
    pub palette_mean_l1: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TargetPairDistance {
    pub left: String,
    pub right: String,
    pub raw_l1: f32,
    pub raw_l2: f32,
    pub coarse_spatial_l1: f32,
    pub edge_l1: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PairwiseSeparability {
    pub target_count: usize,
    pub pair_count: usize,
    pub mean_raw_l1: f32,
    pub minimum_raw_l1: f32,
    pub mean_coarse_spatial_l1: f32,
    pub minimum_coarse_spatial_l1: f32,
    pub closest_pair: Option<[String; 2]>,
    pub pairs: Vec<TargetPairDistance>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BenchmarkReferenceReport {
    pub benchmark_version: u32,
    pub resolution: usize,
    pub references: Vec<TargetReferenceMetrics>,
    pub separability: PairwiseSeparability,
}

/// Create one deterministic synthetic benchmark target on the CPU.
pub fn synthetic_target(kind: SyntheticTargetKind, resolution: usize) -> Result<Tensor> {
    validate_resolution(resolution)?;
    Tensor::from_vec(
        synthetic_planar(kind, resolution),
        (1, 3, resolution, resolution),
        &Device::Cpu,
    )
}

/// Create the complete deterministic suite in stable order.
pub fn benchmark_suite(resolution: usize) -> Result<Vec<BenchmarkTarget>> {
    validate_resolution(resolution)?;
    SYNTHETIC_TARGETS
        .iter()
        .copied()
        .enumerate()
        .map(|(index, kind)| {
            Ok(BenchmarkTarget {
                kind,
                metadata: BenchmarkTargetMetadata {
                    benchmark_version: BENCHMARK_VERSION,
                    index,
                    id: kind.id(),
                    description: kind.description(),
                    asymmetric_by_design: true,
                    width: resolution,
                    height: resolution,
                },
                image: synthetic_target(kind, resolution)?,
            })
        })
        .collect()
}

/// Compute cheap reference statistics for a planar RGB tensor on CPU.
pub fn target_reference_metrics(
    id: impl Into<String>,
    image: &Tensor,
) -> Result<TargetReferenceMetrics> {
    let image = CpuImage::from_tensor(image)?;
    let pixel_count = (image.height * image.width) as f32;
    let mut mean_rgb = [0.0f32; 3];
    let mut variance_rgb = [0.0f32; 3];
    for (channel, mean) in mean_rgb.iter_mut().enumerate() {
        *mean = image.channel(channel).iter().sum::<f32>() / pixel_count;
    }
    for (channel, variance) in variance_rgb.iter_mut().enumerate() {
        *variance = image
            .channel(channel)
            .iter()
            .map(|value| (value - mean_rgb[channel]).powi(2))
            .sum::<f32>()
            / pixel_count;
    }
    let luminance = image.luminance();
    let luminance_sum = luminance.iter().sum::<f32>().max(1e-8);
    let mut centroid = [0.0f32; 2];
    for y in 0..image.height {
        for x in 0..image.width {
            let weight = luminance[y * image.width + x];
            centroid[0] += weight * normalized_coordinate(x, image.width);
            centroid[1] += weight * normalized_coordinate(y, image.height);
        }
    }
    centroid[0] /= luminance_sum;
    centroid[1] /= luminance_sum;
    let luminance_rms =
        (luminance.iter().map(|value| value * value).sum::<f32>() / pixel_count).sqrt();
    Ok(TargetReferenceMetrics {
        id: id.into(),
        width: image.width,
        height: image.height,
        mean_rgb,
        variance_rgb,
        luminance_centroid: centroid,
        luminance_rms,
        edge_energy: mean(&edge_magnitude(&luminance, image.height, image.width)),
        horizontal_asymmetry: horizontal_asymmetry(&image),
        fingerprint: format!("{:016x}", quantized_fingerprint(&image.values)),
    })
}

/// Compare a candidate and reference without discarding spatial registration.
pub fn reconstruction_reference_metrics(
    candidate: &Tensor,
    reference: &Tensor,
) -> Result<ReconstructionReferenceMetrics> {
    let candidate = CpuImage::from_tensor(candidate)?;
    let reference = CpuImage::from_tensor(reference)?;
    ensure_same_shape(&candidate, &reference)?;
    let count = candidate.values.len() as f32;
    let raw_l1 = candidate
        .values
        .iter()
        .zip(&reference.values)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / count;
    let raw_l2 = (candidate
        .values
        .iter()
        .zip(&reference.values)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / count)
        .sqrt();
    let coarse_spatial_l1 =
        mean_absolute_distance(&area_grid(&candidate, 8), &area_grid(&reference, 8));
    let candidate_edges = edge_magnitude(&candidate.luminance(), candidate.height, candidate.width);
    let reference_edges = edge_magnitude(&reference.luminance(), reference.height, reference.width);
    let candidate_mean = channel_mean(&candidate);
    let reference_mean = channel_mean(&reference);
    Ok(ReconstructionReferenceMetrics {
        raw_l1,
        raw_l2,
        coarse_spatial_l1,
        edge_l1: mean_absolute_distance(&candidate_edges, &reference_edges),
        palette_mean_l1: candidate_mean
            .iter()
            .zip(reference_mean)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / 3.0,
    })
}

/// Compute every registered target-pair distance in stable suite order.
pub fn pairwise_separability(targets: &[BenchmarkTarget]) -> Result<PairwiseSeparability> {
    let mut pairs = Vec::new();
    for left_index in 0..targets.len() {
        for right_index in (left_index + 1)..targets.len() {
            let left = &targets[left_index];
            let right = &targets[right_index];
            let metrics = reconstruction_reference_metrics(&left.image, &right.image)?;
            pairs.push(TargetPairDistance {
                left: left.metadata.id.to_owned(),
                right: right.metadata.id.to_owned(),
                raw_l1: metrics.raw_l1,
                raw_l2: metrics.raw_l2,
                coarse_spatial_l1: metrics.coarse_spatial_l1,
                edge_l1: metrics.edge_l1,
            });
        }
    }
    let pair_count = pairs.len();
    let mean_raw_l1 = pair_mean(&pairs, |pair| pair.raw_l1);
    let mean_coarse_spatial_l1 = pair_mean(&pairs, |pair| pair.coarse_spatial_l1);
    let closest = pairs.iter().min_by(|a, b| a.raw_l1.total_cmp(&b.raw_l1));
    Ok(PairwiseSeparability {
        target_count: targets.len(),
        pair_count,
        mean_raw_l1,
        minimum_raw_l1: closest.map_or(0.0, |pair| pair.raw_l1),
        mean_coarse_spatial_l1,
        minimum_coarse_spatial_l1: pairs
            .iter()
            .map(|pair| pair.coarse_spatial_l1)
            .min_by(f32::total_cmp)
            .unwrap_or(0.0),
        closest_pair: closest.map(|pair| [pair.left.clone(), pair.right.clone()]),
        pairs,
    })
}

/// Build the immutable baseline report that trained runs can compare against.
pub fn benchmark_reference_report(resolution: usize) -> Result<BenchmarkReferenceReport> {
    let suite = benchmark_suite(resolution)?;
    let references = suite
        .iter()
        .map(|target| target_reference_metrics(target.metadata.id, &target.image))
        .collect::<Result<Vec<_>>>()?;
    Ok(BenchmarkReferenceReport {
        benchmark_version: BENCHMARK_VERSION,
        resolution,
        separability: pairwise_separability(&suite)?,
        references,
    })
}

fn validate_resolution(resolution: usize) -> Result<()> {
    if !(MIN_BENCHMARK_RESOLUTION..=MAX_BENCHMARK_RESOLUTION).contains(&resolution) {
        bail!(
            "benchmark resolution must be in {MIN_BENCHMARK_RESOLUTION}..={MAX_BENCHMARK_RESOLUTION}"
        );
    }
    Ok(())
}

fn synthetic_planar(kind: SyntheticTargetKind, resolution: usize) -> Vec<f32> {
    let plane = resolution * resolution;
    let mut values = vec![0.0f32; 3 * plane];
    let antialias = 1.25 / resolution as f32;
    for y in 0..resolution {
        for x in 0..resolution {
            let px = normalized_coordinate(x, resolution);
            let py = normalized_coordinate(y, resolution);
            // A common radial canvas reduces empty-region dominance without
            // giving any target a target-specific palette shortcut.
            let radius = ((px - 0.5).powi(2) + (py - 0.5).powi(2)).sqrt();
            let base = 0.035 + 0.025 * (1.0 - (1.5 * radius).min(1.0));
            let mut rgb = [base * 0.72, base * 0.90, base];
            paint_target(kind, px, py, antialias, &mut rgb);
            let index = y * resolution + x;
            for channel in 0..3 {
                values[channel * plane + index] = rgb[channel].clamp(0.0, 1.0);
            }
        }
    }
    values
}

fn paint_target(kind: SyntheticTargetKind, x: f32, y: f32, aa: f32, rgb: &mut [f32; 3]) {
    match kind {
        SyntheticTargetKind::OffsetCircle => {
            circle(rgb, x, y, 0.30, 0.39, 0.205, aa, [0.94, 0.30, 0.18]);
            circle(rgb, x, y, 0.73, 0.70, 0.075, aa, [0.98, 0.76, 0.20]);
        }
        SyntheticTargetKind::OffsetSquare => {
            let body = coverage(rect_sdf(x, y, 0.60, 0.16, 0.88, 0.48), aa);
            let notch = coverage(rect_sdf(x, y, 0.77, 0.16, 0.88, 0.27), aa);
            paint(rgb, [0.20, 0.72, 0.96], body * (1.0 - notch));
            rectangle(rgb, x, y, [0.53, 0.41, 0.66, 0.56], aa, [0.75, 0.32, 0.92]);
        }
        SyntheticTargetKind::Diagonal => {
            segment(
                rgb,
                x,
                y,
                [0.10, 0.82, 0.79, 0.21],
                0.027,
                aa,
                [0.92, 0.82, 0.20],
            );
            circle(rgb, x, y, 0.10, 0.82, 0.075, aa, [0.98, 0.38, 0.18]);
            circle(rgb, x, y, 0.79, 0.21, 0.045, aa, [0.30, 0.84, 0.70]);
        }
        SyntheticTargetKind::BrokenGrid => {
            let column = ((x - 0.11) / 0.14).floor() as i32;
            let row = ((y - 0.22) / 0.14).floor() as i32;
            if (0..6).contains(&column) && (0..5).contains(&row) {
                let missing = matches!((column, row), (0, 0) | (3, 1) | (1, 3) | (5, 4));
                if !missing && (column + row) % 2 == 0 {
                    let x0 = 0.11 + column as f32 * 0.14;
                    let y0 = 0.22 + row as f32 * 0.14;
                    rectangle(
                        rgb,
                        x,
                        y,
                        [x0, y0, x0 + 0.105, y0 + 0.105],
                        aa,
                        [0.34, 0.88, 0.45],
                    );
                }
            }
            circle(rgb, x, y, 0.84, 0.15, 0.047, aa, [0.88, 0.35, 0.68]);
        }
        SyntheticTargetKind::NestedShapes => {
            let outer = coverage(rect_sdf(x, y, 0.12, 0.15, 0.83, 0.84), aa);
            let inner = coverage(rect_sdf(x, y, 0.17, 0.20, 0.78, 0.79), aa);
            paint(rgb, [0.90, 0.32, 0.72], outer * (1.0 - inner));
            paint(
                rgb,
                [0.22, 0.74, 0.96],
                ellipse_coverage(x, y, [0.43, 0.48], [0.22, 0.16], aa),
            );
            circle(rgb, x, y, 0.51, 0.43, 0.052, aa, [0.98, 0.76, 0.24]);
        }
        SyntheticTargetKind::AsymmetricBlobs => {
            let a = ellipse_coverage(x, y, [0.30, 0.58], [0.22, 0.14], aa);
            let b = ellipse_coverage(x, y, [0.52, 0.47], [0.17, 0.24], aa);
            let c = ellipse_coverage(x, y, [0.69, 0.61], [0.12, 0.10], aa);
            paint(rgb, [0.34, 0.88, 0.64], a.max(0.8 * b).max(0.65 * c));
            circle(rgb, x, y, 0.43, 0.52, 0.065, aa, [0.16, 0.34, 0.52]);
            circle(rgb, x, y, 0.71, 0.60, 0.038, aa, [0.95, 0.48, 0.22]);
        }
        SyntheticTargetKind::Branching => {
            let branches = [
                ([0.48, 0.88, 0.47, 0.54], 0.025),
                ([0.47, 0.57, 0.28, 0.37], 0.022),
                ([0.47, 0.57, 0.66, 0.32], 0.022),
                ([0.29, 0.38, 0.18, 0.22], 0.016),
                ([0.29, 0.38, 0.36, 0.18], 0.014),
                ([0.65, 0.33, 0.62, 0.15], 0.015),
                ([0.65, 0.33, 0.82, 0.24], 0.018),
                ([0.47, 0.69, 0.64, 0.61], 0.013),
            ];
            let branch = branches.iter().fold(0.0f32, |mask, (line, radius)| {
                mask.max(coverage(segment_sdf(x, y, *line, *radius), aa))
            });
            paint(rgb, [0.72, 0.94, 0.30], branch);
            circle(rgb, x, y, 0.82, 0.24, 0.032, aa, [0.95, 0.55, 0.20]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn circle(rgb: &mut [f32; 3], x: f32, y: f32, cx: f32, cy: f32, r: f32, aa: f32, color: [f32; 3]) {
    paint(rgb, color, coverage((x - cx).hypot(y - cy) - r, aa));
}

fn rectangle(rgb: &mut [f32; 3], x: f32, y: f32, bounds: [f32; 4], aa: f32, color: [f32; 3]) {
    paint(
        rgb,
        color,
        coverage(
            rect_sdf(x, y, bounds[0], bounds[1], bounds[2], bounds[3]),
            aa,
        ),
    );
}

#[allow(clippy::too_many_arguments)]
fn segment(
    rgb: &mut [f32; 3],
    x: f32,
    y: f32,
    line: [f32; 4],
    radius: f32,
    aa: f32,
    color: [f32; 3],
) {
    paint(rgb, color, coverage(segment_sdf(x, y, line, radius), aa));
}

fn paint(destination: &mut [f32; 3], color: [f32; 3], alpha: f32) {
    let alpha = alpha.clamp(0.0, 1.0);
    for channel in 0..3 {
        destination[channel] = destination[channel] * (1.0 - alpha) + color[channel] * alpha;
    }
}

fn coverage(signed_distance: f32, antialias: f32) -> f32 {
    let t = ((antialias - signed_distance) / (2.0 * antialias)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn ellipse_coverage(x: f32, y: f32, center: [f32; 2], radii: [f32; 2], aa: f32) -> f32 {
    let normalized =
        (((x - center[0]) / radii[0]).powi(2) + ((y - center[1]) / radii[1]).powi(2)).sqrt();
    coverage((normalized - 1.0) * radii[0].min(radii[1]), aa)
}

fn rect_sdf(x: f32, y: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let dx = (x - 0.5 * (x0 + x1)).abs() - 0.5 * (x1 - x0);
    let dy = (y - 0.5 * (y0 + y1)).abs() - 0.5 * (y1 - y0);
    dx.max(0.0).hypot(dy.max(0.0)) + dx.max(dy).min(0.0)
}

fn segment_sdf(x: f32, y: f32, line: [f32; 4], radius: f32) -> f32 {
    let [x0, y0, x1, y1] = line;
    let vx = x1 - x0;
    let vy = y1 - y0;
    let t = (((x - x0) * vx + (y - y0) * vy) / (vx * vx + vy * vy)).clamp(0.0, 1.0);
    (x - (x0 + t * vx)).hypot(y - (y0 + t * vy)) - radius
}

fn normalized_coordinate(index: usize, extent: usize) -> f32 {
    (index as f32 + 0.5) / extent as f32
}

struct CpuImage {
    values: Vec<f32>,
    height: usize,
    width: usize,
}

impl CpuImage {
    fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let (batch, channels, height, width) = tensor.dims4()?;
        if batch != 1 || channels != 3 {
            bail!("benchmark image must have shape (1, 3, height, width)");
        }
        let values = tensor.detach().flatten_all()?.to_vec1::<f32>()?;
        if values.iter().any(|value| !value.is_finite()) {
            bail!("benchmark image contains NaN or infinity");
        }
        Ok(Self {
            values,
            height,
            width,
        })
    }

    fn channel(&self, channel: usize) -> &[f32] {
        let plane = self.height * self.width;
        &self.values[channel * plane..(channel + 1) * plane]
    }

    fn luminance(&self) -> Vec<f32> {
        self.channel(0)
            .iter()
            .zip(self.channel(1))
            .zip(self.channel(2))
            .map(|((red, green), blue)| 0.2126 * red + 0.7152 * green + 0.0722 * blue)
            .collect()
    }
}

fn ensure_same_shape(left: &CpuImage, right: &CpuImage) -> Result<()> {
    if left.height != right.height || left.width != right.width {
        bail!(
            "benchmark comparison requires matching image sizes; left={}x{}, right={}x{}",
            left.width,
            left.height,
            right.width,
            right.height
        );
    }
    Ok(())
}

fn channel_mean(image: &CpuImage) -> [f32; 3] {
    let pixels = (image.height * image.width) as f32;
    [0, 1, 2].map(|channel| image.channel(channel).iter().sum::<f32>() / pixels)
}

fn area_grid(image: &CpuImage, requested: usize) -> Vec<f32> {
    let grid_h = requested.min(image.height);
    let grid_w = requested.min(image.width);
    let mut values = vec![0.0f32; 3 * grid_h * grid_w];
    let mut counts = vec![0usize; grid_h * grid_w];
    for y in 0..image.height {
        let gy = y * grid_h / image.height;
        for x in 0..image.width {
            let gx = x * grid_w / image.width;
            let grid_index = gy * grid_w + gx;
            let pixel_index = y * image.width + x;
            counts[grid_index] += 1;
            for channel in 0..3 {
                values[channel * grid_h * grid_w + grid_index] +=
                    image.channel(channel)[pixel_index];
            }
        }
    }
    for channel in 0..3 {
        for (index, count) in counts.iter().enumerate() {
            values[channel * grid_h * grid_w + index] /= *count as f32;
        }
    }
    values
}

fn edge_magnitude(luminance: &[f32], height: usize, width: usize) -> Vec<f32> {
    let mut edges = vec![0.0f32; height * width];
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let right = luminance[y * width + (x + 1).min(width - 1)];
            let down = luminance[(y + 1).min(height - 1) * width + x];
            edges[index] = (right - luminance[index]).hypot(down - luminance[index]);
        }
    }
    edges
}

fn horizontal_asymmetry(image: &CpuImage) -> f32 {
    let mut total = 0.0f32;
    for channel in 0..3 {
        let values = image.channel(channel);
        for y in 0..image.height {
            for x in 0..image.width {
                total += (values[y * image.width + x]
                    - values[y * image.width + image.width - 1 - x])
                    .abs();
            }
        }
    }
    total / image.values.len() as f32
}

fn quantized_fingerprint(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, value| {
        let quantized = (value.clamp(0.0, 1.0) * 65_535.0).round() as u16;
        quantized.to_le_bytes().iter().fold(hash, |inner, byte| {
            (inner ^ *byte as u64).wrapping_mul(0x100_0000_01b3)
        })
    })
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len().max(1) as f32
}

fn mean_absolute_distance(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / left.len().max(1) as f32
}

fn pair_mean(pairs: &[TargetPairDistance], value: impl Fn(&TargetPairDistance) -> f32) -> f32 {
    pairs.iter().map(value).sum::<f32>() / pairs.len().max(1) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suite_is_deterministic_and_has_stable_metadata() -> Result<()> {
        let first = benchmark_suite(48)?;
        let second = benchmark_suite(48)?;
        assert_eq!(first.len(), SYNTHETIC_TARGETS.len());
        for (index, (left, right)) in first.iter().zip(&second).enumerate() {
            assert_eq!(left.metadata.index, index);
            assert_eq!(left.metadata.id, SYNTHETIC_TARGETS[index].id());
            assert_eq!(left.metadata, right.metadata);
            assert_eq!(
                left.image.flatten_all()?.to_vec1::<f32>()?,
                right.image.flatten_all()?.to_vec1::<f32>()?
            );
        }
        Ok(())
    }

    #[test]
    fn every_target_is_asymmetric_in_pixels_not_just_metadata() -> Result<()> {
        for target in benchmark_suite(64)? {
            let metrics = target_reference_metrics(target.metadata.id, &target.image)?;
            assert!(
                metrics.horizontal_asymmetry > 0.008,
                "{} asymmetry was only {}",
                target.metadata.id,
                metrics.horizontal_asymmetry
            );
        }
        Ok(())
    }

    #[test]
    fn registered_reference_metrics_are_zero_for_identity() -> Result<()> {
        for target in benchmark_suite(32)? {
            let metrics = reconstruction_reference_metrics(&target.image, &target.image)?;
            assert_eq!(metrics.raw_l1, 0.0);
            assert_eq!(metrics.raw_l2, 0.0);
            assert_eq!(metrics.coarse_spatial_l1, 0.0);
            assert_eq!(metrics.edge_l1, 0.0);
            assert_eq!(metrics.palette_mean_l1, 0.0);
        }
        Ok(())
    }

    #[test]
    fn suite_has_nontrivial_pixel_and_coarse_spatial_separation() -> Result<()> {
        let report = benchmark_reference_report(64)?;
        assert_eq!(report.separability.pair_count, 21);
        assert!(report.separability.minimum_raw_l1 > 0.025);
        assert!(report.separability.minimum_coarse_spatial_l1 > 0.02);
        assert!(report.separability.closest_pair.is_some());
        assert!(report
            .references
            .iter()
            .all(|metrics| metrics.edge_energy > 0.002));
        Ok(())
    }

    #[test]
    fn spatial_mismatch_is_visible_when_palette_is_preserved() -> Result<()> {
        let target = synthetic_target(SyntheticTargetKind::OffsetCircle, 48)?;
        let cpu = CpuImage::from_tensor(&target)?;
        let mut shifted = vec![0.0f32; cpu.values.len()];
        let plane = cpu.height * cpu.width;
        for channel in 0..3 {
            for y in 0..cpu.height {
                for x in 0..cpu.width {
                    let source_x = (x + cpu.width / 3) % cpu.width;
                    shifted[channel * plane + y * cpu.width + x] =
                        cpu.values[channel * plane + y * cpu.width + source_x];
                }
            }
        }
        let shifted = Tensor::from_vec(shifted, (1, 3, 48, 48), &Device::Cpu)?;
        let metrics = reconstruction_reference_metrics(&shifted, &target)?;
        assert!(metrics.palette_mean_l1 < 1e-6);
        assert!(metrics.raw_l1 > 0.05);
        assert!(metrics.coarse_spatial_l1 > 0.04);
        Ok(())
    }

    #[test]
    fn invalid_resolution_and_shape_are_rejected() -> Result<()> {
        assert!(synthetic_target(SyntheticTargetKind::Branching, 8).is_err());
        let invalid = Tensor::zeros((1, 1, 32, 32), candle_core::DType::F32, &Device::Cpu)?;
        assert!(target_reference_metrics("invalid", &invalid).is_err());
        Ok(())
    }
}
