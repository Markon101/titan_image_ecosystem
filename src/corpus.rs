use crate::config::{RunConfig, TrainingMode};
use crate::render::SpatialView;
use crate::tensor_ops::splitmix64;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use serde::Serialize;
use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};

pub struct TargetSample {
    pub image: Tensor,
    pub reference_micro: Tensor,
    pub flow_image: Tensor,
    pub reference_macro: Tensor,
    pub genome: Vec<f32>,
    pub genome_tensor: Tensor,
    pub index: usize,
    pub name: String,
    pub source_width: u32,
    pub source_height: u32,
    pub square_crop_x: u32,
    pub square_crop_y: u32,
    pub square_crop_size: u32,
    pub pyramid_levels: Vec<usize>,
    pub fingerprint: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CorpusSourceMetadata {
    pub index: usize,
    pub name: String,
    pub fingerprint: String,
    pub width: u32,
    pub height: u32,
    pub aspect_ratio: f32,
    pub square_crop_x: u32,
    pub square_crop_y: u32,
    pub square_crop_size: u32,
    pub pyramid_levels: Vec<usize>,
}

#[derive(Clone, Debug, Serialize)]
pub struct CorpusSummary {
    pub images: usize,
    pub min_width: u32,
    pub min_height: u32,
    pub max_width: u32,
    pub max_height: u32,
    pub median_megapixels: f32,
    pub total_megapixels: f64,
    pub min_aspect_ratio: f32,
    pub max_aspect_ratio: f32,
    pub cached_pyramid_bytes: u64,
}

struct SourceImage {
    path: PathBuf,
    name: String,
    fingerprint: u64,
    genome: Vec<f32>,
    width: u32,
    height: u32,
    square_crop_x: u32,
    square_crop_y: u32,
    square_crop_size: u32,
    pyramid_levels: Vec<usize>,
}

struct CachedImage {
    index: usize,
    image: Tensor,
    flow_image: Tensor,
    reference_micro: Tensor,
    reference_macro: Tensor,
}

pub struct DetailObservation {
    pub target: Tensor,
    pub local_reference_micro: Tensor,
    pub local_reference_macro: Tensor,
    pub view: SpatialView,
    pub pyramid_level: usize,
    pub zoom: f32,
    pub fingerprint: u64,
}

pub struct ImageCorpus {
    sources: Vec<SourceImage>,
    cache: VecDeque<CachedImage>,
    cache_capacity: usize,
    resolution: usize,
    micro_size: usize,
    flow_resolution: usize,
    macro_size: usize,
    seed: u64,
    mode: TrainingMode,
    fingerprint: u64,
    schedule_epoch: Option<u64>,
    schedule: Vec<usize>,
    cache_root: PathBuf,
    summary: CorpusSummary,
}

impl ImageCorpus {
    pub fn new(config: &RunConfig, device: &Device) -> Result<Self> {
        let mut paths = Vec::new();
        collect_sources(
            &config.corpus_dir,
            &config.output_dir,
            config.recursive_corpus,
            &mut paths,
        )?;
        if paths.is_empty() {
            bail!(
                "no PNG, JPEG, or WebP source images found in {}",
                config.corpus_dir.display()
            );
        }

        let mut sources = Vec::with_capacity(paths.len());
        for path in paths {
            let fingerprint = file_fingerprint(&path)?;
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("unknown")
                .to_owned();
            sources.push(SourceImage {
                genome: genome_for(fingerprint, config.mode, config.genome_dim),
                path,
                name,
                fingerprint,
                width: 0,
                height: 0,
                square_crop_x: 0,
                square_crop_y: 0,
                square_crop_size: 0,
                pyramid_levels: Vec::new(),
            });
        }
        // Content-first ordering makes target identities stable across harmless
        // renames while preserving a deterministic tie-break for duplicates.
        sources.sort_by(|a, b| {
            a.fingerprint
                .cmp(&b.fingerprint)
                .then_with(|| a.path.cmp(&b.path))
        });
        let duplicate_content = sources
            .windows(2)
            .filter(|pair| pair[0].fingerprint == pair[1].fingerprint)
            .count();
        if duplicate_content > 0 {
            println!(
                "Corpus warning: {duplicate_content} exact duplicate source entr{} will receive repeated schedule weight.",
                if duplicate_content == 1 { "y" } else { "ies" }
            );
        }
        if config.mode == TrainingMode::Single && sources.len() != 1 {
            bail!(
                "single mode requires exactly one source image; found {} in {}",
                sources.len(),
                config.corpus_dir.display()
            );
        }

        let fingerprint = corpus_fingerprint(&sources);
        let mut cache = VecDeque::with_capacity(config.image_cache);
        // Decode every source before training so a malformed file cannot fail
        // after a long run. Sources within the configured cache are resized and
        // retained as tensors during the same decode rather than decoded twice.
        for (index, source) in sources.iter_mut().enumerate() {
            let decoded = decode_source(&source.path).with_context(|| {
                format!("corpus preflight failed for {}", source.path.display())
            })?;
            source.width = decoded.width();
            source.height = decoded.height();
            source.square_crop_size = source.width.min(source.height);
            source.square_crop_x = (source.width - source.square_crop_size) / 2;
            source.square_crop_y = (source.height - source.square_crop_size) / 2;
            source.pyramid_levels = available_pyramid_levels(
                source.width.max(source.height) as usize,
                config.detail.cache_max_level,
            );
            if index < config.image_cache {
                let image = Tensor::from_vec(
                    planar_from_image(decoded.clone(), config.train_resolution),
                    (1, 3, config.train_resolution, config.train_resolution),
                    device,
                )?;
                let reference_micro = Tensor::from_vec(
                    planar_from_image(decoded.clone(), config.micro_size),
                    (1, 3, config.micro_size, config.micro_size),
                    device,
                )?;
                let reference_macro = Tensor::from_vec(
                    planar_from_image(decoded.clone(), config.macro_size),
                    (1, 3, config.macro_size, config.macro_size),
                    device,
                )?;
                let flow_image = Tensor::from_vec(
                    planar_from_image(decoded, config.flow.resolution),
                    (1, 3, config.flow.resolution, config.flow.resolution),
                    device,
                )?;
                cache.push_back(CachedImage {
                    index,
                    image,
                    flow_image,
                    reference_micro,
                    reference_macro,
                });
            }
        }
        let cache_root = config.pyramid_cache_root();
        std::fs::create_dir_all(&cache_root)?;
        let cached_pyramid_bytes = directory_bytes(&cache_root)?;
        let summary = corpus_summary(&sources, cached_pyramid_bytes);

        Ok(Self {
            sources,
            cache,
            cache_capacity: config.image_cache,
            resolution: config.train_resolution,
            micro_size: config.micro_size,
            macro_size: config.macro_size,
            seed: config.seed,
            mode: config.mode,
            fingerprint,
            schedule_epoch: None,
            schedule: Vec::new(),
            cache_root,
            flow_resolution: config.flow.resolution,
            summary,
        })
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub fn cached_images(&self) -> usize {
        self.cache.len()
    }

    pub fn source_manifest(&self) -> Vec<CorpusSourceMetadata> {
        self.sources
            .iter()
            .enumerate()
            .map(|(index, source)| CorpusSourceMetadata {
                index,
                name: source.name.clone(),
                fingerprint: format!("{:016x}", source.fingerprint),
                width: source.width,
                height: source.height,
                aspect_ratio: source.width as f32 / source.height.max(1) as f32,
                square_crop_x: source.square_crop_x,
                square_crop_y: source.square_crop_y,
                square_crop_size: source.square_crop_size,
                pyramid_levels: source.pyramid_levels.clone(),
            })
            .collect()
    }

    pub fn summary(&self) -> &CorpusSummary {
        &self.summary
    }

    pub fn summary_snapshot(&self) -> Result<CorpusSummary> {
        let mut summary = self.summary.clone();
        summary.cached_pyramid_bytes = directory_bytes(&self.cache_root)?;
        Ok(summary)
    }

    pub fn sample(&mut self, episode: u64, device: &Device) -> Result<TargetSample> {
        let index = self.index_for_episode(episode);
        self.sample_index(index, device)
    }

    pub fn sample_index(&mut self, index: usize, device: &Device) -> Result<TargetSample> {
        if index >= self.sources.len() {
            bail!("target index {index} is outside the corpus");
        }
        let (image, flow_image, reference_micro, reference_macro) = if let Some(position) =
            self.cache.iter().position(|entry| entry.index == index)
        {
            let entry = self.cache.remove(position).expect("cache position exists");
            let tensors = (
                entry.image.clone(),
                entry.flow_image.clone(),
                entry.reference_micro.clone(),
                entry.reference_macro.clone(),
            );
            self.cache.push_back(entry);
            tensors
        } else {
            let decoded = decode_source(&self.sources[index].path)
                .with_context(|| format!("cannot decode {}", self.sources[index].path.display()))?;
            let image = Tensor::from_vec(
                planar_from_image(decoded.clone(), self.resolution),
                (1, 3, self.resolution, self.resolution),
                device,
            )?;
            let reference_micro = Tensor::from_vec(
                planar_from_image(decoded.clone(), self.micro_size),
                (1, 3, self.micro_size, self.micro_size),
                device,
            )?;
            let reference_macro = Tensor::from_vec(
                planar_from_image(decoded.clone(), self.macro_size),
                (1, 3, self.macro_size, self.macro_size),
                device,
            )?;
            let flow_image = Tensor::from_vec(
                planar_from_image(decoded, self.flow_resolution),
                (1, 3, self.flow_resolution, self.flow_resolution),
                device,
            )?;
            if self.cache.len() == self.cache_capacity {
                self.cache.pop_front();
            }
            self.cache.push_back(CachedImage {
                index,
                image: image.clone(),
                flow_image: flow_image.clone(),
                reference_micro: reference_micro.clone(),
                reference_macro: reference_macro.clone(),
            });
            (image, flow_image, reference_micro, reference_macro)
        };
        let source = &self.sources[index];
        let genome = source.genome.clone();
        Ok(TargetSample {
            image,
            reference_micro,
            reference_macro,
            genome_tensor: Tensor::from_vec(genome.clone(), genome.len(), device)?,
            flow_image,
            genome,
            index,
            name: source.name.clone(),
            source_width: source.width,
            source_height: source.height,
            square_crop_x: source.square_crop_x,
            square_crop_y: source.square_crop_y,
            square_crop_size: source.square_crop_size,
            pyramid_levels: source.pyramid_levels.clone(),
            fingerprint: source.fingerprint,
        })
    }

    pub fn detail_observation(
        &self,
        index: usize,
        world_step: u64,
        age_phase: f32,
        config: &RunConfig,
        device: &Device,
    ) -> Result<Option<DetailObservation>> {
        if config.detail.probability <= 0.0 || age_phase < config.detail.curriculum_start {
            return Ok(None);
        }
        let curriculum = ((age_phase - config.detail.curriculum_start)
            / (1.0 - config.detail.curriculum_start).max(1e-6))
        .clamp(0.0, 1.0);
        let key = splitmix64(
            config.seed
                ^ world_step.wrapping_mul(0xd1b5_4a32_d192_ed03)
                ^ self.sources[index].fingerprint
                ^ 0x9d37_4f10_b31c_8a25,
        );
        if unit_float(key) >= config.detail.probability * curriculum {
            return Ok(None);
        }
        let zoom_unit = unit_float(splitmix64(key ^ 0x5a17_91e3));
        let log_zoom = config.detail.min_zoom.log2()
            + zoom_unit * (config.detail.max_zoom.log2() - config.detail.min_zoom.log2());
        let zoom = 2.0f32.powf(log_zoom);
        let size = 1.0 / zoom;
        let x = unit_float(splitmix64(key ^ 0x71c3_0b55)) * (1.0 - size);
        let y = unit_float(splitmix64(key ^ 0xa91f_d217)) * (1.0 - size);
        let view = SpatialView { x, y, size, zoom };
        let desired_level = ((config.detail.resolution as f32 * zoom).ceil() as usize)
            .min(config.detail.cache_max_level);
        let source = &self.sources[index];
        let pyramid_level = source
            .pyramid_levels
            .iter()
            .copied()
            .find(|level| *level >= desired_level)
            .or_else(|| source.pyramid_levels.last().copied())
            .unwrap_or(source.width.max(source.height) as usize);
        let level = self.load_pyramid_level(index, pyramid_level)?;
        let square = level.width().min(level.height());
        let square_x = (level.width() - square) / 2;
        let square_y = (level.height() - square) / 2;
        let crop_size = ((square as f32 * size).round() as u32).clamp(2, square);
        let max_start = square - crop_size;
        let crop_x = square_x + ((x * square as f32).round() as u32).min(max_start);
        let crop_y = square_y + ((y * square as f32).round() as u32).min(max_start);
        let crop = level.crop_imm(crop_x, crop_y, crop_size, crop_size);
        let target_values = planar_from_image_exact(crop.clone(), config.detail.resolution);
        let target = Tensor::from_vec(
            target_values,
            (1, 3, config.detail.resolution, config.detail.resolution),
            device,
        )?;
        let local_reference_micro =
            local_reference_overlay(&crop, config.micro_size, view, device)?;
        let local_reference_macro =
            local_reference_overlay(&crop, config.macro_size, view, device)?;
        Ok(Some(DetailObservation {
            target,
            local_reference_micro,
            local_reference_macro,
            view,
            pyramid_level,
            zoom,
            fingerprint: splitmix64(key ^ pyramid_level as u64),
        }))
    }

    fn load_pyramid_level(&self, index: usize, level: usize) -> Result<image::DynamicImage> {
        let source = &self.sources[index];
        let path = self
            .cache_root
            .join(format!("{:016x}_{level}.png", source.fingerprint));
        if path.exists() {
            return decode_source(&path)
                .with_context(|| format!("cannot decode cached pyramid {}", path.display()));
        }
        let decoded = decode_source(&source.path)?;
        let resized = if decoded.width().max(decoded.height()) as usize <= level {
            decoded
        } else {
            decoded.resize(
                level as u32,
                level as u32,
                image::imageops::FilterType::Lanczos3,
            )
        };
        let temporary = path.with_extension("png.tmp");
        resized.save_with_format(&temporary, image::ImageFormat::Png)?;
        std::fs::rename(&temporary, &path)?;
        Ok(resized)
    }
    /// Interpolate between two source genomes and add a small deterministic
    /// mutation. Interpolation stays close to conditioning seen during training,
    /// unlike unconstrained random vectors.
    pub fn gallery_genome(&self, variant: usize, gallery_seed: u64) -> Vec<f32> {
        let key = splitmix64(gallery_seed ^ variant as u64);
        let first = key as usize % self.sources.len();
        let second = splitmix64(key) as usize % self.sources.len();
        let alpha = 0.15 + 0.70 * unit_float(splitmix64(key ^ 0xa17e_51d5));
        self.sources[first]
            .genome
            .iter()
            .zip(&self.sources[second].genome)
            .enumerate()
            .map(|(dimension, (a, b))| {
                let mutation = 0.08
                    * (2.0
                        * unit_float(splitmix64(
                            key ^ (dimension as u64).wrapping_mul(0x9e37_79b9),
                        ))
                        - 1.0);
                ((1.0 - alpha) * a + alpha * b + mutation).clamp(-1.0, 1.0)
            })
            .collect()
    }

    fn index_for_episode(&mut self, episode: u64) -> usize {
        if self.mode == TrainingMode::Single {
            return 0;
        }
        let count = self.sources.len();
        let epoch = episode / count as u64;
        if self.schedule_epoch != Some(epoch) {
            self.schedule = (0..count).collect();
            self.schedule.sort_by_key(|index| {
                splitmix64(
                    self.seed
                        ^ epoch.wrapping_mul(0xd1b5_4a32_d192_ed03)
                        ^ self.sources[*index].fingerprint,
                )
            });
            self.schedule_epoch = Some(epoch);
        }
        self.schedule[episode as usize % count]
    }
}

fn collect_sources(
    directory: &Path,
    output_dir: &Path,
    recursive: bool,
    paths: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in std::fs::read_dir(directory)
        .with_context(|| format!("cannot read corpus directory {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            if recursive && !path.starts_with(output_dir) {
                collect_sources(&path, output_dir, true, paths)?;
            }
        } else if file_type.is_file()
            && !path.starts_with(output_dir)
            && !generated_name(&path)
            && supported(&path)
        {
            paths.push(path);
        }
    }
    Ok(())
}

fn supported(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "png" | "jpg" | "jpeg" | "webp"
            )
        })
        .unwrap_or(false)
}

fn generated_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_ascii_lowercase().starts_with("titan_image_"))
        .unwrap_or(false)
}

fn available_pyramid_levels(native_max: usize, cache_max: usize) -> Vec<usize> {
    let mut levels: Vec<usize> = [192usize, 384, 768, 1536]
        .into_iter()
        .filter(|level| *level <= native_max && *level <= cache_max)
        .collect();
    levels.push(native_max.min(cache_max).max(1));
    levels.sort_unstable();
    levels.dedup();
    levels
}

fn corpus_summary(sources: &[SourceImage], cached_pyramid_bytes: u64) -> CorpusSummary {
    let mut megapixels: Vec<f32> = sources
        .iter()
        .map(|source| source.width as f32 * source.height as f32 / 1_000_000.0)
        .collect();
    megapixels.sort_by(f32::total_cmp);
    let median_megapixels = megapixels[megapixels.len() / 2];
    CorpusSummary {
        images: sources.len(),
        min_width: sources.iter().map(|source| source.width).min().unwrap_or(0),
        min_height: sources
            .iter()
            .map(|source| source.height)
            .min()
            .unwrap_or(0),
        max_width: sources.iter().map(|source| source.width).max().unwrap_or(0),
        max_height: sources
            .iter()
            .map(|source| source.height)
            .max()
            .unwrap_or(0),
        median_megapixels,
        total_megapixels: megapixels.iter().map(|value| *value as f64).sum(),
        min_aspect_ratio: sources
            .iter()
            .map(|source| source.width as f32 / source.height.max(1) as f32)
            .fold(f32::INFINITY, f32::min),
        max_aspect_ratio: sources
            .iter()
            .map(|source| source.width as f32 / source.height.max(1) as f32)
            .fold(0.0, f32::max),
        cached_pyramid_bytes,
    }
}

fn directory_bytes(path: &Path) -> Result<u64> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            total = total.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(total)
}

fn planar_from_image_exact(image: image::DynamicImage, resolution: usize) -> Vec<f32> {
    let image = image
        .resize_exact(
            resolution as u32,
            resolution as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    planar_from_rgb(&image)
}

fn planar_from_rgb(image: &image::RgbImage) -> Vec<f32> {
    let plane = image.width() as usize * image.height() as usize;
    let mut values = vec![0.0f32; 3 * plane];
    for (index, pixel) in image.pixels().enumerate() {
        values[index] = pixel[0] as f32 / 255.0;
        values[plane + index] = pixel[1] as f32 / 255.0;
        values[2 * plane + index] = pixel[2] as f32 / 255.0;
    }
    values
}

fn local_reference_overlay(
    crop: &image::DynamicImage,
    field_size: usize,
    view: SpatialView,
    device: &Device,
) -> Result<Tensor> {
    let cells = ((field_size as f32 * view.size).round() as usize).clamp(2, field_size);
    let max_start = field_size - cells;
    let start_x = ((view.x * field_size as f32).round() as usize).min(max_start);
    let start_y = ((view.y * field_size as f32).round() as usize).min(max_start);
    let resized = crop
        .resize_exact(
            cells as u32,
            cells as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    let patch = planar_from_rgb(&resized);
    let patch_plane = cells * cells;
    let field_plane = field_size * field_size;
    let mut values = vec![0.0f32; 3 * field_plane];
    for channel in 0..3 {
        for y in 0..cells {
            let destination = channel * field_plane + (start_y + y) * field_size + start_x;
            let source = channel * patch_plane + y * cells;
            values[destination..destination + cells]
                .copy_from_slice(&patch[source..source + cells]);
        }
    }
    Tensor::from_vec(values, (1, 3, field_size, field_size), device).map_err(Into::into)
}
#[cfg(test)]
fn load_planar(path: &Path, resolution: usize) -> Result<Vec<f32>> {
    Ok(planar_from_image(decode_source(path)?, resolution))
}

fn planar_from_image(image: image::DynamicImage, resolution: usize) -> Vec<f32> {
    let image = image
        .resize_to_fill(
            resolution as u32,
            resolution as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    let plane = resolution * resolution;
    let mut values = vec![0.0f32; 3 * plane];
    for (index, pixel) in image.pixels().enumerate() {
        values[index] = pixel[0] as f32 / 255.0;
        values[plane + index] = pixel[1] as f32 / 255.0;
        values[2 * plane + index] = pixel[2] as f32 / 255.0;
    }
    values
}

fn decode_source(path: &Path) -> Result<image::DynamicImage> {
    image::ImageReader::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("cannot identify image format from {}", path.display()))?
        .decode()
        .with_context(|| format!("cannot decode {}", path.display()))
}

fn file_fingerprint(path: &Path) -> Result<u64> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("cannot hash corpus source {}", path.display()))?;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        for byte in &buffer[..read] {
            hash = (hash ^ *byte as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
    Ok(hash)
}

fn corpus_fingerprint(sources: &[SourceImage]) -> u64 {
    sources.iter().fold(
        0xcbf2_9ce4_8422_2325 ^ sources.len() as u64,
        |hash, source| (hash ^ source.fingerprint).wrapping_mul(0x100_0000_01b3),
    )
}

fn genome_for(fingerprint: u64, mode: TrainingMode, dimensions: usize) -> Vec<f32> {
    if mode == TrainingMode::Single {
        return vec![0.0; dimensions];
    }
    (0..dimensions)
        .map(|index| {
            let hash = splitmix64(fingerprint ^ index as u64);
            2.0 * unit_float(hash) - 1.0
        })
        .collect()
}

fn unit_float(value: u64) -> f32 {
    (value >> 40) as f32 / (1u32 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn genomes_are_stable_bounded_and_content_keyed() {
        let a = genome_for(123, TrainingMode::Texture, 8);
        let b = genome_for(123, TrainingMode::Texture, 8);
        let c = genome_for(124, TrainingMode::Texture, 8);
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.iter().all(|value| (-1.0..=1.0).contains(value)));
    }

    #[test]
    fn content_detection_accepts_mislabeled_jpeg() -> Result<()> {
        let path = std::env::temp_dir().join(format!(
            "titan-mislabeled-{}-{}.png",
            std::process::id(),
            splitmix64(17)
        ));
        let image = image::RgbImage::from_pixel(8, 8, image::Rgb([12, 34, 56]));
        image.save_with_format(&path, image::ImageFormat::Jpeg)?;
        let planar = load_planar(&path, 8)?;
        assert_eq!(planar.len(), 3 * 8 * 8);
        std::fs::remove_file(path)?;
        Ok(())
    }
}
