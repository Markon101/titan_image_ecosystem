use crate::config::{RunConfig, TrainingMode};
use crate::tensor_ops::splitmix64;
use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use serde::Serialize;
use std::collections::VecDeque;
use std::io::Read;
use std::path::{Path, PathBuf};

pub struct TargetSample {
    pub image: Tensor,
    pub genome: Vec<f32>,
    pub genome_tensor: Tensor,
    pub index: usize,
    pub name: String,
    pub fingerprint: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CorpusSourceMetadata {
    pub index: usize,
    pub name: String,
    pub fingerprint: String,
}

struct SourceImage {
    path: PathBuf,
    name: String,
    fingerprint: u64,
    genome: Vec<f32>,
}

struct CachedImage {
    index: usize,
    image: Tensor,
}

pub struct ImageCorpus {
    sources: Vec<SourceImage>,
    cache: VecDeque<CachedImage>,
    cache_capacity: usize,
    resolution: usize,
    seed: u64,
    mode: TrainingMode,
    fingerprint: u64,
    schedule_epoch: Option<u64>,
    schedule: Vec<usize>,
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
        for (index, source) in sources.iter().enumerate() {
            let decoded = decode_source(&source.path).with_context(|| {
                format!("corpus preflight failed for {}", source.path.display())
            })?;
            if index < config.image_cache {
                let values = planar_from_image(decoded, config.train_resolution);
                cache.push_back(CachedImage {
                    index,
                    image: Tensor::from_vec(
                        values,
                        (1, 3, config.train_resolution, config.train_resolution),
                        device,
                    )?,
                });
            }
        }

        Ok(Self {
            sources,
            cache,
            cache_capacity: config.image_cache,
            resolution: config.train_resolution,
            seed: config.seed,
            mode: config.mode,
            fingerprint,
            schedule_epoch: None,
            schedule: Vec::new(),
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
            })
            .collect()
    }

    pub fn sample(&mut self, episode: u64, device: &Device) -> Result<TargetSample> {
        let index = self.index_for_episode(episode);
        let image = if let Some(position) = self.cache.iter().position(|entry| entry.index == index)
        {
            let entry = self.cache.remove(position).expect("cache position exists");
            let image = entry.image.clone();
            self.cache.push_back(entry);
            image
        } else {
            let values = load_planar(&self.sources[index].path, self.resolution)?;
            let image = Tensor::from_vec(values, (1, 3, self.resolution, self.resolution), device)?;
            if self.cache.len() == self.cache_capacity {
                self.cache.pop_front();
            }
            self.cache.push_back(CachedImage {
                index,
                image: image.clone(),
            });
            image
        };
        let source = &self.sources[index];
        let genome = source.genome.clone();
        Ok(TargetSample {
            image,
            genome_tensor: Tensor::from_vec(genome.clone(), genome.len(), device)?,
            genome,
            index,
            name: source.name.clone(),
            fingerprint: source.fingerprint,
        })
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
