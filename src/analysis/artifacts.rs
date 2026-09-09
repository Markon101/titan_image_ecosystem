//! Write-once evaluation outputs. Existence alone never establishes ownership.
use super::provenance::{file_identity, ArtifactIdentity};
use super::AnalysisSummary;
use crate::config::RunConfig;
use crate::persistence::write_json_atomic;
use anyhow::{ensure, Context, Result};
use candle_core::Tensor;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

pub struct EvaluationArtifacts {
    pub(super) evaluation_id: String,
    root: PathBuf,
    records: BTreeMap<String, ArtifactIdentity>,
    failed: bool,
}

impl EvaluationArtifacts {
    pub fn new(config: &RunConfig, step: u64) -> Result<Self> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let evaluation_id = format!(
            "{nanos}-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        );
        let history = config
            .output_dir
            .join(format!("analysis_history_v9{}", config.suffix()));
        std::fs::create_dir_all(&history)?;
        let root = history.join(format!("step_{step}_{evaluation_id}"));
        // Fail closed on a collision, even across processes or clock resets.
        std::fs::create_dir(&root)?;
        Ok(Self {
            evaluation_id,
            root,
            records: BTreeMap::new(),
            failed: false,
        })
    }

    pub fn emitted_paths(&self) -> impl Iterator<Item = &String> {
        self.records.keys()
    }

    pub fn archive(&self) -> PathBuf {
        self.root.join("evaluation.json")
    }

    /// The canonical name is only a naming template; its bytes are never read.
    /// A reserved destination prevents duplicate writes, including rounded probe
    /// fidelity filename collisions. Generation must replace the empty reservation.
    pub fn write(
        &mut self,
        canonical: &Path,
        generate: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<PathBuf> {
        ensure!(!self.failed, "evaluation has a failed write");
        self.failed = true;
        let name = canonical.file_name().context("artifact needs a filename")?;
        let path = self.root.join(name);
        ensure!(
            path != self.archive(),
            "evaluation.json is reserved for publication"
        );
        let reservation = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| {
                format!("refusing to replace evaluation artifact {}", path.display())
            })?;
        drop(reservation);
        generate(&path)?;
        let identity = file_identity(&path)?;
        ensure!(
            identity.bytes > 0,
            "artifact writer emitted no bytes: {}",
            path.display()
        );
        self.records.insert(path.display().to_string(), identity);
        self.failed = false;
        Ok(path)
    }

    pub fn png(&mut self, image: &Tensor, canonical: &Path) -> Result<PathBuf> {
        self.write(canonical, |path| crate::render::save_png(image, path))
    }

    pub fn montage(&mut self, images: &[PathBuf], canonical: &Path, size: u32) -> Result<PathBuf> {
        ensure!(!images.is_empty(), "cannot emit an empty analysis montage");
        for path in images {
            let expected = self
                .records
                .get(&path.display().to_string())
                .context("montage input was not emitted by this evaluation")?;
            ensure!(
                &file_identity(path)? == expected,
                "montage input changed: {}",
                path.display()
            );
        }
        self.write(canonical, |path| {
            crate::render::save_contact_sheet_resized(images, path, size)
        })
    }

    pub fn json<T: Serialize>(&mut self, canonical: &Path, value: &T) -> Result<PathBuf> {
        self.write(canonical, |path| write_json_atomic(path, value))
    }

    // Probe reports need their own destination before serialization.
    pub fn output_path(&self, canonical: &Path) -> PathBuf {
        self.root
            .join(canonical.file_name().expect("artifact filename"))
    }

    /// Consumes the writer: no later write can mutate a published evaluation.
    /// Validate registered bytes again, then publish the archive and latest JSON.
    pub fn publish(self, mut summary: AnalysisSummary, latest: &Path) -> Result<AnalysisSummary> {
        ensure!(
            summary.provenance.evaluation_id == self.evaluation_id,
            "evaluation ownership mismatch"
        );
        ensure!(
            Path::new(&summary.provenance.archive) == self.archive(),
            "evaluation archive mismatch"
        );
        for expected in self.records.values() {
            ensure!(
                file_identity(Path::new(&expected.path))? == *expected,
                "artifact changed before publication: {}",
                expected.path
            );
        }
        ensure!(
            !self.failed,
            "cannot publish an evaluation with a failed write"
        );
        // This validates references against write receipts, never against existence.
        let owned = |path: &str| -> Result<()> {
            ensure!(
                self.records.contains_key(path),
                "unowned analysis output: {path}"
            );
            Ok(())
        };
        for path in [&summary.decomposition_montage, &summary.flow_sample]
            .into_iter()
            .flatten()
        {
            owned(path)?;
        }
        for label in &summary.decomposition_labels {
            let (_, path) = label
                .split_once(':')
                .context("invalid decomposition label")?;
            owned(path)?;
        }
        if let Some(probe) = &summary.natural_image_probe {
            owned(&probe.report)?;
            owned(&probe.montage)?;
            for point in probe.targets.iter().flat_map(|target| &target.points) {
                owned(&point.output)?;
            }
        }
        if let Some(panel) = &summary.experimental_panel {
            for point in panel["points"]
                .as_array()
                .context("invalid experimental panel")?
            {
                for frame in point["autonomous"]
                    .as_array()
                    .context("invalid autonomous panel")?
                {
                    owned(
                        frame["raw_frame"]
                            .as_str()
                            .context("missing panel raw frame")?,
                    )?;
                }
            }
        }
        summary.provenance.artifacts = self.records;
        // Publish only a complete JSON; readers ignore partial evaluation folders.
        let archive = Path::new(&summary.provenance.archive);
        ensure!(!archive.try_exists()?, "evaluation archive already exists");
        write_json_atomic(archive, &summary)?;
        write_json_atomic(latest, &summary)?;
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writer_rejects_noops_collisions_and_unowned_montage_inputs() -> Result<()> {
        let config = RunConfig {
            output_dir: std::env::temp_dir().join(format!(
                "titan-artifact-writer-{}-{}",
                std::process::id(),
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
            )),
            ..RunConfig::default()
        };
        let canonical = config.output_dir.join("old.png");
        let mut writer = EvaluationArtifacts::new(&config, 1)?;
        std::fs::write(&canonical, b"stale")?;
        assert!(writer.write(&canonical, |_| Ok(())).is_err());
        assert!(writer.records.is_empty());
        assert!(writer.failed);
        let mut writer = EvaluationArtifacts::new(&config, 1)?;
        assert!(writer
            .montage(std::slice::from_ref(&canonical), Path::new("sheet.png"), 16)
            .is_err());
        assert!(writer.records.is_empty());
        let output = writer.json(Path::new("fresh.json"), &serde_json::json!({"fresh": true}))?;
        let bytes = std::fs::read(&output)?;
        assert!(writer.json(Path::new("fresh.json"), &false).is_err());
        assert_eq!(std::fs::read(&output)?, bytes);
        let mut writer = EvaluationArtifacts::new(&config, 1)?;
        let occupied = writer.output_path(Path::new("occupied.png"));
        std::fs::write(&occupied, b"foreign")?;
        assert!(writer
            .write(&occupied, |_| panic!(
                "must not generate over existing files"
            ))
            .is_err());
        assert!(writer.records.is_empty());
        assert_eq!(std::fs::read(&occupied)?, b"foreign");
        std::fs::remove_dir_all(config.output_dir)?;
        Ok(())
    }
}
