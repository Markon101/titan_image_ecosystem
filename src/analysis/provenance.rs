//! Additive analysis provenance. Checkpoint and legacy sidecar formats stay intact.
use super::AnalysisSummary;
use crate::config::RunConfig;
use crate::corpus::{ImageCorpus, TargetSample};
use crate::persistence::ArtifactPaths;
use crate::state::WorldState;
use crate::telemetry::tensor_fingerprint;
use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Serialize)]
pub struct ArtifactIdentity {
    pub path: String,
    pub bytes: u64,
    /// Identity/error-detection only, not a cryptographic integrity guarantee.
    pub fnv1a64: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct AnalysisProvenance {
    pub analysis_version: u32,
    pub evaluation_id: String,
    pub archive: String,
    pub build: serde_json::Value,
    pub config: RunConfig,
    pub corpus_fingerprint: String,
    pub initial_world: serde_json::Value,
    pub checkpoint_manifest: Option<serde_json::Value>,
    pub checkpoint_files: BTreeMap<String, ArtifactIdentity>,
    pub completed: BTreeMap<String, bool>,
    pub policies: serde_json::Value,
    pub artifacts: BTreeMap<String, ArtifactIdentity>,
}

impl AnalysisProvenance {
    pub(super) fn new(
        config: &RunConfig,
        paths: &ArtifactPaths,
        corpus: &ImageCorpus,
        world: &WorldState,
        sample: &TargetSample,
    ) -> Result<Self> {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let evaluation_id = format!(
            "{}-{}-{}",
            nanos,
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        );
        let archive = config
            .output_dir
            .join(format!("analysis_history_v9{}", config.suffix()))
            .join(format!("step_{}_{}.json", world.step, evaluation_id));
        let mut checkpoint_files = BTreeMap::new();
        for (name, path) in [
            ("model", &paths.model),
            ("world", &paths.world),
            ("optimizer", &paths.optimizer),
            ("manifest", &paths.checkpoint_manifest),
        ] {
            if path.try_exists()? {
                checkpoint_files.insert(name.to_owned(), file_identity(path)?);
            }
        }
        let checkpoint_manifest = if paths.checkpoint_manifest.try_exists()? {
            Some(serde_json::from_slice(&std::fs::read(
                &paths.checkpoint_manifest,
            )?)?)
        } else {
            None
        };
        Ok(Self {
            analysis_version: 2,
            evaluation_id,
            archive: archive.display().to_string(),
            build: serde_json::json!({
                "commit": env!("TITAN_BUILD_COMMIT"),
                "dirty": env!("TITAN_BUILD_DIRTY") == "true",
                "release": !cfg!(debug_assertions),
                "target_arch": std::env::consts::ARCH,
                "rustflags": env!("TITAN_BUILD_RUSTFLAGS"),
                "package_version": env!("CARGO_PKG_VERSION"),
            }),
            config: config.clone(),
            corpus_fingerprint: format!("{:016x}", corpus.fingerprint()),
            initial_world: serde_json::json!({
                "step": world.step, "age": world.age, "episode": world.episode,
                "target_index": world.target_index,
                "active_morph_depth": world.morph_active_depth,
                "morph_generation": world.morph_generation,
                "micro": tensor_fingerprint(&world.micro)?,
                "macro": tensor_fingerprint(&world.macro_field)?,
                "memory": tensor_fingerprint(&world.memory)?,
                "genome": tensor_fingerprint(&sample.genome_tensor)?,
                "target_source_fingerprint": format!("{:016x}", sample.fingerprint),
            }),
            checkpoint_manifest,
            checkpoint_files,
            completed: BTreeMap::new(),
            policies: serde_json::json!({
                "autonomous_reference": "none; fidelity zero; saved state and fixed genome retained",
                "autonomous_age": "saved age raised to at least developmental horizon",
                "perturbation_reference": "guided; configured maximum fidelity",
                "recurrence_signature": "legacy channel spatial means plus memory; heuristic, not full-state recurrence",
                "checkpoint_identity": "on-disk checkpoint bytes; runtime anatomy and state recorded separately",
                "artifact_identity": "FNV-1a 64-bit plus byte length; non-cryptographic",
                "archive": "complete JSON results retained; image paths remain mutable, verify fingerprints",
            }),
            artifacts: BTreeMap::new(),
        })
    }
}

pub(super) fn completion_status(summary: &AnalysisSummary) -> BTreeMap<String, bool> {
    [
        ("decomposition", summary.decomposition_montage.is_some()),
        ("frontier", !summary.frontier.is_empty()),
        (
            "resolution_consistency",
            summary.provenance.config.analysis.only,
        ),
        ("target_separability", summary.target_separability.is_some()),
        ("autonomous_rollout", !summary.autonomous_rollout.is_empty()),
        ("perturbations", !summary.perturbations.is_empty()),
        ("dynamics_ablation", !summary.dynamics_ablation.is_empty()),
        ("benchmark", summary.benchmark.is_some()),
        ("natural_image_probe", summary.natural_image_probe.is_some()),
        ("flow_sample", summary.flow_sample.is_some()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect()
}

pub(super) fn completed_artifacts(
    paths: &ArtifactPaths,
    summary: &AnalysisSummary,
) -> Result<BTreeMap<String, ArtifactIdentity>> {
    let mut files: Vec<PathBuf> = Vec::new();
    if let Some(path) = &summary.decomposition_montage {
        files.push(PathBuf::from(path));
        files.push(paths.emergence_frontier.clone());
        for label in &summary.decomposition_labels {
            if let Some((_, path)) = label.split_once(':') {
                files.push(PathBuf::from(path));
            }
        }
    }
    if summary.provenance.config.analysis.only {
        files.push(paths.resolution_ladder.clone());
    }
    if summary.target_separability.is_some() {
        files.push(paths.target_comparison.clone());
    }
    if !summary.autonomous_rollout.is_empty() {
        files.push(paths.attractor_analysis.clone());
        files.push(paths.attractor_analysis.with_extension("png"));
        for record in &summary.autonomous_rollout {
            files.push(super::sibling_png(
                &paths.attractor_analysis,
                &format!("{:05}", record.offset),
            ));
        }
    }
    if !summary.perturbations.is_empty() {
        files.push(paths.perturbation_analysis.clone());
    }
    if summary.benchmark.is_some() {
        files.push(paths.benchmark.clone());
    }
    if let Some(probe) = &summary.natural_image_probe {
        files.push(PathBuf::from(&probe.report));
        files.push(PathBuf::from(&probe.montage));
        for target in &probe.targets {
            files.extend(target.points.iter().map(|p| PathBuf::from(&p.output)));
        }
    }
    if let Some(path) = &summary.flow_sample {
        files.push(PathBuf::from(path));
        files.push(paths.flow_trajectory.clone());
    }
    files
        .into_iter()
        .map(|path| {
            let identity = file_identity(&path)?;
            Ok((path.display().to_string(), identity))
        })
        .collect()
}

fn file_identity(path: &Path) -> Result<ArtifactIdentity> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("cannot fingerprint {}", path.display()))?;
    let mut buffer = [0u8; 65536];
    let mut hash = 0xcbf29ce484222325u64;
    let mut bytes = 0u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        for byte in &buffer[..n] {
            hash = (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
    }
    Ok(ArtifactIdentity {
        path: path.display().to_string(),
        bytes,
        fnv1a64: format!("{hash:016x}"),
    })
}
