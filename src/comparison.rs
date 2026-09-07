use crate::analysis::AnalysisSummary;
use crate::persistence::ArtifactPaths;
use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize)]
pub struct RunComparisonSnapshot {
    pub label: String,
    pub world_step: Option<f64>,
    pub reconstruction_content: Option<f64>,
    pub grounding_loss: Option<f64>,
    pub structure_loss: Option<f64>,
    pub palette_loss: Option<f64>,
    pub emergent_residual_rms: Option<f64>,
    pub target_separability: Option<f64>,
    pub micro_near_bound_fraction: Option<f64>,
    pub macro_near_bound_fraction: Option<f64>,
    pub memory_rms: Option<f64>,
    pub gradient_rms: Option<f64>,
    pub clip_scale: Option<f64>,
    pub seam_energy: Option<f64>,
    pub gamut_excess: Option<f64>,
    pub development_steps_per_second: Option<f64>,
    pub peak_rss_kib: Option<f64>,
    pub parameter_count: Option<f64>,
    pub active_parameter_count: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct V8V9Comparison {
    pub v8_metrics: String,
    pub v8_metadata: String,
    pub v9_metrics: String,
    pub v9_metadata: String,
    pub v9_analysis_archive: Option<String>,
    pub v8: RunComparisonSnapshot,
    pub v9: RunComparisonSnapshot,
    pub interpretation: &'static str,
}

pub fn compare_v8_v9(
    v8_dir: &Path,
    v9_paths: &ArtifactPaths,
    analysis: Option<&AnalysisSummary>,
) -> Result<V8V9Comparison> {
    let v8_metrics = newest_with_prefix(v8_dir, "titan_image_metrics_v8", "csv")?;
    let v8_metadata = newest_with_prefix(v8_dir, "titan_image_run_metadata_v8", "json")?;
    if !v9_paths.metrics.exists() || !v9_paths.metadata.exists() {
        bail!("v9 comparison requires completed v9 metrics and metadata");
    }
    let v8 = snapshot("v8", &v8_metrics, &v8_metadata, None)?;
    let v9 = snapshot(
        "v9",
        &v9_paths.metrics,
        &v9_paths.metadata,
        analysis
            .and_then(|summary| summary.target_separability.as_ref())
            .map(|report| f64::from(report.mean_output_l1)),
    )?;
    Ok(V8V9Comparison {
        v8_metrics: v8_metrics.display().to_string(),
        v8_metadata: v8_metadata.display().to_string(),
        v9_metrics: v9_paths.metrics.display().to_string(),
        v9_metadata: v9_paths.metadata.display().to_string(),
        v9_analysis_archive: analysis.map(|summary| summary.provenance.archive.clone()),
        v8,
        v9,
        interpretation:
            "Developmental comparison only. Different schemas/checkpoints are not weight-compatible and metric improvements do not alone prove emergence or causality.",
    })
}

fn snapshot(
    label: &str,
    metrics_path: &Path,
    metadata_path: &Path,
    target_separability: Option<f64>,
) -> Result<RunComparisonSnapshot> {
    let row = last_csv_row(metrics_path)?;
    let metadata: Value = serde_json::from_slice(&std::fs::read(metadata_path)?)?;
    Ok(RunComparisonSnapshot {
        label: label.to_owned(),
        world_step: field(&row, "step"),
        reconstruction_content: field(&row, "loss_content"),
        grounding_loss: field(&row, "loss_grounding"),
        structure_loss: field(&row, "loss_structure"),
        palette_loss: field(&row, "loss_palette"),
        emergent_residual_rms: field(&row, "emergent_output_rms"),
        target_separability,
        micro_near_bound_fraction: field(&row, "micro_clamp_fraction"),
        macro_near_bound_fraction: field(&row, "macro_clamp_fraction"),
        memory_rms: field(&row, "interface_memory_rms"),
        gradient_rms: field(&row, "gradient_rms"),
        clip_scale: field(&row, "gradient_clip_scale"),
        seam_energy: field(&row, "seam_energy"),
        gamut_excess: field(&row, "gamut_excess"),
        development_steps_per_second: field(&row, "development_steps_per_second"),
        peak_rss_kib: metadata.get("peak_rss_kib").and_then(Value::as_f64),
        parameter_count: metadata.get("parameter_count").and_then(Value::as_f64),
        active_parameter_count: metadata
            .get("active_parameter_count")
            .and_then(Value::as_f64),
    })
}

fn field(row: &HashMap<String, String>, name: &str) -> Option<f64> {
    row.get(name)?.parse().ok()
}

fn last_csv_row(path: &Path) -> Result<HashMap<String, String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read metrics {}", path.display()))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .context("metrics CSV has no header")?
        .split(',')
        .collect();
    let values: Vec<&str> = lines
        .rfind(|line| !line.trim().is_empty())
        .context("metrics CSV has no data rows")?
        .split(',')
        .collect();
    if header.len() != values.len() {
        bail!(
            "metrics CSV header/row length mismatch in {}",
            path.display()
        );
    }
    Ok(header
        .into_iter()
        .zip(values)
        .map(|(name, value)| (name.to_owned(), value.to_owned()))
        .collect())
}

fn newest_with_prefix(directory: &Path, prefix: &str, extension: &str) -> Result<PathBuf> {
    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(directory)?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            (name.starts_with(prefix)
                && path.extension().and_then(|value| value.to_str()) == Some(extension))
            .then(|| {
                (
                    entry
                        .metadata()
                        .and_then(|metadata| metadata.modified())
                        .unwrap_or(std::time::UNIX_EPOCH),
                    path,
                )
            })
        })
        .collect();
    candidates.sort_by_key(|(modified, _)| *modified);
    candidates
        .pop()
        .map(|(_, path)| path)
        .with_context(|| format!("no {prefix}*.{extension} in {}", directory.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_does_not_load_a_stale_canonical_analysis() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "titan-compare-provenance-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        std::fs::create_dir_all(&root)?;
        let config = crate::config::RunConfig {
            output_dir: root.clone(),
            ..Default::default()
        };
        let paths = ArtifactPaths::new(&config);
        for path in [&paths.metrics, &root.join("titan_image_metrics_v8.csv")] {
            std::fs::write(path, "step,loss_content\n12,0.25\n")?;
        }
        for path in [
            &paths.metadata,
            &root.join("titan_image_run_metadata_v8.json"),
        ] {
            std::fs::write(path, b"{}")?;
        }
        let stale = br#"{"target_separability":{"mean_output_l1":999.0}}"#;
        std::fs::write(&paths.analysis, stale)?;
        let comparison = compare_v8_v9(&root, &paths, None)?;
        assert_eq!(comparison.v9.target_separability, None);
        assert_eq!(comparison.v9_analysis_archive, None);
        assert_eq!(comparison.v9.reconstruction_content, Some(0.25));
        assert_eq!(std::fs::read(&paths.analysis)?, stale);
        assert!(
            !paths.comparison.exists(),
            "only the evaluation writer publishes outputs"
        );
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn parses_flat_metrics_without_external_csv_dependency() -> Result<()> {
        let root = std::env::temp_dir().join(format!("titan-v9-compare-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let path = root.join("metrics.csv");
        std::fs::write(&path, "a,b\n1.5,2\n")?;
        let row = last_csv_row(&path)?;
        assert_eq!(field(&row, "a"), Some(1.5));
        assert_eq!(field(&row, "b"), Some(2.0));
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
