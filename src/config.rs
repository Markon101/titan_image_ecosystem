use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const SCHEMA_VERSION: u32 = 9;
pub const MIN_PHYSICAL_CHANNELS: usize = 12;
const CLI_HELP_V9: &str = r#"TITAN Image 0.9.1 (schema 9) - compact recurrent morphogenic learner
Usage: titan_image --corpus-dir PATH [options]

Lifecycle and presets:
  --corpus-dir PATH             PNG/JPEG/WebP corpus (required)
  --output-dir PATH             v9 artifact directory
  --run-tag NAME                Artifact/checkpoint namespace
  --fresh                       Start a new v9 organism
  --render-only                 Load checkpoint and render without training
  --analysis-only               Load checkpoint and run frozen-state analysis
  --profile NAME                s25-fast | s25-balanced | s25-quality
  --style NAME                  alien-fluid | fractal-flame | reaction-garden | quasicrystal | pure-nca
  --research-preset NAME        strict-reconstruct | reconstruction-plus | grounded-emergent | free-morph | flow-reconstruct
  --list-presets                Explain compute/style/research presets

Reconstruction++:
  --objective NAME              endpoint | reconstruction-plus | flow | hybrid-flow
  --conditioning NAME           generate | hybrid | reconstruct
  --grounding-strength X        Early grounded-objective scale
  --grounding-floor X           Mature fraction of grounding retained
  --emergence-strength X        Mature residual contribution
  --emergence-start X           Normalized developmental start age
  --emergence-ramp X            Smooth-ramp width
  --emergent-limit X            Smooth residual-state bound
  --emergence-low-budget X      Low-frequency residual access, 0..1
  --emergence-mid-budget X      Mesoscale residual access, 0..1
  --local-reference-gain X      Bounded direct RGB-to-state drive gain
  --loss-ground-coarse X        Coarse grounded reconstruction weight
  --loss-ground-mid X           Medium grounded reconstruction weight
  --loss-ground-fine X          Fine grounded reconstruction weight
  --loss-emergent-fit X         Compatible residual-fit weight
  --loss-emergent-low X         Low-band residual regularizer
  --loss-emergent-tv X          Residual total-variation regularizer
  --loss-head-redundancy X      Ground/emergent correlation penalty
  --loss-cross-resolution X     Scheduled same-anatomy consistency weight

Native detail and boundaries:
  --detail-crop-probability X   Mature probability of a native-detail window
  --detail-resolution N         Differentiable crop render edge
  --detail-min-zoom X           Minimum crop zoom
  --detail-max-zoom X           Maximum crop zoom
  --detail-curriculum-start X   Normalized age before crop windows begin
  --pyramid-cache-max-level N   Largest cached source pyramid edge
  --pyramid-cache-dir PATH      Persistent pyramid cache override
  --target-boundary NAME        natural | periodic | crop

Morphic capacity:
  --morph-layers N              Allocated append-preserving blocks
  --morph-depth N               Fresh fixed-mode active depth
  --morph-depth-mode NAME       fixed | capacity | adaptive
  --morph-min-depth N           Fresh adaptive starting depth
  --morph-max-depth N           Maximum active reserve
  --morph-growth-interval N     Conservative activation check cadence
  --morph-plateau-window N      Grounding-history window
  --morph-plateau-epsilon X     Maximum improvement treated as plateau
  --morph-seam-threshold X      Assimilation seam safety threshold

Experimental conditional rectified flow:
  --flow-weight X               Conditional flow-loss weight
  --flow-endpoint-weight X      Hybrid Reconstruction++ weight
  --flow-resolution N           Global Oklab flow edge (balanced: 64)
  --flow-cadence N              Hybrid active-window cadence
  --flow-min-time X             Minimum sampled path time
  --flow-max-time X             Maximum sampled path time
  --flow-hidden N               Velocity-head width
  --flow-sample-steps N         Analysis midpoint ODE steps

Analysis:
  --probe-dir PATH              Held-out natural images; requires --analysis-only
  --probe-ages LIST             Comma-separated developmental ages (default 1,8,16,32,64)
  --probe-reference-fidelities LIST
                                Comma-separated reference strengths (default 1,0.5,0.25,0.1,0)
  --render-attribution          Frozen-state decomposition and emergence sweep
  --model-stats                 Request checkpoint model statistics
  --autonomous-rollout N        Mature frozen-weight continuation horizon
  --perturbation-analysis N     Deterministic damage/recovery horizon
  --dynamics-ablation N         Causal cloned-trajectory horizon
  --analysis-stride N           Heavy-analysis sampling stride
  --benchmark                   Asymmetric deterministic reconstruction suite
  --compare-v8-dir PATH         Compare compatible saved v8/v9 run artifacts
  --no-emergence-gallery        Disable final decomposition/frontier sweep

Training, architecture, and phone controls:
  --mode MODE                   single | family | texture
  --steps N                     Additional development steps
  -t, --threads N               CPU threads
  --terminal MODE               compact | rich | quiet
  --train-resolution N          Whole-image supervision edge
  --output-resolution N         Final/gallery edge
  --snapshot-resolution N       Preview/ladder edge
  --episode-steps N --bptt N --core-update-every N
  --snapshot-every N --checkpoint-every N --log-every N
  --micro-size N --macro-size N --channels N --genome-dim N
  --interface-grid N            2..16; 8 is balanced reconstruction default
  --interface-width N --interface-loops N --interface-gain X
  --render-hidden N --render-blocks N --coord-bands N --coord-gain X
  --state-skip X --chroma X --gamma X
  --reference-fidelity-min X --reference-fidelity-max X --reference-dropout X

Stability and optimizer:
  --dt X --nca-gain X --state-leak X --state-limit X --state-soft-limit X
  --loss-state X --loss-memory X --grad-clip X --stability-patience N
  --optimizer adamw|hybrid-muon --learning-rate X --weight-decay X
  --warmup-updates N --muon-momentum X --muon-ns-steps N

Dynamics/output flags remain available; use README.md for equations and full examples.
  -h, --help                    Show this help
  -V, --version                 Show package/schema version"#;

const PRESETS_HELP_V9: &str = r#"Compute profiles:
  s25-fast      48/24 fields, 16ch, 4x4 interface, morph L2/4, 128px global, 96px crop
  s25-balanced  64/32 fields, 24ch, 8x8 interface, morph L3/6, 192px global, 128px crop
  s25-quality   80/40 fields, 32ch, 8x8 interface, morph L4/8, 256px global, 192px crop

Research presets:
  strict-reconstruct    maximum grounding, minimal synthesis
  reconstruction-plus  stable grounded/emergent default
  grounded-emergent     more late and mesoscale freedom
  free-morph            weak reference, synthesis-oriented
  flow-reconstruct      experimental endpoint + exact conditional rectified flow

Style bases:
  alien-fluid | fractal-flame | reaction-garden | quasicrystal | pure-nca

Preset categories are resolved in profile -> style -> research order, then every
explicit scalar flag overrides them regardless of CLI argument order."#;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    Single,
    Family,
    Texture,
}

impl TrainingMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "single" => Ok(Self::Single),
            "family" => Ok(Self::Family),
            "texture" => Ok(Self::Texture),
            _ => bail!("invalid --mode {value}; expected single, family, or texture"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum PhoneProfile {
    S25Fast,
    S25Balanced,
    S25Quality,
}

impl PhoneProfile {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "s25-fast" => Ok(Self::S25Fast),
            "s25-balanced" | "balanced" => Ok(Self::S25Balanced),
            "s25-quality" | "quality" => Ok(Self::S25Quality),
            _ => {
                bail!("invalid --profile {value}; expected s25-fast, s25-balanced, or s25-quality")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum StylePreset {
    AlienFluid,
    FractalFlame,
    ReactionGarden,
    Quasicrystal,
    PureNca,
}

impl StylePreset {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "alien-fluid" | "alien" => Ok(Self::AlienFluid),
            "fractal-flame" | "fractal" => Ok(Self::FractalFlame),
            "reaction-garden" | "reaction" => Ok(Self::ReactionGarden),
            "quasicrystal" | "quasi" => Ok(Self::Quasicrystal),
            "pure-nca" => Ok(Self::PureNca),
            _ => bail!(
                "invalid --style {value}; expected alien-fluid, fractal-flame, reaction-garden, quasicrystal, or pure-nca"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Integrator {
    Euler,
    Midpoint,
}

impl Integrator {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "euler" => Ok(Self::Euler),
            "midpoint" | "rk2" => Ok(Self::Midpoint),
            _ => bail!("invalid --integrator {value}; expected euler or midpoint"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConditioningMode {
    Generate,
    Hybrid,
    Reconstruct,
}

impl ConditioningMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "generate" | "unconditioned" => Ok(Self::Generate),
            "hybrid" => Ok(Self::Hybrid),
            "reconstruct" | "reconstruction" => Ok(Self::Reconstruct),
            _ => bail!("invalid --conditioning {value}; expected generate, hybrid, or reconstruct"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    AdamW,
    HybridMuon,
}

impl OptimizerKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "adamw" => Ok(Self::AdamW),
            "hybrid-muon" | "muon" => Ok(Self::HybridMuon),
            _ => bail!("invalid --optimizer {value}; expected adamw or hybrid-muon"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ObjectiveMode {
    Endpoint,
    ReconstructionPlus,
    Flow,
    HybridFlow,
}

impl ObjectiveMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "endpoint" => Ok(Self::Endpoint),
            "reconstruction-plus" | "reconstruction++" | "reconstruction" => {
                Ok(Self::ReconstructionPlus)
            }
            "flow" | "flow-matching" => Ok(Self::Flow),
            "hybrid-flow" | "endpoint-flow" => Ok(Self::HybridFlow),
            _ => bail!(
                "invalid --objective {value}; expected endpoint, reconstruction-plus, flow, or hybrid-flow"
            ),
        }
    }

    pub fn uses_endpoint(self) -> bool {
        !matches!(self, Self::Flow)
    }

    pub fn uses_flow(self) -> bool {
        matches!(self, Self::Flow | Self::HybridFlow)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MorphDepthMode {
    Fixed,
    Capacity,
    Adaptive,
}

impl MorphDepthMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "fixed" => Ok(Self::Fixed),
            "capacity" => Ok(Self::Capacity),
            "adaptive" => Ok(Self::Adaptive),
            _ => bail!("invalid --morph-depth-mode {value}; expected fixed, capacity, or adaptive"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BoundaryMode {
    Natural,
    Periodic,
    Crop,
}

impl BoundaryMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "natural" => Ok(Self::Natural),
            "periodic" | "texture" => Ok(Self::Periodic),
            "crop" | "interior" => Ok(Self::Crop),
            _ => bail!("invalid --target-boundary {value}; expected natural, periodic, or crop"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TerminalMode {
    Compact,
    Rich,
    Quiet,
}

impl TerminalMode {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "compact" => Ok(Self::Compact),
            "rich" => Ok(Self::Rich),
            "quiet" => Ok(Self::Quiet),
            _ => bail!("invalid --terminal {value}; expected compact, rich, or quiet"),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ResearchPreset {
    StrictReconstruct,
    ReconstructionPlus,
    GroundedEmergent,
    FreeMorph,
    FlowReconstruct,
}

impl ResearchPreset {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "strict-reconstruct" => Ok(Self::StrictReconstruct),
            "reconstruction-plus" | "default" => Ok(Self::ReconstructionPlus),
            "grounded-emergent" => Ok(Self::GroundedEmergent),
            "free-morph" => Ok(Self::FreeMorph),
            "flow-reconstruct" => Ok(Self::FlowReconstruct),
            _ => bail!(
                "invalid --research-preset {value}; expected strict-reconstruct, reconstruction-plus, grounded-emergent, free-morph, or flow-reconstruct"
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReconstructionConfig {
    pub grounding_strength: f32,
    pub emergence_strength: f32,
    pub emergence_start: f32,
    pub emergence_ramp: f32,
    pub grounding_floor: f32,
    pub emergent_limit: f32,
    pub emergence_low_budget: f32,
    pub emergence_mid_budget: f32,
    pub local_reference_gain: f32,
    pub loss_composite: f32,
    pub loss_ground_coarse: f32,
    pub loss_ground_mid: f32,
    pub loss_ground_fine: f32,
    pub loss_emergent_low: f32,
    pub loss_emergent_tv: f32,
    pub loss_emergent_fit: f32,
    pub loss_head_redundancy: f32,
    pub loss_cross_resolution: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetailConfig {
    pub probability: f32,
    pub resolution: usize,
    pub min_zoom: f32,
    pub max_zoom: f32,
    pub curriculum_start: f32,
    pub cache_max_level: usize,
    pub cache_dir: Option<PathBuf>,
    pub boundary: BoundaryMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MorphGrowthConfig {
    pub mode: MorphDepthMode,
    pub min_depth: usize,
    pub max_depth: usize,
    pub interval: usize,
    pub plateau_window: usize,
    pub plateau_epsilon: f32,
    pub seam_threshold: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlowConfig {
    pub weight: f32,
    pub endpoint_weight: f32,
    pub resolution: usize,
    pub cadence: usize,
    pub min_time: f32,
    pub max_time: f32,
    pub hidden: usize,
    pub sample_steps: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisConfig {
    pub only: bool,
    pub render_attribution: bool,
    pub model_stats: bool,
    pub autonomous_horizon: usize,
    pub perturbation_horizon: usize,
    pub dynamics_horizon: usize,
    pub stride: usize,
    pub emergence_gallery: bool,
    pub benchmark: bool,
    pub compare_v8_dir: Option<PathBuf>,
    pub probe_dir: Option<PathBuf>,
    pub probe_ages: Vec<usize>,
    pub probe_reference_fidelities: Vec<f32>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub corpus_dir: PathBuf,
    pub output_dir: PathBuf,
    pub run_tag: Option<String>,
    pub mode: TrainingMode,
    pub profile: PhoneProfile,
    pub style: StylePreset,
    pub research_preset: ResearchPreset,
    pub objective: ObjectiveMode,
    pub reconstruction: ReconstructionConfig,
    pub detail: DetailConfig,
    pub morph_growth: MorphGrowthConfig,
    pub flow: FlowConfig,
    pub analysis: AnalysisConfig,
    pub steps: usize,
    pub threads: usize,
    pub seed: u64,
    pub micro_size: usize,
    pub macro_size: usize,
    pub channels: usize,
    pub genome_dim: usize,
    pub ca_hidden: usize,
    pub interface_grid: usize,
    pub interface_width: usize,
    pub interface_loops: usize,
    pub morph_layers: usize,
    pub morph_depth: usize,
    pub morph_residual_gain: f32,
    pub memory_limit: f32,
    pub interface_gain: f32,
    pub conditioning: ConditioningMode,
    pub reference_fidelity_min: f32,
    pub reference_fidelity_max: f32,
    pub reference_dropout: f32,
    pub render_hidden: usize,
    pub render_blocks: usize,
    pub coord_bands: usize,
    pub coord_gain: f32,
    pub state_skip: f32,
    pub chroma: f32,
    pub gamma: f32,
    pub train_resolution: usize,
    pub output_resolution: usize,
    pub snapshot_resolution: usize,
    pub episode_steps: usize,
    pub episode_reset: f32,
    pub memory_reset: f32,
    pub bptt: usize,
    pub core_update_every: usize,
    pub macro_update_every: usize,
    pub snapshot_every: usize,
    pub checkpoint_every: usize,
    pub log_every: usize,
    pub terminal: TerminalMode,
    pub image_cache: usize,
    pub recursive_corpus: bool,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub adam_epsilon: f64,
    pub grad_clip: f64,
    pub warmup_updates: usize,
    pub optimizer: OptimizerKind,
    pub muon_momentum: f64,
    pub muon_ns_steps: usize,
    pub integrator: Integrator,
    pub dt: f32,
    pub state_limit: f32,
    pub state_soft_limit: f32,
    pub clock_probability: f32,
    pub nca_gain: f32,
    pub reaction_gain: f32,
    pub state_leak: f32,
    pub phase_gain: f32,
    pub fractal_gain: f32,
    pub quasiperiodic_gain: f32,
    pub cyclic_gain: f32,
    pub rd_diffusion_u: f32,
    pub rd_diffusion_v: f32,
    pub rd_feed: f32,
    pub rd_kill: f32,
    pub phase_growth: f32,
    pub phase_saturation: f32,
    pub phase_frequency: f32,
    pub phase_diffusion: f32,
    pub phase_dispersion: f32,
    pub loss_content: f32,
    pub loss_palette: f32,
    pub loss_structure: f32,
    pub loss_seam: f32,
    pub loss_gamut: f32,
    pub loss_state: f32,
    pub loss_memory: f32,
    pub max_saturation_fraction: f32,
    pub stability_patience: usize,
    pub mastering_strength: f32,
    pub save_state_atlas: bool,
    pub gallery: usize,
    pub gallery_steps: usize,
    pub gallery_stride: usize,
    pub gallery_seed: u64,
    pub fresh: bool,
    pub render_only: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4);
        let mut config = Self {
            corpus_dir: PathBuf::from("/sdcard/Download/titan_image_sources"),
            output_dir: PathBuf::from("/sdcard/Download/titan_image_v9"),
            run_tag: None,
            mode: TrainingMode::Family,
            profile: PhoneProfile::S25Balanced,
            style: StylePreset::AlienFluid,
            research_preset: ResearchPreset::ReconstructionPlus,
            objective: ObjectiveMode::ReconstructionPlus,
            reconstruction: ReconstructionConfig {
                grounding_strength: 1.0,
                emergence_strength: 0.35,
                emergence_start: 0.25,
                emergence_ramp: 0.50,
                grounding_floor: 0.70,
                emergent_limit: 0.30,
                emergence_low_budget: 0.05,
                emergence_mid_budget: 0.35,
                local_reference_gain: 0.12,
                loss_composite: 1.0,
                loss_ground_coarse: 2.5,
                loss_ground_mid: 1.5,
                loss_ground_fine: 0.75,
                loss_emergent_low: 0.20,
                loss_emergent_tv: 0.015,
                loss_emergent_fit: 0.25,
                loss_head_redundancy: 0.01,
                loss_cross_resolution: 0.05,
            },
            detail: DetailConfig {
                probability: 0.20,
                resolution: 128,
                min_zoom: 2.0,
                max_zoom: 8.0,
                curriculum_start: 0.35,
                cache_max_level: 1536,
                cache_dir: None,
                boundary: BoundaryMode::Natural,
            },
            morph_growth: MorphGrowthConfig {
                mode: MorphDepthMode::Fixed,
                min_depth: 3,
                max_depth: 6,
                interval: 512,
                plateau_window: 128,
                plateau_epsilon: 0.002,
                seam_threshold: 0.05,
            },
            flow: FlowConfig {
                weight: 0.10,
                endpoint_weight: 1.0,
                min_time: 0.0,
                max_time: 1.0,
                hidden: 64,
                sample_steps: 16,
                resolution: 64,
                cadence: 4,
            },
            analysis: AnalysisConfig {
                only: false,
                render_attribution: false,
                model_stats: false,
                autonomous_horizon: 0,
                perturbation_horizon: 0,
                dynamics_horizon: 0,
                stride: 32,
                emergence_gallery: true,
                benchmark: false,
                compare_v8_dir: None,
                probe_dir: None,
                probe_ages: vec![1, 8, 16, 32, 64],
                probe_reference_fidelities: vec![1.0, 0.5, 0.25, 0.1, 0.0],
            },
            steps: 1600,
            threads,
            seed: 42,
            micro_size: 64,
            macro_size: 32,
            channels: 24,
            genome_dim: 8,
            ca_hidden: 128,
            render_hidden: 128,
            render_blocks: 4,
            interface_grid: 8,
            interface_width: 128,
            interface_loops: 3,
            morph_layers: 6,
            morph_depth: 3,
            morph_residual_gain: 0.02,
            memory_limit: 3.0,
            interface_gain: 0.18,
            conditioning: ConditioningMode::Reconstruct,
            reference_fidelity_min: 0.15,
            reference_fidelity_max: 0.85,
            reference_dropout: 0.15,
            coord_bands: 4,
            coord_gain: 0.12,
            state_skip: 0.75,
            chroma: 0.20,
            gamma: 2.2,
            train_resolution: 192,
            output_resolution: 768,
            snapshot_resolution: 384,
            episode_steps: 64,
            episode_reset: 0.85,
            bptt: 4,
            memory_reset: 1.0,
            core_update_every: 2,
            macro_update_every: 4,
            snapshot_every: 50,
            checkpoint_every: 400,
            log_every: 10,
            terminal: TerminalMode::Compact,
            image_cache: 32,
            recursive_corpus: false,
            learning_rate: 6e-4,
            weight_decay: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            adam_epsilon: 1e-8,
            grad_clip: 1.0,
            warmup_updates: 48,
            integrator: Integrator::Euler,
            dt: 0.12,
            state_limit: 3.5,
            optimizer: OptimizerKind::AdamW,
            muon_momentum: 0.95,
            state_soft_limit: 1.5,
            muon_ns_steps: 5,
            clock_probability: 0.72,
            nca_gain: 0.25,
            reaction_gain: 0.35,
            phase_gain: 0.85,
            fractal_gain: 0.08,
            quasiperiodic_gain: 0.12,
            state_leak: 0.10,
            cyclic_gain: 0.22,
            rd_diffusion_u: 0.16,
            rd_diffusion_v: 0.08,
            rd_feed: 0.035,
            rd_kill: 0.061,
            phase_growth: 0.28,
            phase_saturation: 0.42,
            phase_frequency: 0.72,
            phase_diffusion: 0.12,
            phase_dispersion: 0.035,
            loss_content: 1.0,
            loss_palette: 1.1,
            loss_structure: 0.75,
            loss_seam: 0.08,
            loss_gamut: 0.18,
            mastering_strength: 0.75,
            save_state_atlas: true,
            loss_state: 0.02,
            loss_memory: 0.002,
            max_saturation_fraction: 0.25,
            stability_patience: 8,
            gallery: 4,
            gallery_steps: 64,
            gallery_stride: 16,
            gallery_seed: 0x5eed_5eed,
            fresh: false,
            render_only: false,
        };
        config.apply_profile(PhoneProfile::S25Balanced);
        config.apply_style(StylePreset::AlienFluid);
        config.apply_research_preset(ResearchPreset::ReconstructionPlus);
        config
    }
}

impl RunConfig {
    pub fn parse_env() -> Result<Option<Self>> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.iter().any(|arg| arg == "--help" || arg == "-h") {
            println!("{}", Self::help());
            return Ok(None);
        }
        if args.iter().any(|arg| arg == "--version" || arg == "-V") {
            println!(
                "titan_image {} (schema v{SCHEMA_VERSION})",
                env!("CARGO_PKG_VERSION")
            );
            return Ok(None);
        }
        if args.iter().any(|arg| arg == "--list-presets") {
            println!("{}", Self::presets_help());
            return Ok(None);
        }

        let mut cfg = Self::default();
        preapply_presets(&args, &mut cfg)?;
        let mut corpus_was_explicit = false;
        let mut i = 0;
        while i < args.len() {
            let flag = &args[i];
            let mut value = || -> Result<String> {
                i += 1;
                args.get(i)
                    .cloned()
                    .with_context(|| format!("missing value for {flag}"))
            };
            match flag.as_str() {
                "--corpus-dir" => {
                    cfg.corpus_dir = PathBuf::from(value()?);
                    corpus_was_explicit = true;
                }
                "--output-dir" => cfg.output_dir = PathBuf::from(value()?),
                "--run-tag" => cfg.run_tag = Some(validate_tag(&value()?)?),
                "--mode" => cfg.mode = TrainingMode::parse(&value()?)?,
                "--profile" => {
                    let _ = PhoneProfile::parse(&value()?)?;
                }
                "--style" => {
                    let _ = StylePreset::parse(&value()?)?;
                }
                "--research-preset" => {
                    let _ = ResearchPreset::parse(&value()?)?;
                }
                "--objective" => cfg.objective = ObjectiveMode::parse(&value()?)?,
                "--grounding-strength" => {
                    cfg.reconstruction.grounding_strength = parse(&value()?, flag)?
                }
                "--emergence-strength" => {
                    cfg.reconstruction.emergence_strength = parse(&value()?, flag)?
                }
                "--emergence-low-budget" => {
                    cfg.reconstruction.emergence_low_budget = parse(&value()?, flag)?
                }
                "--emergence-mid-budget" => {
                    cfg.reconstruction.emergence_mid_budget = parse(&value()?, flag)?
                }
                "--emergence-start" => cfg.reconstruction.emergence_start = parse(&value()?, flag)?,
                "--emergence-ramp" => cfg.reconstruction.emergence_ramp = parse(&value()?, flag)?,
                "--grounding-floor" => cfg.reconstruction.grounding_floor = parse(&value()?, flag)?,
                "--emergent-limit" => cfg.reconstruction.emergent_limit = parse(&value()?, flag)?,
                "--local-reference-gain" => {
                    cfg.reconstruction.local_reference_gain = parse(&value()?, flag)?
                }
                "--loss-composite" => cfg.reconstruction.loss_composite = parse(&value()?, flag)?,
                "--loss-ground-coarse" => {
                    cfg.reconstruction.loss_ground_coarse = parse(&value()?, flag)?
                }
                "--loss-ground-mid" => cfg.reconstruction.loss_ground_mid = parse(&value()?, flag)?,
                "--loss-ground-fine" => {
                    cfg.reconstruction.loss_ground_fine = parse(&value()?, flag)?
                }
                "--loss-emergent-low" => {
                    cfg.reconstruction.loss_emergent_low = parse(&value()?, flag)?
                }
                "--loss-emergent-tv" => {
                    cfg.reconstruction.loss_emergent_tv = parse(&value()?, flag)?
                }
                "--detail-crop-probability" => cfg.detail.probability = parse(&value()?, flag)?,
                "--loss-emergent-fit" => {
                    cfg.reconstruction.loss_emergent_fit = parse(&value()?, flag)?
                }
                "--loss-head-redundancy" => {
                    cfg.reconstruction.loss_head_redundancy = parse(&value()?, flag)?
                }
                "--loss-cross-resolution" => {
                    cfg.reconstruction.loss_cross_resolution = parse(&value()?, flag)?
                }
                "--detail-resolution" => cfg.detail.resolution = parse(&value()?, flag)?,
                "--detail-min-zoom" => cfg.detail.min_zoom = parse(&value()?, flag)?,
                "--detail-max-zoom" => cfg.detail.max_zoom = parse(&value()?, flag)?,
                "--detail-curriculum-start" => {
                    cfg.detail.curriculum_start = parse(&value()?, flag)?
                }
                "--pyramid-cache-max-level" => cfg.detail.cache_max_level = parse(&value()?, flag)?,
                "--pyramid-cache-dir" => cfg.detail.cache_dir = Some(PathBuf::from(value()?)),
                "--target-boundary" => cfg.detail.boundary = BoundaryMode::parse(&value()?)?,
                "--morph-depth-mode" => cfg.morph_growth.mode = MorphDepthMode::parse(&value()?)?,
                "--morph-min-depth" => cfg.morph_growth.min_depth = parse(&value()?, flag)?,
                "--morph-max-depth" => cfg.morph_growth.max_depth = parse(&value()?, flag)?,
                "--morph-growth-interval" => cfg.morph_growth.interval = parse(&value()?, flag)?,
                "--morph-plateau-window" => {
                    cfg.morph_growth.plateau_window = parse(&value()?, flag)?
                }
                "--morph-plateau-epsilon" => {
                    cfg.morph_growth.plateau_epsilon = parse(&value()?, flag)?
                }
                "--morph-seam-threshold" => {
                    cfg.morph_growth.seam_threshold = parse(&value()?, flag)?
                }
                "--flow-weight" => cfg.flow.weight = parse(&value()?, flag)?,
                "--flow-endpoint-weight" => cfg.flow.endpoint_weight = parse(&value()?, flag)?,
                "--flow-min-time" => cfg.flow.min_time = parse(&value()?, flag)?,
                "--flow-max-time" => cfg.flow.max_time = parse(&value()?, flag)?,
                "--flow-hidden" => cfg.flow.hidden = parse(&value()?, flag)?,
                "--flow-resolution" => cfg.flow.resolution = parse(&value()?, flag)?,
                "--flow-cadence" => cfg.flow.cadence = parse(&value()?, flag)?,
                "--flow-sample-steps" => cfg.flow.sample_steps = parse(&value()?, flag)?,
                "--terminal" => cfg.terminal = TerminalMode::parse(&value()?)?,
                "--analysis-only" => cfg.analysis.only = true,
                "--render-attribution" => cfg.analysis.render_attribution = true,
                "--model-stats" => cfg.analysis.model_stats = true,
                "--benchmark" => cfg.analysis.benchmark = true,
                "--probe-dir" => cfg.analysis.probe_dir = Some(PathBuf::from(value()?)),
                "--probe-ages" => cfg.analysis.probe_ages = parse_csv(&value()?, flag)?,
                "--probe-reference-fidelities" => {
                    cfg.analysis.probe_reference_fidelities = parse_csv(&value()?, flag)?
                }
                "--compare-v8-dir" => cfg.analysis.compare_v8_dir = Some(PathBuf::from(value()?)),
                "--autonomous-rollout" => cfg.analysis.autonomous_horizon = parse(&value()?, flag)?,
                "--perturbation-analysis" => {
                    cfg.analysis.perturbation_horizon = parse(&value()?, flag)?
                }
                "--dynamics-ablation" => cfg.analysis.dynamics_horizon = parse(&value()?, flag)?,
                "--analysis-stride" => cfg.analysis.stride = parse(&value()?, flag)?,
                "--no-emergence-gallery" => cfg.analysis.emergence_gallery = false,
                "--steps" => cfg.steps = parse(&value()?, flag)?,
                "--threads" | "-t" => cfg.threads = parse(&value()?, flag)?,
                "--seed" => cfg.seed = parse(&value()?, flag)?,
                "--micro-size" => cfg.micro_size = parse(&value()?, flag)?,
                "--macro-size" => cfg.macro_size = parse(&value()?, flag)?,
                "--channels" => cfg.channels = parse(&value()?, flag)?,
                "--genome-dim" => cfg.genome_dim = parse(&value()?, flag)?,
                "--ca-hidden" => cfg.ca_hidden = parse(&value()?, flag)?,
                "--interface-grid" => cfg.interface_grid = parse(&value()?, flag)?,
                "--interface-width" => cfg.interface_width = parse(&value()?, flag)?,
                "--interface-loops" => cfg.interface_loops = parse(&value()?, flag)?,
                "--morph-layers" => cfg.morph_layers = parse(&value()?, flag)?,
                "--morph-depth" => cfg.morph_depth = parse(&value()?, flag)?,
                "--morph-residual-gain" => cfg.morph_residual_gain = parse(&value()?, flag)?,
                "--memory-limit" => cfg.memory_limit = parse(&value()?, flag)?,
                "--interface-gain" => cfg.interface_gain = parse(&value()?, flag)?,
                "--conditioning" => cfg.conditioning = ConditioningMode::parse(&value()?)?,
                "--reference-fidelity-min" => cfg.reference_fidelity_min = parse(&value()?, flag)?,
                "--reference-fidelity-max" => cfg.reference_fidelity_max = parse(&value()?, flag)?,
                "--reference-dropout" => cfg.reference_dropout = parse(&value()?, flag)?,
                "--render-hidden" => cfg.render_hidden = parse(&value()?, flag)?,
                "--render-blocks" => cfg.render_blocks = parse(&value()?, flag)?,
                "--coord-bands" => cfg.coord_bands = parse(&value()?, flag)?,
                "--coord-gain" => cfg.coord_gain = parse(&value()?, flag)?,
                "--state-skip" => cfg.state_skip = parse(&value()?, flag)?,
                "--chroma" => cfg.chroma = parse(&value()?, flag)?,
                "--gamma" => cfg.gamma = parse(&value()?, flag)?,
                "--train-resolution" => cfg.train_resolution = parse(&value()?, flag)?,
                "--output-resolution" => cfg.output_resolution = parse(&value()?, flag)?,
                "--snapshot-resolution" => cfg.snapshot_resolution = parse(&value()?, flag)?,
                "--episode-steps" => cfg.episode_steps = parse(&value()?, flag)?,
                "--episode-reset" => cfg.episode_reset = parse(&value()?, flag)?,
                "--memory-reset" => cfg.memory_reset = parse(&value()?, flag)?,
                "--bptt" => cfg.bptt = parse(&value()?, flag)?,
                "--core-update-every" => cfg.core_update_every = parse(&value()?, flag)?,
                "--macro-update-every" => cfg.macro_update_every = parse(&value()?, flag)?,
                "--snapshot-every" => cfg.snapshot_every = parse(&value()?, flag)?,
                "--checkpoint-every" => cfg.checkpoint_every = parse(&value()?, flag)?,
                "--log-every" => cfg.log_every = parse(&value()?, flag)?,
                "--image-cache" => cfg.image_cache = parse(&value()?, flag)?,
                "--recursive-corpus" => cfg.recursive_corpus = true,
                "--learning-rate" => cfg.learning_rate = parse(&value()?, flag)?,
                "--weight-decay" => cfg.weight_decay = parse(&value()?, flag)?,
                "--beta1" => cfg.beta1 = parse(&value()?, flag)?,
                "--beta2" => cfg.beta2 = parse(&value()?, flag)?,
                "--adam-epsilon" => cfg.adam_epsilon = parse(&value()?, flag)?,
                "--grad-clip" => cfg.grad_clip = parse(&value()?, flag)?,
                "--warmup-updates" => cfg.warmup_updates = parse(&value()?, flag)?,
                "--optimizer" => cfg.optimizer = OptimizerKind::parse(&value()?)?,
                "--muon-momentum" => cfg.muon_momentum = parse(&value()?, flag)?,
                "--muon-ns-steps" => cfg.muon_ns_steps = parse(&value()?, flag)?,
                "--integrator" => cfg.integrator = Integrator::parse(&value()?)?,
                "--dt" => cfg.dt = parse(&value()?, flag)?,
                "--state-limit" => cfg.state_limit = parse(&value()?, flag)?,
                "--state-soft-limit" => cfg.state_soft_limit = parse(&value()?, flag)?,
                "--clock-probability" => cfg.clock_probability = parse(&value()?, flag)?,
                "--nca-gain" => cfg.nca_gain = parse(&value()?, flag)?,
                "--reaction-gain" => cfg.reaction_gain = parse(&value()?, flag)?,
                "--state-leak" => cfg.state_leak = parse(&value()?, flag)?,
                "--phase-gain" => cfg.phase_gain = parse(&value()?, flag)?,
                "--fractal-gain" => cfg.fractal_gain = parse(&value()?, flag)?,
                "--quasiperiodic-gain" => {
                    cfg.quasiperiodic_gain = parse(&value()?, flag)?;
                }
                "--cyclic-gain" => cfg.cyclic_gain = parse(&value()?, flag)?,
                "--rd-diffusion-u" => cfg.rd_diffusion_u = parse(&value()?, flag)?,
                "--rd-diffusion-v" => cfg.rd_diffusion_v = parse(&value()?, flag)?,
                "--rd-feed" => cfg.rd_feed = parse(&value()?, flag)?,
                "--rd-kill" => cfg.rd_kill = parse(&value()?, flag)?,
                "--phase-growth" => cfg.phase_growth = parse(&value()?, flag)?,
                "--phase-saturation" => cfg.phase_saturation = parse(&value()?, flag)?,
                "--phase-frequency" => cfg.phase_frequency = parse(&value()?, flag)?,
                "--phase-diffusion" => cfg.phase_diffusion = parse(&value()?, flag)?,
                "--phase-dispersion" => cfg.phase_dispersion = parse(&value()?, flag)?,
                "--loss-content" => cfg.loss_content = parse(&value()?, flag)?,
                "--loss-palette" => cfg.loss_palette = parse(&value()?, flag)?,
                "--loss-structure" => cfg.loss_structure = parse(&value()?, flag)?,
                "--loss-seam" => cfg.loss_seam = parse(&value()?, flag)?,
                "--loss-gamut" => cfg.loss_gamut = parse(&value()?, flag)?,
                "--loss-state" => cfg.loss_state = parse(&value()?, flag)?,
                "--loss-memory" => cfg.loss_memory = parse(&value()?, flag)?,
                "--max-saturation-fraction" => {
                    cfg.max_saturation_fraction = parse(&value()?, flag)?
                }
                "--stability-patience" => cfg.stability_patience = parse(&value()?, flag)?,
                "--mastering-strength" => cfg.mastering_strength = parse(&value()?, flag)?,
                "--gallery" => cfg.gallery = parse(&value()?, flag)?,
                "--gallery-steps" => cfg.gallery_steps = parse(&value()?, flag)?,
                "--gallery-stride" => cfg.gallery_stride = parse(&value()?, flag)?,
                "--gallery-seed" => cfg.gallery_seed = parse(&value()?, flag)?,
                "--fresh" => cfg.fresh = true,
                "--render-only" => cfg.render_only = true,
                "--no-reaction-diffusion" => cfg.reaction_gain = 0.0,
                "--no-complex-phase" => cfg.phase_gain = 0.0,
                "--no-fractal" => cfg.fractal_gain = 0.0,
                "--no-quasiperiodic" => cfg.quasiperiodic_gain = 0.0,
                "--no-cyclic" => cfg.cyclic_gain = 0.0,
                "--no-mastering" => cfg.mastering_strength = 0.0,
                "--no-state-atlas" => cfg.save_state_atlas = false,
                _ => bail!("unknown option {flag}; run with --help"),
            }
            i += 1;
        }
        if !corpus_was_explicit {
            bail!(
                "--corpus-dir is required; TITAN Image will not scan /sdcard/Download implicitly"
            );
        }
        cfg.validate()?;
        Ok(Some(cfg))
    }

    fn apply_profile(&mut self, profile: PhoneProfile) {
        self.profile = profile;
        match profile {
            PhoneProfile::S25Fast => {
                self.micro_size = 48;
                self.macro_size = 24;
                self.channels = 16;
                self.genome_dim = 6;
                self.ca_hidden = 96;
                self.interface_grid = 4;
                self.interface_width = 96;
                self.interface_loops = 2;
                self.morph_layers = 4;
                self.morph_depth = 2;
                self.render_hidden = 80;
                self.render_blocks = 3;
                self.coord_bands = 3;
                self.train_resolution = 128;
                self.output_resolution = 512;
                self.morph_growth.min_depth = 2;
                self.morph_growth.max_depth = 4;
                self.detail.resolution = 96;
                self.detail.probability = 0.15;
                self.snapshot_resolution = 256;
                self.bptt = 4;
                self.core_update_every = 2;
                self.episode_steps = 48;
                self.snapshot_every = 50;
                self.checkpoint_every = 480;
                self.gallery_steps = 48;
            }
            PhoneProfile::S25Balanced => {
                self.micro_size = 64;
                self.macro_size = 32;
                self.channels = 24;
                self.genome_dim = 8;
                self.ca_hidden = 128;
                self.render_hidden = 128;
                self.interface_grid = 8;
                self.interface_width = 128;
                self.interface_loops = 3;
                self.morph_layers = 6;
                self.morph_depth = 3;
                self.render_blocks = 4;
                self.coord_bands = 4;
                self.train_resolution = 192;
                self.output_resolution = 768;
                self.snapshot_resolution = 384;
                self.morph_growth.min_depth = 3;
                self.morph_growth.max_depth = 6;
                self.detail.resolution = 128;
                self.detail.probability = 0.20;
                self.bptt = 4;
                self.core_update_every = 2;
                self.episode_steps = 64;
                self.snapshot_every = 50;
                self.checkpoint_every = 400;
                self.gallery_steps = 64;
            }
            PhoneProfile::S25Quality => {
                self.micro_size = 80;
                self.macro_size = 40;
                self.channels = 32;
                self.genome_dim = 12;
                self.ca_hidden = 192;
                self.render_hidden = 160;
                self.interface_grid = 8;
                self.interface_width = 160;
                self.interface_loops = 4;
                self.morph_layers = 8;
                self.morph_depth = 4;
                self.render_blocks = 5;
                self.coord_bands = 5;
                self.train_resolution = 256;
                self.output_resolution = 1024;
                self.snapshot_resolution = 512;
                self.bptt = 4;
                self.core_update_every = 1;
                self.episode_steps = 80;
                self.morph_growth.min_depth = 4;
                self.morph_growth.max_depth = 8;
                self.detail.resolution = 192;
                self.detail.probability = 0.25;
                self.snapshot_every = 50;
                self.checkpoint_every = 400;
                self.gallery_steps = 80;
            }
        }
    }

    fn apply_style(&mut self, style: StylePreset) {
        self.style = style;
        match style {
            StylePreset::AlienFluid => {
                self.reaction_gain = 0.35;
                self.phase_gain = 0.85;
                self.fractal_gain = 0.08;
                self.quasiperiodic_gain = 0.12;
                self.cyclic_gain = 0.22;
                self.coord_gain = 0.12;
                self.state_skip = 0.75;
                self.chroma = 0.20;
            }
            StylePreset::FractalFlame => {
                self.reaction_gain = 0.22;
                self.phase_gain = 0.42;
                self.fractal_gain = 0.34;
                self.quasiperiodic_gain = 0.15;
                self.cyclic_gain = 0.16;
                self.coord_gain = 0.18;
                self.state_skip = 0.95;
                self.chroma = 0.22;
            }
            StylePreset::ReactionGarden => {
                self.reaction_gain = 0.95;
                self.phase_gain = 0.18;
                self.fractal_gain = 0.035;
                self.quasiperiodic_gain = 0.05;
                self.cyclic_gain = 0.28;
                self.coord_gain = 0.08;
                self.state_skip = 0.82;
                self.chroma = 0.18;
            }
            StylePreset::Quasicrystal => {
                self.reaction_gain = 0.16;
                self.phase_gain = 0.52;
                self.fractal_gain = 0.07;
                self.quasiperiodic_gain = 0.34;
                self.cyclic_gain = 0.20;
                self.coord_gain = 0.28;
                self.state_skip = 0.72;
                self.chroma = 0.21;
            }
            StylePreset::PureNca => {
                self.reaction_gain = 0.0;
                self.phase_gain = 0.0;
                self.fractal_gain = 0.0;
                self.quasiperiodic_gain = 0.0;
                self.cyclic_gain = 0.0;
                self.coord_gain = 0.08;
                self.state_skip = 0.60;
                self.chroma = 0.18;
            }
        }
    }

    fn apply_research_preset(&mut self, preset: ResearchPreset) {
        self.research_preset = preset;
        match preset {
            ResearchPreset::StrictReconstruct => {
                self.objective = ObjectiveMode::ReconstructionPlus;
                self.conditioning = ConditioningMode::Reconstruct;
                self.reference_fidelity_max = 1.0;
                self.reconstruction.grounding_strength = 1.25;
                self.reconstruction.emergence_strength = 0.05;
                self.reconstruction.grounding_floor = 0.95;
                self.reconstruction.emergence_low_budget = 0.0;
                self.reconstruction.emergence_mid_budget = 0.10;
                self.reconstruction.loss_ground_coarse = 3.0;
                self.reconstruction.loss_ground_mid = 2.0;
                self.reconstruction.loss_ground_fine = 1.25;
                self.detail.probability = 0.25;
            }
            ResearchPreset::ReconstructionPlus => {
                self.objective = ObjectiveMode::ReconstructionPlus;
                self.conditioning = ConditioningMode::Reconstruct;
                self.reference_fidelity_max = 1.0;
            }
            ResearchPreset::GroundedEmergent => {
                self.objective = ObjectiveMode::ReconstructionPlus;
                self.conditioning = ConditioningMode::Reconstruct;
                self.reference_fidelity_max = 1.0;
                self.reconstruction.emergence_strength = 0.70;
                self.reconstruction.emergence_start = 0.20;
                self.reconstruction.grounding_floor = 0.65;
                self.reconstruction.loss_emergent_low = 0.08;
                self.reconstruction.emergence_low_budget = 0.15;
                self.reconstruction.emergence_mid_budget = 0.65;
                self.morph_growth.mode = MorphDepthMode::Adaptive;
            }
            ResearchPreset::FreeMorph => {
                self.objective = ObjectiveMode::ReconstructionPlus;
                self.conditioning = ConditioningMode::Hybrid;
                self.reference_fidelity_min = 0.0;
                self.reference_fidelity_max = 0.45;
                self.reference_dropout = 0.35;
                self.reconstruction.emergence_strength = 1.0;
                self.reconstruction.emergence_start = 0.10;
                self.reconstruction.grounding_floor = 0.25;
                self.reconstruction.loss_emergent_low = 0.02;
                self.reconstruction.emergence_low_budget = 0.50;
                self.reconstruction.emergence_mid_budget = 1.0;
                self.morph_growth.mode = MorphDepthMode::Adaptive;
            }
            ResearchPreset::FlowReconstruct => {
                self.objective = ObjectiveMode::HybridFlow;
                self.conditioning = ConditioningMode::Reconstruct;
                self.reference_fidelity_max = 1.0;
                self.reconstruction.emergence_strength = 0.20;
                self.flow.weight = 0.10;
                self.flow.endpoint_weight = 1.0;
            }
        }
    }

    pub fn developmental_schedule(&self, age_phase: f32) -> (f32, f32) {
        let x = ((age_phase - self.reconstruction.emergence_start)
            / self.reconstruction.emergence_ramp.max(1e-6))
        .clamp(0.0, 1.0);
        let smooth = x * x * (3.0 - 2.0 * x);
        let grounding = self.reconstruction.grounding_strength
            * (1.0 - (1.0 - self.reconstruction.grounding_floor) * smooth);
        let emergence = self.reconstruction.emergence_strength * smooth;
        (grounding, emergence)
    }

    pub fn initial_morph_depth(&self) -> usize {
        match self.morph_growth.mode {
            MorphDepthMode::Fixed => self.morph_depth,
            MorphDepthMode::Capacity => self.morph_growth.max_depth,
            MorphDepthMode::Adaptive => self.morph_growth.min_depth,
        }
        .min(self.morph_layers)
    }

    pub fn pyramid_cache_root(&self) -> PathBuf {
        self.detail
            .cache_dir
            .clone()
            .unwrap_or_else(|| self.output_dir.join("pyramid_cache_v9"))
    }

    pub fn analysis_requested(&self) -> bool {
        self.analysis.only
            || self.analysis.render_attribution
            || self.analysis.model_stats
            || self.analysis.autonomous_horizon > 0
            || self.analysis.perturbation_horizon > 0
            || self.analysis.benchmark
            || self.analysis.probe_dir.is_some()
            || self.analysis.compare_v8_dir.is_some()
            || self.analysis.dynamics_horizon > 0
    }
    pub fn validate(&self) -> Result<()> {
        if !self.render_only && !self.analysis.only && self.steps == 0 {
            bail!("--steps must be positive unless render-only or analysis-only is selected");
        }
        if self.bptt == 0 || self.episode_steps == 0 {
            bail!("--bptt and --episode-steps must be positive");
        }
        if !self.render_only && !self.analysis.only && !self.steps.is_multiple_of(self.bptt) {
            bail!("--steps must be a multiple of --bptt");
        }
        if !self.episode_steps.is_multiple_of(self.bptt) {
            bail!("--episode-steps must be a multiple of --bptt");
        }
        if !(1..=64).contains(&self.threads) {
            bail!("--threads must be in 1..=64");
        }
        if !(24..=128).contains(&self.micro_size)
            || !(8..=self.micro_size).contains(&self.macro_size)
        {
            bail!("--micro-size must be 24..=128 and --macro-size must be 8..=micro-size");
        }
        if !(MIN_PHYSICAL_CHANNELS..=64).contains(&self.channels) {
            bail!("--channels must be in {MIN_PHYSICAL_CHANNELS}..=64");
        }
        if !(2..=32).contains(&self.genome_dim)
            || !(32..=512).contains(&self.ca_hidden)
            || !(32..=512).contains(&self.render_hidden)
            || !(1..=8).contains(&self.render_blocks)
            || !(1..=8).contains(&self.coord_bands)
        {
            bail!("architecture controls exceed their safe phone ranges");
        }
        if !(2..=16).contains(&self.interface_grid)
            || !self.micro_size.is_multiple_of(self.interface_grid)
            || !self.macro_size.is_multiple_of(self.interface_grid)
        {
            bail!("--interface-grid must be 2..=16 and divide both field sizes");
        }
        if !(32..=512).contains(&self.interface_width)
            || !self.interface_width.is_multiple_of(32)
            || !(1..=8).contains(&self.interface_loops)
            || !(1..=32).contains(&self.morph_layers)
            || !(1..=self.morph_layers).contains(&self.morph_depth)
        {
            bail!("recurrent-interface controls exceed their safe phone ranges");
        }
        if !(1..=self.morph_layers).contains(&self.morph_growth.min_depth)
            || !(self.morph_growth.min_depth..=self.morph_layers)
                .contains(&self.morph_growth.max_depth)
        {
            bail!("morph growth depths must satisfy 1 <= min <= max <= morph-layers");
        }
        if self.morph_growth.mode == MorphDepthMode::Fixed
            && !(self.morph_growth.min_depth..=self.morph_growth.max_depth)
                .contains(&self.morph_depth)
        {
            bail!("fixed --morph-depth must lie inside morph min/max depth");
        }
        if self.morph_growth.interval == 0 || self.morph_growth.plateau_window < 4 {
            bail!("morph growth interval must be positive and plateau window >= 4");
        }
        finite_range(
            self.morph_growth.plateau_epsilon,
            0.0,
            1.0,
            "--morph-plateau-epsilon",
        )?;
        finite_range(
            self.morph_growth.seam_threshold,
            0.0,
            10.0,
            "--morph-seam-threshold",
        )?;
        for (name, value) in [
            (
                "--grounding-strength",
                self.reconstruction.grounding_strength,
            ),
            (
                "--emergence-strength",
                self.reconstruction.emergence_strength,
            ),
            ("--emergence-start", self.reconstruction.emergence_start),
            ("--emergence-ramp", self.reconstruction.emergence_ramp),
            ("--grounding-floor", self.reconstruction.grounding_floor),
            ("--emergent-limit", self.reconstruction.emergent_limit),
            (
                "--local-reference-gain",
                self.reconstruction.local_reference_gain,
            ),
        ] {
            finite_range(value, 0.0, 4.0, name)?;
        }
        finite_range(
            self.reconstruction.emergence_low_budget,
            0.0,
            1.0,
            "--emergence-low-budget",
        )?;
        finite_range(
            self.reconstruction.emergence_mid_budget,
            0.0,
            1.0,
            "--emergence-mid-budget",
        )?;
        if self.reconstruction.emergence_start > 1.0
            || self.reconstruction.emergence_ramp <= 0.0
            || self.reconstruction.emergence_start + self.reconstruction.emergence_ramp > 2.0
            || self.reconstruction.grounding_floor > 1.0
        {
            bail!("reconstruction curriculum must use normalized, nonzero schedule controls");
        }
        for (name, value) in [
            ("--loss-composite", self.reconstruction.loss_composite),
            (
                "--loss-ground-coarse",
                self.reconstruction.loss_ground_coarse,
            ),
            ("--loss-ground-mid", self.reconstruction.loss_ground_mid),
            ("--loss-ground-fine", self.reconstruction.loss_ground_fine),
            ("--loss-emergent-low", self.reconstruction.loss_emergent_low),
            ("--loss-emergent-tv", self.reconstruction.loss_emergent_tv),
            ("--loss-emergent-fit", self.reconstruction.loss_emergent_fit),
            (
                "--loss-head-redundancy",
                self.reconstruction.loss_head_redundancy,
            ),
            (
                "--loss-cross-resolution",
                self.reconstruction.loss_cross_resolution,
            ),
        ] {
            finite_range(value, 0.0, 100.0, name)?;
        }
        finite_range(
            self.detail.probability,
            0.0,
            1.0,
            "--detail-crop-probability",
        )?;
        if !(32..=384).contains(&self.detail.resolution)
            || self.detail.min_zoom < 1.0
            || self.detail.max_zoom < self.detail.min_zoom
            || self.detail.max_zoom > 32.0
            || !(64..=4096).contains(&self.detail.cache_max_level)
        {
            bail!("native-detail resolution/zoom/cache controls exceed safe ranges");
        }
        finite_range(
            self.detail.curriculum_start,
            0.0,
            1.0,
            "--detail-curriculum-start",
        )?;
        if !(16..=512).contains(&self.flow.hidden)
            || !(1..=256).contains(&self.flow.sample_steps)
            || !(16..=128).contains(&self.flow.resolution)
            || self.flow.cadence == 0
        {
            bail!("flow hidden/resolution/cadence/sample-step controls exceed safe ranges");
        }
        finite_range(self.flow.weight, 0.0, 100.0, "--flow-weight")?;
        finite_range(
            self.flow.endpoint_weight,
            0.0,
            100.0,
            "--flow-endpoint-weight",
        )?;
        finite_range(self.flow.min_time, 0.0, 1.0, "--flow-min-time")?;
        finite_range(
            self.flow.max_time,
            self.flow.min_time,
            1.0,
            "--flow-max-time",
        )?;
        if self.analysis.stride == 0
            || self.analysis.autonomous_horizon > 4096
            || self.analysis.perturbation_horizon > 4096
            || self.analysis.dynamics_horizon > 4096
        {
            bail!("analysis stride must be positive and horizons <= 4096");
        }
        finite_range(self.interface_gain, 0.0, 1.0, "--interface-gain")?;
        finite_range(self.morph_residual_gain, 0.0, 0.25, "--morph-residual-gain")?;
        finite_range(self.memory_limit, 0.5, 16.0, "--memory-limit")?;
        finite_range(
            self.reference_fidelity_min,
            0.0,
            1.0,
            "--reference-fidelity-min",
        )?;
        finite_range(
            self.reference_fidelity_max,
            self.reference_fidelity_min,
            1.0,
            "--reference-fidelity-max",
        )?;
        finite_range(self.reference_dropout, 0.0, 1.0, "--reference-dropout")?;
        finite_range(self.coord_gain, 0.0, 2.0, "--coord-gain")?;
        finite_range(self.state_skip, 0.0, 4.0, "--state-skip")?;
        finite_range(self.chroma, 0.0, 0.4, "--chroma")?;
        finite_range(self.gamma, 1.0, 3.0, "--gamma")?;
        if self.train_resolution < self.micro_size || self.train_resolution > 512 {
            bail!("--train-resolution must be between micro-size and 512");
        }
        if self.output_resolution < self.train_resolution || self.output_resolution > 4096 {
            bail!("--output-resolution must be between train resolution and 4096");
        }
        if self.snapshot_resolution < self.micro_size || self.snapshot_resolution > 4096 {
            bail!("--snapshot-resolution must be between micro-size and 4096");
        }
        if self.core_update_every == 0 || self.macro_update_every == 0 || self.log_every == 0 {
            bail!("update and logging cadences must be positive");
        }
        if self.checkpoint_every > 0 && !self.checkpoint_every.is_multiple_of(self.bptt) {
            bail!("--checkpoint-every must be zero or a multiple of --bptt");
        }
        if !(1..=256).contains(&self.image_cache) {
            bail!("--image-cache must be in 1..=256");
        }
        finite_positive(self.learning_rate, "--learning-rate")?;
        finite_nonnegative(self.weight_decay, "--weight-decay")?;
        if !(0.0..1.0).contains(&self.beta1) || !(0.0..1.0).contains(&self.beta2) {
            bail!("--beta1 and --beta2 must be finite and in [0,1)");
        }
        finite_positive(self.adam_epsilon, "--adam-epsilon")?;
        finite_nonnegative(self.grad_clip, "--grad-clip")?;
        if !(0.0..1.0).contains(&self.muon_momentum) || !(1..=10).contains(&self.muon_ns_steps) {
            bail!("--muon-momentum must be in [0,1) and --muon-ns-steps in 1..=10");
        }
        finite_range(self.dt, 0.005, 0.25, "--dt")?;
        finite_range(self.state_limit, 1.0, 12.0, "--state-limit")?;
        finite_range(
            self.state_soft_limit,
            0.25,
            self.state_limit,
            "--state-soft-limit",
        )?;
        finite_range(self.clock_probability, 0.05, 1.0, "--clock-probability")?;
        finite_range(self.episode_reset, 0.0, 1.0, "--episode-reset")?;
        finite_range(self.memory_reset, 0.0, 1.0, "--memory-reset")?;
        finite_range(self.state_leak, 0.0, 0.5, "--state-leak")?;
        for (name, value) in [
            ("--nca-gain", self.nca_gain),
            ("--reaction-gain", self.reaction_gain),
            ("--phase-gain", self.phase_gain),
            ("--fractal-gain", self.fractal_gain),
            ("--quasiperiodic-gain", self.quasiperiodic_gain),
            ("--cyclic-gain", self.cyclic_gain),
        ] {
            finite_range(value, 0.0, 4.0, name)?;
        }
        for (name, value) in [
            ("--rd-diffusion-u", self.rd_diffusion_u),
            ("--rd-diffusion-v", self.rd_diffusion_v),
            ("--rd-feed", self.rd_feed),
            ("--rd-kill", self.rd_kill),
            ("--phase-growth", self.phase_growth),
            ("--phase-saturation", self.phase_saturation),
            ("--phase-frequency", self.phase_frequency),
            ("--phase-diffusion", self.phase_diffusion),
            ("--phase-dispersion", self.phase_dispersion),
        ] {
            finite_range(value, 0.0, 2.0, name)?;
        }
        let mut positive_loss = false;
        for (name, value) in [
            ("--loss-content", self.loss_content),
            ("--loss-palette", self.loss_palette),
            ("--loss-structure", self.loss_structure),
            ("--loss-seam", self.loss_seam),
            ("--loss-gamut", self.loss_gamut),
        ] {
            finite_range(value, 0.0, 100.0, name)?;
            positive_loss |= value > 0.0;
        }
        if !positive_loss {
            bail!("at least one visual loss weight must be positive");
        }
        for (name, value) in [
            ("--loss-state", self.loss_state),
            ("--loss-memory", self.loss_memory),
        ] {
            finite_range(value, 0.0, 100.0, name)?;
        }
        finite_range(
            self.max_saturation_fraction,
            0.0,
            1.0,
            "--max-saturation-fraction",
        )?;
        if self.stability_patience > 10_000 {
            bail!("--stability-patience must be <= 10000");
        }
        finite_range(self.mastering_strength, 0.0, 2.0, "--mastering-strength")?;
        if self.gallery > 64 || self.gallery_steps > 1024 || self.gallery_stride > 256 {
            bail!("--gallery must be <= 64, --gallery-steps <= 1024, and --gallery-stride <= 256");
        }
        if self.analysis.probe_dir.is_some() {
            if !self.analysis.only {
                bail!(
                    "--probe-dir requires --analysis-only so probing cannot take optimizer steps"
                );
            }
            if self.analysis.probe_ages.is_empty()
                || self.analysis.probe_ages.len() > 64
                || self
                    .analysis
                    .probe_ages
                    .iter()
                    .any(|age| *age == 0 || *age > 4096)
            {
                bail!("--probe-ages must contain 1..=64 unique ages in 1..=4096");
            }
            if self.analysis.probe_reference_fidelities.is_empty()
                || self.analysis.probe_reference_fidelities.len() > 32
            {
                bail!("--probe-reference-fidelities must contain 1..=32 values");
            }
            for fidelity in &self.analysis.probe_reference_fidelities {
                finite_range(*fidelity, 0.0, 1.0, "--probe-reference-fidelities")?;
            }
            if !strictly_increasing(&self.analysis.probe_ages) {
                bail!("--probe-ages must be sorted in strictly increasing order");
            }
            if has_duplicate_f32(&self.analysis.probe_reference_fidelities) {
                bail!("--probe-reference-fidelities must not contain duplicates");
            }
        }
        if (self.render_only || self.analysis.only) && self.fresh {
            bail!("render-only/analysis-only and --fresh are mutually exclusive");
        }
        Ok(())
    }

    pub fn suffix(&self) -> String {
        self.run_tag
            .as_ref()
            .map(|tag| format!("_{tag}"))
            .unwrap_or_default()
    }

    /// Fingerprint every setting that changes learned evolution or gradients.
    /// Corpus identity is stored separately; paths and output-only controls are excluded.
    pub fn checkpoint_signature(&self) -> u64 {
        let mode = match self.mode {
            TrainingMode::Single => 0u64,
            TrainingMode::Family => 1,
            TrainingMode::Texture => 2,
        };
        let integrator = match self.integrator {
            Integrator::Euler => 0u64,
            Integrator::Midpoint => 1,
        };
        let style = match self.style {
            StylePreset::AlienFluid => 0u64,
            StylePreset::FractalFlame => 1,
            StylePreset::ReactionGarden => 2,
            StylePreset::Quasicrystal => 3,
            StylePreset::PureNca => 4,
        };
        let values = [
            SCHEMA_VERSION as u64,
            mode,
            integrator,
            style,
            self.objective as u64,
            self.reconstruction.grounding_strength.to_bits() as u64,
            self.reconstruction.emergence_strength.to_bits() as u64,
            self.reconstruction.emergence_start.to_bits() as u64,
            self.reconstruction.emergence_ramp.to_bits() as u64,
            self.reconstruction.grounding_floor.to_bits() as u64,
            self.reconstruction.emergent_limit.to_bits() as u64,
            self.reconstruction.emergence_low_budget.to_bits() as u64,
            self.reconstruction.emergence_mid_budget.to_bits() as u64,
            self.reconstruction.local_reference_gain.to_bits() as u64,
            self.reconstruction.loss_composite.to_bits() as u64,
            self.reconstruction.loss_ground_coarse.to_bits() as u64,
            self.reconstruction.loss_ground_mid.to_bits() as u64,
            self.reconstruction.loss_ground_fine.to_bits() as u64,
            self.reconstruction.loss_emergent_low.to_bits() as u64,
            self.reconstruction.loss_emergent_tv.to_bits() as u64,
            self.reconstruction.loss_emergent_fit.to_bits() as u64,
            self.reconstruction.loss_head_redundancy.to_bits() as u64,
            self.reconstruction.loss_cross_resolution.to_bits() as u64,
            self.detail.probability.to_bits() as u64,
            self.detail.resolution as u64,
            self.detail.min_zoom.to_bits() as u64,
            self.detail.max_zoom.to_bits() as u64,
            self.detail.curriculum_start.to_bits() as u64,
            self.detail.cache_max_level as u64,
            self.detail.boundary as u64,
            self.morph_growth.mode as u64,
            self.morph_growth.interval as u64,
            self.morph_growth.plateau_window as u64,
            self.morph_growth.plateau_epsilon.to_bits() as u64,
            self.morph_growth.seam_threshold.to_bits() as u64,
            self.flow.weight.to_bits() as u64,
            self.flow.endpoint_weight.to_bits() as u64,
            self.flow.min_time.to_bits() as u64,
            self.flow.max_time.to_bits() as u64,
            self.flow.hidden as u64,
            self.flow.sample_steps as u64,
            self.flow.resolution as u64,
            self.flow.cadence as u64,
            self.seed,
            self.micro_size as u64,
            self.macro_size as u64,
            self.channels as u64,
            self.genome_dim as u64,
            self.ca_hidden as u64,
            self.interface_grid as u64,
            self.interface_width as u64,
            self.interface_loops as u64,
            self.morph_residual_gain.to_bits() as u64,
            self.memory_limit.to_bits() as u64,
            self.interface_gain.to_bits() as u64,
            self.conditioning as u64,
            self.reference_fidelity_min.to_bits() as u64,
            self.reference_fidelity_max.to_bits() as u64,
            self.reference_dropout.to_bits() as u64,
            self.render_hidden as u64,
            self.render_blocks as u64,
            self.coord_bands as u64,
            self.coord_gain.to_bits() as u64,
            self.state_skip.to_bits() as u64,
            self.chroma.to_bits() as u64,
            self.gamma.to_bits() as u64,
            self.train_resolution as u64,
            self.episode_steps as u64,
            self.episode_reset.to_bits() as u64,
            self.bptt as u64,
            self.memory_reset.to_bits() as u64,
            self.core_update_every as u64,
            self.macro_update_every as u64,
            self.learning_rate.to_bits(),
            self.weight_decay.to_bits(),
            self.beta1.to_bits(),
            self.beta2.to_bits(),
            self.adam_epsilon.to_bits(),
            self.grad_clip.to_bits(),
            self.warmup_updates as u64,
            self.optimizer as u64,
            self.muon_momentum.to_bits(),
            self.muon_ns_steps as u64,
            self.dt.to_bits() as u64,
            self.state_limit.to_bits() as u64,
            self.clock_probability.to_bits() as u64,
            self.state_soft_limit.to_bits() as u64,
            self.nca_gain.to_bits() as u64,
            self.state_leak.to_bits() as u64,
            self.reaction_gain.to_bits() as u64,
            self.phase_gain.to_bits() as u64,
            self.fractal_gain.to_bits() as u64,
            self.quasiperiodic_gain.to_bits() as u64,
            self.cyclic_gain.to_bits() as u64,
            self.rd_diffusion_u.to_bits() as u64,
            self.rd_diffusion_v.to_bits() as u64,
            self.rd_feed.to_bits() as u64,
            self.rd_kill.to_bits() as u64,
            self.phase_growth.to_bits() as u64,
            self.phase_saturation.to_bits() as u64,
            self.phase_frequency.to_bits() as u64,
            self.phase_diffusion.to_bits() as u64,
            self.phase_dispersion.to_bits() as u64,
            self.loss_content.to_bits() as u64,
            self.loss_palette.to_bits() as u64,
            self.loss_structure.to_bits() as u64,
            self.loss_seam.to_bits() as u64,
            self.loss_gamut.to_bits() as u64,
            self.loss_state.to_bits() as u64,
            self.loss_memory.to_bits() as u64,
            self.max_saturation_fraction.to_bits() as u64,
            self.stability_patience as u64,
        ];
        values
            .into_iter()
            .fold(0xcbf2_9ce4_8422_2325, |hash, value| {
                (hash ^ value).wrapping_mul(0x100_0000_01b3)
            })
    }

    pub fn resolved_config_signature(&self) -> u64 {
        [
            self.checkpoint_signature(),
            self.morph_layers as u64,
            self.morph_depth as u64,
            self.morph_growth.min_depth as u64,
            self.morph_growth.max_depth as u64,
            self.research_preset as u64,
        ]
        .into_iter()
        .fold(0xcbf2_9ce4_8422_2325, |hash, value| {
            (hash ^ value).wrapping_mul(0x100_0000_01b3)
        })
    }
    pub fn help() -> &'static str {
        CLI_HELP_V9
        /* v8/v9-pre-R++ help retained in source history only:
        "TITAN Image v9 - recurrently stable S25-first morphogenic visual dynamics\n\
         Usage: titan_image [options]\n\n\
         Required and lifecycle:\n\
           --corpus-dir PATH           Source PNG/JPEG/WebP directory (required)\n\
           --output-dir PATH           v9 artifact directory\n\
           --run-tag NAME              Isolate checkpoint and output artifacts\n\
           --fresh                     Start a new v9 organism for this tag\n\
           --render-only               Load without training; render/gallery only\n\
           --profile NAME              s25-fast | s25-balanced | s25-quality\n\
           --style NAME                alien-fluid | fractal-flame | reaction-garden | quasicrystal | pure-nca\n\
           --list-presets              Describe profiles and styles, then exit\n\n\
         Training and phone controls:\n\
           --mode MODE                 single | family | texture\n\
           --steps N                   Additional development steps\n\
           -t, --threads N             Rayon/Candle CPU threads (default <= 8)\n\
           --seed N                    Deterministic training/world seed\n\
           --train-resolution N        Differentiable render resolution\n\
           --output-resolution N       Final/gallery render resolution\n\
           --snapshot-resolution N     Lower-cost periodic preview resolution\n\
           --episode-steps N           Steps per coherent source episode\n\
           --episode-reset X           Seed-state blend at target changes, 0..1\n\
           --bptt N                    Recurrent gradient horizon\n\
           --core-update-every N       Full-core cadence in optimizer windows\n\
           --memory-reset X            Independent recurrent-memory reset, 0..1\n\
           --macro-update-every N      Slow-field cadence in development steps\n\
           --snapshot-every N          Nominal step cadence; 0 disables\n\
           --checkpoint-every N        Checkpoint cadence; 0 disables periodic saves\n\
           --log-every N               Console cadence in optimizer updates\n\
           --image-cache N             Resized source images retained in RAM\n\
           --recursive-corpus          Include supported files below subdirectories\n\n\
         Runtime architecture:\n\
           --micro-size N              Micro field edge, 24..128\n\
           --macro-size N              Macro field edge, 8..micro-size\n\
           --channels N                Recurrent channels, 12..64\n\
           --genome-dim N              Conditioning dimensions, 2..32\n\
           --ca-hidden N               NCA pointwise hidden width\n\
           --interface-grid N          Pooled token-grid edge; divides both fields\n\
           --interface-width N         Recurrent token/GRU width\n\
           --interface-loops N         Shared transformer passes per world step\n\
           --morph-layers N            Physical append-preserving memory blocks\n\
           --morph-depth N             Active memory blocks, <= morph-layers\n\
           --interface-gain X          Spatial writeback gain, 0..1\n\
           --render-hidden N           Implicit renderer hidden width\n\
           --morph-residual-gain X     Bounded MorphicStack residual gain\n\
           --memory-limit X            Smooth recurrent-memory bound\n\
           --render-blocks N           Renderer residual blocks, 1..8\n\
           --coord-bands N             Global Fourier coordinate octaves, 1..8\n\
           --coord-gain X              Coordinate amplitude; low values avoid shortcuts\n\
           --state-skip X              Direct physical-state color path gain\n\
           --chroma X                  OKLab-inspired chroma bound, 0..0.4\n\
           --gamma X                   Display transfer exponent, 1..3\n\
           --integrator NAME           euler | midpoint (rk2 alias)\n\
           --dt X                      Development integration step\n\
           --state-limit X             Symmetric recurrent-state bound\n\
           --clock-probability X       Deterministic asynchronous cell rate\n\
           --nca-gain X                Learned update gain\n\n\
           --state-leak X              Contractive recurrent-state restoring gain\n\
           --state-soft-limit X        State-energy barrier threshold\n\
           --conditioning NAME         generate | hybrid | reconstruct\n\
           --reference-fidelity-min X  Hybrid reference-strength floor, 0..1\n\
           --reference-fidelity-max X  Hybrid/reconstruction ceiling, 0..1\n\
           --reference-dropout X       Null-reference probability in hybrid mode\n\n\
         Optimizer:\n\
           --learning-rate X           AdamW peak learning rate\n\
           --optimizer NAME            adamw | hybrid-muon\n\
           --weight-decay X            AdamW decoupled weight decay\n\
           --beta1 X --beta2 X         Adam moment coefficients\n\
           --adam-epsilon X             Adam denominator epsilon\n\
           --grad-clip X               Global L2 clip; 0 disables\n\
           --warmup-updates N          Linear optimizer warmup length\n\n\
           --muon-momentum X           Muon momentum coefficient\n\
           --muon-ns-steps N           Newton-Schulz iterations, 1..10\n\n\
         Dynamics and ablations:\n\
           --reaction-gain X           Gray-Scott contribution\n\
           --phase-gain X              Complex-phase contribution\n\
           --fractal-gain X            Stable IFS target attraction\n\
           --quasiperiodic-gain X      Stable quasiperiodic target attraction\n\
           --cyclic-gain X             Three-field cyclic chemistry\n\
           --rd-diffusion-u X --rd-diffusion-v X --rd-feed X --rd-kill X\n\
           --phase-growth X --phase-saturation X --phase-frequency X\n\
           --phase-diffusion X --phase-dispersion X\n\
           --no-reaction-diffusion --no-complex-phase --no-fractal\n\
           --no-quasiperiodic --no-cyclic\n\n\
         Objective, output, and exploration:\n\
           --loss-content X --loss-palette X --loss-structure X\n\
           --loss-seam X --loss-gamut X\n\
           --mastering-strength X      Toroidal bloom/local contrast, 0..2\n\
           --no-mastering              Save mastered image without enhancement\n\
           --no-state-atlas            Do not save micro/macro diagnostic atlases\n\
           --loss-state X              Soft state-energy barrier weight\n\
           --loss-memory X             Soft memory-energy barrier weight\n\
           --max-saturation-fraction X Sustained near-bound watchdog threshold\n\
           --stability-patience N      Violating windows before safe stop; 0 disables\n\
           --gallery N                 Interpolated-genome variants; 0 disables\n\
           --gallery-steps N           Fresh development steps per variant\n\
           --gallery-stride N          Extra development steps between variants\n\
           --gallery-seed N            Deterministic gallery seed\n\
           -h, --help                  Show this help\n\
           -V, --version               Show program/schema version"
        */
    }

    fn presets_help() -> &'static str {
        PRESETS_HELP_V9
        /* legacy preset text:
        "Compute profiles:\n\
           s25-fast      48/24 fields, 16 channels, 96/80x3 model, 128 train\n\
           s25-balanced  64/32 fields, 24 channels, 128/128x4 model, 192 train\n\
           s25-quality   80/40 fields, 32 channels, 192/160x5 model, 256 train\n\n\
         Style bases (all individual gains remain overrideable):\n\
           alien-fluid      complex-phase dominant, flowing organic forms\n\
           fractal-flame    stronger stable IFS geometry and phase color\n\
           reaction-garden  Turing spots/stripes plus cyclic competition\n\
           quasicrystal     incommensurate forcing and interference\n\
           pure-nca         learned near/far NCA only, useful as a control\n\n\
         Presets are applied before explicit flags regardless of argument order."
        */
    }
}

fn preapply_presets(args: &[String], config: &mut RunConfig) -> Result<()> {
    let mut profile = None;
    let mut style = None;
    let mut research = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                let value = args.get(i + 1).context("missing value for --profile")?;
                profile = Some(PhoneProfile::parse(value)?);
                i += 2;
            }
            "--style" => {
                let value = args.get(i + 1).context("missing value for --style")?;
                style = Some(StylePreset::parse(value)?);
                i += 2;
            }
            "--research-preset" => {
                let value = args
                    .get(i + 1)
                    .context("missing value for --research-preset")?;
                research = Some(ResearchPreset::parse(value)?);
                i += 2;
            }
            _ => i += 1,
        }
    }
    // Apply categories in a stable order, then let the normal parser apply all
    // explicit scalar flags. CLI order must never change resolved semantics.
    if let Some(profile) = profile {
        config.apply_profile(profile);
    }
    if let Some(style) = style {
        config.apply_style(style);
    }
    if let Some(research) = research {
        config.apply_research_preset(research);
    }
    Ok(())
}

fn parse<T: std::str::FromStr>(value: &str, flag: &str) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|err| anyhow::anyhow!("invalid value for {flag}: {err}"))
}

fn parse_csv<T: std::str::FromStr>(value: &str, flag: &str) -> Result<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    if value.trim().is_empty() {
        bail!("empty value for {flag}");
    }
    value
        .split(",")
        .map(|item| parse(item.trim(), flag))
        .collect()
}

fn strictly_increasing(values: &[usize]) -> bool {
    values.windows(2).all(|pair| pair[0] < pair[1])
}

fn has_duplicate_f32(values: &[f32]) -> bool {
    values
        .iter()
        .enumerate()
        .any(|(index, value)| values[..index].iter().any(|prior| prior == value))
}

fn finite_positive(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        bail!("{name} must be finite and positive");
    }
    Ok(())
}

fn finite_nonnegative(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        bail!("{name} must be finite and nonnegative");
    }
    Ok(())
}

fn finite_range(value: f32, min: f32, max: f32, name: &str) -> Result<()> {
    if !value.is_finite() || !(min..=max).contains(&value) {
        bail!("{name} must be finite and in {min}..={max}");
    }
    Ok(())
}

fn validate_tag(value: &str) -> Result<String> {
    if value.is_empty()
        || value.len() > 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        bail!("--run-tag must be 1..=64 ASCII letters, digits, '-' or '_'");
    }
    Ok(value.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_obey_guards() {
        RunConfig::default().validate().unwrap();
    }

    #[test]
    fn profile_and_style_are_applied_before_overrides() -> Result<()> {
        let mut config = RunConfig::default();
        preapply_presets(
            &[
                "--style".to_owned(),
                "fractal-flame".to_owned(),
                "--profile".to_owned(),
                "s25-fast".to_owned(),
            ],
            &mut config,
        )?;
        assert_eq!(config.micro_size, 48);
        assert_eq!(config.fractal_gain, 0.34);
        Ok(())
    }

    #[test]
    fn preset_category_order_is_deterministic() -> Result<()> {
        let mut first = RunConfig::default();
        preapply_presets(
            &[
                "--research-preset".to_owned(),
                "grounded-emergent".to_owned(),
                "--profile".to_owned(),
                "s25-fast".to_owned(),
                "--style".to_owned(),
                "pure-nca".to_owned(),
            ],
            &mut first,
        )?;
        let mut second = RunConfig::default();
        preapply_presets(
            &[
                "--style".to_owned(),
                "pure-nca".to_owned(),
                "--profile".to_owned(),
                "s25-fast".to_owned(),
                "--research-preset".to_owned(),
                "grounded-emergent".to_owned(),
            ],
            &mut second,
        )?;
        assert_eq!(
            first.resolved_config_signature(),
            second.resolved_config_signature()
        );
        assert_eq!(first.interface_grid, 4);
        assert_eq!(first.reaction_gain, 0.0);
        assert_eq!(first.reconstruction.emergence_strength, 0.70);
        assert_eq!(first.morph_growth.mode, MorphDepthMode::Adaptive);
        Ok(())
    }

    #[test]
    fn research_presets_resolve_distinct_operating_regimes() {
        let mut strict = RunConfig::default();
        strict.apply_research_preset(ResearchPreset::StrictReconstruct);
        let mut balanced = RunConfig::default();
        balanced.apply_research_preset(ResearchPreset::ReconstructionPlus);
        let mut free = RunConfig::default();
        free.apply_research_preset(ResearchPreset::FreeMorph);
        let mut flow = RunConfig::default();
        flow.apply_research_preset(ResearchPreset::FlowReconstruct);

        assert!(strict.reconstruction.grounding_floor > balanced.reconstruction.grounding_floor);
        assert!(
            strict.reconstruction.emergence_strength < balanced.reconstruction.emergence_strength
        );
        assert!(
            free.reconstruction.emergence_low_budget > balanced.reconstruction.emergence_low_budget
        );
        assert!(free.reference_fidelity_max < balanced.reference_fidelity_max);
        assert_eq!(flow.objective, ObjectiveMode::HybridFlow);
        assert_eq!(flow.flow.weight, 0.10);
    }

    #[test]
    fn append_only_morph_capacity_has_separate_resolved_signature() {
        let base = RunConfig::default();
        let mut expanded = base.clone();
        expanded.morph_layers += 2;
        expanded.morph_growth.max_depth += 2;
        assert_eq!(base.checkpoint_signature(), expanded.checkpoint_signature());
        assert_ne!(
            base.resolved_config_signature(),
            expanded.resolved_config_signature()
        );
    }

    #[test]
    fn tag_cannot_escape_output_dir() {
        assert!(validate_tag("alien-garden_01").is_ok());
        assert!(validate_tag("../escape").is_err());
    }

    #[test]
    fn continuation_signature_tracks_training_but_not_output_size() {
        let base = RunConfig::default();
        let changed_dt = RunConfig {
            dt: 0.11,
            ..base.clone()
        };
        let changed_output = RunConfig {
            output_resolution: 1024,
            ..base.clone()
        };
        let changed_preview = RunConfig {
            snapshot_resolution: 512,
            ..base.clone()
        };
        assert_ne!(
            base.checkpoint_signature(),
            changed_dt.checkpoint_signature()
        );
        assert_eq!(
            base.checkpoint_signature(),
            changed_output.checkpoint_signature()
        );
        assert_eq!(
            base.checkpoint_signature(),
            changed_preview.checkpoint_signature()
        );
    }

    #[test]
    fn continuation_signature_tracks_stability_semantics() {
        let base = RunConfig::default();
        let changed = RunConfig {
            memory_limit: 2.5,
            ..base.clone()
        };
        assert_ne!(base.checkpoint_signature(), changed.checkpoint_signature());
    }

    #[test]
    fn probe_controls_require_frozen_analysis_and_preserve_checkpoint_identity() {
        let base = RunConfig::default();
        let mut probe = base.clone();
        probe.analysis.only = true;
        probe.analysis.probe_dir = Some(PathBuf::from("held-out-probes"));
        probe.analysis.probe_ages = vec![1, 8, 16, 64];
        probe.analysis.probe_reference_fidelities = vec![1.0, 0.25, 0.0];
        assert!(probe.validate().is_ok());
        assert_eq!(base.checkpoint_signature(), probe.checkpoint_signature());

        let mut training_probe = probe.clone();
        training_probe.analysis.only = false;
        assert!(training_probe.validate().is_err());

        let mut unsorted_ages = probe.clone();
        unsorted_ages.analysis.probe_ages = vec![1, 16, 8];
        assert!(unsorted_ages.validate().is_err());

        let mut invalid_fidelity = probe;
        invalid_fidelity.analysis.probe_reference_fidelities = vec![1.0, -0.1];
        assert!(invalid_fidelity.validate().is_err());
    }

    #[test]
    fn stability_controls_obey_guards() {
        let invalid_loss = RunConfig {
            loss_state: -0.1,
            ..RunConfig::default()
        };
        assert!(invalid_loss.validate().is_err());
        let invalid_soft_limit = RunConfig {
            state_soft_limit: 4.0,
            state_limit: 3.5,
            ..RunConfig::default()
        };
        assert!(invalid_soft_limit.validate().is_err());
        let disabled_watchdog = RunConfig {
            stability_patience: 0,
            ..RunConfig::default()
        };
        assert!(disabled_watchdog.validate().is_ok());
    }
}
