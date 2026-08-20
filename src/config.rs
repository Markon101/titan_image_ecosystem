use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const SCHEMA_VERSION: u32 = 5;
pub const MIN_PHYSICAL_CHANNELS: usize = 12;

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub corpus_dir: PathBuf,
    pub output_dir: PathBuf,
    pub run_tag: Option<String>,
    pub mode: TrainingMode,
    pub profile: PhoneProfile,
    pub style: StylePreset,
    pub steps: usize,
    pub threads: usize,
    pub seed: u64,
    pub micro_size: usize,
    pub macro_size: usize,
    pub channels: usize,
    pub genome_dim: usize,
    pub ca_hidden: usize,
    pub render_hidden: usize,
    pub render_blocks: usize,
    pub coord_bands: usize,
    pub coord_gain: f32,
    pub state_skip: f32,
    pub chroma: f32,
    pub gamma: f32,
    pub train_resolution: usize,
    pub output_resolution: usize,
    pub episode_steps: usize,
    pub episode_reset: f32,
    pub bptt: usize,
    pub core_update_every: usize,
    pub macro_update_every: usize,
    pub snapshot_every: usize,
    pub checkpoint_every: usize,
    pub log_every: usize,
    pub image_cache: usize,
    pub recursive_corpus: bool,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub adam_epsilon: f64,
    pub grad_clip: f64,
    pub warmup_updates: usize,
    pub integrator: Integrator,
    pub dt: f32,
    pub state_limit: f32,
    pub clock_probability: f32,
    pub nca_gain: f32,
    pub reaction_gain: f32,
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
            output_dir: PathBuf::from("/sdcard/Download/titan_image_v5"),
            run_tag: None,
            mode: TrainingMode::Texture,
            profile: PhoneProfile::S25Balanced,
            style: StylePreset::AlienFluid,
            steps: 1600,
            threads,
            seed: 42,
            micro_size: 64,
            macro_size: 32,
            channels: 24,
            genome_dim: 8,
            ca_hidden: 96,
            render_hidden: 64,
            render_blocks: 3,
            coord_bands: 4,
            coord_gain: 0.12,
            state_skip: 0.75,
            chroma: 0.20,
            gamma: 2.2,
            train_resolution: 192,
            output_resolution: 768,
            episode_steps: 64,
            episode_reset: 0.85,
            bptt: 4,
            core_update_every: 4,
            macro_update_every: 4,
            snapshot_every: 200,
            checkpoint_every: 400,
            log_every: 10,
            image_cache: 32,
            recursive_corpus: false,
            learning_rate: 6e-4,
            weight_decay: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            adam_epsilon: 1e-8,
            grad_clip: 1.0,
            warmup_updates: 24,
            integrator: Integrator::Euler,
            dt: 0.16,
            state_limit: 3.5,
            clock_probability: 0.72,
            nca_gain: 0.38,
            reaction_gain: 0.35,
            phase_gain: 0.85,
            fractal_gain: 0.08,
            quasiperiodic_gain: 0.12,
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
            loss_structure: 0.9,
            loss_seam: 0.08,
            loss_gamut: 0.18,
            mastering_strength: 0.75,
            save_state_atlas: true,
            gallery: 4,
            gallery_steps: 64,
            gallery_stride: 16,
            gallery_seed: 0x5eed_5eed,
            fresh: false,
            render_only: false,
        };
        config.apply_profile(PhoneProfile::S25Balanced);
        config.apply_style(StylePreset::AlienFluid);
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
                "--steps" => cfg.steps = parse(&value()?, flag)?,
                "--threads" | "-t" => cfg.threads = parse(&value()?, flag)?,
                "--seed" => cfg.seed = parse(&value()?, flag)?,
                "--micro-size" => cfg.micro_size = parse(&value()?, flag)?,
                "--macro-size" => cfg.macro_size = parse(&value()?, flag)?,
                "--channels" => cfg.channels = parse(&value()?, flag)?,
                "--genome-dim" => cfg.genome_dim = parse(&value()?, flag)?,
                "--ca-hidden" => cfg.ca_hidden = parse(&value()?, flag)?,
                "--render-hidden" => cfg.render_hidden = parse(&value()?, flag)?,
                "--render-blocks" => cfg.render_blocks = parse(&value()?, flag)?,
                "--coord-bands" => cfg.coord_bands = parse(&value()?, flag)?,
                "--coord-gain" => cfg.coord_gain = parse(&value()?, flag)?,
                "--state-skip" => cfg.state_skip = parse(&value()?, flag)?,
                "--chroma" => cfg.chroma = parse(&value()?, flag)?,
                "--gamma" => cfg.gamma = parse(&value()?, flag)?,
                "--train-resolution" => cfg.train_resolution = parse(&value()?, flag)?,
                "--output-resolution" => cfg.output_resolution = parse(&value()?, flag)?,
                "--episode-steps" => cfg.episode_steps = parse(&value()?, flag)?,
                "--episode-reset" => cfg.episode_reset = parse(&value()?, flag)?,
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
                "--integrator" => cfg.integrator = Integrator::parse(&value()?)?,
                "--dt" => cfg.dt = parse(&value()?, flag)?,
                "--state-limit" => cfg.state_limit = parse(&value()?, flag)?,
                "--clock-probability" => cfg.clock_probability = parse(&value()?, flag)?,
                "--nca-gain" => cfg.nca_gain = parse(&value()?, flag)?,
                "--reaction-gain" => cfg.reaction_gain = parse(&value()?, flag)?,
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
                self.ca_hidden = 64;
                self.render_hidden = 48;
                self.render_blocks = 2;
                self.coord_bands = 3;
                self.train_resolution = 128;
                self.output_resolution = 512;
                self.bptt = 4;
                self.core_update_every = 8;
                self.episode_steps = 48;
                self.snapshot_every = 240;
                self.checkpoint_every = 480;
                self.gallery_steps = 48;
            }
            PhoneProfile::S25Balanced => {
                self.micro_size = 64;
                self.macro_size = 32;
                self.channels = 24;
                self.genome_dim = 8;
                self.ca_hidden = 96;
                self.render_hidden = 64;
                self.render_blocks = 3;
                self.coord_bands = 4;
                self.train_resolution = 192;
                self.output_resolution = 768;
                self.bptt = 4;
                self.core_update_every = 4;
                self.episode_steps = 64;
                self.snapshot_every = 200;
                self.checkpoint_every = 400;
                self.gallery_steps = 64;
            }
            PhoneProfile::S25Quality => {
                self.micro_size = 80;
                self.macro_size = 40;
                self.channels = 32;
                self.genome_dim = 12;
                self.ca_hidden = 128;
                self.render_hidden = 96;
                self.render_blocks = 4;
                self.coord_bands = 5;
                self.train_resolution = 256;
                self.output_resolution = 1024;
                self.bptt = 4;
                self.core_update_every = 2;
                self.episode_steps = 80;
                self.snapshot_every = 200;
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

    pub fn validate(&self) -> Result<()> {
        if !self.render_only && self.steps == 0 {
            bail!("--steps must be positive unless --render-only is selected");
        }
        if self.bptt == 0 || self.episode_steps == 0 {
            bail!("--bptt and --episode-steps must be positive");
        }
        if !self.render_only && !self.steps.is_multiple_of(self.bptt) {
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
        if self.core_update_every == 0 || self.macro_update_every == 0 || self.log_every == 0 {
            bail!("update and logging cadences must be positive");
        }
        for (name, cadence) in [
            ("--snapshot-every", self.snapshot_every),
            ("--checkpoint-every", self.checkpoint_every),
        ] {
            if cadence > 0 && !cadence.is_multiple_of(self.bptt) {
                bail!("{name} must be zero or a multiple of --bptt");
            }
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
        finite_range(self.dt, 0.005, 0.25, "--dt")?;
        finite_range(self.state_limit, 1.0, 12.0, "--state-limit")?;
        finite_range(self.clock_probability, 0.05, 1.0, "--clock-probability")?;
        finite_range(self.episode_reset, 0.0, 1.0, "--episode-reset")?;
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
            bail!("at least one loss weight must be positive");
        }
        finite_range(self.mastering_strength, 0.0, 2.0, "--mastering-strength")?;
        if self.gallery > 64 || self.gallery_steps > 1024 || self.gallery_stride > 256 {
            bail!("--gallery must be <= 64, --gallery-steps <= 1024, and --gallery-stride <= 256");
        }
        if self.render_only && self.fresh {
            bail!("--render-only and --fresh are mutually exclusive");
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
            self.seed,
            self.micro_size as u64,
            self.macro_size as u64,
            self.channels as u64,
            self.genome_dim as u64,
            self.ca_hidden as u64,
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
            self.core_update_every as u64,
            self.macro_update_every as u64,
            self.learning_rate.to_bits(),
            self.weight_decay.to_bits(),
            self.beta1.to_bits(),
            self.beta2.to_bits(),
            self.adam_epsilon.to_bits(),
            self.grad_clip.to_bits(),
            self.warmup_updates as u64,
            self.dt.to_bits() as u64,
            self.state_limit.to_bits() as u64,
            self.clock_probability.to_bits() as u64,
            self.nca_gain.to_bits() as u64,
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
        ];
        values
            .into_iter()
            .fold(0xcbf2_9ce4_8422_2325, |hash, value| {
                (hash ^ value).wrapping_mul(0x100_0000_01b3)
            })
    }

    pub fn help() -> &'static str {
        "TITAN Image v5 - S25-first morphogenic visual dynamics\n\
         Usage: titan_image [options]\n\n\
         Required and lifecycle:\n\
           --corpus-dir PATH           Source PNG/JPEG/WebP directory (required)\n\
           --output-dir PATH           v5 artifact directory\n\
           --run-tag NAME              Isolate checkpoint and output artifacts\n\
           --fresh                     Start a new v5 organism for this tag\n\
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
           --output-resolution N       Final/snapshot render resolution\n\
           --episode-steps N           Steps per coherent source episode\n\
           --episode-reset X           Seed-state blend at target changes, 0..1\n\
           --bptt N                    Recurrent gradient horizon\n\
           --core-update-every N       Full-core cadence in optimizer windows\n\
           --macro-update-every N      Slow-field cadence in development steps\n\
           --snapshot-every N          Snapshot cadence; 0 disables\n\
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
           --render-hidden N           Implicit renderer hidden width\n\
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
         Optimizer:\n\
           --learning-rate X           AdamW peak learning rate\n\
           --weight-decay X            AdamW decoupled weight decay\n\
           --beta1 X --beta2 X         Adam moment coefficients\n\
           --adam-epsilon X             Adam denominator epsilon\n\
           --grad-clip X               Global L2 clip; 0 disables\n\
           --warmup-updates N          Linear optimizer warmup length\n\n\
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
           --gallery N                 Interpolated-genome variants; 0 disables\n\
           --gallery-steps N           Fresh development steps per variant\n\
           --gallery-stride N          Extra development steps between variants\n\
           --gallery-seed N            Deterministic gallery seed\n\
           -h, --help                  Show this help\n\
           -V, --version               Show program/schema version"
    }

    fn presets_help() -> &'static str {
        "Compute profiles:\n\
           s25-fast      48/24 fields, 16 channels, 128 train, 512 output, sparse core\n\
           s25-balanced  64/32 fields, 24 channels, 192 train, 768 output\n\
           s25-quality   80/40 fields, 32 channels, 256 train, 1024 output\n\n\
         Style bases (all individual gains remain overrideable):\n\
           alien-fluid      complex-phase dominant, flowing organic forms\n\
           fractal-flame    stronger stable IFS geometry and phase color\n\
           reaction-garden  Turing spots/stripes plus cyclic competition\n\
           quasicrystal     incommensurate forcing and interference\n\
           pure-nca         learned near/far NCA only, useful as a control\n\n\
         Presets are applied before explicit flags regardless of argument order."
    }
}

fn preapply_presets(args: &[String], config: &mut RunConfig) -> Result<()> {
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" => {
                let value = args.get(i + 1).context("missing value for --profile")?;
                config.apply_profile(PhoneProfile::parse(value)?);
                i += 2;
            }
            "--style" => {
                let value = args.get(i + 1).context("missing value for --style")?;
                config.apply_style(StylePreset::parse(value)?);
                i += 2;
            }
            _ => i += 1,
        }
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
    fn tag_cannot_escape_output_dir() {
        assert!(validate_tag("alien-garden_01").is_ok());
        assert!(validate_tag("../escape").is_err());
    }

    #[test]
    fn continuation_signature_tracks_training_but_not_output_size() {
        let base = RunConfig::default();
        let changed_dt = RunConfig {
            dt: 0.12,
            ..base.clone()
        };
        let changed_output = RunConfig {
            output_resolution: 1024,
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
    }
}
