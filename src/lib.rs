#![recursion_limit = "256"]

pub mod analysis;
pub mod benchmark;
pub mod comparison;
pub mod config;
pub mod corpus;
pub mod dynamics;
pub mod engine;
pub mod flow;
pub mod interface;
pub mod metrics;
pub mod objectives;
pub mod optimizer;
pub mod persistence;
pub mod render;
pub mod state;
pub mod telemetry;
pub mod tensor_ops;
pub mod terminal;

pub use config::{
    BoundaryMode, ConditioningMode, Integrator, MorphDepthMode, ObjectiveMode, OptimizerKind,
    PhoneProfile, ResearchPreset, RunConfig, StylePreset, TerminalMode, TrainingMode,
};
pub use engine::run;
