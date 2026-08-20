#![recursion_limit = "256"]

pub mod config;
pub mod corpus;
pub mod dynamics;
pub mod engine;
pub mod metrics;
pub mod objectives;
pub mod optimizer;
pub mod persistence;
pub mod render;
pub mod state;
pub mod tensor_ops;

pub use config::{Integrator, PhoneProfile, RunConfig, StylePreset, TrainingMode};
pub use engine::run;
