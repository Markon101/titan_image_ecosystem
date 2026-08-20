use anyhow::Result;
use titan_image::{run, RunConfig};

fn main() -> Result<()> {
    if let Some(config) = RunConfig::parse_env()? {
        run(config)?;
    }
    Ok(())
}
