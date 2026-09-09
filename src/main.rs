use anyhow::Result;
use titan_image::{run, RunConfig};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "fork") {
        return titan_image::training_fork::cli(&args[1..]);
    }
    if args.first().is_some_and(|a| a == "gradient-check") {
        anyhow::ensure!(args.len() == 2, "gradient-check expects config.json");
        let config: RunConfig = serde_json::from_slice(&std::fs::read(&args[1])?)?;
        config.validate()?;
        println!(
            "{}",
            serde_json::to_string_pretty(&titan_image::training_fork::gradient_probe(&config)?)?
        );
        return Ok(());
    }
    if let Some(config) = RunConfig::parse_env()? {
        run(config)?;
    }
    Ok(())
}
