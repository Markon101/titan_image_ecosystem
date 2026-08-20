use std::process::Command;

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn main() {
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");

    let commit = git_output(&["rev-parse", "--short=12", "HEAD"])
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "uncommitted".to_owned());
    let dirty = git_output(&["status", "--porcelain"])
        .map(|value| !value.is_empty())
        .unwrap_or(true);
    println!("cargo:rustc-env=TITAN_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=TITAN_BUILD_DIRTY={dirty}");
    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .replace('\u{1f}', " ");
    println!("cargo:rustc-env=TITAN_BUILD_RUSTFLAGS={rustflags}");
}
