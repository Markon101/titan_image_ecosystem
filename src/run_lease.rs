//! Advisory ownership of a run's mutable artifact namespace.
//! Keep the lock inode: unlinking it could allow a second writer to lock a new file.
use crate::config::RunConfig;
use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static INVOCATIONS: AtomicU64 = AtomicU64::new(0);

pub struct RunLease {
    _file: File,
    pub invocation_id: String,
}

impl Drop for RunLease {
    fn drop(&mut self) {
        // Closing alone can leave flock held by a descriptor inherited across fork.
        // Explicit unlock releases this lease before an immediate resume acquires it.
        let _ = self._file.unlock();
    }
}

impl RunLease {
    /// The output directory must already exist. No checkpoint is read or written here.
    pub fn acquire(config: &RunConfig, operation: &str) -> Result<Self> {
        let path = config
            .output_dir
            .join(format!(".titan_image_writer{}.lock", config.suffix()));
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .with_context(|| format!("cannot open run ownership file {}", path.display()))?;
        let (locked, lock_scope) = acquire_lock(&file, &config.output_dir, file.try_lock())
            .with_context(|| format!(
                "run namespace is busy or cannot be locked: {}; another training/analysis writer may own it; do not delete the lock file",
                path.display()))?;
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let invocation_id = format!(
            "{nanos}-{}-{}",
            std::process::id(),
            INVOCATIONS.fetch_add(1, Ordering::Relaxed)
        );
        // Truncation happens only after exclusive ownership is established.
        file.set_len(0)?;
        serde_json::to_writer(
            &mut file,
            &serde_json::json!({
                "invocation_id": invocation_id, "pid": std::process::id(), "operation": operation,
                "run_tag": config.run_tag, "lock_scope": lock_scope, "output_dir": std::fs::canonicalize(&config.output_dir)?
            }),
        )?;
        file.write_all(b"\n")?;
        file.flush()?;
        Ok(Self {
            _file: locked,
            invocation_id,
        })
    }
}

// Some Android shared-storage files reject flock with ENOSYS/EOPNOTSUPP while
// directory descriptors support it. Never interpret an unsupported lock as ownership.
fn acquire_lock(
    file: &File,
    directory: &std::path::Path,
    attempt: std::result::Result<(), std::fs::TryLockError>,
) -> Result<(File, &'static str)> {
    match attempt {
        Ok(()) => Ok((file.try_clone()?, "run_tag")),
        Err(std::fs::TryLockError::Error(err))
            if err.kind() == std::io::ErrorKind::Unsupported
                || err.raw_os_error() == Some(38)
                || err.raw_os_error() == Some(95) =>
        {
            let directory_lock = File::open(directory)?;
            directory_lock.try_lock().with_context(|| {
                format!(
                    "output-directory fallback lock is busy or unsupported: {}",
                    directory.display()
                )
            })?;
            Ok((directory_lock, "output_directory"))
        }
        Err(err) => Err(err.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optional_existing_storage_probe_is_read_only() -> Result<()> {
        let Some(path) = std::env::var_os("TITAN_LOCK_TEST_FILE") else {
            return Ok(());
        };
        let path = std::path::PathBuf::from(path);
        let directory = path.parent().unwrap();
        let before = std::fs::read(&path)?;
        let file = File::open(&path)?;
        let (guard, scope) = acquire_lock(&file, directory, file.try_lock())?;
        let competitor = File::open(&path)?;
        assert!(acquire_lock(&competitor, directory, competitor.try_lock()).is_err());
        guard.unlock()?;
        drop(guard);
        let (next, _) = acquire_lock(&competitor, directory, competitor.try_lock())?;
        next.unlock()?;
        assert_eq!(before, std::fs::read(&path)?);
        println!("read-only storage lock scope: {scope}");
        Ok(())
    }

    #[test]
    fn unsupported_file_locks_use_exclusive_directory_ownership() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "titan-directory-lease-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        std::fs::create_dir(&root)?;
        let path = root.join("owner.lock");
        std::fs::write(&path, b"original owner")?;
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let unsupported = || {
            Err(std::fs::TryLockError::Error(
                std::io::Error::from_raw_os_error(38),
            ))
        };
        let (guard, scope) = acquire_lock(&file, &root, unsupported())?;
        assert_eq!(scope, "output_directory");
        assert!(acquire_lock(&file, &root, unsupported()).is_err());
        assert_eq!(std::fs::read(&path)?, b"original owner");
        let inherited = guard.try_clone()?;
        let lease = RunLease {
            _file: guard,
            invocation_id: "test".into(),
        };
        drop(lease);
        drop(acquire_lock(&file, &root, unsupported())?);
        drop(inherited);
        // Busy and permission failures must never switch locking mechanisms.
        assert!(acquire_lock(&file, &root, Err(std::fs::TryLockError::WouldBlock)).is_err());
        assert!(acquire_lock(
            &file,
            &root,
            Err(std::fs::TryLockError::Error(std::io::Error::from(
                std::io::ErrorKind::PermissionDenied
            )))
        )
        .is_err());
        drop(file);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }

    #[test]
    fn excludes_competitors_without_truncation_and_releases_on_drop() -> Result<()> {
        let root = std::env::temp_dir().join(format!(
            "titan-lease-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        ));
        std::fs::create_dir(&root)?;
        let config = RunConfig {
            output_dir: root.clone(),
            run_tag: Some("same".into()),
            ..Default::default()
        };
        let first = RunLease::acquire(&config, "test")?;
        let path = root.join(".titan_image_writer_same.lock");
        let bytes = std::fs::read(&path)?;
        assert!(RunLease::acquire(&config, "competitor").is_err());
        assert_eq!(bytes, std::fs::read(&path)?);
        let other = RunConfig {
            run_tag: Some("other".into()),
            ..config.clone()
        };
        drop(RunLease::acquire(&other, "independent namespace")?);
        let id = first.invocation_id.clone();
        // A forked child can briefly retain the same open file description.
        let inherited = first._file.try_clone()?;
        drop(first);
        assert!(path.exists());
        let next = RunLease::acquire(&config, "resume")?;
        assert_ne!(id, next.invocation_id);
        drop(next);
        drop(inherited);
        std::fs::remove_dir_all(root)?;
        Ok(())
    }
}
