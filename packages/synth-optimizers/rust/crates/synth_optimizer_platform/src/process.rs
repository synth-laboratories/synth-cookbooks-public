use std::fs::{self, File, OpenOptions};
use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use crate::config::ContainerConfig;
use crate::error::{OptimizerError, Result};
use crate::http::ContainerClient;
use fs2::FileExt;
use sha2::{Digest, Sha256};

const CONTAINER_TERM_GRACE: Duration = Duration::from_secs(2);
const CONTAINER_KILL_GRACE: Duration = Duration::from_secs(3);

pub struct ManagedContainerProcess {
    child: Option<Child>,
    lock: Option<ContainerLock>,
}

struct ContainerLock {
    file: File,
    path: PathBuf,
}

impl ManagedContainerProcess {
    pub fn maybe_start(config: &ContainerConfig) -> Result<Option<Self>> {
        if config.command.is_empty() {
            return Ok(None);
        }
        let lock = match &config.url {
            Some(url) => Some(acquire_container_lock(url, config.startup_timeout_seconds)?),
            None => None,
        };
        if let Some(url) = &config.url {
            if is_healthy(url) {
                return Ok(Some(Self { child: None, lock }));
            }
        }
        let program = &config.command[0];
        let args = &config.command[1..];
        let mut command = Command::new(program);
        command.args(args);
        #[cfg(unix)]
        {
            command.process_group(0);
        }
        command.env_remove("VIRTUAL_ENV");
        if let Some(cwd) = &config.cwd {
            command.current_dir(cwd);
        }
        command.stdout(Stdio::inherit());
        command.stderr(Stdio::inherit());
        let child = command.spawn().map_err(|source| {
            OptimizerError::io(
                config.cwd.clone().unwrap_or_else(|| PathBuf::from(".")),
                source,
            )
        })?;
        let process = Self {
            child: Some(child),
            lock,
        };
        if let Some(url) = &config.url {
            wait_for_health(url, config.startup_timeout_seconds)?;
        }
        Ok(Some(process))
    }
}

impl Drop for ManagedContainerProcess {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            stop_container_child(child);
        }
        if let Some(lock) = &self.lock {
            let _ = lock.file.unlock();
            let _ = fs::remove_file(&lock.path);
        }
    }
}

#[cfg(unix)]
fn stop_container_child(child: &mut Child) {
    let pgid = child.id() as i32;
    // Negative pid targets the process group created with CommandExt::process_group above.
    unsafe {
        libc::kill(-pgid, libc::SIGTERM);
    }
    if wait_for_child(child, CONTAINER_TERM_GRACE) {
        return;
    }
    let _ = child.kill();
    unsafe {
        libc::kill(-pgid, libc::SIGKILL);
    }
    let _ = wait_for_child(child, CONTAINER_KILL_GRACE);
}

#[cfg(not(unix))]
fn stop_container_child(child: &mut Child) {
    let _ = child.kill();
    let _ = wait_for_child(child, CONTAINER_KILL_GRACE);
}

fn wait_for_child(child: &mut Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => return true,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(50)),
            Ok(None) => return false,
            Err(_) => return true,
        }
    }
}

fn acquire_container_lock(url: &str, timeout_seconds: u64) -> Result<ContainerLock> {
    let timeout_seconds = timeout_seconds.max(600);
    let root = std::env::temp_dir().join("synth-optimizers-container-locks");
    fs::create_dir_all(&root).map_err(|source| OptimizerError::io(root.clone(), source))?;
    let path = root.join(format!("{}.lock", stable_lock_id(url)));
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|source| OptimizerError::io(path.clone(), source))?;
    let deadline = Instant::now() + Duration::from_secs(timeout_seconds.max(1));
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => return Ok(ContainerLock { file, path }),
            Err(err) if err.kind() == ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(OptimizerError::Container(format!(
                        "timed out waiting for container run lock at {} for {}",
                        path.display(),
                        url
                    )));
                }
                thread::sleep(Duration::from_millis(250));
            }
            Err(source) => return Err(OptimizerError::io(path.clone(), source)),
        }
    }
}

fn stable_lock_id(url: &str) -> String {
    let mut digest = Sha256::new();
    digest.update(url.as_bytes());
    let hex = format!("{:x}", digest.finalize());
    hex[..24].to_string()
}

fn is_healthy(url: &str) -> bool {
    ContainerClient::new(url.to_string())
        .and_then(|client| client.health())
        .is_ok()
}

fn wait_for_health(url: &str, timeout_seconds: u64) -> Result<()> {
    let client = ContainerClient::new(url.to_string())?;
    let deadline = Instant::now() + Duration::from_secs(timeout_seconds.max(1));
    loop {
        if client.health().is_ok() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(OptimizerError::Container(format!(
                "container did not become healthy within {}s at {}",
                timeout_seconds, url
            )));
        }
        thread::sleep(Duration::from_millis(250));
    }
}
