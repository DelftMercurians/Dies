use anyhow::{bail, Result};
use std::{
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::{
    ipc::{IpcConnection, IpcListener},
    rye_runner::RyeRunner,
    RuntimeEvent, RuntimeMsg,
};

/// Python runtime configuration
#[derive(Debug, Clone)]
pub struct PyRuntimeConfig {
    /// Path to the workspace
    pub workspace: PathBuf,
    /// Name of the package
    pub package: String,
    /// Name of the module
    pub module: String,
    /// Whether to run `rye sync` before starting the runtime
    pub sync: bool,
}

impl Default for PyRuntimeConfig {
    fn default() -> Self {
        Self {
            workspace: std::env::current_dir().expect("Failed to get current directory"),
            package: String::from("dies"),
            module: String::from("__main__"),
            sync: true,
        }
    }
}

/// Python runtime.
///
/// When the runtime is dropped, it sends a termination message to the child process
/// and waits for it to exit. Attempting to receive from the child process after
/// the runtime has been dropped will result in an error.
///
/// This struct is created with [`create_py_runtime`].
pub struct PyRuntime {
    child_proc: Arc<Mutex<Child>>,
    ipc: IpcConnection,
}

impl PyRuntime {
    /// Run the given python module of the given package in the virtual environment,
    /// making sure that the virtual environment is up to date.
    ///
    /// If the module is `__main__`, the package is run as a script.
    ///
    /// # Errors
    ///
    /// - [`RyeRunner::sync`] fails
    /// - the given path is not a valid python module
    /// - the command cannot be run
    pub async fn new(config: PyRuntimeConfig) -> Result<PyRuntime> {
        if config.package.contains("-") {
            bail!("Package name cannot contain dashes");
        }
        let rye = RyeRunner::new(&config.workspace)?;
        if config.sync {
            rye.sync()?;
        }

        let target = if config.module == "__main__" {
            config.package.to_owned()
        } else {
            format!("{}.{}", config.package, config.module)
        };

        log::debug!("Running python module {}", target);

        let listener = IpcListener::new().await?;
        let host = listener.host().to_owned();
        let port = listener.port();

        let rye_bin = rye.get_rye_bin();
        let child_proc = Command::new(rye_bin)
            .args(["run", "python", "-m", &target])
            .current_dir(&config.workspace)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .stdin(Stdio::null())
            .env("DIES_IPC_HOST", host)
            .env("DIES_IPC_PORT", port.to_string())
            .spawn()?;

        // Wait for the child process to connect to the socket
        let ipc = listener.wait_for_conn(Duration::from_secs(10)).await?;

        log::debug!("Python process started");

        let child_proc = Arc::new(Mutex::new(child_proc));
        Ok(PyRuntime {
            child_proc: Arc::clone(&child_proc),
            ipc,
        })
    }

    pub async fn send(&mut self, data: &RuntimeMsg) -> Result<()> {
        let data = serde_json::to_string(&data)?;
        self.ipc.send(&data).await?;
        Ok(())
    }

    pub async fn recv(&mut self) -> Result<RuntimeEvent> {
        // Return error if child process has exited
        {
            let mut child_proc = self.child_proc.lock().unwrap();
            if !is_proc_alive(&mut child_proc) {
                bail!("Python process exited unexpectedly");
            }
        };

        let data = self.ipc.recv().await?;
        Ok(serde_json::from_str(&data)?)
    }
}

impl Drop for PyRuntime {
    fn drop(&mut self) {
        // Kill child process
        let mut child_proc = self.child_proc.lock().unwrap();
        if let Err(e) = child_proc.kill() {
            log::error!("Failed to kill python process: {}", e);
        }
    }
}

fn is_proc_alive(proc: &mut Child) -> bool {
    match proc.try_wait() {
        Ok(Some(_)) => false,
        Ok(None) => true,
        Err(_) => false,
    }
}
