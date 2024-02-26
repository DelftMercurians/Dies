use anyhow::{bail, Result};
use std::{
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::{
    env_manager::{PythonDistro, PythonDistroConfig},
    ipc::{IpcConnection, IpcListener},
    RuntimeEvent, RuntimeMsg,
};

#[derive(Debug, Clone)]
pub enum PyExecute {
    /// Execute a python script with `python <script>`.
    /// The path can be relative to the workspace.
    Script(PathBuf),
    /// Execute a python package as a module with `python -m <package>`. The package
    /// is first installed in the virtual environment.
    Package { path: PathBuf, name: String },
}

/// Python runtime configuration
#[derive(Debug, Clone)]
pub struct PyRuntimeConfig {
    /// Path to the workspace -- this is where the virtual environment will be created
    pub workspace: PathBuf,
    /// The specific Python version to use eg. "3.8.5"
    pub python_version: String,
    /// The Python build number to use eg. `20240107`.
    /// See [https://github.com/indygreg/python-build-standalone/tags] for the list
    /// of available builds.
    pub python_build: u32,
    /// The target to run
    pub execute: PyExecute,
    /// Whether to install any packages before running the target. Recommended to keep
    /// this as `true`.
    pub install: bool,
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
        let python = PythonDistro::new(PythonDistroConfig::from_version_and_build(
            &config.python_version,
            config.python_build,
        ))
        .await?;
        let mut venv = python.create_venv(&config.workspace).await?;

        if config.install {
            // First, install dies-py
            tracing::debug!("Installing dies-py");
            venv.install_editable(&config.workspace.join("py").join("dies-py"))
                .await?;

            // Then, install the target package
            if let PyExecute::Package { path, .. } = &config.execute {
                let path = if path.is_relative() {
                    config.workspace.join(path)
                } else {
                    path.to_owned()
                };
                if !path.exists() {
                    bail!("Package not found at {}", path.display());
                }
                let path = path.canonicalize()?;
                tracing::debug!("Installing package from {}", path.display());
                venv.install_editable(&path).await?;
            }
        }

        let target = match &config.execute {
            PyExecute::Script(path) => {
                let path = if path.is_relative() {
                    config.workspace.join(path)
                } else {
                    path.to_owned()
                };
                vec![path.canonicalize()?.to_string_lossy().to_string()]
            }
            PyExecute::Package { name, .. } => {
                let name = name.replace("-", "_");
                vec!["-m".to_string(), name]
            }
        };

        tracing::info!("Running python {}", target.join(" "));

        let listener = IpcListener::new().await?;
        let host = listener.host().to_owned();
        let port = listener.port();

        let child_proc = Command::new(venv.python_bin())
            .args(target)
            .current_dir(&config.workspace)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .stdin(Stdio::null())
            .env("DIES_IPC_HOST", host)
            .env("DIES_IPC_PORT", port.to_string())
            .spawn()?;

        // Wait for the child process to connect to the socket
        let ipc = listener.wait_for_conn(Duration::from_secs(5)).await?;

        tracing::debug!("Python process started");

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

    pub async fn wait_with_timeout(&self, timeout: Duration) -> Result<bool> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let child_proc = Arc::clone(&self.child_proc);
        tokio::task::spawn_blocking(move || {
            let mut child_proc = child_proc.lock().unwrap();
            let start = Instant::now();
            let res = loop {
                match child_proc.try_wait() {
                    Ok(Some(_)) => break Ok(true),
                    Ok(None) => {
                        if start.elapsed() > timeout {
                            break Ok(false);
                        }
                    }
                    Err(e) => break Err(e),
                }
            };
            tx.send(res).ok();
        });
        rx.await?.map_err(|e| e.into())
    }

    pub fn is_alive(&self) -> bool {
        let mut child_proc = self.child_proc.lock().unwrap();
        is_proc_alive(&mut child_proc)
    }

    pub fn kill(&mut self) {
        let mut child_proc = self.child_proc.lock().unwrap();
        if let Err(e) = child_proc.kill() {
            tracing::error!("Failed to kill python process: {}", e);
        }
    }
}

// impl Drop for PyRuntime {
//     fn drop(&mut self) {
//         // Kill child process
//         let mut child_proc = self.child_proc.lock().unwrap();
//         if let Err(e) = child_proc.kill() {
//             tracing::error!("Failed to kill python process: {}", e);
//         }
//     }
// }

fn is_proc_alive(proc: &mut Child) -> bool {
    match proc.try_wait() {
        Ok(Some(_)) => false,
        Ok(None) => true,
        Err(_) => false,
    }
}
