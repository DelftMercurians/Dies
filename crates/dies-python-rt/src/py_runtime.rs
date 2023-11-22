use std::{
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};

use crate::ipc_codec::{IpcListener, IpcReceiver, IpcSender};

use super::rye_runner::RyeRunner;
use anyhow::{bail, Result};
use dies_core::{RuntimeConfig, RuntimeEvent, RuntimeMsg, RuntimeReceiver, RuntimeSender};

/// Python runtime configuration
///
/// Runs the given python module of the given package in the virtual environment,
/// making sure that the virtual environment is up to date.
///
/// If the module is `__main__`, the package is run as a script.
///
/// # Errors
///
/// Calling `build()` will fail if:
/// - [`RyeRunner::sync`] fails
/// - the given path is not a valid python module
/// - the command cannot be run
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

/// Sender side of the python runtime.
///
/// When the sender is dropped, it sends a termination message to the child process
/// and waits for it to exit. Attempting to receive from the child process after
/// the sender has been dropped will result in an error.
///
/// This struct is created by [`create_py_runtime`].
pub struct PyRuntimeSender {
    child_proc: Arc<Mutex<Child>>,
    sender: IpcSender,
}

/// Receiver side of the python runtime.
///
/// Trying to call `recv` after the child process has exited will result in an error.
/// This can also happen if the corresponding [`PyRuntimeSender`] is dropped.
///
/// This struct is created by [`create_py_runtime`].
pub struct PyRuntimeReceiver {
    child_proc: Arc<Mutex<Child>>,
    receiver: IpcReceiver,
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

impl RuntimeConfig for PyRuntimeConfig {
    fn build(self) -> Result<(Box<dyn RuntimeSender>, Box<dyn RuntimeReceiver>)> {
        if self.package.contains("-") {
            bail!("Package name cannot contain dashes");
        }
        let rye = RyeRunner::new(&self.workspace)?;
        if self.sync {
            rye.sync()?;
        }

        let target = if self.module == "__main__" {
            self.package.to_owned()
        } else {
            format!("{}.{}", self.package, self.module)
        };

        log::debug!("Running python module {}", target);

        let listener = IpcListener::new()?;
        let host = listener.host().to_owned();
        let port = listener.port();

        let rye_bin = rye.get_rye_bin();
        let child_proc = Command::new(rye_bin)
            .args(["run", "python", "-m", &target])
            .current_dir(&self.workspace)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .stdin(Stdio::null())
            .env("DIES_IPC_HOST", host)
            .env("DIES_IPC_PORT", port.to_string())
            .spawn()?;

        // Wait for the child process to connect to the socket
        let (sender, receiver) = listener.wait_for_conn(Duration::from_secs(3))?;

        log::debug!("Python process started");

        let child_proc = Arc::new(Mutex::new(child_proc));
        Ok((
            Box::new(PyRuntimeSender {
                child_proc: Arc::clone(&child_proc),
                sender,
            }),
            Box::new(PyRuntimeReceiver {
                child_proc,
                receiver,
            }),
        ))
    }
}

impl RuntimeSender for PyRuntimeSender {
    fn send(&mut self, data: &RuntimeMsg) -> Result<()> {
        let data = serde_json::to_string(&data)?;
        self.sender.send(&data)?;
        Ok(())
    }
}

impl Drop for PyRuntimeSender {
    fn drop(&mut self) {
        // TODO: Is there a better way to handle this?
        // 1. Check if child process is still alive
        {
            // Only hold the lock for a short time
            let mut child_proc = self.child_proc.lock().unwrap();
            if !is_proc_alive(&mut child_proc) {
                return;
            }
        }

        // 2. Send termination message
        if let Err(e) = self.send(&RuntimeMsg::Term) {
            log::error!("Failed to send termination message: {}", e);
        }

        // 3. Wait for child process to exit
        std::thread::sleep(Duration::from_secs(1));

        // 4. Kill child process
        let mut child_proc = self.child_proc.lock().unwrap();
        if let Err(e) = child_proc.kill() {
            log::error!("Failed to kill python process: {}", e);
        }
    }
}

impl RuntimeReceiver for PyRuntimeReceiver {
    fn recv(&mut self) -> Result<RuntimeEvent> {
        // Return error if child process has exited
        let is_alive = {
            let mut child_proc = self.child_proc.lock().unwrap();
            is_proc_alive(&mut child_proc)
        };
        if !is_alive {
            bail!("Python process exited unexpectedly");
        }
        let data = self.receiver.recv()?;
        Ok(serde_json::from_str(&data)?)
    }
}

fn is_proc_alive(proc: &mut Child) -> bool {
    match proc.try_wait() {
        Ok(Some(_)) => false,
        Ok(None) => true,
        Err(_) => false,
    }
}
