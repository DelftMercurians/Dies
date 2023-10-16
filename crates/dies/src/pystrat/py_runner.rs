use std::{
    io::{BufRead, Write},
    ops::Add,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread::sleep,
    time::Duration,
};

use super::messages::{StratCmd, StratMsg};

use super::rye_runner::RyeRunner;
use anyhow::{anyhow, bail, Result};
use serde::{de::DeserializeOwned, Serialize};
use zmq::{Context, Socket, SocketType};

#[derive(Debug, Serialize)]
struct SocketInfo {
    pub_port: u16,
    pull_port: u16,
}

///
pub struct PyRunner {
    child_proc: Child,
    zmq_ctx: Context,
    pub_sock: zmq::Socket,
    pull_sock: zmq::Socket,
}

impl PyRunner {
    /// Runs the given python module of the given package in the virtual environment,
    /// making sure that the virtual environment is up to date.
    ///
    /// If the module is `__main__`, the package is run as a script.
    ///
    /// # Errors
    ///
    /// Returns an error if
    /// - [`RyeRunner::sync`] fails
    /// - the given path is not a valid python module
    /// - the command cannot be run
    /// - the command exits with a non-zero exit code
    pub fn new(workspace: impl AsRef<Path>, package: &str, module: &str) -> Result<PyRunner> {
        if package.contains("-") {
            bail!("Package name cannot contain dashes");
        }
        let rye = RyeRunner::new(&workspace)?;
        // rye.sync()?;

        let zmq_ctx = Context::new();
        let (pub_sock, pub_port) = create_socket_with_random_port(&zmq_ctx, SocketType::PUB)?;
        let (pull_sock, pull_port) = create_socket_with_random_port(&zmq_ctx, SocketType::PULL)?;

        let target = if module == "__main__" {
            package.to_owned()
        } else {
            format!("{}.{}", package, module)
        };

        log::info!("Running python module {}", target);

        let rye_bin = rye.get_rye_bin();
        let mut child_proc = Command::new(rye_bin)
            .args(["run", "python", "-m", &target])
            .current_dir(&workspace)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .stdin(Stdio::piped())
            .spawn()?;

        // // Spawn thread that logs stdout and stderr
        // let stdout = child_proc.stdout.take().unwrap();
        // let stderr = child_proc.stderr.take().unwrap();
        // std::thread::spawn(move || {
        //     let mut stdout = std::io::BufReader::new(stdout);
        //     let mut stderr = std::io::BufReader::new(stderr);
        //     loop {
        //         let mut buf = String::new();
        //         if stdout.read_line(&mut buf).is_ok() {
        //             // Skip if whitespace only
        //             if buf.trim().is_empty() {
        //                 continue;
        //             }
        //             log::info!("[python]: {}", buf.trim_end());
        //         }
        //         let mut buf = String::new();
        //         if stderr.read_line(&mut buf).is_ok() {
        //             // Skip if whitespace only
        //             if buf.trim().is_empty() {
        //                 continue;
        //             }
        //             log::error!("[python]: {}", buf.trim_end());
        //         }
        //     }
        // });

        // Send socket info through to child process
        let mut stdin = child_proc.stdin.take().unwrap();
        let socket_info = SocketInfo {
            pub_port,
            pull_port,
        };
        log::info!("Sending socket info: {:?}", socket_info);
        stdin.write_all(serde_json::to_string(&socket_info)?.add("\n").as_bytes())?;

        Ok(PyRunner {
            child_proc,
            pub_sock,
            pull_sock,
            zmq_ctx,
        })
    }

    pub fn send(&self, data: &StratMsg) -> Result<()> {
        log::info!("Sending message {:?}", data);
        let data = serde_json::to_string(&data)?;
        self.pub_sock.send(data.as_bytes(), 0)?;
        Ok(())
    }

    pub fn recv(&mut self) -> Result<StratCmd> {
        // Return error if child process has exited
        if !self.is_alive() {
            bail!("Python process exited");
        }
        let data = self.pull_sock.recv_string(0)?;
        let data = data.map_err(|_| anyhow!("Failed to receive data from python process"))?;
        log::info!("Received message {}", data);
        Ok(serde_json::from_str(&data)?)
    }

    pub fn is_alive(&mut self) -> bool {
        if let Some(status) = self.child_proc.try_wait().unwrap() {
            log::error!("Python process exited with status {}", status);
            return false;
        }
        true
    }
}

impl Drop for PyRunner {
    fn drop(&mut self) {
        // TODO: Is there a better way to handle this?
        // 1. Check if child process is still alive
        if !self.is_alive() {
            return;
        }

        // 2. Send termination message
        if let Err(e) = self.send(&StratMsg::Term) {
            log::error!("Failed to send termination message: {}", e);
        }

        // 3. Wait for child process to exit
        sleep(Duration::from_secs(1));

        // 4. Kill child process
        if let Err(e) = self.child_proc.kill() {
            log::error!("Failed to kill python process: {}", e);
        }
    }
}

fn create_socket_with_random_port(ctx: &Context, socket_type: SocketType) -> Result<(Socket, u16)> {
    let sock = ctx.socket(socket_type)?;
    sock.bind("tcp://*:*")?;
    let addr = sock
        .get_last_endpoint()?
        .map_err(|_| anyhow!("Failed to bind socket"))?;
    let port = addr
        .split(":")
        .collect::<Vec<&str>>()
        .get(2)
        .ok_or(anyhow!("Failed to get socket port"))?
        .parse::<u16>()?;

    Ok((sock, port))
}
