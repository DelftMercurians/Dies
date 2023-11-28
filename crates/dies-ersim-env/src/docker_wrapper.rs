use anyhow::{anyhow, Result};
use std::{
    io::ErrorKind,
    process::{Command, Stdio},
};

#[cfg(target_os = "linux")]
const DOCKER_COMPOSE_YML: &str = include_str!("docker-compose-linux.yml");
#[cfg(target_os = "windows")]
const DOCKER_COMPOSE_YML: &str = include_str!("docker-compose-win.yml");

const BRIDGE_DOCKERFILE: &str = include_str!("bridge/Dockerfile");
const BRIDGE_RS: &str = include_str!("bridge/mod.rs");

pub struct DockerWrapper {
    project_name: String,
}

impl DockerWrapper {
    pub fn new(project_name: String) -> Result<Self> {
        log::debug!("Starting docker compose with project name {}", project_name);

        let tmpdir = tempfile::tempdir()?;

        // Create docker-compose.yml
        let docker_compose_yml = tmpdir.path().join("docker-compose.yml");
        std::fs::write(&docker_compose_yml, DOCKER_COMPOSE_YML)?;

        // Create bridge image
        let bridge_dir = tmpdir.path().join("bridge");
        std::fs::create_dir(&bridge_dir)?;
        let bridge_dockerfile = bridge_dir.join("Dockerfile");
        std::fs::write(&bridge_dockerfile, BRIDGE_DOCKERFILE)?;
        let bridge_rs = bridge_dir.join("bridge.rs");
        std::fs::write(&bridge_rs, BRIDGE_RS)?;

        // Start docker compose
        let args = &[
            "-l",
            "warn",
            "compose",
            "-p",
            &project_name,
            "up",
            "--build",
            "-d",
            "--remove-orphans",
            "--quiet-pull",
            "--no-color",
        ];
        log::debug!("Running docker {}", args.join(" "));
        println!("Running docker compose up -- this may take a while");

        let child = Command::new("docker")
            .current_dir(tmpdir.path())
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let child = match child {
            Ok(child) => child,
            Err(err) => {
                if let ErrorKind::NotFound = err.kind() {
                    log::error!("Docker not found");
                    return Err(anyhow!("Docker not found"));
                } else {
                    log::error!("Failed to execute docker compose: {}", err);
                    return Err(anyhow!("Failed to execute docker compose: {}", err));
                }
            }
        };

        log::debug!("Waiting for docker compose to finish starting");

        // Wait for docker compose to start
        let output = child.wait_with_output()?;

        if output.status.success() {
            log::debug!("Docker compose started successfully");
            Ok(Self {
                project_name: project_name.to_owned(),
            })
        } else {
            log::error!(
                "Docker compose exited with error: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            Err(anyhow!(
                "Docker compose exited with error: {}",
                String::from_utf8_lossy(&output.stderr)
            ))
        }
    }
}

impl Drop for DockerWrapper {
    fn drop(&mut self) {
        log::debug!(
            "Stopping docker compose with project name {}",
            self.project_name
        );
        // Stop docker compose project
        let out = Command::new("docker")
            .args(&[
                "-l",
                "warn",
                "compose",
                "-p",
                &self.project_name,
                "down",
                "--remove-orphans",
                "-t",
                "3",
            ])
            .output()
            .expect("Failed to run docker compose down");
        if out.status.success() {
            log::debug!("Docker compose stopped successfully");
        } else {
            log::error!(
                "Failed to stop docker compose: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}
