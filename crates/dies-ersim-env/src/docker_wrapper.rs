use anyhow::{anyhow, Result};
use std::{
    io::{ErrorKind, Write},
    process::{Command, Stdio},
};

const DOCKER_COMPOSE_YML: &str = include_str!("docker-compose.yml");

pub struct DockerWrapper {
    project_name: String,
}

impl DockerWrapper {
    pub fn new(project_name: String) -> Result<Self> {
        log::info!("Starting docker compose with project name {}", project_name);

        // Start docker compose
        let args = &[
            "-l",
            "warn",
            "compose",
            "-f",
            "-",
            "-p",
            &project_name,
            "up",
            "-d",
            "--remove-orphans",
            "--quiet-pull",
            "--no-color",
        ];
        log::info!("Running docker with arguments: {}", args.join(" "));

        let child = Command::new("docker")
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn();

        let mut child = match child {
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

        // Write docker compose file to stdin
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(DOCKER_COMPOSE_YML.as_bytes())?;
        // We need to drop stdin before we wait for the child to exit
        drop(stdin);

        log::info!("Waiting for docker compose to finish starting");

        // Wait for docker compose to start
        let output = child.wait_with_output()?;

        if output.status.success() {
            log::info!("Docker compose started successfully");
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
        log::info!(
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
            log::info!("Docker compose stopped successfully");
        } else {
            log::error!(
                "Failed to stop docker compose: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        }
    }
}
