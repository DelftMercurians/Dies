use anyhow::{bail, Result};
use std::{
    collections::HashSet,
    fs::read_to_string,
    path::{Path, PathBuf},
};
use toml::Table;

use tokio::process::Command;

pub struct Venv {
    venv_dir: PathBuf,
    installed_deps: HashSet<String>,
}

impl Venv {
    /// Create a handle to an existing venv
    pub(super) async fn from_venv_path(venv_dir: PathBuf) -> Result<Self> {
        Ok(Self {
            venv_dir: venv_dir.clone(),
            installed_deps: get_installed_deps(
                venv_dir.join("bin").join("python").as_path(),
                &venv_dir,
            )
            .await?,
        })
    }

    pub fn python_bin(&self) -> PathBuf {
        self.venv_dir.join("bin").join("python")
    }

    /// Install a local package in editable mode
    pub async fn install_editable(&mut self, path: &Path) -> Result<()> {
        let workspace_dir = self.venv_dir.parent().unwrap();
        let relative_path = path.strip_prefix(workspace_dir).unwrap();
        let package_name = format!("-e {}", relative_path.to_str().unwrap());
        let pyproject_toml = path.join("pyproject.toml");
        if !pyproject_toml.exists() {
            bail!("pyproject.toml not found in {}", path.display());
        }

        let mut deps = HashSet::from_iter(get_deps_from_pyproject_toml(&pyproject_toml)?);
        deps.insert(package_name);
        let diff = deps.difference(&self.installed_deps).collect::<Vec<_>>();

        if diff.is_empty() {
            log::debug!("No new dependencies to install");
            return Ok(());
        }
        log::debug!("New dependencies to install: {:?}", diff);

        log::info!("Installing editable package: {}", relative_path.display());
        pip_install(&self.python_bin(), path, &["-e", "."]).await?;
        log::info!("Done installing editable package");

        // Update installed_deps
        self.installed_deps = get_installed_deps(&self.python_bin(), &self.venv_dir).await?;
        Ok(())
    }

    /// Install a package from PyPI
    #[allow(dead_code)]
    pub async fn install(&mut self, package_name: &str, version: &str) -> Result<()> {
        let spec = format!("{}=={}", package_name, version);
        if self.installed_deps.contains(package_name) {
            log::debug!("{} is already installed", spec);
            return Ok(());
        }

        log::info!("Installing package: {}", spec);
        pip_install(&self.python_bin(), self.venv_dir.as_path(), &[&spec]).await?;
        log::info!("Done installing package");

        // Update installed_deps
        self.installed_deps = get_installed_deps(&self.python_bin(), &self.venv_dir).await?;
        Ok(())
    }

    /// Install a list of packages from a requirements.txt format
    #[allow(dead_code)]
    pub async fn install_from_requirements_list(
        &mut self,
        requirements: Vec<String>,
    ) -> Result<()> {
        let requirements = requirements.iter().filter(|req| {
            if req.starts_with("#") {
                return false;
            }
            if req.starts_with("-e") {
                return true;
            }
            let spec = pep_508::parse(req);
            if let Ok(spec) = spec {
                !self.installed_deps.contains(spec.name)
            } else {
                true
            }
        });

        for req in requirements {
            log::info!("Installing package: {}", req);
            pip_install(&self.python_bin(), self.venv_dir.as_path(), &[req]).await?;
            log::info!("Done installing package");
        }

        // Update installed_deps
        self.installed_deps = get_installed_deps(&self.python_bin(), &self.venv_dir).await?;
        Ok(())
    }

    /// Install a list of packages from a requirements.txt file
    #[allow(dead_code)]
    pub async fn install_from_requirements_file(&mut self, path: &Path) -> Result<()> {
        let requirements = read_to_string(path)?
            .lines()
            .map(|line| line.to_string())
            .collect();
        self.install_from_requirements_list(requirements).await
    }
}

/// List the installed dependencies in the venv
async fn get_installed_deps(python_bin: &Path, venv_dir: &Path) -> Result<HashSet<String>> {
    let cmd = Command::new(python_bin)
        .arg("-m")
        .arg("pip")
        .arg("freeze")
        .arg("--local")
        .output()
        .await?;

    if !cmd.status.success() {
        log::error!("Failed to run pip freeze");
        log::error!("stdout: {}", String::from_utf8_lossy(&cmd.stdout));
        log::error!("stderr: {}", String::from_utf8_lossy(&cmd.stderr));
        anyhow::bail!("Failed to run pip freeze");
    }

    let root_dir = venv_dir.parent().unwrap();
    let deps = String::from_utf8_lossy(&cmd.stdout)
        .lines()
        .filter_map(|line| {
            if line.starts_with("-e") {
                let name = line.strip_prefix("-e ").unwrap(); // Safe, we know it starts with "-e"
                if name.starts_with("git+") {
                    // Parse subdirectory from "-e git+ssh://git@github.com/DelftMercurians/Dies.git@...#egg=dies_py&subdirectory=py/dies-py"
                    let subdirectory = name
                        .split("&")
                        .find(|s| s.starts_with("subdirectory="))?
                        .split("=")
                        .last()?
                        .to_owned();
                    Some(format!("-e {}", subdirectory))
                } else {
                    let path = Path::new(name);
                    if path.exists() {
                        let path = path.strip_prefix(root_dir).ok()?;
                        Some(format!("-e {}", path.display()))
                    } else {
                        None
                    }
                }
            } else {
                pep_508::parse(line).map(|spec| spec.name.to_owned()).ok()
            }
        })
        .collect();

    Ok(deps)
}

/// Get the dependencies from the pyproject.toml file
fn get_deps_from_pyproject_toml(path: &Path) -> Result<Vec<String>> {
    let toml: Table = std::fs::read_to_string(path)?.parse()?;

    // Get project.dependencies
    let deps = toml
        .get("project")
        .and_then(|project| project.get("dependencies"))
        .and_then(|deps| deps.as_array())
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|dep| dep.as_str().map(|s| s.to_string()))
        .filter_map(|dep| {
            let spec = pep_508::parse(&dep).ok()?;
            Some(spec.name.to_owned())
        })
        .collect();

    Ok(deps)
}

/// Run pip install
async fn pip_install(python_bin: &Path, cwd: &Path, args: &[&str]) -> Result<()> {
    let cmd = Command::new(python_bin)
        .current_dir(cwd)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .args(args)
        .output()
        .await?;

    if !cmd.status.success() {
        log::error!("Failed to run pip install `pip install {}`", args.join(" "));
        log::error!("stdout: {}", String::from_utf8_lossy(&cmd.stdout));
        log::error!("stderr: {}", String::from_utf8_lossy(&cmd.stderr));
        anyhow::bail!("Failed to run pip install");
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use tempfile::tempdir;

    use crate::env_manager::{PythonDistro, PythonDistroConfig};

    #[test_log::test(tokio::test)]
    #[ignore = "Takes too long"]
    async fn test_install_editable() {
        let dir = tempdir().unwrap();
        let mut venv = PythonDistro::new(PythonDistroConfig::from_version_and_build(
            "3.10.13", 20240107,
        ))
        .await
        .unwrap()
        .create_venv(dir.path())
        .await
        .unwrap();

        // Create a fake package
        let package = dir.path().join("py").join("dies-py");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(
            package.join("pyproject.toml"),
            indoc::indoc! {r#"
                [project]
                name = "dies-py"
                version = "0.1.0"
                dependencies = ["msgspec>=0.18.4"]

                [build-system]
                requires = ["setuptools"]
                build-backend = "setuptools.build_meta"
            "#},
        )
        .unwrap();
        std::fs::create_dir(package.join("src")).unwrap();
        std::fs::write(
            package.join("src").join("main.py"),
            "print('Hello, world!')",
        )
        .unwrap();

        venv.install_editable(&package).await.unwrap();

        // Re-run to test that it doesn't install twice
        let start = std::time::Instant::now();
        venv.install_editable(&package).await.unwrap();
        assert!(start.elapsed().as_millis() < 500); // Kind of arbitrary, but it should be fast
    }

    #[test_log::test(tokio::test)]
    #[ignore = "Takes too long"]
    async fn test_install_editable_in_git_repo() {
        let dir = dies_core::workspace_utils::get_workspace_root();
        let mut venv = PythonDistro::new(PythonDistroConfig::from_version_and_build(
            "3.10.13", 20240107,
        ))
        .await
        .unwrap()
        .create_venv(dir)
        .await
        .unwrap();

        // Create a fake package
        let package = dir.join("py").join("dies-py");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(
            package.join("pyproject.toml"),
            indoc::indoc! {r#"
                [project]
                name = "dies-py"
                version = "0.1.0"
                dependencies = ["msgspec>=0.18.4"]

                [build-system]
                requires = ["setuptools"]
                build-backend = "setuptools.build_meta"
            "#},
        )
        .unwrap();
        std::fs::create_dir(package.join("src")).unwrap();
        std::fs::write(
            package.join("src").join("main.py"),
            "print('Hello, world!')",
        )
        .unwrap();

        venv.install_editable(&package).await.unwrap();

        // Re-run to test that it doesn't install twice
        let start = std::time::Instant::now();
        venv.install_editable(&package).await.unwrap();
        assert!(start.elapsed().as_millis() < 500); // Kind of arbitrary, but it should be fast
    }

    #[test_log::test(tokio::test)]
    #[ignore = "Takes too long"]
    async fn test_intall_pypi_package() {
        let dir = tempdir().unwrap();
        let mut venv = PythonDistro::new(PythonDistroConfig::from_version_and_build(
            "3.10.13", 20240107,
        ))
        .await
        .unwrap()
        .create_venv(dir.path())
        .await
        .unwrap();

        venv.install("msgspec", "0.18.4").await.unwrap();
    }
}
