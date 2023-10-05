use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Error};

use super::pyproject::latest_available_python_version;
use super::sources::{PythonVersion, PythonVersionRequest};

static APP_DIR: Mutex<Option<&'static PathBuf>> = Mutex::new(None);

/// Returns the application directory.
pub fn get_app_dir() -> &'static Path {
    APP_DIR.lock().unwrap().expect("platform not initialized")
}

/// Runs a check if symlinks are supported.
pub fn symlinks_supported() -> bool {
    #[cfg(unix)]
    {
        true
    }
    #[cfg(windows)]
    {
        use once_cell::sync::Lazy;

        fn probe() -> Result<(), std::io::Error> {
            let dir = tempfile::tempdir()?;
            let a_path = dir.path().join("a");
            fs::write(&a_path, "")?;
            std::os::windows::fs::symlink_file(&a_path, dir.path().join("b"))?;
            Ok(())
        }

        static SUPPORTED: Lazy<bool> = Lazy::new(|| probe().is_ok());
        *SUPPORTED
    }
}

/// Returns the cache directory for a particular python version that can be downloaded.
pub fn get_canonical_py_path(version: &PythonVersion) -> Result<PathBuf, Error> {
    let mut rv = get_app_dir().to_path_buf();
    rv.push("py");
    rv.push(version.to_string());
    Ok(rv)
}

/// Returns the path of the python binary for the given version.
pub fn get_toolchain_python_bin(version: &PythonVersion) -> Result<PathBuf, Error> {
    let mut p = get_canonical_py_path(version)?;

    // It's permissible to link Python binaries directly in two ways.  It can either be
    // a symlink in which case it's used directly, it can be a non-executable text file
    // in which case the contents are the location of the interpreter, or it can be an
    // executable file on unix.
    if p.is_file() {
        if p.is_symlink() {
            return Ok(p.canonicalize()?);
        }
        #[cfg(unix)]
        {
            use std::os::unix::prelude::MetadataExt;
            if p.metadata().map_or(false, |x| x.mode() & 0o001 != 0) {
                return Ok(p);
            }
        }
        let contents = fs::read_to_string(&p).context("could not read toolchain file")?;
        return Ok(PathBuf::from(contents.trim_end()));
    }

    // we support install/bin/python, install/python and bin/python
    p.push("install");
    if !p.is_dir() {
        p.pop();
    }
    p.push("bin");
    if !p.is_dir() {
        p.pop();
    }

    #[cfg(unix)]
    {
        p.push("python3");
    }
    #[cfg(windows)]
    {
        p.push("python.exe");
    }

    Ok(p)
}

/// Returns a list of all registered toolchains.
pub fn list_known_toolchains() -> Result<Vec<(PythonVersion, PathBuf)>, Error> {
    let folder = get_app_dir().join("py");
    let mut rv = Vec::new();
    if let Ok(iter) = folder.read_dir() {
        for entry in iter {
            let entry = entry?;
            if let Ok(ver) = entry
                .file_name()
                .as_os_str()
                .to_string_lossy()
                .parse::<PythonVersion>()
            {
                let target = get_toolchain_python_bin(&ver)?;
                if !target.exists() {
                    continue;
                }
                rv.push((ver, target));
            }
        }
    }
    Ok(rv)
}

/// Reads the current `.python-version` file.
pub fn get_python_version_request_from_pyenv_pin(root: &Path) -> Option<PythonVersionRequest> {
    let mut here = root.to_owned();

    loop {
        here.push(".python-version");
        if let Ok(contents) = fs::read_to_string(&here) {
            let ver = contents.trim().parse().ok()?;
            return Some(ver);
        }

        // pop filename
        here.pop();

        // pop parent
        if !here.pop() {
            break;
        }
    }

    None
}

/// Returns the most recent cpython release.
pub fn get_latest_cpython_version() -> Result<PythonVersion, Error> {
    latest_available_python_version(&PythonVersionRequest {
        name: None,
        arch: None,
        os: None,
        major: 3,
        minor: None,
        patch: None,
        suffix: None,
    })
    .context("unsupported platform")
}
