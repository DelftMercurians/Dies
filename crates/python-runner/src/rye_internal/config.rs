use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Error};
use once_cell::sync::Lazy;
use regex::Regex;
use toml_edit::Document;

use super::platform::get_latest_cpython_version;
use super::pyproject::{SourceRef, SourceRefType};
use super::sources::PythonVersionRequest;

static CONFIG: Mutex<Option<Arc<Config>>> = Mutex::new(None);
static AUTHOR_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*(.*?)\s*<\s*(.*?)\s*>\s*$").unwrap());

#[derive(Clone)]
pub struct Config {
    doc: Document,
    path: PathBuf,
}

impl Config {
    /// Returns the current config
    pub fn current() -> Arc<Config> {
        CONFIG
            .lock()
            .unwrap()
            .as_ref()
            .expect("config not initialized")
            .clone()
    }

    /// Returns the default python toolchain
    pub fn default_toolchain(&self) -> Result<PythonVersionRequest, Error> {
        match self
            .doc
            .get("default")
            .and_then(|x| x.get("toolchain"))
            .and_then(|x| x.as_str())
        {
            Some(ver) => ver.parse(),
            None => get_latest_cpython_version().map(Into::into),
        }
        .context("failed to get default toolchain")
    }

    /// Returns the HTTP proxy that should be used.
    pub fn http_proxy_url(&self) -> Option<String> {
        std::env::var("http_proxy").ok().or_else(|| {
            self.doc
                .get("proxy")
                .and_then(|x| x.get("http"))
                .and_then(|x| x.as_str())
                .map(|x| x.to_string())
        })
    }

    /// Returns the HTTPS proxy that should be used.
    pub fn https_proxy_url(&self) -> Option<String> {
        std::env::var("HTTPS_PROXY")
            .ok()
            .or_else(|| std::env::var("https_proxy").ok())
            .or_else(|| {
                self.doc
                    .get("proxy")
                    .and_then(|x| x.get("https"))
                    .and_then(|x| x.as_str())
                    .map(|x| x.to_string())
            })
    }

    /// Returns the list of default sources.
    pub fn sources(&self) -> Result<Vec<SourceRef>, Error> {
        let mut rv = Vec::new();
        let mut need_default = true;
        if let Some(sources) = self.doc.get("sources").and_then(|x| x.as_array_of_tables()) {
            for source in sources {
                let source_ref = SourceRef::from_toml_table(source)?;
                if source_ref.name == "default" {
                    need_default = false;
                }
                rv.push(source_ref);
            }
        }

        if need_default {
            rv.push(SourceRef::from_url(
                "default".to_string(),
                "https://pypi.org/simple/".into(),
                SourceRefType::Index,
            ));
        }

        Ok(rv)
    }
}
