//! Client for the log vault (`dies-logvault` server, deployed on mercury11):
//! push `.dieslog` files, list what's stored, and pull a shared log by link.
//!
//! A share link looks like `http://<host>/l/<id>`; the server base URL is
//! derived from the link itself, so a link from any vault works regardless of
//! the locally configured one.

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncReadExt;

/// Default vault (mercury11 via Tailscale MagicDNS). Override with
/// `DIES_LOGVAULT_URL` or `--logvault-url`.
pub const DEFAULT_URL: &str = "http://mercury11:8420";

/// Subdirectory of the log directory where pulled logs are cached.
pub const CACHE_SUBDIR: &str = "vault";

/// One stored log, as the server reports it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteLog {
    pub id: String,
    pub name: String,
    pub sha256: String,
    pub size_uploaded: u64,
    pub size_stored: u64,
    pub uploaded_at_unix: f64,
    pub session_start_unix: Option<f64>,
    pub duration_s: Option<f64>,
    pub frame_count: Option<u64>,
    pub is_simulation: bool,
    pub is_match: bool,
    pub blue_strategy: Option<String>,
    pub yellow_strategy: Option<String>,
    pub blue_team_name: Option<String>,
    pub yellow_team_name: Option<String>,
    /// Share link (`<base>/l/<id>`).
    pub url: String,
}

/// Outcome of a push.
#[derive(Debug, Clone)]
pub struct Pushed {
    pub log: RemoteLog,
    /// True if the vault already had this exact file (nothing was uploaded).
    pub already_stored: bool,
}

#[derive(Deserialize)]
struct UploadResponse {
    #[serde(flatten)]
    log: RemoteLog,
    deduped: bool,
}

#[derive(Clone)]
pub struct VaultClient {
    base: String,
    http: reqwest::Client,
}

impl VaultClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            base: base_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base
    }

    /// List stored logs, newest session first. `query` filters by name,
    /// strategy or team name on the server.
    pub async fn list(&self, query: Option<&str>) -> Result<Vec<RemoteLog>> {
        let mut req = self.http.get(format!("{}/api/logs", self.base));
        if let Some(q) = query.filter(|q| !q.is_empty()) {
            req = req.query(&[("q", q)]);
        }
        let resp = req.send().await.with_context(|| self.unreachable())?;
        Ok(check(resp).await?.json().await?)
    }

    /// Upload a `.dieslog`. A log *directory* is resolved to its sibling
    /// `.dieslog` — a directory without one is still recording (or crashed
    /// before compaction) and is refused. Skips the transfer if the vault
    /// already holds a byte-identical file.
    pub async fn push(&self, path: &Path) -> Result<Pushed> {
        let file = resolve_dieslog(path)?;
        let sha = sha256_file(&file).await?;

        let resp = self
            .http
            .get(format!("{}/api/by-hash/{sha}", self.base))
            .send()
            .await
            .with_context(|| self.unreachable())?;
        if resp.status().is_success() {
            return Ok(Pushed {
                log: resp.json().await?,
                already_stored: true,
            });
        }

        let name = file
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        let f = tokio::fs::File::open(&file).await?;
        let len = f.metadata().await?.len();
        let body = reqwest::Body::wrap_stream(tokio_util::io::ReaderStream::new(f));
        let resp = self
            .http
            .post(format!("{}/api/logs", self.base))
            .query(&[("name", name.as_str())])
            .header(reqwest::header::CONTENT_LENGTH, len)
            .body(body)
            .send()
            .await
            .with_context(|| self.unreachable())?;
        let up: UploadResponse = check(resp).await?.json().await?;
        Ok(Pushed {
            log: up.log,
            already_stored: up.deduped,
        })
    }

    fn unreachable(&self) -> String {
        format!("log vault at {} unreachable", self.base)
    }
}

/// Split a share link (`http://host:port/l/<id>`) into `(base, id)`. Also
/// accepts the API forms `…/api/logs/<id>[/file]`.
pub fn parse_link(link: &str) -> Result<(String, String)> {
    let link = link.trim().trim_end_matches('/');
    let link = link.split(['?', '#']).next().unwrap_or(link);
    let link = link.strip_suffix("/file").unwrap_or(link);
    for marker in ["/l/", "/api/logs/"] {
        if let Some(i) = link.rfind(marker) {
            let base = &link[..i];
            let id = &link[i + marker.len()..];
            if (base.starts_with("http://") || base.starts_with("https://"))
                && !id.is_empty()
                && id.chars().all(|c| c.is_ascii_alphanumeric())
            {
                return Ok((base.to_string(), id.to_string()));
            }
        }
    }
    Err(anyhow!("not a log vault link: {link}"))
}

/// Download the log behind a share link into `<log_dir>/vault/<id>.dieslog`
/// (reused if already cached) and return its path.
pub async fn pull(link: &str, log_dir: &Path) -> Result<PathBuf> {
    let (base, id) = parse_link(link)?;
    let cache = log_dir.join(CACHE_SUBDIR);
    let dst = cache.join(format!("{id}.dieslog"));
    if dst.is_file() {
        return Ok(dst);
    }
    tokio::fs::create_dir_all(&cache).await?;

    let resp = reqwest::get(format!("{base}/api/logs/{id}/file"))
        .await
        .with_context(|| format!("log vault at {base} unreachable"))?;
    let resp = check(resp).await?;

    // Stream to a temp name, rename when complete so a partial download is
    // never mistaken for a cached log.
    let tmp = cache.join(format!(".{id}.part"));
    let mut out = tokio::fs::File::create(&tmp).await?;
    let mut stream = resp.bytes_stream();
    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;
    while let Some(chunk) = stream.next().await {
        out.write_all(&chunk?).await?;
    }
    out.flush().await?;
    drop(out);
    tokio::fs::rename(&tmp, &dst).await?;
    Ok(dst)
}

/// Every pushable log under `log_dir` (top level only — not `vault/` or
/// `selfplay/`): the `.dieslog` zips.
pub fn local_logs(log_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(log_dir)
        .with_context(|| format!("reading {}", log_dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "dieslog"))
        .collect();
    out.sort();
    Ok(out)
}

fn resolve_dieslog(path: &Path) -> Result<PathBuf> {
    if path.is_file() {
        if path.extension().is_some_and(|e| e == "dieslog") {
            return Ok(path.to_path_buf());
        }
        bail!("{} is not a .dieslog", path.display());
    }
    if path.is_dir() {
        let zip = PathBuf::from(format!(
            "{}.dieslog",
            path.to_string_lossy().trim_end_matches('/')
        ));
        if zip.is_file() {
            return Ok(zip);
        }
        bail!(
            "{} has no .dieslog yet (still recording, or the session crashed before compaction)",
            path.display()
        );
    }
    bail!("{} does not exist", path.display())
}

async fn sha256_file(path: &Path) -> Result<String> {
    let mut f = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

async fn check(resp: reqwest::Response) -> Result<reqwest::Response> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    bail!("log vault returned {status}: {body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_links() {
        let want = ("http://mercury11:8420".to_string(), "abc23".to_string());
        assert_eq!(parse_link("http://mercury11:8420/l/abc23").unwrap(), want);
        assert_eq!(parse_link(" http://mercury11:8420/l/abc23/ ").unwrap(), want);
        assert_eq!(
            parse_link("http://mercury11:8420/api/logs/abc23/file").unwrap(),
            want
        );
        assert!(parse_link("mercury11/l/abc").is_err());
        assert!(parse_link("http://x/l/../etc").is_err());
    }
}
