//! Build script for dies-webui — folds the old `just webbuild` recipe into the
//! build graph.
//!
//! - TS bindings (`webui/src/bindings.ts`) are regenerated via `typeshare` on
//!   every build (cheap), but only written when the content actually changes so
//!   we don't churn the file and trigger needless vite reloads.
//! - The frontend bundle (`vite build` -> `crates/dies-webui/static`) is rebuilt
//!   whenever the hash of the frontend inputs changes (in any profile), so a
//!   debug binary serves an up-to-date bundle too. Since `bindings.ts` is part
//!   of the hashed inputs, a Rust type change that alters the bindings also
//!   triggers a frontend rebuild. Dev iteration normally uses the vite dev
//!   server (see mprocs.yaml), which doesn't touch this bundle.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Crate `src` dirs that hold `#[typeshare]` types feeding the bindings.
const TYPE_SRC_DIRS: &[&str] = &[
    "crates/dies-core/src",
    "crates/dies-test-driver/src",
    "crates/dies-webui/src",
];

/// Frontend inputs that, when changed, require a bundle rebuild. `webui/src`
/// includes the generated `bindings.ts`.
const FRONTEND_INPUTS: &[&str] = &[
    "webui/src",
    "webui/index.html",
    "webui/package.json",
    "webui/pnpm-lock.yaml",
    "webui/vite.config.ts",
];

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("dies-webui should live at <root>/crates/dies-webui")
        .to_path_buf();

    // Declare build-script inputs (switches cargo to explicit dependency mode).
    for dir in TYPE_SRC_DIRS {
        emit_rerun_recursive(&root.join(dir));
    }
    for input in FRONTEND_INPUTS {
        emit_rerun_recursive(&root.join(input));
    }

    // Bindings are frontend source, so regenerate them before hashing the
    // frontend: a Rust type change that alters the bindings then rebuilds the
    // bundle too.
    generate_bindings(&root);
    maybe_build_frontend(&root);
}

/// Emit `cargo:rerun-if-changed` for a path; if it is a directory, recurse so
/// nested edits are tracked (cargo only watches the directory entry otherwise).
fn emit_rerun_recursive(path: &Path) {
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                emit_rerun_recursive(&entry.path());
            }
        }
    }
    println!("cargo:rerun-if-changed={}", path.display());
}

/// Regenerate `webui/src/bindings.ts` from the workspace's `#[typeshare]` types.
/// Best-effort: a missing/failing `typeshare` warns rather than fails the build,
/// so a checked-in bindings file still lets the crate compile.
fn generate_bindings(root: &Path) {
    let output = root.join("webui/src/bindings.ts");
    // Write the temp file to OUT_DIR, NOT into `webui/src`. Creating/deleting an
    // entry inside `webui/src` bumps that directory's mtime, and since we emit
    // `rerun-if-changed` for it, cargo would mark dies-webui dirty on every
    // subsequent build — forcing a needless recompile + relink of dies-cli even
    // when nothing changed.
    let tmp = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bindings.ts.tmp");

    let status = Command::new("typeshare")
        .current_dir(root)
        .arg(".")
        .arg("--lang=typescript")
        .arg(format!("--output-file={}", tmp.display()))
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            let _ = std::fs::remove_file(&tmp);
            println!("cargo:warning=typeshare failed ({s}); TS bindings may be stale");
            return;
        }
        Err(e) => {
            println!(
                "cargo:warning=could not run typeshare ({e}); install with \
                 `cargo install typeshare-cli`. TS bindings may be stale"
            );
            return;
        }
    }

    let mut contents = match std::fs::read_to_string(&tmp) {
        Ok(c) => c,
        Err(e) => {
            println!("cargo:warning=could not read generated bindings ({e})");
            return;
        }
    };
    let _ = std::fs::remove_file(&tmp);

    // Strip the typeshare artifact and append the hand-maintained type aliases
    // (these map Rust types typeshare leaves unresolved).
    contents = contents.replace("data?: undefined", "");
    contents.push_str(
        "\nexport type Vector2 = [number, number];\n\
         export type Vector3 = [number, number, number];\n\
         export type Duration = number;\n\
         export type HashSet<T> = Array<T>;\n",
    );

    // Only write when changed, to avoid churning mtime / vite reloads.
    let unchanged = std::fs::read_to_string(&output)
        .map(|existing| existing == contents)
        .unwrap_or(false);
    if !unchanged {
        if let Err(e) = std::fs::write(&output, contents) {
            println!("cargo:warning=could not write bindings.ts ({e})");
        }
    }
}

/// Rebuild the frontend bundle if the frontend inputs changed since the last
/// build, or if the bundle is missing. The last-built hash is cached in
/// `OUT_DIR` (wiped by `cargo clean -p dies-webui`, forcing a rebuild).
fn maybe_build_frontend(root: &Path) {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let hash_file = out_dir.join("frontend.hash");

    let current = frontend_hash(root).to_string();
    let previous = std::fs::read_to_string(&hash_file).unwrap_or_default();
    let bundle_present = root.join("crates/dies-webui/static/index.html").exists();

    if current == previous && bundle_present {
        return;
    }

    build_frontend(root);

    if let Err(e) = std::fs::write(&hash_file, &current) {
        println!("cargo:warning=could not record frontend hash ({e})");
    }
}

/// Content hash of all frontend input files (path + bytes), order-independent.
fn frontend_hash(root: &Path) -> u64 {
    let mut files = Vec::new();
    for input in FRONTEND_INPUTS {
        collect_files(&root.join(input), &mut files);
    }
    files.sort();

    let mut hasher = DefaultHasher::new();
    for path in &files {
        path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .hash(&mut hasher);
        match std::fs::read(path) {
            Ok(bytes) => bytes.hash(&mut hasher),
            Err(_) => 0u8.hash(&mut hasher),
        }
    }
    hasher.finish()
}

/// Recursively collect file paths under `path` (directories are descended).
fn collect_files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                collect_files(&entry.path(), out);
            }
        }
    } else if path.is_file() {
        out.push(path.to_path_buf());
    }
}

/// Build the production frontend bundle into `crates/dies-webui/static`.
/// Failure aborts the build — a stale/missing bundle would be served silently.
fn build_frontend(root: &Path) {
    let webui = root.join("webui");
    let status = Command::new("pnpm")
        .current_dir(&webui)
        .args(["run", "build"])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => panic!("`pnpm run build` failed ({s})"),
        Err(e) => panic!("could not run `pnpm run build` (is pnpm installed?): {e}"),
    }
}
