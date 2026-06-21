//! Build script for dies-webui — folds the old `just webbuild` recipe into the
//! build graph.
//!
//! - TS bindings (`webui/src/bindings.ts`) are regenerated via `typeshare` on
//!   every build (cheap), but only written when the content actually changes so
//!   we don't churn the file and trigger needless vite reloads.
//! - The frontend bundle (`vite build` -> `crates/dies-webui/static`) is only
//!   produced for release builds; dev uses the vite dev server (see mprocs.yaml).

use std::path::{Path, PathBuf};
use std::process::Command;

/// Crate `src` dirs that hold `#[typeshare]` types feeding the bindings.
const TYPE_SRC_DIRS: &[&str] = &[
    "crates/dies-core/src",
    "crates/dies-test-driver/src",
    "crates/dies-webui/src",
];

/// Frontend inputs that, when changed, require a release rebuild.
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

    generate_bindings(&root);

    if std::env::var("PROFILE").as_deref() == Ok("release") {
        build_frontend(&root);
    }
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
    let tmp = root.join("webui/src/.bindings.ts.tmp");

    let status = Command::new("typeshare")
        .current_dir(root)
        .arg(".")
        .arg("--lang=typescript")
        .arg("--output-file=webui/src/.bindings.ts.tmp")
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

/// Build the production frontend bundle into `crates/dies-webui/static`.
/// Failure here aborts the (release) build — the bundle is required to serve UI.
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
