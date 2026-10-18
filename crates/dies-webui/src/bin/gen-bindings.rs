//! Regenerates `webui/src/bindings.ts` from the Rust types (see
//! `dies_webui::bindings`). Run after changing a type the web UI uses:
//! `cargo run -p dies-webui --bin gen-bindings`.

use std::path::Path;

use anyhow::Context;

fn main() -> anyhow::Result<()> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../webui/src/bindings.ts");
    std::fs::write(&path, dies_webui::bindings::render())
        .with_context(|| format!("writing {}", path.display()))
}
