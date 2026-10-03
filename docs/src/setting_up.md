# Setting Up

## Requirements

- A recent stable Rust toolchain, managed with [rustup](https://rustup.rs/). Install `rustfmt` and `clippy` with `rustup component add rustfmt clippy`.
- On Linux: `sudo apt install libudev-dev libssl-dev pkg-config clang`.
- [`pnpm`](https://pnpm.io/) and [`typeshare`](https://github.com/1Password/typeshare) (`cargo install typeshare-cli`) on your `PATH`. The `dies-webui` build script uses them to regenerate the TypeScript bindings and build the frontend bundle.

## Running

Dies is a plain Cargo workspace. The `dies-cli` crate is the entry point; it also builds the strategy binary for you.

```sh
cargo run
```

Open the web UI at <http://localhost:5555>.

For a headless, faster-than-realtime match between two strategies (no UI, deterministic for a given seed):

```sh
cargo run -- self-play --blue-strategy concerto --yellow-strategy concerto
```

## Web UI development

Run `pnpm install` in `webui/` once, then start [`mprocs`](https://github.com/pvolok/mprocs) from the repository root. It runs the Vite dev server (with hot module reloading) next to `dies --watch`.

## Tests and lints

```sh
cargo test                    # all tests
cargo test -p dies-world      # one crate
cargo fmt && cargo clippy
cd webui && pnpm run tsc      # TypeScript type check
```

## Debugging hangs

Install [`tokio-console`](https://github.com/tokio-rs/console) with `cargo install --locked tokio-console` and run it while Dies is running. It shows the state of every async task.
