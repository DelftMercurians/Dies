# Setting Up

This page describes how to set up Dies for development.

## Requirements

To compile the project all you need a recent version of stable Rust. We recommend using [rustup](https://rustup.rs/) to manage your Rust installation.

For development, you should make sure that `rustfmt` and `clippy` are installed. You can install them with `rustup component add rustfmt clippy`.

To use the ER-Force simulator, you will also need to install a recent version of [Docker](https://docs.docker.com/engine/install/) and ensure that the Docker daemon is running.

## Building and Running

Dies uses [cargo workspaces](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html) to manage its crates. The `dies-cli` crate contains the Dies CLI, which can be used to run matches.

To run tests for all crates, run `cargo test` in the root directory. You can run specific tests by specifying the crate name, eg. `cargo test -p dies-core`.

To run the CLI from the workspace root, use `cargo run`.

_TODO: describe web UI._
