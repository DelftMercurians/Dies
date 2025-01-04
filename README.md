<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://delftmercurians.nl/images/logo_dark.svg">
  <img src="https://delftmercurians.nl/images/logo.svg" style="max-width: 40%;">
</picture>
</p>

<h1 align="center">🌩 Dies 🌩</h1>

In Roman mythology [Dies](<https://en.wikipedia.org/wiki/Dies_(mythology)>) (_dai-ez_) was the personification of day and the mother of Mercury.

Dies is the framework powering the [Delft Mercurian](https://delftmercurians.nl/)'s RoboCup AI. It consists of a core written in Rust, which includes a physics simulator, networking, vision data processing, game state management, and a Python API and executor. Strategies -- the high-level logic that governs the player's behaviors -- are written in Python and can be run in a separate process, allowing for hot reloading and easy debugging.

See [http://docs.delftmercurians.nl/](http://docs.delftmercurians.nl/) for the latest documentation.

## Getting Started

You'll need the following dependencies on your system:

- Stable [Rust](https://www.rust-lang.org/tools/install) toolchain. Use rustup if you can.
- On Linux, you'll need to install the `pkg-config`, `clang`, `libudev-dev` and `libssl-dev` packages: `sudo apt install libudev-dev libssl-dev pkg-config clang`.

To run Dies locally, simply use `cargo run -- <options>`.

### Debugging hanging issues

If you are experiencing hanging issues, you can use `tokio-console` to see the state of running tasks. To do this, you need to install `tokio-console`:

```sh
cargo install --locked tokio-console
```

Then while Dies is running, you can run:

```sh
tokio-console
```

You will see a `top`-like interface that shows the state of all running tasks. You can use `q` to quit.

### Web UI Development

If you do not indent to make changes to the Typescript code of the web UI, feel free to skip this section.

If you want to work on the web UI (`./webui`) you will also need:

- Node.JS v18+ - Recommended to use [`fnm`](https://github.com/Schniz/fnm) for installing it.
- [cargo-make](https://sagiegurari.github.io/cargo-make) - Install with `cargo install cargo-make`.

Then you will need to install the web dependencies

```bash
cd webui
npm install
```

To build the webui after making changes in `./webui`:

```bash
# From the project root:
cargo make webui
```

You can also pass the `--web-build` flag to `./run.sh` for the same effect (you still need all the other dependencies):

```bash
./run.sh --web-build <options>
```

## Crates

Dies is split into several crates, each with a specific purpose:

- ![`dies-core`](./crates/dies-core): Contains the most widely used types and traits. Should be kept slim.
- ![`dies-team`](./crates/dies-team): Contains the `Executor`, `TeamController`, `PlayerController`, `Role`, and `Strategy` types.
- ![`dies-basestation-client`](./crates/dies-basestation-client): Contains the `BasestationClient` type.
- ![`dies-ssl-client`](./crates/dies-ssl-client): Contains the `SSLClient` type.
- ![`dies-protos`](./crates/dies-protos): Contains the protobuf definitions used for communication with the game controller and vision.
- ![`dies-simulator`](./crates/dies-simulator): Contains the `Simulator` type, which can be used to replace the `SSLClient` and `BasestationClient` with a simulator for testing.
- ![`dies-tracker`](./crates/dies-tracker): Contains the `World` type, which is used to represent the state of the game world, as well as filters and processors for incoming data.
- ![`dies-webui`](./crates/dies-webui): Contains the backend for the web interface, which can be used to monitor and control the AI. The frotend is in the `webui` directory.
- ![`dies-cli`](./crates/dies-cli): Contains the command line interface for running the AI. This is the main entry point for the framework.
