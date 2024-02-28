<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://delftmercurians.nl/images/logo_dark.svg">
  <img src="https://delftmercurians.nl/images/logo.svg" style="max-width: 40%;">
</picture>
</p>

<h1 align="center">🌩 Dies 🌩</h1>

In Roman mythology [Dies](<https://en.wikipedia.org/wiki/Dies_(mythology)>) (_dai-ez_) was the personification of day and the mother of Mercury.

Dies is the framework powering the [Delft Mercurian](https://delftmercurians.nl/)'s RoboCup AI. It consists of a core written in Rust, which includes a physics simulator, networking, vision data processing, game state management, and a Python API and executor. Strategies -- the high-level logic that governs the player's behaviors -- are written in Python and can be run in a separate process, allowing for hot reloading and easy debugging.

## Getting Started

You'll need the following dependencies on your system:

- Stable [Rust](https://www.rust-lang.org/tools/install) toolchain. Use rustup if you can.
- On Linux, you'll need to install the  `pkg-config`, `libudev-dev` and `libssl-dev` packages: `sudo apt install libudev-dev libssl-dev pkg-config`.
- To run scripts, you should install cargo-make: `cargo install cargo-make`.

To run Dies locally, simply use `cargo run`.

If you are connected to the team VPN, you can run your local copy of Dies on the server. First, you'll need an ssh client:
 - Windows: [OpenSSH](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse?tabs=guide)
 - Ubuntu: `sudo apt install openssh-client`
 - Arch: `sudo pacman -S openssh`

You'll also need to make sure that your ssh key is added to the server's `~/.ssh/authorized_keys` file.

Then, you can run the following command to copy your local Dies to the server and run it there:

```sh
cargo make run-on-server
```

## Documentation

See [http://docs.delftmercurians.nl/](http://docs.delftmercurians.nl/) for the latest documentation.
