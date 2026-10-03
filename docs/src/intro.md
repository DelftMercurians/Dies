# Introduction

Dies is the Delft Mercurians' framework for RoboCup Small Size League. It runs on a central computer, receives camera and referee data from the league software, decides what our robots should do, and sends commands to them over radio. It also contains a physics simulator with an auto-referee, so the whole stack can run without any hardware.

The core is written in Rust. Strategies, the team-level logic, are separate Rust binaries that run in their own process and talk to the core over IPC, so they can be rebuilt and hot-swapped while Dies keeps running. A web UI shows the field, debug output and settings, and lets you control the executor and simulator.

- [Setting Up](./setting_up.md): install the toolchain and run Dies.
- [Codebase Map](./codebase_map.md): follow one frame through the system and find where a change belongs.
- [Developer Guide](./dev.md): conventions for git, code style, errors and logging.

The source lives at [github.com/DelftMercurians/Dies](https://github.com/DelftMercurians/Dies).
