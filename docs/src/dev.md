# Developer Guide

This document provides some guidelines for developing Dies.

## Using Git

The main branch should remain stable at all times. All new features and fixes should be developed in feature branches, and merged into main only after they are complete and have been tested. Feature branches should be named `<name of author>/<feature name>`, eg. `balint/fix-ersim-env-win`.

The CI will run tests and formatting checks on all branches, but you should run these locally before pushing your changes.

Commit names should adhere to a lite version of the [Conventional Commits spec](https://www.conventionalcommits.org/en/v1.0.0/), ie. they should follow the format: `<commit type>: <short description>`. Scopes are not needed. The commit type should be one of the following:

- `feat`: new feature
- `fix`: bug fix
- `refactor`: refactoring
- `docs`: documentation
- `test`: adding tests
- `chore`: other changes (eg. CI, formatting, etc.)
- `wip`: work in progress
- `misc`: other changes

## Code Style

Dies uses the [rustfmt](https://github.com/rust-lang/rustfmt) formatter. It is recommended to use the [rust-analyzer](https://rust-analyzer.github.io/) extension for VS Code, and configure it to automatically format the code on save.

We also use [clippy](https://github.com/rust-lang/rust-clippy) for linting. The VS Code extension should also automatically run clippy on save.

In general, the Rust code should be idiomatic, and should attempt to follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).

Generics should be used with consideration. They are powerful, but they make the code more complex and significantly increase compile times. As a rule of thumb, remember YAGNI (you ain't gonna need it) -- don't overengineer, you can always add generics later if you need them.

## Crate Structure

Dies is split into multiple crates. We use [cargo workspaces](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html) to manage these. Having a lot of crates is not necessarily a bad thing -- crates are the unit of compilation in Rust, so more granular crates help caching and parallel compilation.

In general, the dependency graph between our crates should resemble a diamond:

```
     +-  feature crates  -+
    /                      \
dies-core ------------> dies-cli
    \                      /
     +-  feature crates  -+
```

There should be a few root crates (`dies-core`) which will rarely change, a number of specialized feature crates (`dies-basestation-client`, `dies-team`, etc.), and a few small leaf crates which just glue things together (`dies-cli`).

For an up-to-date list of crates, see the README.

## Error Handling

Errors should be handled according to Rust idioms. In general, _we should avoid panicking_, and instead propagate `Result` to the very top, where it can be handled by the executor.

Throughout the codebase, the [`anyhow`](https://docs.rs/anyhow/latest/anyhow/) crate should be used for error handling. Anwhow provides a convenient dynamic wrapper type for errors, so we do not have to define our own error types. `anyhow::Result` should be the return type of all functions which can fail.

## Logging

Logging is really important for Dies, as it is the only way to debug issues that occured during a match. Logs should contain all the information needed to replay matches, including all the data we receive from the vision system, the world state, and the commands we send to the robots.

Dies uses the [`tracing`](https://docs.rs/tracing/latest/tracing/) crate for logging. The following log levels should be used:

- `error`: should be used only for fatal errors, which cause the executor to stop, ie. errors should only be logged in the executor
- `warn`: should be used for unexpected events or recoverable errors, which do not cause the executor to stop and do not get propagated up
- `info`: should be used rarely, only for information that the user needs to know while running Dies - for example when long operations start and finish, or when error-prone operations succeed
- `debug`: should be used most of the time: everything from the data we receive from the vision system, commands we send to the robots, to all minor events should be logged at this level

We do not use `trace`.

`println!` and all other forms of direct output should be avoided, as they are not captured by the logger.

## Testing

Reliability is very important for Dies, so we aim for high test coverage. All new features should come with corresponding unit tests. Unit tests should test the code located in the module they are in, and should avoid relying on other modules.

Property-based testing should be explored for testing the core logic of Dies. See the [proptest](https://proptest-rs.github.io/proptest/intro.html) crate for more details on this technique.

## Documentation

_This section is WIP._
