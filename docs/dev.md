# Developer Documentation

This document provides some guidelines for developing Dies.

It is recommended to first get an overview of Dies' architecture, which is described in [docs/architecture.md](./architecture.md).

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

There should be a few root crates (`dies-core`) which will rarely change, a number of specialized feature crates (`dies-ersim-env`, `dies-python-rt`, etc.), and a few small leaf crates which just glue things together (`dies-cli`).

For an up-to-date list of crates, see the README.

## Testing

Reliability is very important for Dies, so we aim for high test coverage. All new features should come with corresponding unit tests. Unit tests should test the code located in the module they are in, and should avoid relying on other modules.

Property-based testing should be explored for testing the core logic of Dies. See the [proptest](https://proptest-rs.github.io/proptest/intro.html) crate for more details on this technique.

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
- `revert`: reverting a previous commit
- `misc`: other changes

## Documentation

_This section is WIP._
