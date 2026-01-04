one:
  uv run cargo run -- --auto-start

both:
  uv run cargo run -- --auto-start --controlled-teams=both

mpc:
  cd mpc_jax && uv run pytest -vv -s

# Build all strategies and copy them to target/strategies
build-strategies:
  cargo build -p test-strategy
  mkdir -p target/strategies
  cp target/debug/test-strategy target/strategies/

# Build strategies in release mode
build-strategies-release:
  cargo build -p test-strategy --release
  mkdir -p target/strategies
  cp target/release/test-strategy target/strategies/
