one:
  uv run cargo run -- --auto-start

both:
  uv run cargo run -- --auto-start --controlled-teams=both

mpc:
  cd mpc_jax && uv run pytest -vv -s
