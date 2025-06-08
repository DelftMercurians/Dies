setup:
  uv venv
  uv pip install -e mpc_jax

run:
  uv run cargo run
