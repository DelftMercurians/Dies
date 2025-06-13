setup:
  uv venv --python 3.12
  uv pip install -e mpc_jax

run:
  uv run cargo run
