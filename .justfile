mpc:
  cd mpc_jax && uv run pytest -vv

run:
  uv pip install -e mpc_jax
  uv run cargo run
