mpc:
  cd mpc_jax && uv run pytest -vv

run:
  cd mpc_jax && uv run pytest -vv -x
  uv pip install -e mpc_jax
  uv run cargo run
