mpc:
  cd mpc_jax && uv run pytest -vv -s && uv run -m mpc_jax.debug

run:
  uv pip install -e mpc_jax
  uv run cargo run
