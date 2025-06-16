setup:
  uv venv --python 3.12
  uv pip install -e mpc_jax

run:
  cd mpc_jax && uv run pytest -vv -x
  uv run cargo run
