set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

mpc-dev:
  # Add common binary paths
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

  # ---- Project-specific setup ----
  cd crates/dies-executor/src/control/mpc_jax

  # Create / reuse venv with uv and activate it
  uv venv
  source .venv/bin/activate

  # Environment for PyO3
  export PYO3_PYTHON="$(which python)"
  export PYO3_CROSS_LIB_DIR="$(python -c 'import sysconfig, os; print(sysconfig.get_config_var("LIBDIR") or os.path.dirname(sysconfig.get_config_var("LIBRARY")))')"
  python -m pip install -e .

  # ----- Activate the venv with the right script and drop into that shell -----
  shell_name=$(basename "$SHELL")

  case "$shell_name" in
    fish)
      source .venv/bin/activate.fish
      echo "🐟  Dropping you into fish with the venv active…"
      exec fish  -i
      ;;
    zsh)
      source .venv/bin/activate.zsh
      echo "🐚  Dropping you into zsh with the venv active…"
      exec zsh   -i
      ;;
    *)
      # Covers bash, dash, ksh, etc.
      source .venv/bin/activate
      echo "🌀  Dropping you into $shell_name with the venv active…"
      exec "$SHELL" -i
      ;;
  esac
