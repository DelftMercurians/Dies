"""Turn an `ilqr_codegen.Model` into a dict of Python-friendly callables.

For each `m.function` registered on the model, `build()` produces a callable
whose signature mirrors the model's Rust `m.arg(...)` declarations. The
callable:

  1. Accepts structured Python args (the dataclasses from `pysim.types`,
     or plain ndarrays/scalars).
  2. Evaluates each binding's `rust_source` against those args to get a
     scalar (e.g. `p.tau[FWD]` → 0.08). Sources are author-controlled
     formulation files, so the use of `eval` is safe by construction.
  3. Calls a SymPy-lambdified inner function that does the actual math.
  4. Reshapes outputs (vector exports squeezed from Nx1 → N,).
  5. Returns scalar / ndarray / tuple matching the export(s).

Multi-export functions return a tuple in declaration order, matching the
generated Rust tuple ABI.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import sympy as sp

from ilqr_codegen.model import Arg, Binding, Export, Function, Model


# Names available inside binding source-string eval context. Keep in sync
# with the constants exported from `crates/dies-mpc/src/types.rs`.
_EVAL_GLOBALS: dict[str, Any] = {
    "FWD": 0,
    "STRAFE": 1,
    "__builtins__": {},  # no built-ins; bindings only need attribute/index access
}


def _kind(expr) -> str:
    """One of: 'scalar', 'vector', 'matrix' — drives output reshape."""

    if isinstance(expr, sp.MatrixBase):
        r, c = expr.shape
        if r == 1 and c == 1:
            return "scalar"
        if r == 1 or c == 1:
            return "vector"
        return "matrix"
    return "scalar"


def _to_lambdify_arg(expr):
    """sp.lambdify needs Matrix for vector/matrix outputs; scalar Expr for scalars."""

    if isinstance(expr, sp.MatrixBase):
        r, c = expr.shape
        if r == 1 and c == 1:
            return expr[0, 0]
        return expr
    return expr


def _reshape_output(raw, kind: str):
    if kind == "scalar":
        # SymPy may return a 0-d ndarray or a Python float; normalise to float.
        return float(np.asarray(raw).item())
    if kind == "vector":
        return np.asarray(raw).reshape(-1)
    return np.asarray(raw)


def _build_one(
    model: Model,
    fn: Function,
) -> Callable:
    exports: list[Export] = [model.exports[r] for r in fn.returns]
    required_args: list[Arg] = model.args_for(exports)
    arg_names = [a.name for a in required_args]
    bindings: list[Binding] = model.bindings_for(exports)

    # Lambdify: sym_list in binding-insertion order, outputs in export order.
    sym_list = [b.symbol for b in bindings]
    out_specs = [(e.name, _kind(e.expr)) for e in exports]
    lambdify_args = [_to_lambdify_arg(e.expr) for e in exports]
    inner = sp.lambdify(sym_list, lambdify_args, modules="numpy", cse=True)

    # Compile each binding's source string once for fast eval at call time.
    compiled: list[tuple[str, Any]] = [
        (str(b.symbol), compile(b.rust_source, f"<{b.symbol}>", "eval"))
        for b in bindings
    ]

    def wrapper(*args, **kwargs):
        if len(args) > len(arg_names):
            raise TypeError(
                f"{fn.name}: too many positional args "
                f"(expected {len(arg_names)}: {arg_names})"
            )
        bound = dict(zip(arg_names, args))
        for k, v in kwargs.items():
            if k not in arg_names:
                raise TypeError(f"{fn.name}: unexpected arg {k!r}; expected {arg_names}")
            if k in bound:
                raise TypeError(f"{fn.name}: arg {k!r} passed both positionally and by name")
            bound[k] = v
        missing = [n for n in arg_names if n not in bound]
        if missing:
            raise TypeError(f"{fn.name}: missing args {missing}")

        # Evaluate every binding source string to get scalars in sym_list order.
        scalars = []
        for sym_name, code in compiled:
            try:
                scalars.append(eval(code, _EVAL_GLOBALS, bound))
            except Exception as exc:
                raise RuntimeError(
                    f"{fn.name}: failed to evaluate binding {sym_name}: {exc}"
                ) from exc

        raw = inner(*scalars)
        if len(out_specs) == 1:
            return _reshape_output(raw, out_specs[0][1])
        # Lambdify returns a list/tuple when given a list of outputs.
        return tuple(_reshape_output(r, kind) for r, (_, kind) in zip(raw, out_specs))

    wrapper.__name__ = fn.name
    wrapper.__qualname__ = f"pysim.{model.name}.{fn.name}"
    wrapper.__doc__ = (
        f"Lambdified `{fn.name}` from model `{model.name}`. "
        f"Args: {arg_names}. Returns: {[n for n, _ in out_specs]}."
    )
    return wrapper


def build(model: Model) -> dict[str, Callable]:
    """For each registered `m.function`, return a Python callable.

    The returned dict is keyed by function name and contains callables
    whose signature mirrors the corresponding Rust function (modulo the
    Python-Rust args mapping — see `pysim.types`).
    """

    return {fn.name: _build_one(model, fn) for fn in model.functions}
