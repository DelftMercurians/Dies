"""Per-function Rust emission: arg inference, CSE, vector/matrix assignment."""

from __future__ import annotations

from typing import Union

import sympy as sp

from .model import Arg, Export, Function, Model
from .rust import rust_expr


# Map nalgebra type strings to their constructor identifier when they differ.
# Type aliases (State, Control, StateJac, ControlJac) double as constructors.
_TYPE_CTOR: dict[str, str] = {
    "Vector2<f64>": "Vector2",
    "Vector3<f64>": "Vector3",
    "Vector4<f64>": "Vector4",
    "Vector5<f64>": "Vector5",
    "Matrix2<f64>": "Matrix2",
    "Matrix3<f64>": "Matrix3",
    "Matrix4<f64>": "Matrix4",
    "Matrix5<f64>": "Matrix5",
    "Matrix2x4<f64>": "Matrix2x4",
    "Matrix4x2<f64>": "Matrix4x2",
    "Matrix3x5<f64>": "Matrix3x5",
    "Matrix5x3<f64>": "Matrix5x3",
}


def _ctor_for(rust_type: str) -> str:
    return _TYPE_CTOR.get(rust_type, rust_type)


def _shape(expr) -> tuple[int, int]:
    if isinstance(expr, sp.MatrixBase):
        return expr.shape
    return (1, 1)


def _flat(expr) -> list[sp.Expr]:
    if isinstance(expr, sp.MatrixBase):
        return [sp.sympify(e) for e in expr]
    return [sp.sympify(expr)]


def _classify(expr) -> str:
    """One of: 'scalar', 'vector', 'matrix'."""

    r, c = _shape(expr)
    if r == 1 and c == 1:
        return "scalar"
    if r == 1 or c == 1:
        return "vector"
    return "matrix"


def emit_function(model: Model, fn: Function) -> str:
    exports: list[Export] = [model.exports[r] for r in fn.returns]
    args: list[Arg] = model.args_for(exports)
    bindings = model.bindings_for(exports)

    # Concatenate all flat output entries; remember slices per export.
    flat: list[sp.Expr] = []
    slices: list[tuple[int, int]] = []
    for exp in exports:
        entries = _flat(exp.expr)
        slices.append((len(flat), len(flat) + len(entries)))
        flat.extend(entries)

    replacements, reduced = sp.cse(flat, symbols=sp.numbered_symbols("z"))

    # Build signature.
    args_text = ", ".join(f"{a.name}: {a.rust_type}" for a in args)
    if len(exports) == 1:
        ret_type = exports[0].rust_type
    else:
        ret_type = "(" + ", ".join(e.rust_type for e in exports) + ")"

    lines: list[str] = [f"pub(crate) fn {fn.name}({args_text}) -> {ret_type} {{"]

    # Input bindings (sym = rust source), in insertion order.
    # Skip identity bindings — the arg is already in scope under that name.
    for b in bindings:
        if b.rust_source == str(b.symbol):
            continue
        lines.append(f"    let {b.symbol} = {b.rust_source};")

    # CSE replacements.
    for sym, expr in replacements:
        lines.append(f"    let {sym} = {rust_expr(expr)};")

    # Export assignments. For multi-export functions every export becomes a
    # `let <name> = ...;` so we can package them into the tuple return. For
    # single-export functions we emit scalars and vectors as the trailing
    # block expression (avoids clippy::let_and_return); matrices still need
    # the let-then-mutate dance.
    single = len(exports) == 1

    def emit_matrix(exp: Export, chunk: list[sp.Expr]) -> None:
        rows, cols = _shape(exp.expr)
        nonzero: list[tuple[int, int, sp.Expr]] = []
        for r in range(rows):
            for c in range(cols):
                e = sp.simplify(chunk[r * cols + c])
                if e != 0:
                    nonzero.append((r, c, e))
        mut = "mut " if nonzero else ""
        lines.append(f"    let {mut}{exp.name} = {_ctor_for(exp.rust_type)}::zeros();")
        for r, c, e in nonzero:
            lines.append(f"    {exp.name}[({r}, {c})] = {rust_expr(e)};")

    out_names: list[str] = []
    for exp, (lo, hi) in zip(exports, slices):
        chunk = reduced[lo:hi]
        kind = _classify(exp.expr)
        if single and kind == "scalar":
            lines.append(f"    {rust_expr(chunk[0])}")
        elif single and kind == "vector":
            args_str = ", ".join(rust_expr(e) for e in chunk)
            lines.append(f"    {_ctor_for(exp.rust_type)}::new({args_str})")
        elif kind == "scalar":
            lines.append(f"    let {exp.name} = {rust_expr(chunk[0])};")
            out_names.append(exp.name)
        elif kind == "vector":
            args_str = ", ".join(rust_expr(e) for e in chunk)
            lines.append(f"    let {exp.name} = {_ctor_for(exp.rust_type)}::new({args_str});")
            out_names.append(exp.name)
        else:
            emit_matrix(exp, chunk)
            out_names.append(exp.name)

    if not single:
        lines.append(f"    ({', '.join(out_names)})")
    elif _classify(exports[0].expr) == "matrix":
        # single matrix export: trailing identifier returns it
        lines.append(f"    {exports[0].name}")

    lines.append("}")
    return "\n".join(lines) + "\n"
