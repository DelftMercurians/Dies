"""Small Rust expression printer for the scalar algebra emitted by codegen.

SymPy's Rust printer currently misses parentheses for some `Mul(Add(...))`
forms, so this printer handles the limited expression grammar we generate:
symbols, numbers, add, multiply, integer powers, sin, cos, and tanh.
"""

from __future__ import annotations

import sympy as sp


def rust_number(expr: sp.Expr) -> str:
    if expr.is_Integer:
        return f"{int(expr)}.0"
    if expr.is_Rational:
        return f"{float(expr):.17g}"
    if expr.is_Float:
        return f"{float(expr):.17g}"
    raise TypeError(f"not a number: {expr!r}")


def rust_expr(expr: sp.Expr, parent_prec: int = 0) -> str:
    expr = sp.sympify(expr)
    if expr.is_Number:
        return rust_number(expr)
    if expr.is_Symbol:
        return str(expr)

    if isinstance(expr, sp.Add):
        pieces: list[str] = []
        for term in expr.as_ordered_terms():
            if term.could_extract_minus_sign():
                printed = rust_expr(-term, parent_prec=1)
                pieces.append(("-" if not pieces else " - ") + printed)
            else:
                printed = rust_expr(term, parent_prec=1)
                pieces.append(("" if not pieces else " + ") + printed)
        text = "".join(pieces)
        return f"({text})" if parent_prec > 0 else text

    if isinstance(expr, sp.Mul):
        sign = ""
        if expr.could_extract_minus_sign():
            sign = "-"
            expr = -expr
        factors = [rust_expr(arg, parent_prec=2) for arg in expr.as_ordered_factors()]
        text = sign + "*".join(factors)
        return f"({text})" if parent_prec > 2 else text

    if isinstance(expr, sp.Pow):
        base, exp = expr.as_base_exp()
        base_text = rust_expr(base, parent_prec=3)
        if exp == -1:
            return f"{base_text}.recip()"
        if exp.is_Integer:
            return f"{base_text}.powi({int(exp)})"
        return f"{base_text}.powf({rust_expr(exp)})"

    if expr.func in (sp.sin, sp.cos, sp.tanh):
        arg = rust_expr(expr.args[0], parent_prec=3)
        return f"{arg}.{expr.func.__name__}()"

    raise TypeError(f"unsupported expression for Rust generation: {expr!r}")
