"""Thin sympy aliases used by formulation modules."""

from __future__ import annotations

from typing import Iterable, Union

import sympy as sp

half = sp.Rational(1, 2)

ExprOrMatrix = Union[sp.Expr, sp.Matrix]


def _as_matrix(expr: ExprOrMatrix) -> sp.Matrix:
    if isinstance(expr, sp.MatrixBase):
        return sp.Matrix(expr)
    return sp.Matrix([expr])


def _as_list(vars_: Union[sp.Matrix, Iterable[sp.Symbol]]) -> list[sp.Symbol]:
    if isinstance(vars_, sp.MatrixBase):
        return list(vars_)
    return list(vars_)


def jac(f: ExprOrMatrix, v: Union[sp.Matrix, Iterable[sp.Symbol]]) -> sp.Matrix:
    """Jacobian of `f` w.r.t. `v`. Works for scalar or matrix `f`."""

    return _as_matrix(f).jacobian(_as_list(v))


def hessian(f: ExprOrMatrix, v: Union[sp.Matrix, Iterable[sp.Symbol]]) -> sp.Matrix:
    """Hessian of scalar `f` w.r.t. `v`."""

    scalar = sp.sympify(f)
    if isinstance(scalar, sp.MatrixBase):
        if scalar.shape != (1, 1):
            raise ValueError(f"hessian() expects scalar f, got shape {scalar.shape}")
        scalar = scalar[0, 0]
    return sp.hessian(scalar, _as_list(v))
