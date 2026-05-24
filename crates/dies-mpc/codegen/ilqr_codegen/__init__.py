"""Declarative iLQR code generator.

Public surface:

    Model       — declarative formulation builder (model.py)
    jac         — sp.Matrix(f).jacobian(v) wrapper
    hessian     — sp.hessian wrapper
    half        — sp.Rational(1, 2)
"""

from .derive import half, hessian, jac
from .model import Model

__all__ = ["Model", "jac", "hessian", "half"]
