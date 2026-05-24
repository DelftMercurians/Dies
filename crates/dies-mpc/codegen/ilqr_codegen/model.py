"""Declarative model: args, symbol↔Rust-source bindings, equations, exports, functions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Union

import sympy as sp


@dataclass
class Arg:
    name: str
    rust_type: str


@dataclass
class Binding:
    symbol: sp.Symbol
    rust_source: str
    deps: frozenset[str]


@dataclass
class Export:
    name: str
    expr: Union[sp.Expr, sp.MatrixBase]
    rust_type: str


@dataclass
class Function:
    name: str
    returns: list[str]


@dataclass
class Equation:
    label: str
    expr: Union[sp.Expr, sp.MatrixBase]  # the user-provided RHS (in terms of prior placeholders)


_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")


def _deps_for(source: str, arg_names: set[str]) -> frozenset[str]:
    return frozenset(tok for tok in _IDENT.findall(source) if tok in arg_names)


class Model:
    def __init__(self, name: str, title: Optional[str] = None):
        self.name = name
        self.title = title
        self.args: list[Arg] = []
        self._arg_names: set[str] = set()
        self.bindings: dict[sp.Symbol, Binding] = {}
        self.equations: list[Equation] = []
        self.exports: dict[str, Export] = {}
        self.functions: list[Function] = []
        # Placeholder substitution map: each entry maps an intermediate
        # placeholder symbol (introduced by m.eq) to its underlying expression
        # in terms of bindings and earlier placeholders. `expand` walks this
        # to a fixed point.
        self._intermediate_subs: dict[sp.Symbol, sp.Expr] = {}

    # ── args + bindings ─────────────────────────────────────────────
    def arg(self, name: str, rust_type: str) -> None:
        if name in self._arg_names:
            raise ValueError(f"duplicate arg: {name}")
        self.args.append(Arg(name, rust_type))
        self._arg_names.add(name)

    def scalar(self, name: str, source: Optional[str] = None) -> sp.Symbol:
        src = name if source is None else source
        sym = sp.Symbol(name)
        self._add_binding(sym, src)
        return sym

    def scalars(self, mapping: dict[str, str]) -> SimpleNamespace:
        ns = SimpleNamespace()
        for n, src in mapping.items():
            setattr(ns, n, self.scalar(n, src))
        return ns

    def vec(
        self,
        name: str,
        components: str,
        source: Union[str, list[str]],
    ) -> sp.Matrix:
        names = components.split()
        if isinstance(source, str):
            sources = [source.format(i=i) for i in range(len(names))]
        else:
            if len(source) != len(names):
                raise ValueError(
                    f"vec {name!r}: source list length {len(source)} != components {len(names)}"
                )
            sources = list(source)
        syms = [sp.Symbol(n) for n in names]
        for sym, src in zip(syms, sources):
            self._add_binding(sym, src)
        return sp.Matrix(syms)

    def _add_binding(self, sym: sp.Symbol, source: str) -> None:
        if sym in self.bindings:
            raise ValueError(f"duplicate binding for symbol: {sym}")
        self.bindings[sym] = Binding(sym, source, _deps_for(source, self._arg_names))

    # ── equations / exports / functions ─────────────────────────────
    def eq(self, name: str, expr, label: Optional[str] = None):
        """Record a named intermediate; return a placeholder so downstream
        equations (and the auto-generated doc comment) reference the name
        instead of the expanded expression.

        `name` must be a Python identifier — it's used as the placeholder
        symbol name. `label` is what's shown in the doc comment (defaults
        to `name`)."""

        if not name.isidentifier():
            raise ValueError(f"m.eq name must be a Python identifier, got {name!r}")
        ident = name
        self.equations.append(Equation(label if label is not None else name, expr))
        if isinstance(expr, sp.MatrixBase):
            rows, cols = expr.shape
            if cols == 1:
                syms = [sp.Symbol(f"{ident}_{i}") for i in range(rows)]
                placeholder = sp.Matrix(syms)
            elif rows == 1:
                syms = [sp.Symbol(f"{ident}_{j}") for j in range(cols)]
                placeholder = sp.Matrix([syms])
            else:
                syms2 = [[sp.Symbol(f"{ident}_{i}_{j}") for j in range(cols)]
                         for i in range(rows)]
                placeholder = sp.Matrix(syms2)
            for ph_sym, rhs in zip(list(placeholder), list(expr)):
                if ph_sym in self._intermediate_subs:
                    raise ValueError(f"duplicate placeholder symbol: {ph_sym}")
                self._intermediate_subs[ph_sym] = sp.sympify(rhs)
            return placeholder
        else:
            ph = sp.Symbol(ident)
            if ph in self._intermediate_subs:
                raise ValueError(f"duplicate placeholder symbol: {ph}")
            self._intermediate_subs[ph] = sp.sympify(expr)
            return ph

    def expand(self, expr):
        """Recursively substitute every placeholder until none remain."""

        if not self._intermediate_subs:
            return expr
        cur = sp.Matrix(expr) if isinstance(expr, sp.MatrixBase) else sp.sympify(expr)
        # Fixed-point iteration. Bounded by the longest substitution chain;
        # we cap at len(subs) + 1 to detect any unexpected cycle.
        for _ in range(len(self._intermediate_subs) + 1):
            new = cur.subs(self._intermediate_subs)
            if new == cur:
                return new
            cur = new
        raise RuntimeError("intermediate substitution did not converge — cycle?")

    def jac(self, f, v) -> sp.Matrix:
        """Jacobian of `f` w.r.t. `v`, with placeholders expanded first."""

        from .derive import jac as _jac
        return _jac(self.expand(f), v)

    def hessian(self, f, v) -> sp.Matrix:
        """Hessian of scalar `f` w.r.t. `v`, with placeholders expanded first."""

        from .derive import hessian as _hessian
        return _hessian(self.expand(f), v)

    def export(self, name: str, expr, rust_type: str) -> None:
        if name in self.exports:
            raise ValueError(f"duplicate export: {name}")
        # Expand placeholders so emit.py sees fully-concrete expressions.
        self.exports[name] = Export(name, self.expand(expr), rust_type)

    def function(self, name: str, returns: list[str]) -> None:
        for r in returns:
            if r not in self.exports:
                raise ValueError(f"function {name!r} returns unknown export: {r}")
        self.functions.append(Function(name, list(returns)))

    # ── derived helpers (used by emit.py) ───────────────────────────
    def args_for(self, exports: list[Export]) -> list[Arg]:
        """Args needed to compute the given exports, in declaration order."""

        needed: set[str] = set()
        for exp in exports:
            for sym in self._free_symbols(exp.expr):
                binding = self.bindings.get(sym)
                if binding is None:
                    raise ValueError(
                        f"export {exp.name!r} references unbound symbol {sym!r}"
                    )
                needed.update(binding.deps)
        return [a for a in self.args if a.name in needed]

    def bindings_for(self, exports: list[Export]) -> list[Binding]:
        """Bindings referenced by the given exports, in insertion order."""

        used: set[sp.Symbol] = set()
        for exp in exports:
            used.update(self._free_symbols(exp.expr))
        return [b for sym, b in self.bindings.items() if sym in used]

    @staticmethod
    def _free_symbols(expr) -> set[sp.Symbol]:
        if isinstance(expr, sp.MatrixBase):
            syms: set[sp.Symbol] = set()
            for e in expr:
                syms.update(sp.sympify(e).free_symbols)
            return syms
        return set(sp.sympify(expr).free_symbols)
