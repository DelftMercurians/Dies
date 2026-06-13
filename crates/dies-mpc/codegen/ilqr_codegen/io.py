"""File I/O for codegen: rustfmt, write-or-check, the CLI entry point."""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from .docgen import HEADER, render_doc
from .emit import emit_function
from .model import Model


# Type tokens emitted in generated files. Anything found in arg types,
# binding sources, or export types is auto-added to the corresponding
# `use` line. Keep these in sync with `crates/dies-mpc/src/types.rs` and
# the nalgebra types referenced in formulations.
_CRATE_TYPES = {
    "State", "Control", "StateJac", "ControlJac",
    "RobotParams", "MpcTarget",
    "FWD", "STRAFE",
}
_NALGEBRA_TYPES = {
    "Vector2", "Vector3", "Vector4", "Vector5",
    "Matrix2", "Matrix3", "Matrix4", "Matrix5",
    "Matrix2x4", "Matrix4x2", "Matrix3x5", "Matrix5x3",
}

_IDENT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")

# crates/dies-mpc/codegen/ilqr_codegen/io.py → up 2 to codegen/ → up 1 to dies-mpc/
ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "src" / "generated"


def _collect_uses(model: Model) -> tuple[set[str], set[str]]:
    crate: set[str] = set()
    nalg: set[str] = set()

    def scan(text: str) -> None:
        for tok in _IDENT.findall(text):
            if tok in _CRATE_TYPES:
                crate.add(tok)
            elif tok in _NALGEBRA_TYPES:
                nalg.add(tok)

    for a in model.args:
        scan(a.rust_type)
    for b in model.bindings.values():
        scan(b.rust_source)
    for e in model.exports.values():
        scan(e.rust_type)
    return crate, nalg


def _render_uses(model: Model) -> str:
    crate, nalg = _collect_uses(model)
    lines: list[str] = []
    if crate:
        lines.append(f"use crate::types::{{{', '.join(sorted(crate))}}};")
    if nalg:
        lines.append(f"use nalgebra::{{{', '.join(sorted(nalg))}}};")
    return "\n".join(lines) + ("\n" if lines else "")


def _render_model(model: Model) -> str:
    parts: list[str] = [render_doc(model)]
    uses = _render_uses(model)
    if uses:
        parts.append("\n" + uses)
    for fn in model.functions:
        parts.append("\n" + emit_function(model, fn))
    return "".join(parts)


def _render_mod(models: Iterable[Model]) -> str:
    body = "".join(f"pub(crate) mod {m.name};\n" for m in sorted(models, key=lambda m: m.name))
    return HEADER + body


def _rustfmt(content: str) -> str:
    with tempfile.NamedTemporaryFile("w+", suffix=".rs") as tmp:
        tmp.write(content)
        tmp.flush()
        result = subprocess.run(
            ["rustfmt", "--edition", "2021", tmp.name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        tmp.seek(0)
        return tmp.read()


def _write_or_check(path: Path, content: str, check: bool) -> bool:
    old = path.read_text() if path.exists() else ""
    if old == content:
        return True
    if check:
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=str(path),
                tofile=f"{path} (generated)",
            )
        )
        sys.stdout.write(diff)
        return False
    path.write_text(content)
    return True


def run(models: list[Model], argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args(argv)

    GENERATED.mkdir(parents=True, exist_ok=True)

    files: dict[Path, str] = {GENERATED / "mod.rs": _render_mod(models)}
    for m in models:
        files[GENERATED / f"{m.name}.rs"] = _rustfmt(_render_model(m))

    ok = True
    for path, content in files.items():
        ok = _write_or_check(path, content, args.check) and ok
    if not ok:
        print("generated files are stale; run the generator without --check")
        return 1
    return 0
