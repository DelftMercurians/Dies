"""Plain dataclasses mirroring `crates/dies-mpc/src/types.rs`.

These are the Python-side representations of the structs that show up in
binding source strings (`target.p.x`, `p.tau[FWD]`, ...) — designed so the
Rust source strings parse as valid Python expressions against an instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Body-axis indices, matching `crates/dies-mpc/src/types.rs::{FWD, STRAFE}`.
FWD = 0
STRAFE = 1


@dataclass
class Vec2:
    x: float
    y: float

    def __getitem__(self, i: int) -> float:
        return (self.x, self.y)[i]

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @staticmethod
    def zeros() -> "Vec2":
        return Vec2(0.0, 0.0)

    @staticmethod
    def from_array(a: np.ndarray) -> "Vec2":
        return Vec2(float(a[0]), float(a[1]))


@dataclass
class RobotParams:
    tau: np.ndarray         # shape (2,)
    accel_max: np.ndarray   # shape (2,)

    @staticmethod
    def default_hand_tuned() -> "RobotParams":
        return RobotParams(
            tau=np.array([0.08, 0.10]),
            accel_max=np.array([3500.0, 3500.0]),
        )


@dataclass
class CostWeights:
    position: float = 1.0e-3
    velocity: float = 0.0
    control: float = 5.0e-5
    control_smoothness: float = 1.0e-4


@dataclass
class MpcTarget:
    p: Vec2
    v: Vec2 = field(default_factory=Vec2.zeros)
    weights: CostWeights = field(default_factory=CostWeights)

    @staticmethod
    def goto(p: Vec2) -> "MpcTarget":
        return MpcTarget(p=p, v=Vec2.zeros(), weights=CostWeights())


@dataclass
class SolverConfig:
    horizon: int = 150
    dt: float = 0.06
    max_iters: int = 15
    cost_tol: float = 1.0e-3
    reg_init: float = 1.0e-6
    reg_min: float = 1.0e-8
    reg_max: float = 1.0e3
    reg_factor: float = 4.0


@dataclass
class SolveResult:
    states: np.ndarray       # shape (N+1, 4)
    controls: np.ndarray     # shape (N, 2)
    final_cost: float
    iters: int
    converged: bool
