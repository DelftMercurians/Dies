# dies-mpc

Pure-Rust model-predictive controller for translational motion of the Delft
Mercurians SSL robots. No Python, no external optimiser, no automatic
differentiation — everything is hand-rolled on top of small dense `nalgebra`
matrices so it's fully auditable and fast.

This document explains what the crate does, the math behind it, and — most
importantly — **how to extend the cost function** when a new skill needs a
new behaviour.

---

## 1. Overview

At every 60 Hz control tick, for each of our robots:

1. The caller packages up the robot's current state, a small description of
   what the skill wants (`MpcTarget`), and a snapshot of the world
   (obstacles, field geometry) into a call to `solve()`.
2. `solve()` runs iLQR to optimise a sequence of 10 velocity commands over
   a 600 ms horizon.
3. Only the first command is applied to the robot. The remainder is stored
   and used as a warm-start for the next tick.

Three things are decoupled from the core math and live at the boundary:

- **Heading** — the onboard IMU/magnetometer controller owns heading. MPC
  receives `heading_traj: &[f64]` as a per-step exogenous input.
- **Obstacles** — the caller predicts them forward with constant velocity
  and passes them in as `PredictedObstacle`s.
- **Ball flight** — not modelled in this crate. For `InterceptAt`-style
  use cases, the caller queries a ball-flight model externally and fills in
  `TerminalMode::RelativeVelocity { target_p, target_v }` directly.

### 1.1 Dataflow

```text
 skill (strategy)       ┌─────────────┐
        │               │  MpcTarget  │
        ▼               │  + weights  │
┌───────────────┐       │  + care     │      ┌───────────────┐
│  RobotState   │──────▶│             │─────▶│               │
└───────────────┘       │             │      │  solve()      │
┌───────────────┐       │             │      │  → iLQR       │─▶ controls[0]
│ WorldSnapshot │──────▶│             │      │  → traj       │
└───────────────┘       │             │      │               │
┌───────────────┐       │             │      └───────────────┘
│ heading_traj  │──────▶│             │
└───────────────┘       └─────────────┘
                                                             │
┌───────────────┐                                            │
│ RobotParams   │──────────────────────────────────────────▶ │
└───────────────┘   (sysid-identified dynamics parameters,
                     shared across all robots)
```

### 1.2 What's in each file

| File | Purpose |
|---|---|
| `types.rs` | All public data types. No behaviour. |
| `dynamics.rs` | 4-state / 2-input dynamics `f(x, u, heading)` + analytic Jacobians. |
| `barrier.rs` | Smooth soft-barrier function + signed distances for Circle / Rectangle / Line. |
| `cost.rs` | Stage and terminal cost, residual-by-residual with Gauss-Newton Hessian. |
| `solver.rs` | iLQR backward/forward passes, regularisation, multi-start wrapper. |
| `sysid.rs` | Levenberg-Marquardt fit of the 7 dynamics parameters. |
| `lib.rs` | Thin re-export shell. |
| `examples/goto_target.rs` | Sanity-check binary that dumps a trajectory as TSV. |

---

## 2. Problem formulation

### 2.1 State and control

```text
x = [p_x, p_y, v_x, v_y]    ∈ ℝ⁴    (global frame, mm and mm/s)
u = [v_cmd_x, v_cmd_y]      ∈ ℝ²    (global frame, mm/s)
```

Heading θ is **not** part of the state. It's supplied per stage as a scalar
input: `heading_traj[k]` is the robot's heading at stage `k`. The dynamics
rotate the global-frame velocity/command into the body frame internally so
that body-axis parameters (forward vs. strafe) apply correctly.

### 2.2 Dynamics model

In the body frame, per axis `i ∈ {FWD, STRAFE}`:

```text
v̇_b[i] = a_max[i] · tanh( (v_cmd_b[i] − v_b[i]) / (τ[i] · a_max[i]) )
         − stiction[i] · tanh( v_b[i] / v_ε )
```

Terms:

- **First term** — smooth-saturated velocity-lag. Near zero error behaves
  like a first-order lag `(v_cmd − v)/τ`. For large errors it saturates at
  `±a_max`. The `tanh` makes this C∞, so iLQR gets clean gradients.
- **Second term** — smooth stiction. Opposes current velocity, saturates
  at `stiction` magnitude, with a C∞ transition at width `v_ε`. This term
  is what the MTP controller lacks and why it can't make accurate small
  adjustments — real robots can't budge below some command threshold
  because of static friction.

The **7 identifiable parameters**:

```text
τ_fwd,      τ_strafe         ∈ (0, ∞)   velocity-lag constants [s]
a_max_fwd,  a_max_strafe     ∈ (0, ∞)   acceleration saturations [mm/s²]
stiction_fwd, stiction_strafe∈ [0, ∞)   stiction magnitudes [mm/s²]
v_ε                          ∈ (0, ∞)   stiction smoothing scale [mm/s]
```

Global (shared across all robots, for now). Mass is folded into `stiction`
because only the μ/m ratio is identifiable from velocity observations.

Integration is forward Euler with `dt = 60 ms`:

```text
x_{k+1} = x_k + dt · f(x_k, u_k, θ_k)
```

At τ ≈ 80–100 ms this is marginally stable but good enough for a 600 ms
MPC predictor. We trade a bit of integration accuracy for clean analytic
Jacobians (`dynamics.rs::step_with_jacobians`).

### 2.3 The cost function

The optimal-control problem is:

```text
min  J = Σ_{k=0}^{N-1} ℓ(x_k, u_k, u_{k-1}, k)  +  ℓ_N(x_N)
 u

s.t. x_{k+1} = f(x_k, u_k, θ_k)
     x_0     = current state  (fixed)
```

Every cost term is expressed as one or more **scalar residuals** `r_i(x, u)`
with contribution `½ r_i²`. This is central to how the code is organised
(see §4).

Stage cost `ℓ(x_k, u_k, u_{k-1}, k)` is the sum of:

| Term | Residual | Weight |
|---|---|---|
| Position tracking | `p_k − p_ref(t_k)` (2-D) | `weights.position` |
| Velocity tracking | `v_k − v_ref(t_k)` (2-D, only if `v_ref` present) | `weights.velocity` |
| Control smoothness | `u_k − u_{k-1}` (2-D) | `weights.control_smoothness · (1 − aggressiveness)` |
| Obstacle barrier | `b(d(p_k, obs))` per obstacle | `weights.obstacle · care · obs.weight_scale` |
| Field boundary | 4× `b(d_plane)` half-plane | `weights.field_boundary · care` |

Terminal cost `ℓ_N(x_N)` dispatches on `TerminalMode`:

| Mode | Residual |
|---|---|
| `Position { p }` | `p_N − p` (2-D) |
| `PositionAndVelocity { p, v }` | `[p_N − p; v_N − v]` (4-D) |
| `RelativeVelocity { target_p, target_v }` | `[p_N − target_p; v_N − target_v]` (4-D) |

All scaled by `weights.terminal`.

---

## 3. The iLQR algorithm

iLQR (iterative Linear-Quadratic Regulator) is basically Newton's method
on the trajectory `(x_0..x_N, u_0..u_{N-1})` under the dynamics constraint.
Each iteration:

1. **Backward pass** — propagate a quadratic model of the cost-to-go
   backwards in time, producing a **local feedback law**
   `δu_k = α · k_k + K_k · δx_k`.
2. **Forward pass** — roll out that feedback law from the current initial
   state, with a line search over α ∈ {1, 0.5, 0.25, ...}, to get a new
   trajectory.

### 3.1 Backward pass (in `solver.rs::backward_pass`)

Starting at the terminal stage:

```text
V_x  = ℓ_N,x   (gradient of terminal cost)
V_xx = ℓ_N,xx  (Gauss-Newton Hessian)
```

Walking back through stages `k = N-1 .. 0`, define the stage Q-function:

```text
Q_x  = ℓ_x  + fₓᵀ · V_x
Q_u  = ℓ_u  + fᵤᵀ · V_x
Q_xx = ℓ_xx + fₓᵀ · V_xx · fₓ
Q_uu = ℓ_uu + fᵤᵀ · V_xx · fᵤ + λ · I      ← Levenberg regularisation
Q_ux = ℓ_ux + fᵤᵀ · V_xx · fₓ
```

Here `fₓ, fᵤ` are the dynamics Jacobians from `step_with_jacobians` and
`ℓ_x, ℓ_u, ℓ_xx, ℓ_uu, ℓ_ux` are the stage-cost derivatives from
`stage_derivs`.

Solve the unconstrained QP `min_δu ½ δuᵀ Q_uu δu + Q_uᵀ δu + δxᵀ Q_uxᵀ δu`
in closed form:

```text
k_k  = − Q_uu⁻¹ · Q_u          feedforward correction
K_k  = − Q_uu⁻¹ · Q_ux         feedback gain
```

Then the Q-function compresses back into an updated value function:

```text
V_x  = Q_x  + Q_uxᵀ · k_k
V_xx = Q_xx + Q_uxᵀ · K_k
```

If `Q_uu` is not positive-definite (determinant ≤ 0 or inverse fails), we
bump λ and restart the backward pass. See `solver.rs:100–118`.

### 3.2 Forward pass (in `solver.rs::forward_pass`)

Given `(k, K)` and the previous trajectory `(x, u)`, roll out at learning
rate α:

```text
δx_0 = 0
u_k_new = u_k + α · k_k + K_k · δx_k
x_{k+1}_new = f(x_k_new, u_k_new, θ_k)
δx_{k+1} = x_{k+1}_new − x_{k+1}
```

We try α ∈ {1, 0.5, 0.25, 0.125, 0.0625} in order. The first α that
sufficiently decreases cost is accepted (see `solver.rs:220–248`).
"Sufficiently" is an Armijo-like test:

```text
expected = α · ΔV₁ + α² · ΔV₂     (accumulated in the backward pass)
actual   = J(new) − J(old)
accept   if  actual < 0.1 · expected    (when expected is negative enough)
```

If no α works, we bump λ and try the backward pass again with heavier
regularisation. If λ saturates at `cfg.reg_max`, we bail out and return
whatever we have. The outer loop runs up to `cfg.max_iters = 15` times.

### 3.3 Multi-start wrapper (in `solver.rs::solve`)

iLQR is a local method. On obstacle-dense problems it can get stuck in
local minima. As a cheap safeguard we run iLQR from **two** initialisations
and pick the lower-cost result:

1. **Warm start** — the previous solve's controls, shifted by one stage
   (drop first, pad last). Essential for smooth, low-iteration tracking
   at 60 Hz.
2. **Straight line** — constant velocity aimed at the terminal target,
   clipped to a sane `3500 mm/s` cap.

If field testing reveals deadlock scenarios — perfectly symmetric
obstacles, for instance — a third *random-perturbation* init is the
natural thing to add here.

---

## 4. Cost function architecture

This is the section to read before touching `cost.rs`. Everything in the
cost is built on one pattern.

### 4.1 The residual pattern

Every cost contribution takes the form:

```text
cost term = ½ · r_i(x, u)²                for some scalar r_i
```

From a residual `r` with gradients `∂r/∂x = g_x`, `∂r/∂u = g_u` we get:

```text
∂cost/∂x       = r · g_x
∂cost/∂u       = r · g_u
∂²cost/∂x²     ≈ g_x · g_xᵀ         (outer product; Gauss-Newton)
∂²cost/∂u²     ≈ g_u · g_uᵀ
∂²cost/∂u∂x    ≈ g_u · g_xᵀ
```

The Gauss-Newton approximation drops the `r · ∂²r/∂…²` term. For least-
squares problems near a local minimum (`r` small), this is almost the true
Hessian. Away from the minimum, it's still positive-semidefinite, which
keeps the backward-pass QP well-posed — a property the exact Hessian
wouldn't have.

The helper `add_scalar_residual` (`cost.rs:112`) accumulates all five
outputs into a `StageDerivs` struct. It's called once per scalar residual.
Multiple residuals from different cost terms just stack.

**The implication for extension**: adding a new cost term means writing its
residual and its two gradients (w.r.t. `x` and `u`). You never write a
Hessian by hand — the Gauss-Newton outer-product handles that.

### 4.2 Stage vs terminal

- **Stage cost** acts at every `k ∈ [0, N-1]` with access to the state,
  control, previous control, and stage index. Added to `StageDerivs`.
- **Terminal cost** acts only at `k = N` with access to just the state.
  Added to `TerminalDerivs` (no `lu`, `luu`, `lux`).

Use stage cost for anything you want enforced *throughout* the trajectory
(e.g. "stay away from this obstacle"). Use terminal cost for anything
that matters specifically *at the end of the horizon* (e.g. "be at this
point moving at this velocity").

### 4.3 Reference trajectory

`ReferenceTrajectory` is the skill's time-varying reference:

- `StaticPoint(p)` — constant target, no velocity reference.
- `Timed(Vec<TimedRef>)` — piecewise-linear interpolation. Each sample
  carries a time and a position; velocity is optional.

At stage `k`, the cost evaluates the reference at time `t_k = k · dt`. For
moving targets (e.g. tracking an intercept point as the ball rolls), the
caller fills in `Timed` samples at every horizon step.

### 4.4 Obstacle barriers

`PredictedObstacle` carries a shape, a constant-velocity drift, two
distance thresholds, and a per-obstacle `weight_scale`:

```text
b(d) = max(0, (no_cost − d) / (no_cost − safe))
```

- `d ≥ no_cost`: `b = 0`, gradient zero. No cost contribution.
- `safe ≤ d < no_cost`: linear ramp from 0 to 1.
- `d < safe`: keeps growing linearly — penetration cost grows quadratically
  in `cost` (because cost = ½b²).

At each stage, the obstacle's position is extrapolated with
`obs.velocity · t_k`. The gradient of `b(d(p))` w.r.t. position is
`db/dd · ∂d/∂p`, where `∂d/∂p` comes from `signed_distance`.

Three shapes are supported (`ObstacleShape::{Circle, Rectangle, Line}`).
Goal-area avoidance is not a first-class concept — goal areas should be
added to `world.obstacles` as Rectangles with a larger `weight_scale`
(e.g. 5–10×).

### 4.5 Field boundary

Four soft walls, one per field edge, handled as half-plane residuals:

```text
d_left(p)   = p_x + half_length        (positive inside)
d_right(p)  = half_length − p_x
d_bottom(p) = p_y + half_width
d_top(p)    = half_width − p_y
```

Each is passed through the same barrier function with `safe = 0`,
`no_cost = margin` (default 100 mm). Scales with `weights.field_boundary`
and `care`.

---

## 5. Extending the cost function

This is the runbook for adding new MPC behaviour from the skill side.

### 5.1 Adding a new *weight* / *parameter* to an existing term

If all you want is to tune an existing cost term differently per skill,
you don't touch `cost.rs` at all. You have three levers:

1. `CostWeights` fields — global scale on each term class.
2. `care: f64` on `MpcTarget` — scales *all* barrier terms (obstacle +
   field-boundary) together. Semantically "how much this skill cares
   about safety". Goal-keeping: high care. Last-ditch interception:
   low care.
3. `aggressiveness: f64 ∈ [0, 1]` on `MpcTarget` — scales the
   control-smoothness penalty by `(1 − aggressiveness)`. 0 = full
   smoothing, 1 = no smoothness cost ⇒ robot will gun it.
4. `obs.weight_scale: f64` on each `PredictedObstacle` — per-obstacle
   multiplier on top of `weights.obstacle · care`.

In 95 % of cases this is enough. **Always check this first before adding
a new cost term.**

### 5.2 Adding a new scalar residual to the stage cost

Example: a "stay behind the ball" term — penalise the component of the
robot's position that lies *ahead of* the ball along the ball's velocity
direction.

**Step 1**: add a skill-facing knob so the term is *opt-in* and
parameterisable. For a simple scalar weight:

```rust
// types.rs, in CostWeights:
pub struct CostWeights {
    ...
    pub stay_behind_ball: f64,  // new, default 0.0
}
```

For something more structural (e.g. the ball state itself and a direction),
add a dedicated field on `MpcTarget`:

```rust
// types.rs:
#[derive(Clone, Debug)]
pub struct StayBehindBall {
    pub ball_pos: Vec2,
    pub ball_vel: Vec2,
}

pub struct MpcTarget {
    ...
    pub stay_behind_ball: Option<StayBehindBall>,
}
```

Make it `Option<…>` unless every skill will supply it. Default to `None`
in the skill-side builder.

**Step 2**: derive the residual and its gradients.

The ball-relative position is `p − ball_pos`. We want to penalise the
component along the ball's unit velocity direction `n = ball_vel / ‖ball_vel‖`
when it's positive (robot ahead of ball). Residual:

```text
r = sqrt(w) · max(0, n · (p − ball_pos))
```

Gradient:

```text
∂r/∂p = sqrt(w) · n   when n·(p − ball_pos) > 0
       = 0            otherwise
∂r/∂u = 0
```

(The `max(0, ·)` kink is fine — the derivative is piecewise constant and
still valid for Gauss-Newton.)

**Step 3**: add the residual to `stage_derivs` in `cost.rs`.

```rust
// In stage_derivs(), after the existing cost terms:
if let Some(sbb) = &target.stay_behind_ball {
    let w = target.weights.stay_behind_ball.max(0.0);
    if w > 0.0 {
        let n_mag = sbb.ball_vel.norm();
        if n_mag > 1.0e-6 {
            let n = sbb.ball_vel / n_mag;
            let ahead = n.dot(&(pos - sbb.ball_pos));
            if ahead > 0.0 {
                let sqrt_w = w.sqrt();
                let r  = sqrt_w * ahead;
                let gx = sqrt_w * Vector4::new(n.x, n.y, 0.0, 0.0);
                let gu = Vector2::zeros();
                add_scalar_residual(&mut d, r, gx, gu);
            }
        }
    }
}
```

**Step 4**: mirror it into `stage_cost_scalar` (the no-derivatives variant
used for rollouts and finite-difference tests):

```rust
if let Some(sbb) = &target.stay_behind_ball {
    let w = target.weights.stay_behind_ball.max(0.0);
    if w > 0.0 {
        let n_mag = sbb.ball_vel.norm();
        if n_mag > 1.0e-6 {
            let n = sbb.ball_vel / n_mag;
            let ahead = n.dot(&(pos - sbb.ball_pos)).max(0.0);
            c += 0.5 * w * ahead * ahead;
        }
    }
}
```

**Step 5**: add a finite-difference test. The existing
`stage_gradient_matches_finite_diff_*` tests are the template:

```rust
#[test]
fn stay_behind_ball_gradient_matches_fd() {
    let mut target = target_goto(Vec2::new(0.0, 0.0));
    target.stay_behind_ball = Some(StayBehindBall {
        ball_pos: Vec2::new(500.0, 0.0),
        ball_vel: Vec2::new(1000.0, 0.0),
    });
    target.weights.stay_behind_ball = 1.0e-3;
    // Robot ahead of ball along +x
    let x = Vector4::new(800.0, 100.0, 0.0, 0.0);
    let u = Vector2::zeros();
    let u_prev = Vector2::zeros();
    let world = world_no_obstacles();
    let d = stage_derivs(0, &x, &u, &u_prev, &target, &world, 0.06);
    let (fd_gx, fd_gu) = finite_diff_gradient(0, &x, &u, &u_prev, &target, &world, 0.06);
    for i in 0..4 { assert_abs_diff_eq!(d.lx[i], fd_gx[i], epsilon = 1.0e-3); }
    for i in 0..2 { assert_abs_diff_eq!(d.lu[i], fd_gu[i], epsilon = 1.0e-3); }
}
```

This catches 90 % of mistakes — a wrong sign, a missing factor, a mis-
indexed gradient entry — in the 30 seconds it takes to run.

### 5.3 Adding a new `TerminalMode` variant

Example: a "pass reception" terminal — robot should arrive at a point,
moving in a specified direction, at a specified speed (not the full
target velocity, just magnitude along direction).

**Step 1**: extend the enum.

```rust
// types.rs:
pub enum TerminalMode {
    Position { p: Vec2 },
    PositionAndVelocity { p: Vec2, v: Vec2 },
    RelativeVelocity { target_p: Vec2, target_v: Vec2 },
    ReceivePass { p: Vec2, direction: Vec2, speed: f64 },  // new
}
```

**Step 2**: derive the residual.

```text
r_p = sqrt(w) · (p_N − p)                            2-D position residual
r_v = sqrt(w) · (v_N − speed · direction)            2-D velocity residual
```

(This is actually the same as `PositionAndVelocity` with
`v = speed · direction` — if you only need that, use the existing variant
and skip this section. The example is here to illustrate the pattern for
genuinely new shapes.)

**Step 3**: add branches in both `terminal_derivs` and
`terminal_cost_scalar` in `cost.rs`. Use `add` (the local closure in
`terminal_derivs`) the same way `add_scalar_residual` is used in stage
cost — four `add(...)` calls for the four residual components.

**Step 4**: handle the new variant in `solver.rs::terminal_target_pos`
(used by the straight-line initialisation).

```rust
fn terminal_target_pos(m: &TerminalMode) -> Vec2 {
    match m {
        TerminalMode::Position { p } => *p,
        TerminalMode::PositionAndVelocity { p, .. } => *p,
        TerminalMode::RelativeVelocity { target_p, .. } => *target_p,
        TerminalMode::ReceivePass { p, .. } => *p,
    }
}
```

**Step 5**: test that the residual is zero when the terminal condition is
exactly satisfied, mirroring `terminal_relative_velocity_zero_at_match`.
Also add a finite-difference gradient test.

### 5.4 Adding a new obstacle class

The three shapes in `ObstacleShape` cover the game fully today. If you
need something more (convex polygon, capsule, …):

1. Add a variant to `ObstacleShape` in `types.rs`.
2. Add a `match` arm in `barrier.rs::signed_distance` returning
   `(d, ∂d/∂p)`. Verify with a finite-difference test in `barrier.rs::tests`
   — every existing shape has one.
3. Add a `match` arm in `cost.rs::predict_shape` that translates the shape
   by `vel · t`.

Nothing in `solver.rs` needs to change.

### 5.5 Tuning weights — scale intuition

Units matter. Positions are in **mm** so squared position errors are
enormous numerically:

- Error of 1 m (1000 mm) → squared error = 10⁶.
- Velocity of 1 m/s (1000 mm/s) → squared error = 10⁶.
- A barrier residual `b ∈ [0, ~2]` → squared ~ [0, 4].

That's why the default weights have `position: 1e-3` and `obstacle: 200`.
The default balance is intentional — if you find yourself bumping
`obstacle` into the thousands, you probably want to dial up `care` on a
per-skill basis instead. If the cost ratios between terms change by
orders of magnitude you may get numerical trouble in the backward pass
(ill-conditioned `Q_uu`).

Rule of thumb when adding a new cost term: ballpark the **typical squared
residual** at a reasonable operating point, pick a weight that makes its
contribution comparable (within 10×) to the position tracking cost. You
can always turn it up via a per-skill parameter.

---

## 6. System identification (`fit_params`)

### 6.1 The residual

For each consecutive sample pair `(t_i, cmd_i, heading_i, state_i)` and
`(t_{i+1}, …, state_{i+1})`, forward-integrate for `Δt = t_{i+1} − t_i`:

```text
v̂_{i+1} = v_i + Δt · a_global(state_i, cmd_i, heading_i; θ)
r_i = v_observed_{i+1} − v̂_{i+1}          2-D
```

We deliberately do **not** fit position residuals. Position drift is just
integrated velocity error and adds no independent information while biasing
the fit toward low-frequency components. Velocity-only is the right
residual for a dynamics model whose state-derivative only touches velocity.

### 6.2 Log-parameterisation

All 7 parameters are strictly positive. To prevent Levenberg-Marquardt from
stepping into negative territory we optimise over
`θ = [log τ_fwd, log τ_strafe, log a_max_fwd, …, log v_ε]`. The residual
and its parameter Jacobian are computed in raw-parameter space, then the
column scaling `∂p/∂θ = p` is applied — a single column-wise multiply in
`log_jacobian`.

### 6.3 The fit

Standard Levenberg-Marquardt with diagonal scaling:

```text
(JᵀJ + λ · diag(JᵀJ)) · Δθ = −Jᵀr

if cost decreases:  θ ← θ + Δθ,  λ ← λ/2
else:               λ ← 2λ
```

On clean synthetic data it recovers the true parameters within 5 %. With
10 mm/s velocity noise it gets within 15–30 %.

### 6.4 What excitation is needed

For the fit to be well-posed the commanded sequence must excite all 7
parameters identifiably:

- **τ**: needs near-linear-regime commands where `|err| < τ · a_max` so
  the `tanh` argument stays small (otherwise the response saturates and
  `a_max` dominates). Low-amplitude steps and ramps.
- **a_max**: needs saturating commands — large steps far above the
  linear-regime threshold. Hard step inputs at ≥ 2000 mm/s.
- **stiction**: needs *low-velocity* data. Commands applied while the
  robot is near rest. Slow ramps from zero are ideal.
- **v_ε**: needs data at *graded* low velocities — various speeds below
  a few hundred mm/s — so the `tanh` transition is resolved.

The test harness in `sysid.rs::tests::rich_command_sequence` is a decent
template to adapt for an on-robot calibration routine.

---

## 7. Integration notes

### 7.1 Coordinate frames

- MPC input (`RobotState.pos/vel`, `MpcTarget.reference`, obstacles, field
  bounds) is all in the **global frame** — the team-relative frame the
  strategy layer already works in.
- Heading is in radians, in the global frame (robot's forward axis pointing
  in that direction).
- Controls returned by the solver are in the **global frame**. The
  basestation-client wrapper is responsible for rotating them into the
  robot's body frame before transmission (matching the existing
  `PlayerCmdUntransformer` convention).

### 7.2 Units

- Distances: **mm**. (Don't mix with metres — the cost weights are tuned
  for mm.)
- Velocities: **mm/s**.
- Accelerations: **mm/s²**.
- Time: **s**.

### 7.3 Public API

```rust
pub fn solve(
    state: RobotState,
    heading_traj: &[f64],
    target: &MpcTarget,
    params: &RobotParams,
    world: &WorldSnapshot,
    warm_start: Option<&Trajectory>,
    cfg: &SolverConfig,
) -> SolveResult;

pub fn fit_params(
    samples: &[Sample],
    init: RobotParams,
    opts: FitOptions,
) -> FitResult;
```

Both are pure functions — no internal state, no background threads. The
per-robot `Trajectory` that the caller keeps for warm-starting is the
only thing that persists across ticks.

### 7.4 What's deliberately *not* in this crate

- Replacing the `MPCController` that currently fronts the JAX bridge in
  `dies-executor`. That's separate plumbing.
- Calibration CLI with the excitation routine. `fit_params` is just the
  math — the runner is a separate binary.
- Skill-side construction of `MpcTarget`. Skills will grow their own
  builders in `dies-strategy-api`.
- Webui telemetry for solve time / iters / cost breakdown. Hook into
  `SolveResult` fields when wiring integration.
- Ball flight model + `InterceptAt` terminal mode. The caller is expected
  to query ball flight externally and supply results as
  `TerminalMode::RelativeVelocity { target_p, target_v }`.

---

## 8. A note on iLQR's failure modes

Keep these in mind when debugging unexpected behaviour:

1. **Symmetric obstacles produce zero y-gradient.** A circular obstacle
   exactly on the robot-to-target line has no side to prefer, so iLQR
   doesn't find the detour. Real obstacles are rarely exactly symmetric,
   but if this bites in practice, add a third perturbed-random init to
   the multi-start wrapper in `solve()`.
2. **Cost weight imbalance blows up the backward pass.** If one term's
   Hessian dominates by orders of magnitude, `Q_uu` gets ill-conditioned
   and you'll see `Q_uu.try_inverse() = None` forcing many regularisation
   bumps. Keep term magnitudes within ~100× of each other.
3. **Warm-start divergence after skill switch.** If the skill changes and
   the previous trajectory is now irrelevant, the warm start is actively
   harmful. The straight-line init rescues this because we pick the
   lower-cost result — but be aware the first post-switch tick may run
   hotter than steady state.
4. **Infeasible problems don't error.** Targets inside obstacle bodies,
   targets outside the field, etc., produce a best-effort trajectory with
   `converged: false`. Check that flag at the integration layer before
   trusting the output for control.
