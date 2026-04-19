# dies-mpc — Deferred work

Living list of things we know we want eventually but explicitly left out of
the v1 push. Keep it short and actionable; delete items as they land.

## Integration-layer

1. **Heading trajectory interpolation.** Currently we pass a constant current
   heading across all N+1 horizon steps. For motions that rotate during
   travel (e.g. dribbling + reorienting, pass reception), the body-frame
   anisotropic dynamics (τ, a_max per axis) will be wrong at later stages.
   Interpolate toward `PlayerControlInput.yaw` using the onboard heading
   controller's expected rate, then feed that as `heading_traj`.
2. **`care` / `aggressiveness` clamps.** The adapter forwards them unchanged;
   check real skills don't send degenerate values (care = 0 → obstacles
   ignored entirely, aggressiveness > 1 → negative smoothness weight).
3. **`avoid_ball_care` semantics.** Today it's folded into `safe_dist`
   directly via a `+ 50·(care + avoid_ball_care)` term. Consider moving the
   scaling onto the per-obstacle `weight_scale` so `MpcTarget.care` alone
   tunes intensity.
4. **`add_vel` composition under iLQR.** MTP adds
   `PlayerControlInput.velocity` on top of the position-controller output.
   iLQR currently replaces the velocity wholesale, losing that nudge
   channel. Add it back if any skill relies on it.

## Params + calibration

5. **Config-file loading of `RobotParams`.** Currently hard-coded
   `RobotParams::default_hand_tuned()`. Load `calibration.toml` at executor
   startup when available; fall back to defaults otherwise.
6. **Per-robot parameters.** One global set today. Real motors/wheels drift;
   the existing per-robot handicap pipeline is the right surface to mirror.
7. **Calibration CLI (`dies-calibrate`).** Excitation routine driver + LM
   fit dump. The math is already in `sysid::fit_params`; this is just the
   runner + serialisation.

## Solver + cost

8. **Perturbed-random-init branch in the multi-start.** Adds robustness for
   symmetric-obstacle deadlocks (see README §8). Cheap to add; wait for
   field evidence that it's needed.
9. **Hard constraints (augmented Lagrangian wrapper).** For ball capture
   where the relative-velocity match must be exact. Today we rely on a
   strong terminal penalty.
10. **Rayon parallelisation of per-robot solves.** Sequential today;
    ~6 ms of budget for the full team. Worth the ~5 lines of rayon if the
    tick budget ever tightens.
11. **Own-robot prediction upgrade.** Currently constant-velocity for every
    other robot (own + opp). Sequential solve ordering (solve highest-
    priority robot first, treat its resulting trajectory as a known
    time-varying obstacle for the next) is the standard fix.

## Dynamics

12. **Asymmetric accel/decel.** Real bots brake harder than they accelerate.
    Add two more params if sysid residuals demand it.
13. **Rotation–translation coupling.** Small lateral drift during rotation
    is ignored. Add once fast-spin manoeuvres show the error.
14. **Online sysid / parameter drift.** Background fit from logged play
    data, no dedicated excitation routine. Research-heavy; defer.

## Skill-side

15. **`MpcTarget` builders per skill family.** `PlayerControlInput` → basic
    `Position` target is all we do today. Add builders for
    `PositionAndVelocity` (pass reception, dribbler pickup) and
    `RelativeVelocity` (ball capture).
16. **Ball flight model + `InterceptAt`.** 3D ballistic model for chip-kick
    interception. The terminal mode is already in `dies-mpc`; we just need
    the ball-flight query and the skill wiring.

## Telemetry + UX

17. **Webui trajectory overlay.** Draw the iLQR trajectory on the field
    view, colour-coded by stage index.
18. **Solve telemetry panel.** Per-robot solve time / iters / final cost /
    active barrier list. Data is already in `SolveResult` + the
    `dies_core::debug_value` calls in `ilqr.rs`.
