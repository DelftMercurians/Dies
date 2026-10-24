# Simulator internals

`Simulation` and `SimulationBuilder` remain the public entry points. All existing
public types are re-exported from `src/lib.rs`; callers do not need to know the
internal module layout.

| Module | Responsibility |
| --- | --- |
| `simulation.rs` | Public API, snapshots, simulation clock, shared seeded RNG, and update order |
| `builder.rs` | Original default/seeded layouts and fluent setup API |
| `config.rs` / `realism.rs` | Configuration defaults, environment profiles, and noise sampling |
| `physics.rs` | Rapier world, field construction, ball creation, physical resets, and integration |
| `robot.rs` | Robot roster, delayed commands, movement limits, kicker/reflex state, and contacts |
| `ball.rs` | Rolling friction, dribbler ownership, contested balls, and snatch peeling |
| `referee/mod.rs` | Referee commands, match bookkeeping, events, and physical action requests |
| `referee/state.rs` / `transitions.rs` | State data, transition dispatch, and state-specific handlers |
| `referee/rules.rs` / `placement.rs` / `cards.rs` | Rule checks, restart placement, cards, and substitutions |
| `packets.rs` | Vision/geometry/referee encoding and scheduled output buffers |
| `time.rs` | Original `Timer` and `IntervalTrigger` behavior |

`PhysicsWorld`, `RobotFleet`, `BallInteraction`, `AutoReferee`, and `SimOutput` own
their corresponding state. Short-lived contexts borrow the state needed by an
update. The referee observes physical state and requests `RefereeAction`s; the
simulation applies them before the next update phase. Robot and ball updates
borrow the same physics world and RNG, preserving player order and random draws.

## Update order

`Simulation::step(dt)` preserves the original sequence:

1. Observe ball touches and double-touch violations.
2. Expire cards, process substitutions, and check robot allowances. Physical
   substitutions are applied before the next phase.
3. Advance the referee state machine and apply requested ball placement.
4. Generate detection if its interval is due, before physics advances.
5. Select delayed commands, update spatial queries, and apply rolling friction.
6. Update each robot in roster order. Feedback is generated before that robot's
   new command, movement, and ball-contact processing.
7. Resolve ball ownership, contested captures, and snatch peeling.
8. Integrate Rapier, clamp the ball, and advance simulation time.

The shared RNG, command overwrite order, duplicate builder entries, packet queue
order, and existing timing quirks are intentionally preserved. In particular,
referee timers and rolling friction still read the previous Rapier `dt`, and the
referee still temporarily takes its stored state during transition dispatch.
Known pre-existing issues and their current source locations are recorded in
[problems.md](problems.md), beside this README.

## Validation

The original unit tests live in `src/simulation/tests/`. Run them and check
formatting with:

```sh
cargo test --locked -p dies-simulator
cargo fmt -p dies-simulator -- --check
```
