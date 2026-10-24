# Existing simulator problems

These issues predate the simulator modularization and are intentionally preserved by that refactor. This file records observations only; it does not authorize behavior changes. It now lives at `crates/dies-simulator/problems.md`. Each issue includes **current locations** in the modular source tree, with links relative to this crate directory, verified on 2026-10-19 unless the entry specifies a later date. Original baseline references to the former `src/lib.rs` and `utils.rs` are retained for comparison; they are historical locations, not current source links.

## Confirmed behavior defects and inconsistencies

- **Non-reflexive game-state equality.** Original baseline `lib.rs:503-530`: `PartialEq` omits both `StopAndForceStart` and `Null`. Either value compares unequal to itself, unlike the other state variants. Equality also deliberately ignores payloads for the variants it recognizes; that separate behavior is preserved.
  Current location: [src/referee/state.rs:118](src/referee/state.rs#L118), `impl PartialEq for SimulationGameState`.

- **Smart-kick counter wrap is not recognized.** Original baseline `lib.rs:2677-2679`: a command fires only when its `u8` counter is greater than the stored counter. After accepting 255, wrapping to 0 cannot trigger another smart kick, because no subsequent `u8` value exceeds 255.
  Current location: [src/robot.rs:335](src/robot.rs#L335), the global-command kick-counter comparison in `RobotFleet::update`.

- **Penalty restart command disagrees with internal state.** Original baseline `lib.rs:1963-1997`: both penalty timeout and wrong-direction handling broadcast `DIRECT_FREE` for the attacking team but transition to `FreeKick` for the defending team. Consumers of referee packets and the internal referee therefore receive different restart ownership.
  Current locations: [src/referee/transitions.rs:430](src/referee/transitions.rs#L430) (timeout) and [src/referee/transitions.rs:448](src/referee/transitions.rs#L448) (wrong direction), in `AutoReferee::update_penalty`.

- **Ball-placement messages can omit the target.** Original baseline `lib.rs:1283-1291`, `1399-1403`, `1616`, `2007-2020`: the initial command is emitted before the placement state is installed. Subsequent placement updates emit while `mem::take` has temporarily replaced the state with default `StartGame`. Since packet construction reads the stored state, these messages omit `designated_position` when entering placement from a non-placement state and on its regular updates. Replacing an existing placement can initially advertise its previous target.
  Current locations: [src/referee/mod.rs:262](src/referee/mod.rs#L262) (command entry), [src/referee/transitions.rs:9](src/referee/transitions.rs#L9) (`mem::take`), [src/referee/transitions.rs:487](src/referee/transitions.rs#L487) (placement rebroadcast), and [src/packets.rs:200](src/packets.rs#L200) (target encoding).

- **Timer and friction calculations use the previous step duration.** Original baseline `lib.rs:1633`, `1836`, `1912`, `2162`, `2596`, `3055`: referee timers, ball-out prediction, and rolling friction read `integration_parameters.dt` before the current `step(dt)` assigns it. The first call uses Rapier's initial duration; changing `dt` introduces a one-step lag in those calculations.
  Current locations: [src/referee/transitions.rs:27](src/referee/transitions.rs#L27) (timer example), [src/referee/rules.rs:138](src/referee/rules.rs#L138) (ball-out prediction), [src/ball.rs:343](src/ball.rs#L343) (friction), and [src/physics.rs:153](src/physics.rs#L153) (current `dt` assignment). Their phase order is in [src/simulation.rs:322](src/simulation.rs#L322), `Simulation::step`.

- **Local commands retain global heading and reflex state.** Original baseline `lib.rs:2645-2656`, `2672-2689`, `2740-2781`: the local-command branch updates angular velocity but does not clear a preceding global heading setpoint or reflex arm. A retained heading continues selecting heading control instead of the local angular-velocity command, and a retained reflex arm can still fire.
  Current locations: [src/robot.rs:309](src/robot.rs#L309) (local command), [src/robot.rs:330](src/robot.rs#L330) (global heading/reflex state), and [src/robot.rs:398](src/robot.rs#L398) (heading-control selection).

- **Heading control retains the body's previous angular velocity.** Verified on 2026-10-22 in both the original baseline (`lib.rs:2740-2781`) and the refactored implementation. The angular-velocity branch calls `set_angvel`, but the finite-heading branch only calls `set_rotation`, leaving the body's angular velocity unchanged. Rapier subsequently integrates that retained velocity after the heading correction. If the heading correction is too slow to counter the retained rotation, the robot can continue spinning while its position remains stationary; a faster correction can hide the rotation while leaving a heading offset. A public-API reproduction used default 30 ms command delay, 40 Hz command updates, 5 ms physics steps, zero commanded translation, and an initial 2 rad/s spin for one second, followed by a fixed zero-radian heading with a 0.5 rad/s turning limit. Both implementations then rotated approximately 1.56 turns during the observation window from 2 to 7 seconds, with zero positional displacement. An explicit zero-angular-velocity command stopped rotation; a fresh robot holding a fixed heading remained still. This confirms a pre-existing mode-switch defect, but does not establish the cause of any particular observed session.
  Current locations: [src/robot.rs:398](src/robot.rs#L398) (finite-heading branch), [src/robot.rs:426](src/robot.rs#L426) (orientation assignment), [src/robot.rs:431](src/robot.rs#L431) (angular-velocity assignment in the other branch), and [src/physics.rs:152](src/physics.rs#L152) (subsequent Rapier integration). Update order is in [src/simulation.rs:322](src/simulation.rs#L322).

- **Command timeout leaves ball-handling state active.** Original baseline `lib.rs:2692-2697`, `2793-2799`, `2847-2849`: timeout clears movement targets but does not clear dribble speed or reflex arming. An expired command can therefore continue requesting ball capture or a reflex kick.
  Current locations: [src/robot.rs:350](src/robot.rs#L350) (timeout reset) and [src/robot.rs:455](src/robot.rs#L455) (subsequent reflex/dribbler processing), in `RobotFleet::update`.

- **Continuous contact can count as double touch.** Original baseline `lib.rs:2429-2440`, `2499-2534`: the first touching robot becomes the kicker, and a subsequent step reporting that same robot immediately counts as a double touch while kick tracking remains active. There is no requirement that contact first end and then resume.
  Current locations: [src/referee/rules.rs:34](src/referee/rules.rs#L34) (per-step contact handling) and [src/referee/rules.rs:452](src/referee/rules.rs#L452), `AutoReferee::check_double_touch`.

- **Repeated builder ball creation leaves earlier bodies active.** Original baseline `lib.rs:3304-3326`: each `add_ball` inserts a body and collider and overwrites the tracked ball handles without removing the previous pair. Earlier balls remain in the physics world but are absent from the single tracked-ball detection and control paths. This creation/cleanup behavior remains preserved even when a shared spawning primitive is used.
  Current locations: [src/builder.rs:51](src/builder.rs#L51), `SimulationBuilder::add_ball`, and [src/physics.rs:209](src/physics.rs#L209), `PhysicsWorld::spawn_ball`.

- **Builder robot identities can collide.** Original baseline `lib.rs:3275-3293`, `3335-3380`, compared with `1107-1108`: builder insertion appends without removing an existing same-team ID, while runtime `add_robot` removes the prior robot. Explicit builder IDs also do not advance the automatic counters. Combining explicit and automatic insertion can therefore create duplicate identities, with ambiguous identity-based commands and lookups. Both creation paths retain their existing deduplication behavior in the refactor.
  Current locations: [src/builder.rs:22](src/builder.rs#L22) (explicit/automatic IDs), [src/builder.rs:60](src/builder.rs#L60) (builder insertion), [src/robot.rs:89](src/robot.rs#L89) (append-only spawning), and [src/robot.rs:141](src/robot.rs#L141) (runtime replacement).

- **Detection frame numbers never advance.** Original baseline `lib.rs:3149-3154`: every generated detection frame has `frame_number = 1`, despite capture and send timestamps advancing. Packet consumers cannot use this field to distinguish frames.
  Current location: [src/packets.rs:71](src/packets.rs#L71), `SimOutput::new_detection_packet`.

- **Ground geometry disagrees with its surface comment.** Original baseline `lib.rs:902-908`: the fixed ground body is translated to `-GROUND_THICKNESS / 2`, but the cuboid's half-height is `GROUND_THICKNESS`. With the 10 mm constant, its top is at +5 mm rather than the commented z=0 surface. This is a confirmed geometry/comment mismatch; its intended physical correction is not established here.
  Current location: [src/physics.rs:54](src/physics.rs#L54), the ground construction in `PhysicsWorld::new`.

## Likely inefficiencies and limitations

- **The ignored usage example is out of date.** Original baseline `lib.rs:735-758`: the example passes a `Duration` to `step`, whose actual argument is `f64`, and omits the team argument when pushing commands. The code block is marked `ignore`, so documentation tests do not catch these mismatches. The original example is preserved during the split.
  Current location: [src/simulation.rs:36](src/simulation.rs#L36), the ignored usage example on `Simulation`; current method signatures are in [src/simulation.rs:292](src/simulation.rs#L292) (`push_cmd`) and [src/simulation.rs:322](src/simulation.rs#L322) (`step`).

- **Apparently redundant query-pipeline update.** Original baseline `lib.rs:2578-2579`, `3056-3067`: the pipeline updates before player processing and is also passed to Rapier's physics step. The original simulator performs no intervening query operations. The first update appears unnecessary, but no performance measurement was made.
  Current locations: [src/physics.rs:147](src/physics.rs#L147), `PhysicsWorld::update_queries`, and [src/physics.rs:165](src/physics.rs#L165), the query pipeline passed to Rapier. Call order is in [src/simulation.rs:322](src/simulation.rs#L322).

- **Shot-failure randomness is consumed without a shot attempt.** Original baseline `lib.rs:2793-2799`: a realism-enabled robot with nonzero failure probability samples a misfire whenever the ball is in its dribbler, before checking whether a kick is requested. This adds draws during ordinary contact and couples later stochastic results to those contacts. It is an observed RNG-stream property and potential inefficiency, not a proven physical bug; draw order must remain unchanged during modularization.
  Current location: [src/robot.rs:458](src/robot.rs#L458), where the misfire is sampled before the kick-request check.

- **Periodic triggers catch up one call at a time.** Original baseline `utils.rs:25-33`: a successful trigger advances its deadline by one interval regardless of how far time has jumped. A large time jump can cause repeated triggers on later calls, including repeated geometry reads at the same simulated time, until the deadline catches up. This is a scheduling limitation rather than a demonstrated regression.
  Current location: [src/time.rs:53](src/time.rs#L53), `IntervalTrigger::trigger`.
