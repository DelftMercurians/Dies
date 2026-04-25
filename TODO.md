# Roadmap

## Milestones
- [ ] **M1** — Smooth and reliable motion control with MPC
- [ ] **M2** — Reliable passing
- [ ] **M3** — Aggressive strategy with plan-waypoints-replan loop validated in sim
- [ ] **M4** — Strategy + skills validated IRL (friendly match or 3v3 self-play)
- [ ] **M5** — Strategy improvement loop: many self-plays, tweak, improve, repeat (stretch)

---

## 1. MPC: field validation (M1)

**Goal:** iLQR motion control tuned and validated on real robots. Subjective bar:
uses max available acceleration and speed, zero overshoot / oscillation, no
steady-state error, precise small adjustments that overcome wheel slippage at
low speed commands.

- [ ] Run sysid scenarios on the field (`sysid_forward_chirp.js`,
      `sysid_strafe_step.js`); collect trajectories
- [ ] Fit dynamics params from field data (offline via `dies-mpc/src/sysid.rs`);
      update config
- [ ] Extend iLQR to SE(2): state `[x, y, θ, ẋ, ẏ, θ̇]`, body-frame control
      with anisotropic accel limits. Replace separate `YawController`.
- [ ] Decide override-vs-primary: currently velocity override on top of MTP.
      Widen authority or make iLQR primary with MTP as fallback.
- [ ] Field validation pass: step response, small-adjustment, full-speed
      trajectories. Iterate on cost weights and constraints until subjective
      bar is met.

---

## 2. Ball tracker rebuild (M2 dep)

**Goal:** Track pos/vel accurately plus possession (both teams), dribbling
state, and kick events.

**Data:** historical match logs across tournaments, using autoref signals as
authoritative possession labels.

- [ ] Build processing pipeline: load historical logs → extract vision frames,
      autoref signals, own-robot kicker/dribbler/breakbeam telemetry → aligned
      dataset for tracker fitting and evaluation
- [ ] Keep KF for in-flight / free-rolling dynamics (already works well)
- [ ] Add discrete event detectors on top of KF:
  - [ ] Kick event detection
  - [ ] Possession state per team (own + opponent)
  - [ ] Dribbling vs. free-rolling
- [ ] Tune/fit detectors against autoref labels; measure precision/recall on
      held-out matches
- [ ] Wire into `dies-world`; replace strategy-layer possession logic in
      `concerto/possession.rs`

---

## 3. Skill set validation (M2)

**Goal:** Validate full skill set including ball handling and passing.
Scripted scenarios + subjective validation.

Current skills: `GoToPosSkill`, `DribbleSkill`, `PickupBallSkill`,
`ReflexShootSkill`. Missing: passing, receiving.

- [ ] Build passing skill (kick-to-target with lead calc for moving receiver)
- [ ] Build receive skill (intercept + capture)
- [ ] Write scripted test-driver scenarios for each skill (pattern:
      `scenarios/skill_pickup_ball.js`)
- [ ] Subjective validation pass in sim, then on field
- [ ] Prereq for field validation: fix dribble and shoot in simulator
      (see §6)

---

## 4. Logging redesign (independent)

**Goal:** Cut log file size and build tooling for match post-mortems and
debugging. Clean break — no backward compat.

**Current problem:** massive JSON payloads dumped every frame, wrapped in
protobuf lines. Protobuf is no longer a must.

- [ ] Audit what's actually logged per frame; identify redundant / rarely-read
      fields
- [ ] Design new format (binary, likely msgpack or custom; delta-encode
      slow-changing state; separate streams for hot vs. cold data)
- [ ] Implement writer + reader in `dies-logger`
- [ ] Build post-mortem tooling: load log → filter/query → visualize. Scope
      and UI TBD during build.

---

## 5. Self-play strategy validation (M3, stretch → M5)

**Goal:** Validate strategy design by building a simple strat and doing
self-play in sim. Concerto vs. concerto, dynamic play.

MVP definition is itself part of the task — aim for stripped-down ruleset,
not full SSL compliance.

- [ ] Define MVP harness: spawn 2 strategy procs, run sim headless, dump
      match result + key metrics. Iterate on what "useful" means.
- [ ] Stripped-down ruleset: what's minimum needed for dynamic play (kickoff,
      ball-out handling, goals)? See §6.
- [ ] Faster-than-real-time sim support (see §6)
- [ ] Run concerto vs. concerto; observe; decide what analysis is worth
      automating
- [ ] (Stretch, M5) Orchestration: many runs, param sweeps, regression
      tracking

---

## 6. Simulator fidelity (prereq for M3)

- [ ] Fix dribble and shoot in simulator
- [ ] Automated referee actions:
  - [ ] Safe ball teleportation
  - [ ] Stop game on ball out of bounds; proper free kick
  - [ ] Kick-off after goal
  - [ ] All simulation logic supports faster-than-real-time

---

## 7. UI / misc (independent)

- [ ] Debug visualization: new types — heatmap, curve
- [ ] Debug visualization: allow toggling in UI (in progress per recent commit)
- [ ] UI: move settings to modal

---

## Critical path

M1 ← §1
M2 ← §2 + §3 (needs §6 dribble/shoot fix)
M3 ← §5 (needs §6 full)
M4 ← M1 + M2 + M3
M5 ← §5 stretch
