# Concerto Strategy — Implementation Brief

Status as of the current build. This is the authoritative description of the
`concerto` strategy (`strategies/concerto/`) and every design decision behind it.
It supersedes the earlier `concerto-high-level-brief`, `concerto-formation-driver-brief`,
`strategy_evolution`, and `strategy-refactor-plan` documents.

---

## 1. What Concerto is

Concerto is the Delft Mercurians' first competent RoboCup SSL strategy. It runs as a
separate process implementing the `Strategy` trait from `dies-strategy-api`, consuming
a team-relative world snapshot each tick (~60 Hz) and emitting skill commands.

Its shape is a **formation / planner split**:

- **Formation** positions every field robot *except* the goalkeeper and the single
  plan-controlled "active" robot.
- **Planner → Driver** moves the ball toward the opponent goal via ball-state-transition
  waypoints, re-deciding only on discrete events.

The first version (this build) is deliberately a **single-carrier offense + formation
defense**. Passing is deferred but every seam where it attaches is in place.

### Milestone status

| Milestone | Scope | State |
|---|---|---|
| M1 | Possession tracker + bare offensive loop + keeper | **done** |
| M2 | Real formation (role generators, redirect-cost Hungarian, cadence) | **done** |
| M2.5 | Post-kick seam fix (possession feedforward + velocity-aware classification) | **done** |
| — | Game-state handling (kickoff / free kick / penalty), directed release-kick | **done** |
| — | Excessive-dribbling compliance (kick-ahead offense) | **done** |
| M3 | Conservative steal gate, favorable-angle steal, no-progress detection, tuning | **partial** (see §11) |
| M4 | Passing (future) | **seams only** |

Verified: `cargo build/clippy/test -p concerto` clean, 19 unit tests pass. **Not yet
verified in sim** — behavior tuning is the open work.

---

## 2. Core principles (and how they're interpreted)

1. **Formation / planner split.** The planner is deliberately dumb (hardcoded
   possession-based templates). Visible competence comes from formation positioning and
   skill quality, not planner intelligence. The split's value in v1 is the clean seam
   that lets a smarter planner drop in later without touching positioning.

2. **Plan-replan waypoint loop.** The planner runs only on discrete events; the driver
   realizes the current waypoint and reports rich status that triggers replanning.

3. **"No hysteresis" → stability from physics, not stay-bonuses.** This is the load-bearing
   principle. The slogan "no hysteresis" is not literally achievable, so it's reframed:
   every discrete decision must source its stability from one of three legitimate places,
   never a flat bonus for keeping the status quo:
   - **continuity** — the underlying function varies smoothly with the world;
   - **physical cost** — momentum / redirect time makes changing your mind genuinely expensive;
   - **decision cadence** — you don't re-decide every tick.

   Each discrete seam (possession classification, formation assignment, plan selection,
   active-robot choice, driver commit) answers "where does stability come from here?" with
   one of those, not with a stickiness term.

---

## 3. What the framework provides (strategy scope)

A deliberate scoping decision: **the strategy implements no spatial game rules.** The
executor does, downstream. This keeps Concerto small.

- **Coordinate frame.** All world data is team-relative: **+x toward the opponent goal**,
  −x toward our own. The strategy never sees absolute coordinates or team color; the host
  transforms in and out.

- **Compliance (`comply()` in `dies-executor`).** Automatically enforces, keyed off **role
  names** the strategy sets via `set_role`:
  - Halt/Stop speed caps; 800 mm ball avoidance during Stop / opponent free kick;
  - goal-area avoidance for non-keepers;
  - kickoff own-half + center-circle clamps for non-`kickoff_kicker`;
  - penalty sidelining for everyone except `penalty_kicker` (ours) / `goalkeeper` (theirs).
  - Role-name → type mapping is **substring, case-sensitive**: a name containing
    `"goalkeeper"`, `"kickoff_kicker"`, `"free_kick_kicker"`, `"penalty_kicker"`, or
    `"waller"` maps to the corresponding `RoleType`; everything else is `Player`.

- **Kicker / double-touch detection.** The framework auto-detects the restart kicker (first
  toucher) and tracks double-touch via `world.freekick_kicker()`, which Concerto mirrors and
  excludes from active-robot selection.

- **Passing coordinator (for M4).** `ctx.pass(passer, receiver).target_hint(p)` +
  `ctx.pass_result(id)` drive a full executor-side `PassCoordinator` (Secure→Setup→Commit→
  Flight→Settle). The strategy only decides and reacts — passing is a thin future add.

---

## 4. Module map

All under `strategies/concerto/src/`.

| Module | Responsibility | Cross-tick state |
|---|---|---|
| `config.rs` | All physical tunables, one place | — |
| `possession.rs` | Debounced possession classification (root stability surface) | tracker state, release window |
| `geometry.rs` | Stateless helpers (redirect time, shot/lane clearance, threat, shadow arc) | — |
| `matching.rs` | Hand-rolled rectangular Hungarian (min-cost assignment) | — |
| `formation.rs` | Position all field robots except keeper + active robot | role-identity assignment + recalc cadence |
| `planner.rs` | Pick waypoint + active robot on events | current plan, recent-failure memory |
| `driver.rs` | Realize current waypoint via skills, report status | phase/timers, kick + new-active hooks |
| `keeper.rs` | Dedicated goalkeeper positioning | — |
| `lib.rs` | Orchestrator: tick ordering, triggers, kicker designation, debug | game-state edge, double-touch, dribble origin |

### Tick ordering (`ConcertoStrategy::update`)

1. Owned world snapshot (so reads don't borrow `ctx` while issuing commands).
2. Double-touch tracking (mirror `freekick_kicker`).
3. Game-state transition → clear plan + driver.
4. Hard stops (Halt/Unknown/Timeout) → return (executor halts robots).
5. Possession: `classify_raw` → `tracker.update` → stable possession + `changed_this_tick`.
6. Stamp/clear dribble origin; compute `carried`.
7. If ball present and `is_ball_in_play() && we_may_act`: compute replan triggers → replan
   if needed → `driver.update` (emits active-robot skills) → kick feedforward → kicker role
   override.
8. Our set-piece prep (designate kicker for our restarts).
9. Formation (sees active set + plan context) → apply commands.
10. Keeper. 11. Debug overlays.

**Replan triggers:** stable possession changed · driver Succeeded · driver Failed(reason) ·
no current plan · game-state transition. Nothing else (drift, mid-approach, transient ball
dropout are absorbed below).

---

## 5. Possession (`possession.rs`) — the root stability surface

Everything keys off one debounced classification; nothing downstream re-debounces.

- **Stable output:** `Possession::{We(id), Opp(id), Loose}`.
- **Raw (memoryless), `classify_raw`:**
  1. any own `has_ball` (breakbeam) → `We{breakbeam:true}` — authoritative, ignores speed;
  2. ball faster than `POSSESSION_MAX_BALL_SPEED` → `Loose` (a fast ball can't be possessed
     by proximity);
  3. nearest own within `WE_POSSESSION_DIST` → `We{breakbeam:false}`;
  4. nearest opp within `OPP_POSSESSION_DIST` → `Opp`;
  5. else `Loose`. Ball undetected → `Unknown`.

### Decisions

- **Asymmetric debounce.** Gaining `We` via our own breakbeam commits in **1 frame** (hard
  sensor); losing possession, switching carrier id, or `Loose↔Opp` need **`DEBOUNCE_FRAMES`
  (~4 = 66 ms)**. Rationale: true possession can't change every 16 ms, so a single-frame flip
  is almost certainly a sensor artifact. This is *sensor conditioning*, not the forbidden
  decision-hysteresis — it filters a measurement upstream of any decision; the planner carries
  no preference for its prior choice.
- **Detection dropout hold.** `Unknown` holds the last stable state for `POSSESSION_HOLD_SECS`
  (~0.25 s), then decays to `Loose`.
- **Velocity-aware classification.** Proximity possession requires a slow ball. Kills spurious
  We/Opp when a shot whizzes past a robot, and prevents the kicker re-registering possession on
  a receding ball.
- **Kick feedforward (`notify_release`).** When the driver fires a kick, possession is told the
  ball was *commanded* released: `We(kicker)` drops immediately (bypassing the loss-debounce),
  and a `RELEASE_SUPPRESS_SECS` (~0.2 s) window ignores the kicker's *proximity* re-acquisition.
  **Breakbeam is never suppressed**, so a misfire where the ball never actually left
  self-corrects within one frame. This is efference copy — incorporating a known action — the
  same idea as the player-tracker command feedforward in the core.

This fixes the post-kick seam: without it, the debounce kept `We(kicker)` stuck for ~66 ms after
a shot, and the immediate replan re-tasked the shooter against a ball already gone.

---

## 6. Formation (`formation.rs`) — stable assignment without hysteresis

Role generators produce smoothly-varying positions with continuous importance; robots are
matched to roles by minimum total cost. Stability comes from continuity + redirect cost +
cadence — never a stay-bonus.

### Role generators (pure functions of the world)

- **Shadow / goal coverage** (coordinated set). `K = round(SHADOW_MIN + (SHADOW_MAX−SHADOW_MIN)·threat(ball))`,
  K∈[1,3], spaced across the goal mouth as seen from the ball (`geometry::shadow_arc`).
  Importance scales with ball threat. Faces ball.
- **Per-opponent marking** (independent), slot = stable opponent-id index. Positioned
  `MARK_STANDOFF` in front of the opponent toward our goal. Importance =
  `IMP_MARK_BASE · threat(opp) · lane_openness(ball→opp, other opponents)` — i.e. mark opponents
  who are dangerous *and* open to receive. Faces ball.
- **Offensive support** (`SUPPORT_COUNT`=2, flank-split), forward positions nudged away from the
  nearest opponent. Low importance. Faces opponent goal.
- **Plan-context / receiver** (0 or 1) from `driver.plan_context_area()` — **always None in v1**;
  the M4 pass seam. High importance when present.
- **Residual / spread** — tops up to the over-generation target `ceil(OVERGEN_FACTOR · n)`,
  near-zero importance, so every robot always has something to do.

`threat` and `lane_openness` use `smoothstep` (C¹) — **never hard `if x<0` thresholds** (the
stub's jitter bug).

### Assignment

- **Cost** = `redirect_time(robot→role) − importance · SEC_PER_IMPORTANCE`, in seconds; minimize
  total via hand-rolled rectangular Hungarian (`matching.rs`, rows≤cols by over-generation;
  4 tests cover optimality / forced swap / rectangular slack / negatives).
- **`redirect_time`** is momentum-aware: cruise time `dist/v_max` + penalty
  `(2·v_against + v_cross)/a_max` for reversing/cross velocity − a small head-start credit for
  velocity already toward the target. A robot moving toward a role continues near-free; reversing
  is genuinely expensive. This is the physical basis for stability.
- **`SEC_PER_IMPORTANCE` (~0.4 s/point)** is the primary tuning knob ("1 importance point ≈
  300–500 ms of redirect time").

### Role identity + cadence (the stability sources)

- Each role carries a stable `RoleId{kind, slot}` (deterministic slots: shadow by angle order,
  mark by opp-id index, support by flank, residual by generation order). Count changes preserve
  lower slots (K 2→3 keeps Shadow{0,1}).
- The assignment is recomputed **only on triggers** (assignable-set change · plan-slot change ·
  plan-context change · background ~`RECALC_BG_PERIOD` 0.4 s) subject to a **cooldown**
  (`RECALC_COOLDOWN` ~0.18 s). A trigger during cooldown is queued and runs the instant it
  expires.
- **Every tick regardless**, each assigned `RoleId` is re-resolved to its current position and
  the robot is commanded `go_to(pos).facing(...)` — smooth motion from the trajectory controller
  while the assignment stays frozen between recalcs.

### Decision: exclusion vs plan-slot model

The active (plan-controlled) robot is, in v1, **excluded** from the assignable set rather than
modeled as a max-importance "plan slot." For a single active robot the two are equivalent, and
exclusion is simpler. The plan-slot model (which keeps the robot count constant so two
plan-controlled robots during a pass don't cascade) is deferred to **M4**, where it earns its
keep. The keeper is excluded entirely and runs dedicated logic.

---

## 7. Planner (`planner.rs`) — dumb, event-driven, pass-ready

Waypoint types: `Capture{kind, robot}` (`CaptureKind::{Loose, Steal{from}}`), `Dribble{area}`,
`Shoot{target}`, and `Pass{...}` (defined, never emitted in v1 — the M4 seam). A `Plan` is a
length-1 waypoint list + active robot; replan-after-each makes any tail advisory.

### Branches by stable possession

- **`We(id)`** — carrier offense:
  1. Clear shot in range (`is_clear_shot` corridor + `SHOOT_RANGE`) → `Shoot(goal center)`.
  2. Else **kick-ahead** (primary advancement, see §8): `Shoot` at a forward supporter or open
     space. *This is the pass seam* — M4 swaps the supporter kick-ahead for a `Pass`.
  3. Else (congested, no kick target) and under the carry cap → a small corrective `Dribble`.
  4. Else → release (kick at open space) anyway.
- **`Loose`** → `Capture{Loose, select_capturer()}`.
- **`Opp(oid)`** → steal, gated. **v1 gate is crude**: only commit if the chosen challenger is
  within `STEAL_MAX_DIST` of the ball; otherwise emit no active robot and let Formation contain.
  (The proper conservative gate — don't strip a deep defender, prefer a favorable interception
  angle — is M3.)

### Active-robot selection — continuity from physics

`select_capturer` picks the robot with the smallest **momentum-aware time-to-ball**
(`redirect_time`, position **and** velocity). The current carrier / a moving-toward-ball robot
wins naturally — no artificial bonus. Excludes keeper and the double-touch robot.

### Anti-loop (the no-progress trap)

`recent_failures: HashMap<PlayerId, (FailReason, ts)>`. A robot that fails with `NoProgress` is
excluded from selection for `NOPROGRESS_TTL` (~1 s), expiring by time. **Note:** the driver does
not yet *emit* `NoProgress` (M3), so this is currently dormant plumbing waiting on the detection.

---

## 8. Offense & the excessive-dribbling rule (§8.4.1)

**Rule:** a robot may not dribble the ball more than **1 m linearly from the contact point**;
the budget resets only on **observable separation** (ball leaves the dribbler). Violation →
Stop → free kick against us + foul.

**Decision (per direction from the team lead):** dribbling is treated as an *unreliable* form of
ball handling — minimized, used only for small corrections. The **primary advancement action is
a kick-ahead**, preferably toward a supporter, else the goal/open space.

- **Kick-ahead** (`best_kickahead_target`): prefer a forward supporter (at least
  `SUPPORTER_FWD_MARGIN` ahead, lane openness ≥ `SUPPORTER_MIN_OPENNESS`), kicking
  `SUPPORTER_LEAD` past them into space toward goal; else open forward space
  (`best_pass_area`). Realized as `Shoot`, so it reuses the Shoot arm and the kick feedforward —
  the ball separates, the carrier is suppressed from proximity-possession, and a teammate/forward
  collects.
- **Dribble = correction only**: emitted just when there's no kick-ahead target (congested)
  **and** `carried < DRIBBLE_CORRECTION_LIMIT` (~0.35 m). Beyond the cap, release (kick)
  regardless. `DRIBBLE_ARRIVE_DIST` (150 mm) is kept below the correction step so a correction is
  a real move, not an instant "already there" that would spin the replan loop.
- **Carry tracking** (`lib.rs`): `dribble_origin` is stamped when possession becomes `We`,
  cleared when not `We`; `carried = |ball − origin|`, passed to the planner. The M2.5 feedforward
  makes the post-kick separation crisp, so the budget resets cleanly each possession.

**Caveat:** there is no kick-power control (`Shoot` is fixed-power), so a kick-ahead is a genuine
kick-and-chase, not a soft push. Accepted for v1; gentler self-passes arrive with M4 passing /
power control.

---

## 9. Driver (`driver.rs`) — rich failures, kick feedforward

Per-waypoint state machine. `WaypointStatus::{Ongoing, Succeeded, Failed(FailReason)}` with
`FailReason::{Timeout, BallMoved, PossessionLost, NoProgress, SkillFailed, NoReceiver}`.

- **Global failure first:** ball moved > `BALL_MOVED_DIST` from the engagement point and not ours
  → immediate `Failed(BallMoved)`; missing active robot → `Failed(SkillFailed)`.
- **Capture:** `Approach` (`go_to(ball).facing(ball)`) until within `CAPTURE_PICKUP_DIST` →
  `Pickup` (`pickup_ball(heading toward goal)`). **v1 does not yet distinguish Steal from Loose**
  (both drive straight to the ball); the favorable-angle steal approach is M3.
- **Dribble:** `dribble_to`; success on arrival + possession; fail on PossessionLost / Timeout.
- **Shoot:** `reflex_shoot`; on success sets `kick_event = active_id` (consumed by the
  orchestrator for possession feedforward); fail on SkillFailed / Timeout.
- **Hooks:** `take_kick_event()` (feedforward, §5) and `take_new_active()` (the passer→receiver
  handoff seam, always None in v1). `plan_context_area()` returns the receiver area for `Pass`
  waypoints only (None in v1).

`NoProgress` detection (sliding window on distance-to-ball / interception quality) is **not yet
implemented** — it is the key remaining M3 piece that arms the planner anti-loop.

---

## 10. Goalkeeper & game states

- **Keeper** (`keeper.rs`): sits on a line `KEEPER_DEPTH` in front of the goal, y = clamped
  intersection of the ball→goal-center ray with that line (covers the shot angle). Role name
  `"goalkeeper"`; never the active robot; excluded from Formation. A keeper-held ball clearance
  becomes a normal `Loose` capture.

- **Game states** (minimal; rely on `comply()`): Formation + keeper always run. The offensive
  loop runs only when `is_ball_in_play() && we_may_act` (`Run`; our `Kickoff`/`FreeKick`/
  `PenaltyRun`).
  - **Our restarts:** the active robot is named the appropriate kicker so comply exempts it.
    During the not-yet-in-play prep states (`PrepareKickoff`, `PreparePenalty`, `Penalty`) the
    nearest robot is designated and positioned at the ball — `Penalty` is included because comply
    would otherwise sideline our own kicker.
  - **Directed release-kick (decision).** On our in-play kickoff/free kick the kicker must release
    the ball forward (a dribble would be a double-touch foul). It kicks-ahead at a supporter/space;
    a teammate collects, double-touch clears, normal play resumes. The guard
    `is_kicker = our_attacking_restart && (double_touch_robot ∈ {None, carrier})` ensures only the
    actual kicker releases.
  - **Opponent restarts:** `we_may_act = false` → defend with Formation + keeper; comply enforces
    spacing. No explicit wall in v1 (decision: rely on shadow roles + comply's 800 mm).
  - **Penalty (ours):** kicker captures then shoots at goal (normal `We→Shoot`); single shot, no
    double-touch concern.

---

## 11. What is intentionally not done yet

- **No-progress detection** in the driver (and thus the planner anti-loop is dormant).
- **Favorable-angle steal** — steal capture currently tail-chases the ball.
- **Conservative steal gate** — currently a crude distance check, not "don't strip a deep
  defender."
- **Shot aiming** — always aims at goal center, not the open part of the mouth.
- **Plan-slot formation model** — v1 uses exclusion (fine for one active robot).
- **Sim tuning pass** — all constants are first-guess; behavior has not been tuned against the
  simulator.
- **Passing (M4).**

---

## 12. The passing seam (M4)

Passing is fully implemented in the core; turning it on is three localized edits, all marked in
the code:

1. **Planner** `We`-no-clear-shot branch: choose `Pass` (to a Formation-staged receiver) instead
   of the supporter kick-ahead, using `best_pass_area`.
2. **Driver** gains a `PassExec` arm: a thin wrapper over `ctx.pass(...)` that maps `PassResult`
   → `WaypointStatus` and sets `new_active = receiver` on success.
3. **Formation** staffs the receiver via the existing plan-context role (`plan_context_area()`).

Plus the formation switches from exclusion to the plan-slot model so two plan-controlled robots
(passer + receiver) don't cascade. Possession, tick ordering, triggers, keeper, and compliance
are unchanged. `Waypoint::Pass`, `CaptureKind` (future `Intercept`), `FailReason::NoReceiver`,
and the new-active hook already exist for this.

---

## 13. Key tunables (`config.rs`)

| Constant | Value | Meaning |
|---|---|---|
| `V_MAX`, `A_MAX` | 3000, 3000 | robot motion model for redirect-time |
| `WE_/OPP_POSSESSION_DIST` | 120 / 150 | proximity possession ranges (mm) |
| `DEBOUNCE_FRAMES` | 4 | frames to commit a non-breakbeam possession change |
| `POSSESSION_MAX_BALL_SPEED` | 1000 | above this, no proximity possession (mm/s) — watch in sim |
| `RELEASE_SUPPRESS_SECS` | 0.2 | post-kick proximity-reacquire suppression |
| `CLEAR_SHOT_CORRIDOR`, `SHOOT_RANGE` | 700, 4000 | direct-shot gate |
| `SUPPORTER_FWD_MARGIN` / `_MIN_OPENNESS` / `_LEAD` | 400 / 0.5 / 350 | kick-ahead supporter selection |
| `DRIBBLE_CORRECTION_STEP` / `_LIMIT` | 250 / 350 | corrective dribble step / hard carry cap (mm) |
| `SEC_PER_IMPORTANCE` | 0.4 | **primary formation tuning knob** (s per importance point) |
| `IMP_SHADOW/MARK/SUPPORT/RECEIVER/SPREAD` | 8/6/3/12/0.5 | role importance ladder |
| `OVERGEN_FACTOR` | 1.5 | role over-generation vs robot count |
| `RECALC_COOLDOWN` / `_BG_PERIOD` | 0.18 / 0.4 | assignment cadence (s) |
| `SHADOW_MIN/MAX`, `SHADOW_STANDOFF` | 1/3, 1500 | goal-coverage count + standoff |
| `STEAL_MAX_DIST`, `NOPROGRESS_TTL` | 2500, 1.0 | crude steal gate / anti-loop window |
| `KEEPER_DEPTH` | 200 | keeper line in front of goal (mm) |

---

## 14. Verification

- **Build/lint/test:** `cargo build -p concerto`, `cargo clippy -p concerto --tests`,
  `cargo fmt -p concerto`, `cargo test -p concerto` (19 unit tests).
- **Unit coverage:** possession debounce (blip masked, dropout decay, asymmetric commit,
  velocity-aware, feedforward + suppression); Hungarian (optimality, forced swap, rectangular,
  negatives); geometry (redirect-time monotonicity, smoothstep, threat, lane openness); formation
  (one command per assignable robot, frozen within cooldown).
- **Sim:** `just dev concerto` (builds + runs in simulation). Watch the web UI debug overlays
  (`possession` string, `active_robot` circle, `formation_*` crosses, role names). Per-milestone
  checks: M1 single-carrier attack with no clumping; M2 stable defensive shape, no jitter; M2.5
  possession flips to `Loose` on the kick frame and a forward (not the shooter) collects.
