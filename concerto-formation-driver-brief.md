# Concerto — Formation and Driver Design Brief

This document refines and extends the original MVP brief with detailed designs for the Formation system and the Driver layer. It captures architectural decisions, rationale, interaction mechanics, stability guarantees, edge cases, and failure modes. It supersedes the Formation and Execution sections of the original MVP brief.

---

## Design Principles and Constraints

These principles emerged from critical analysis of the original brief and guide every decision in this document.

**No hysteresis, anywhere.** Hysteresis is a patch that masks instability rather than solving it. It introduces a tuning parameter that behaves differently depending on how fast the game state is changing — too little and you still get jitter, too much and robots become sluggish, clinging to stale positions because the artificial bonus for staying outweighs the benefit of moving. Every stability mechanism in this design must derive from physical reality (redirect cost, cooldowns tied to actual timescales) rather than artificial bonuses for maintaining the status quo.

**The trajectory controller handles movement smoothing.** The underlying trajectory controller is perfectly capable of producing smooth robot motion toward a changing target. The Formation system should never attempt to smooth target positions — that solves a problem that's already solved and pushes the real problem (target selection stability) elsewhere. If a target jumps, the robot accelerates smoothly toward the new target. The architecture's job is to ensure targets don't jump unnecessarily, not to hide jumps when they occur.

**Decisions at the timescale of the game, not the tick rate.** The system runs at 60Hz. Making formation decisions every tick is unnecessary overhead and creates more potential for oscillation. The meaningful timescale for formation decisions is a few hundred milliseconds — fast enough to respond to real tactical changes, slow enough that transient fluctuations in the scoring landscape don't cause reactions. This is structural damping that falls out of the problem's actual timescale.

**The planner is the brain. Formation is good enough.** The crux of Concerto's strategy is the planner. Formation needs to be solid, stable, and simple — not tactically brilliant. It must interact seamlessly with the plan layer, handle dynamic robot counts gracefully, and provide a reliable defensive and supportive foundation. Tactical sophistication comes from the planner and (later) from plan generators. Formation is infrastructure.

**Plan-controlled robots stay in Formation.** The original brief described the active robot as being "excluded from Formation assignment." The refined design keeps plan-controlled robots as visible slots within Formation. They are not commanded by Formation, but Formation accounts for their positions when computing assignments for other robots. This eliminates the dynamic robot count problem at the plan boundary and prevents cascading repositioning when the plan claims or releases robots.

---

## Architecture Overview

The system has three layers above the skill level:

**Formation** manages positioning for all field robots at all times. It computes target positions using a role-based system and assigns robots to roles using cost-aware optimal matching. Formation always sees all field robots, including those controlled by the plan.

**The Planner** produces waypoint sequences that describe how to move the ball toward the opponent goal. It thinks exclusively about ball state transitions and does not concern itself with specific robot identities or trajectories.

**The Driver** sits between the Planner and the skills. It takes a waypoint and realizes it — translating the abstract ball-state transition into moment-to-moment skill activations, dynamically adjusting parameters, monitoring progress, and reporting success or failure back to the Planner. The Driver controls up to two robots simultaneously for coordinated tasks (most importantly, passing). The Driver is the sole authority on skill activation for plan-controlled robots and the sole authority on waypoint success/failure.

The Goalkeeper remains outside Formation and runs dedicated positioning logic as described in the original MVP brief. It never becomes a plan-controlled robot.

---

## Layer 1: Formation

### Role-Based Positioning

Formation uses a role-based system rather than a global scoring function. Roles provide semantic stability — "block the most dangerous shooting lane" persists as a concept from tick to tick even as the specific position shifts with the ball and opponents. This semantic persistence is the primary source of assignment stability, without any need for hysteresis.

#### What a Role Is

A role is a function from world state to a position and an importance score. Each evaluation tick, the role produces a continuously-varying position (because the world state it depends on is continuous) and a contextual importance score reflecting how much the team needs that position covered right now.

Role generators are functions that produce zero or more role instances based on the current state. The total role count fluctuates but should always exceed the number of formation-managed robots by a factor of roughly 1.5-2x. This over-generation gives the assignment algorithm flexibility to find cost-effective matches without forcing expensive reassignments.

Each role instance carries a stable identifier across ticks. The shadow-left role this tick corresponds to the shadow-left role last tick. This enables the assignment algorithm to recognize continuity without artificial stickiness — a robot already serving shadow-left has genuinely low redirect cost to shadow-left's new position because the position only moved slightly.

#### Role Generators (MVP)

**Shadow/coverage roles.** These generate positions that maximize angular coverage of our goal from the ball's perspective. This is the one place where coordination between roles matters — two shadow robots covering the same angle are redundant. The shadow generator handles this internally: given K shadow roles to produce, it spaces them to maximize total goal coverage across a ~160-degree arc. K itself scales with threat level — more shadow roles when the ball is near our goal, fewer when it's in the opponent's half. The generator treats the goalkeeper's coverage as a given and computes shadow positions for the uncovered angles.

This is a deliberate design choice: one generator produces a coordinated set of positions rather than multiple independent generators competing. This is the simplest way to solve the joint coverage optimization without requiring a global coordination mechanism. The tradeoff is that the shadow generator is qualitatively different from other generators (it produces coordinated sets, not independent roles). This pattern should not silently spread to other generators — if every generator starts doing internal coordination, the joint optimization problem is rebuilt inside each generator.

**Opponent marking roles.** One role per opponent in a threatening position. Position is between the opponent and our goal, at a distance allowing interception. Importance scales with opponent threat — near our goal with a passing lane from the ball is high importance, far away with no passing option is low. Each marking role is independent; marking different opponents is implicitly handled by the assignment step (each robot marks a different opponent because each role targets a different one).

**Plan context roles.** Generated from the current plan's context. If the plan says "pass to left wing," this produces a role at the left wing with high importance. If the plan has no context requiring formation support, this generator produces nothing. Plan context roles are the primary mechanism by which the plan communicates its needs to Formation. They compete in the assignment just like any other role. See the Plan-Formation Coupling section for lifecycle details.

**Offensive support roles.** Positions in the opponent half offering passing angles toward the goal, preferring areas far from opponent robots. Lower importance than defensive roles in most situations. These ensure the formation always has some forward presence and passing options even when no plan context role is active.

**Residual/coverage roles.** If all the above generators produce fewer roles than needed for adequate over-generation, residual roles fill the gap by maximizing field coverage and spread. These are the lowest-importance roles — they exist so that the assignment always has enough options and every robot has something to do.

#### Role Generator Requirements

Each generator's output must be smooth — small changes in world state produce small changes in role positions. If a generator's output is discontinuous (an opponent crosses a field-half boundary, threat level crosses a threshold), the resulting formation change is a response to a real event, not jitter. Generators should not introduce artificial discontinuities through hard thresholds where continuous functions would serve.

Generators must be deterministic given the same world state. They should not depend on their previous output (no internal memory) — the world state alone determines their output. This ensures that after any disruption (robot failure, game stoppage), the formation recovers to the correct configuration without needing to reconstruct internal state.

#### The Spread Problem

Individual roles optimize their own positions, but the aggregate set of roles might leave structural holes. Three marking roles could pull three robots to the right side, leaving the left open. The shadow generator handles its own coordination, but no mechanism ensures the full set of roles across all generators produces a well-distributed formation.

For the MVP, this is addressed through the residual/coverage roles and the implicit properties of the other generators. The offensive support generator naturally places roles away from opponents (who are likely concentrated somewhere), providing some counter-balance. The shadow generator distributes across angles. The main risk is a pathological opponent formation that concentrates all threats on one side — in that case, it may be correct for our formation to be concentrated there too.

If testing reveals persistent coverage gaps, the simplest structural fix is a minimum-spread constraint in the assignment step: no two robots can be assigned to roles closer than a threshold distance unless all available roles are clustered. This adds one parameter but addresses the problem directly rather than distorting the role generators.

### Assignment

#### Mechanism

Optimal matching with redirect cost. Each formation robot is matched to exactly one role from the available pool. Each role is matched to at most one robot. The matching uses the Hungarian algorithm (trivially cheap for 5-6 robots) to minimize total assignment cost.

The cost for assigning robot R to role X is a blend of two factors:

**Redirect cost.** The estimated time for robot R, given its current position and velocity, to reach role X's current position. This is not Euclidean distance — it accounts for the robot's momentum. A robot traveling at full speed toward position A has near-zero redirect cost to continue and substantial cost to reverse toward B. A stationary robot near both has low cost either way. Redirect cost reflects physical reality and naturally produces assignment stability: robots in motion toward a target have a genuine advantage for that target without any artificial bonus.

**Importance differential.** The value of filling role X versus leaving it unserved. Higher importance roles justify higher redirect costs. This is expressed as a willingness-to-redirect threshold: the system will redirect a robot from its current trajectory if the importance gain exceeds the redirect cost.

The blending of importance and redirect cost is the primary tuning surface of the Formation system. The calibration should be set so that the importance difference between "critical defensive role" and "nice-to-have offensive support" is equivalent to roughly 300-500ms of redirect time. This means the system will pull a robot from across the field for a critical defensive need, but won't twitch for a marginal improvement.

If this calibration is wrong, the symptoms are clear: too much reshuffling means importance dominates (lower the importance weight), too much sluggishness means redirect cost dominates (raise the importance weight). Unlike hysteresis, the tuning has a physical interpretation (how many milliseconds of delay is a role importance point worth?), making it easier to reason about and adjust.

#### Plan Slots in the Assignment

Robots controlled by the Driver occupy "plan slots" in the Formation. A plan slot is a role with special properties: its position is the robot's actual position (updated each tick), its importance is effectively infinite (no formation role can outbid it), and the robot assigned to it uses Driver-commanded skills instead of formation's movement skill.

From the assignment algorithm's perspective, a plan slot is just another role. The algorithm assigns a robot to it, and because its position matches the robot's actual position and its importance is maximal, the same robot stays assigned to it with zero redirect cost. Other roles compute their importance and positions normally, and the remaining robots are matched to them.

This means Formation always manages all N field robots. The robot count never changes due to plan activity. When the Driver claims a robot, a plan slot appears at that robot's position. When the Driver releases a robot, the plan slot disappears. In both cases, the number of robots in the assignment stays the same — what changes is the number of regular roles competing for assignment.

Formation uses plan-slot robots' actual positions for its coordination calculations (spread, coverage) but does not command their movement. A plan-slot robot moving across the field is visible to Formation as a position that other robots should account for (don't cluster near it, don't try to cover the same angle) but Formation doesn't issue it commands.

#### Stability Without Hysteresis

Assignment stability comes from three structural sources:

**Redirect cost reflects physical momentum.** A robot moving toward a target has genuinely low cost to continue and genuinely high cost to reverse. This is not an artificial bonus — it's the actual physics of the situation. Stability emerges from the real cost of changing direction.

**Role identity across ticks.** Roles carry stable identifiers. The shadow-left role this tick is the same role as last tick. A robot assigned to shadow-left remains assigned because (a) the role's position has only shifted slightly, giving the current assignee minimal redirect cost, and (b) other robots are further away, giving them higher redirect cost. The assignment produces consistency as a consequence of spatial continuity, not as a special case.

**Recalculation frequency.** Full assignment recalculation happens at a controlled rate, not every tick. Between recalculations, robots drive toward their current targets and nothing changes. This prevents transient fluctuations in role positions or importance from causing reactions.

The one genuine discontinuity the assignment cannot smooth: when role positions shift enough that the optimal matching flips topology (e.g., three robots in a triangle swap assignments simultaneously because the role positions rotated past a tipping point). This is rare, and when it happens, it reflects a real tactical change. The robots head to nearby positions and arrive within a few hundred milliseconds. This is acceptable.

### Recalculation Timing

#### Event-Driven with Cooldown

Rather than a fixed-rate recalculation, Formation recalculates on meaningful events with a minimum cooldown between recalculations (~150-200ms).

**Discrete triggers (immediate recalculation, subject to cooldown):**
- Possession change (we gain/lose the ball, or ball becomes loose).
- A robot enters or leaves Formation (penalty, hardware failure, return from penalty).
- The plan context changes (new waypoint, plan slot created or released).

**Background rate (~2-3Hz):** If no discrete trigger fires, Formation recalculates at a slow background rate to adapt to continuous changes in the game state (opponents drifting, ball moving slowly). This catches gradual shifts that no single event represents.

**Between recalculations:** Robots drive toward their current targets. Role generators still update role positions every tick (so the target a robot is driving toward moves smoothly), but the assignment — which robot goes to which role — is fixed until the next recalculation.

This means moment-to-moment smoothness is guaranteed by the trajectory controller (targets move smoothly as role positions update) and assignment stability is guaranteed by the recalculation cooldown (assignments can't change faster than every 150-200ms). The two concerns are handled by different mechanisms at different timescales.

**Potential issue:** During the cooldown after a recalculation, a second event fires (e.g., possession changes twice in quick succession). The cooldown should not suppress the second recalculation — it should queue it. When the cooldown expires, the queued recalculation runs with the latest world state. This ensures responsiveness to rapid state changes while preventing oscillation from near-simultaneous events.

### Dynamic Robot Count

#### Hardware Failures and Penalties

When a robot is removed from the field (hardware failure, yellow card), Formation simply has N-1 robots at the next recalculation. The previously-served roles are redistributed. The least cost-effective role goes unserved. This is a genuinely discrete event and the formation should visibly adjust — losing a robot is a real disruption.

When a robot returns from a penalty, it appears at a field edge position. Formation has N+1 robots. The returning robot gets assigned to whichever role it can fill most cheaply from its entry position — often one that was previously unserved and happens to be nearby. Existing robots' assignments experience minimal disruption because the new robot absorbs a role that was either unserved or the cheapest to transfer.

#### Plan Boundary (No Count Change)

The plan claiming or releasing a robot does NOT change Formation's robot count. See Plan Slots in the Assignment above. A plan claim creates a plan slot; a plan release removes a plan slot. The number of robots in the assignment stays constant. This is the key architectural decision that prevents cascade repositioning at the plan boundary.

---

## The Driver Layer

### Purpose and Scope

The Driver sits between the Planner and the skills. Its job is to realize waypoints: take an abstract ball-state transition (capture, dribble, pass, shoot) and produce moment-to-moment skill activations while monitoring progress and adapting to changing conditions.

The Driver is the sole authority on:
- **Skill activation** for plan-controlled robots. Formation does not command plan-slot robots. The Driver decides what skill each plan-slot robot runs and with what parameters.
- **Waypoint success/failure.** The Planner only sees the Driver's report: ongoing, succeeded, or failed-with-reason.
- **Dynamic parameter adjustment.** The Planner specifies intent (pass to left wing area). The Driver continuously refines the concrete parameters (exact kick angle, power, timing) based on the evolving world state.

The Driver can control **up to two robots simultaneously** for coordinated tasks. This is always a short, well-defined coordinated action — most importantly, the passing sequence where both a passer and a receiver need Driver-level control. Single-robot waypoints (capture, dribble, shoot) use one plan slot. The pass waypoint uses two plan slots during its commit-and-execute phase.

### Failure Reporting

The Driver reports one of three states each tick:

**Ongoing.** The waypoint is executing normally. The Planner does nothing.

**Succeeded.** The ball-state transition completed. The Planner advances to the next waypoint or replans.

**Failed with reason.** The waypoint cannot be completed. The reason matters because the Planner should respond differently:
- *Possession lost:* Ball is loose or with the opponent. Planner needs a capture plan.
- *No progress:* The approach isn't working but we haven't lost control. Planner might try a different robot or waypoint type.
- *No receiver:* Formation couldn't supply a robot where the plan needed one. Planner should pick a different passing target or fall back to dribble/shoot.
- *Ball moved:* The ball left the area of engagement (cleared, deflected far away). Planner needs to reassess from the new ball state.
- *Timeout:* The waypoint took too long. Planner should try a different approach.

The Driver should handle tactical adjustments silently — shifting angles, adjusting approach vectors, waiting briefly for a lane to open. It reports failure only when it has genuinely exhausted its tactical options for the current waypoint. A momentary lane closure isn't a failure; a sustained one where the situation is deteriorating is.

The danger: a Driver that's too eager to fail (reports at every hiccup, causing replan cascades) or too stubborn (keeps trying a doomed approach, wasting time). Timeouts and progress metrics should be calibrated per-waypoint-type. The right patience level for a steal attempt is very different from a pass. A steal against a fast opponent might fail within a few hundred milliseconds; a pass might wait over a second for a receiver to arrive.

### Waypoint Realization

#### Capture

The Driver activates movement/intercept skills to drive the plan-slot robot toward the ball. If the ball is with an opponent, the robot attempts to intercept from a favorable angle (ahead of the opponent, not chasing from behind). If the ball is loose, drive directly to it. If the ball is in flight, move to the predicted arrival area.

The Driver monitors progress: is the distance to the ball decreasing? Is the robot achieving a better interception angle? If neither is happening over a sustained period (tunable, a few hundred ms), the waypoint fails with "no progress."

**Key requirement from the original brief:** If a capture with robot X against opponent Y just failed due to no-progress, the Planner should prefer a different robot for the next attempt to avoid infinite loops. This is a Planner-level concern, not a Driver concern — the Driver just reports the failure honestly.

#### Dribble

The Driver activates the dribble skill toward the target area. It handles high-level obstacle avoidance — if an opponent is directly in the path, the Driver adjusts the dribble direction while still heading generally toward the target. The low-level dribble skill handles ball control; the Driver handles strategic direction.

Success: ball is within the target area and under our control. Failure: possession lost, timeout, or no progress (cornered, blocked).

#### Shoot

The Driver positions the robot for the best available shot angle and activates the kick skill aimed at the goal target. It may adjust the robot's position slightly before kicking to improve the angle.

The waypoint "completes" when the shot is taken — the plan resets regardless of whether the goal goes in, since the ball state has fundamentally changed. If the robot cannot achieve a shot angle within a timeout (defenders closing in, angle too tight), the waypoint fails.

#### Pass (Detailed Below)

The pass waypoint is the most complex because it requires coordinating two robots across the Formation/Driver boundary with precise timing and an irreversible physical event (the kick). It has its own section.

---

## The Pass Mechanic

### Why the Pass Is Special

The pass is the only waypoint that fundamentally requires two robots to cooperate, with the Driver controlling both during the critical phase. It involves an irreversible event (the kick) and a handoff of ball control from one robot to another. The boundary between Formation-managed positioning and Driver-controlled execution must be crossed twice — once to claim the receiver, once to release the passer — in the middle of a time-critical sequence.

### The Five Phases of a Pass

#### Phase 1: Receiver Positioning

The Planner has decided on a pass to a target area. The plan context generates a formation role at that area. Formation assigns a robot and drives it there using the movement skill. The Driver manages the passer (one plan slot) — holding the ball, shielding from opponents, buying time.

During this phase, the receiver is purely a Formation robot. It has no awareness that it's a receiver. It's simply moving to a position that Formation has determined is valuable (because the plan context role has high importance). The Driver observes the receiver's position but does not command it.

**Duration:** Variable. Depends on how far the nearest robot is from the target area. Could be near-instant (a robot was already there) or up to a second or more (robots need to reposition).

**What can go wrong:** Formation might reassign the robot moving toward the target area. If a higher-priority role appears (critical defensive need), Formation pulls the robot away and sends a different one — or leaves the plan context role unfilled. The Driver must not track a specific receiver robot during this phase. It tracks the position: is any formation robot near the target area? It doesn't care which one.

#### Phase 2: Commit Decision

The Driver continuously evaluates whether the pass is viable: Is a formation robot in or near the target area? Is the passing lane open (no opponent directly between passer and receiver)? Is the receiver reasonably stationary (not still in transit)? Have these conditions been stable for a sufficient duration (not a momentary gap in coverage)?

The stability requirement is critical. The Driver's commit should require sustained viability — the lane has been open and the receiver has been in position for at least ~100-150ms — before committing. This isn't hysteresis; it's a genuine requirement that conditions be stable enough to execute a pass that takes time to set up and is irreversible once the ball is kicked.

If conditions never become viable (the formation can't get a robot to the target area, the lane is persistently blocked), the waypoint fails with "no receiver" after a timeout. This is the signal for the Planner to try a different target or fall back to dribble/shoot.

#### Phase 3: Preparation (Two Plan Slots)

The Driver commits. At this moment:

1. The Driver claims the receiver robot as a second plan slot. "The receiver" is whichever formation robot is currently nearest the target area. The Driver requests the robot by position, not by identity — "give me the formation robot nearest to this point." If Formation just reassigned the robot during the last recalculation, the Driver gets whoever is there now.

2. The plan context role for this pass target is retired. It is now redundant — the receiver is under Driver control at the target position. If the plan context role persists alongside the plan slot, Formation might send a second robot to the target area, wasting a formation slot. The lifecycle must be crisp: claiming the receiver retires the corresponding plan context role.

3. The Driver activates the receive skill on the receiver robot. The receive skill fine-tunes the robot's position, spins up the dribbler, and orients toward the incoming ball. This is the preparation that benefits from the 200-300ms lead time before the kick.

4. The Driver simultaneously begins kick preparation on the passer. Both robots are now under Driver control, preparing in parallel.

5. Formation drops to N-2 regular formation robots for the duration of this phase (both the passer and receiver are plan slots). Formation recalculates with the new configuration. Three robots cover all formation roles. This is a real reduction in formation coverage and the primary cost of executing a pass.

**Duration:** Short and bounded. The preparation window should have a maximum duration (~300-500ms). If the Driver can't find a good kick moment within this bound (lane closes, receiver gets bumped), it aborts.

**Abort procedure:** The Driver kills the receive skill on the receiver, releases the receiver's plan slot, and reports failure. The plan context role should be re-established so Formation begins positioning a receiver again for a potential retry. The passer remains in its plan slot, still holding the ball. The Planner receives the failure and decides whether to retry the pass, try a different target, or fall back.

**Abort risk: commit-abort oscillation.** If the Driver keeps committing and aborting because the lane keeps almost-opening and almost-closing, the receiver robot constantly switches between receive skill and formation movement. The sustained-viability requirement in Phase 2 mitigates this: the Driver doesn't commit until conditions have been stable, reducing the chance that they collapse during the short preparation window. If aborts are still frequent, the Planner should detect repeated failures and switch strategy.

#### Phase 4: The Kick and Ball In Flight

The Driver fires the kick. The ball is now in flight. This is irreversible.

Immediately after the kick:
1. The Driver releases the passer's plan slot. The passer returns to Formation control. Formation recalculates with N-1 regular robots plus one plan slot (the receiver). The passer gets assigned to whatever role best fits its current position.
2. The Driver monitors the ball trajectory. It tracks whether the ball is heading toward the receiver.
3. The receiver continues running the receive skill, tracking the incoming ball.

**Duration:** Brief — ball flight over typical passing distances lasts well under a second at SSL ball speeds.

**What can go wrong during flight:**
- *Opponent intercepts the ball.* The ball changes trajectory or stops progressing toward the receiver. The Driver detects this and immediately kills the receive skill, releases the receiver's plan slot, and reports failure with "possession lost."
- *Ball goes out of bounds.* Similar — Driver detects the ball leaving the field, releases the receiver, reports failure with "ball moved."
- *Ball deflects but continues roughly toward receiver.* The Driver should be tolerant of minor trajectory changes. The receive skill can adapt to slightly off-target balls. Only a clear deviation (ball heading in a fundamentally different direction) should trigger failure.

#### Phase 5: Resolution

**Success path:** The receiver gains clean possession of the ball. The Driver detects this (ball is under receiver's control). The pass waypoint succeeds. The receiver's plan slot persists — it becomes the active robot for the next waypoint. The Driver reports success to the Planner. The Planner produces the next waypoint (likely shoot or another pass), and the Driver begins executing it with the receiver as its plan-slot robot.

**Failure path — fumble:** The ball arrives near the receiver but the receiver doesn't secure it. The ball is bouncing or rolling nearby. The receive skill is still trying to collect it. The Driver gives it a short window (~300ms). If the receiver gains possession within that window, success. If not, the Driver kills the receive skill, releases the receiver, and reports failure with "possession lost."

**The over-eager receive skill:** The receive skill is designed to be persistent — it keeps trying to receive even in deteriorating conditions. This is correct behavior for the skill in isolation, but it means the Driver must actively manage the skill's lifecycle. The receive skill never decides to stop on its own. The Driver is the sole authority on when to kill it. This requires the Driver to actively monitor the reception and make the judgment call: is reception still plausible? The moment the Driver decides no, it kills the skill. The skill's eagerness is an asset (it doesn't give up on difficult receptions) managed by the Driver's judgment (it knows when to cut losses).

### Pass Timing Summary

| Phase | Duration | Plan Slots | Formation Robots |
|-------|----------|------------|------------------|
| 1. Receiver positioning | Variable (up to ~1s) | 1 (passer) | N-1 |
| 2. Commit decision | Concurrent with Phase 1 | 1 (passer) | N-1 |
| 3. Preparation | ~200-300ms | 2 (passer + receiver) | N-2 |
| 4. Ball in flight | <500ms typically | 1 (receiver) | N-1 |
| 5. Resolution | <300ms | 1 (receiver) or 0 | N-1 or N |

The two-plan-slot window (Phase 3) is the most constrained period. It should be as short as possible — commit, prepare, kick. If it extends (Driver waiting for the perfect moment), Formation is running thin. The bounded preparation timeout prevents this from becoming open-ended.

### Lifecycle of the Plan Context Role

The plan context role for a pass target has a specific lifecycle that must be managed precisely to avoid interference:

1. **Created:** When the Planner produces a pass waypoint targeting an area. The plan context role appears at that area with high importance.
2. **Active:** Formation assigns a robot to the role. The robot moves to the target area. The Driver observes but does not command this robot.
3. **Retired:** When the Driver commits and claims the receiver as a plan slot (Phase 3). The plan context role is removed. Formation no longer tries to fill it.
4. **Re-established on abort:** If the Driver aborts the preparation phase, the plan context role is re-created so Formation begins positioning a receiver again.
5. **Permanently removed:** When the pass succeeds (the receiver has the ball and the plan context is no longer relevant) or when the Planner replans with a different waypoint.

**Critical invariant:** At no point should both the plan context role and the receiver's plan slot exist simultaneously at the same target area. This would cause Formation to assign a second robot to the area, wasting a formation slot and potentially interfering with the reception.

---

## Plan-Formation Coupling

### How the Plan Communicates with Formation

The plan communicates with Formation through exactly two mechanisms:

**Plan context roles.** Created by the plan to request formation support. "I need a robot at left wing for a pass." Formation treats this as a high-priority role and assigns a robot. The plan observes whether and when the role is filled.

**Plan slots.** Created by the Driver to reserve a robot for direct control. "This robot is now under my command." Formation sees the plan slot as an occupied position and doesn't try to command the robot.

These are separate mechanisms that must not cross. Plan context roles affect formation's management of non-plan robots. Plan slots reserve robots from formation control. If a plan context role influences a plan-slot robot's behavior (Formation trying to "help" the Driver), two systems fight over the same robot.

**The rule:** Plan context creates roles for formation robots. Plan slots reserve robots from formation control. Plan context does NOT affect plan-slot robots. The Driver has full authority over plan-slot robots. Formation has full authority over all other robots.

### How Formation Informs the Plan

Formation provides implicit feedback to the plan through the observable positions of formation robots. When the plan creates a plan context role, it can observe whether a formation robot arrives at the target area. If no robot arrives within a reasonable time, the Driver's commit decision (Phase 2 of the pass) will fail, and the waypoint will fail with "no receiver." The Planner interprets this as "the formation couldn't support this pass" and adapts.

Formation does not explicitly signal the plan. There is no "robot ready" notification. The Driver observes robot positions directly and makes its own judgments.

### The Steal Boundary

The steal (capture waypoint against an opponent with the ball) is where the plan-formation boundary is most subtle. Consider:

B1 is in Formation, positioned between opponent O2 and our goal — a high-value defensive position placed by Formation's shadow or marking roles. The Planner decides to steal from O2. The natural choice is B1 — it's nearest, already facing the right direction.

When the Driver creates a plan slot for the steal, B1 becomes the plan-slot robot. Formation recalculates with N-1 regular robots. The defensive position B1 occupied is now unserved. Another robot (B4) may be reassigned to cover it, but B4 was doing something else useful.

**Key insight from the design:** Because B1's plan slot is visible to Formation at B1's actual position, and B1 is moving toward O2 (who is near B1's original defensive position), Formation still "sees" coverage in roughly that area. The gap only becomes significant when B1 moves far from its original position during the chase. This is a natural consequence of committing to a steal — you accept a temporary defensive cost for the chance of winning the ball.

If the steal fails and B1 returns to Formation, it's assigned to whatever role fits its current position. If B1 ended up in an advanced position from the chase, the Planner might recognize B1 as a good passing target for the next attempt. The system exploits the consequences of failure without special logic.

---

## Interaction Between Driver and Formation: Complete State Machine

### Robot Lifecycle States

At any moment, each field robot is in exactly one of these states relative to the Driver/Formation boundary:

**Formation-controlled.** The robot is assigned to a formation role. Formation commands its movement skill. The Driver does not interact with it (though the Driver may observe its position).

**Plan-slot (Driver-controlled).** The robot is reserved by a plan slot. The Driver commands its skill activations. Formation sees it as an occupied position but does not command it.

### Transitions

**Formation → Plan slot:** The Driver creates a plan slot and Formation's assignment places a robot in it. This happens atomically during a formation recalculation. The robot's skill switches from formation's movement skill to whatever the Driver commands.

**Plan slot → Formation:** The Driver releases the plan slot. The robot becomes a regular formation robot at its current position. On the next recalculation, it's assigned to whatever role is cheapest to fill from its current position. Its skill switches back to formation's movement skill.

Both transitions are discrete events. They cause a formation recalculation (subject to cooldown). The robot count in formation never changes due to these transitions — a plan slot is a role in the assignment, not a removal of a robot.

### Sequencing Within a Tick

When a replan trigger fires, the layers execute in this order within a single tick:

1. **Driver reports** waypoint status (ongoing/succeeded/failed).
2. **Planner runs** if a replan trigger occurred (waypoint success/failure, no plan exists). Produces new waypoint sequence. Defines new plan context and plan slot requirements.
3. **Driver updates.** New waypoint means new Driver state. Old plan slots may be released, new ones requested.
4. **Formation recalculates.** Sees the current set of plan slots and plan context roles. Assigns all N robots to roles (including plan slots). Outputs targets for formation-controlled robots.
5. **Skill activations produced.** Driver produces skill activations for plan-slot robots. Formation produces movement skill activations for formation-controlled robots.

This ordering ensures that within a single tick, the Planner's new plan, the Driver's slot management, and Formation's assignment are all consistent. There is no intermediate state where Formation has the "wrong" number of robots or stale plan context.

---

## Detailed Case Studies

### Case Study A: Clean Pass Sequence

**State:** 5 field robots. B3 has the ball at midfield. Planner has decided: pass to left wing area, then shoot. B3 is in a plan slot (passer). Formation has 4 regular robots.

**Tick 0.** Plan context role created at left wing with high importance. Formation recalculates. B5, nearest to left wing, is assigned the plan context role. B5 starts moving toward left wing. Other robots cover shadow, marking, and support roles.

**Ticks 1-35.** B5 approaches left wing. Driver holds B3 (passer), shielding the ball, dribbling gently to maintain possession. Driver observes B5's position each tick. Passing lane is monitored.

**Tick 36.** B5 is within the target area. Lane is open. Driver's viability check: B5 has been in position for ~100ms. Lane has been open. Conditions stable. Driver commits.

**Tick 37.** Driver claims B5 as a second plan slot. Plan context role at left wing is retired. Receive skill activated on B5 — dribbler spinning up, position fine-tuning, orienting toward B3. Kick preparation begins on B3. Formation recalculates with N-2 regular robots (3 robots covering remaining roles).

**Tick 44 (~120ms later).** Preparation complete. Driver fires kick on B3. Ball launched toward B5.

**Tick 45.** Driver releases B3's plan slot. B3 returns to Formation. Formation recalculates: 4 regular robots, 1 plan slot (B5). B3 is assigned to the cheapest available role from its current position (likely a midfield support or shadow role).

**Ticks 45-51.** Ball in flight. B5 tracks it with receive skill. Driver monitors trajectory — ball heading toward B5, no interception.

**Tick 52.** B5 collects the ball. Driver detects possession. Pass waypoint succeeds. B5 remains in plan slot as active robot. Driver reports success.

**Tick 53.** Planner runs. B5 has the ball on the left wing, near the goal. Clear shot? Yes — far post is open. Plan: shoot(far post). Driver begins shoot execution with B5.

### Case Study B: Pass Abort and Recovery

**State:** B2 has the ball. Planner wants to pass to right wing. B4 is moving toward the right wing area (assigned to the plan context role).

**Tick 20.** B4 is near the target area. Lane is open. Driver's viability check passes. Driver commits: claims B4 as plan slot, activates receive skill.

**Tick 22.** During preparation, opponent O3 steps into the passing lane. The lane is now blocked. Driver detects this.

**Tick 24.** Driver aborts. Kills receive skill on B4. Releases B4's plan slot. Re-establishes the plan context role at right wing (so Formation will send a robot there again for potential retry). Driver reports waypoint failure with "lane blocked."

**Tick 25.** Planner replans. B2 still has possession. Right wing pass was blocked. Planner evaluates alternatives: left wing has an open lane with B1 heading there. New plan: pass to left wing area, then shoot.

**Formation recalculates.** B4 (released from plan slot) is back in Formation at the right wing position. It gets assigned to whatever role fits — perhaps the newly unimportant right wing plan context role disappears (the plan changed targets), and B4 gets a marking or support role. B1 approaches the new plan context role at left wing.

**What went right:** The abort was clean. Formation absorbed B4 back without cascade. The Planner got a specific failure reason and adapted. B2 held the ball throughout. The pass context role lifecycle was managed correctly — retired on claim, re-established on abort, retired again when the plan changed targets.

### Case Study C: Steal Transition to Counter-Attack

**State:** Opponent O2 has the ball near midfield. Formation has B1 between O2 and our goal (shadow role), B4 marking O3 on the wing, B2 and B5 in support positions.

**Tick 0.** Planner: capture(steal from O2). Driver creates plan slot near O2's ball position. Formation assigns B1 (nearest). B1 transitions to plan slot. Intercept skill activated. Formation recalculates: B4, B2, B5 cover remaining roles. The shadow role B1 was covering goes to B2 (next cheapest).

**Ticks 1-18.** B1 closes on O2. Driver monitors progress — distance decreasing, good approach angle. B2 covers the defensive gap.

**Tick 19.** B1 wins the ball. Capture waypoint succeeds. B1 now has possession in a plan slot at midfield.

**Tick 20.** Planner replans. B1 has ball at midfield. Clear shot? No, too far. Best pass option? B5 is in the opponent half (offensive support role). Planner: pass to B5's area, then shoot. Plan context role created at B5's area. But B5 is already there — Formation has it in an offensive support role that overlaps with the plan context role position. The plan context role effectively confirms B5's current position, so B5 barely moves.

**Tick 21.** Driver evaluates pass viability. B5 is already in position. Lane is open. Commit almost immediately.

**Tick 22.** Driver claims B5. Pass preparation. Two plan slots (B1 passer, B5 receiver). Formation: B4, B2, plus whoever else cover 3 formation roles.

**Tick 26.** Kick. B1 releases. Ball in flight to B5.

**Tick 31.** B5 receives. Pass succeeds. B5 has ball near the opponent goal. Planner: shoot.

**What this validates:** The steal-to-pass transition was seamless. B1 went from Formation (shadow role) to plan slot (steal) to plan slot (passer) without ever returning to Formation in between. The Planner's replan after the steal naturally led to a counter-attack because B5 was already positioned forward by Formation's offensive support role.

### Case Study D: Failed Steal and Recovery

**State:** Opponent O2 has the ball. B1 is pursuing the steal (plan slot).

**Tick 15.** O2 passes to O4 on the wing. Ball moves far from B1. Driver detects ball moved. Capture waypoint fails with "ball moved."

**Tick 16.** Plan slot released. B1 returns to Formation at its current position (advanced midfield, from the chase). Planner replans: O4 has ball on wing. New steal target: O4. B4 is nearest to O4 (was marking O3 nearby). New plan slot for B4.

**Formation recalculates.** B1 is back, B4 is claimed. Net zero change in regular robots. B1 gets assigned to whatever role fits its advanced position — possibly an offensive support role, since it ended up in opponent territory. B4 activates intercept skill toward O4.

**Tick 30.** B4 wins the ball from O4. Capture succeeds. Planner replans: B4 on the wing with ball. Pass to B1 in central position (B1's aggressive position from the failed steal is now a passing target). Plan: pass to B1's area, shoot.

**What this validates:** Failed steal → clean recovery → new steal → success → exploitation of the first robot's advanced position. No cascade at any transition. The system turned a failed steal into an advantage without any special logic.

### Case Study E: Robot Lost During Attack

**State:** B3 has the ball, executing a dribble waypoint. B5 is in a plan context role at left wing (preparing for a pass). B1, B2, B4 in Formation covering defense and support.

**Tick 10.** B2 suffers a hardware failure. Robot goes offline. Formation detects this on the next recalculation.

**Formation recalculates.** 4 field robots remaining. B3 is in a plan slot (dribbling). B5 is in a plan context role. B1 and B4 cover the remaining regular roles. One fewer shadow/marking role can be served. The formation is thinner defensively.

**The Driver doesn't know or care.** B3 continues dribbling. B5 continues moving to the pass target. The Driver's pass viability check doesn't change — it's still watching for a receiver at the target and an open lane.

**Tick 25.** B3 reaches the dribble target. Dribble waypoint succeeds. Planner replans. With only 4 field robots, the plan might be different — the reduced defensive cover means the Planner might prefer a faster, more direct attack (dribble-shoot rather than pass-shoot) to minimize the time spent with thin coverage.

**What this validates:** Hardware failure is absorbed by Formation without affecting the plan layer. The Driver operates on the robots it has, Formation covers the rest as best it can with reduced numbers. The Planner naturally adapts its strategy to the reduced team (though this is planner intelligence, not formation logic).

### Case Study F: Pass to Nonexistent Receiver

**State:** B2 has the ball. Planner wants to pass to right wing. Plan context role created. But all formation robots are far from the right wing — B4, the nearest, is on the left side of midfield.

**Tick 0.** Formation assigns B4 to the plan context role. B4 starts moving toward right wing — a long journey.

**Ticks 1-50.** B4 traverses the field. Driver holds B2, evaluating pass viability each tick. No robot is near the target area. Viability check fails repeatedly.

**Tick 55.** B4 is getting closer but an opponent is pressuring B2. B2 is at risk of losing possession.

**Tick 65.** Driver's pass waypoint timeout expires. No receiver arrived in time. Waypoint fails with "no receiver."

**Planner replans.** B2 still has possession but under pressure. Right wing isn't covered. Direct shot? Marginal, but better than losing the ball. Plan: shoot (take whatever angle is available).

**What this validates:** The pass execution correctly waits for a receiver and correctly fails when one doesn't arrive in time. Formation-plan coupling isn't guaranteed to work — sometimes Formation can't supply a robot fast enough. The system degrades gracefully into a simpler action.

### Case Study G: The No-Progress Trap

**State:** Opponent O1 has ball near midfield. B1 is pursuing the steal (plan slot). But O1 is faster and keeps dribbling laterally, never advancing, never losing the ball.

**Ticks 0-25.** B1 chases O1. Distance isn't decreasing. Approach angle isn't improving. B1 is running alongside O1 with no prospect of winning the ball.

**Tick 26.** Driver's no-progress detection fires. Waypoint fails with "no progress."

**Tick 27.** Planner replans. O1 still has ball. B1 just failed against O1 due to no-progress. Planner should prefer a different robot — B4 has a different approach angle and might succeed. Plan: capture(steal from O1) — but the system places the plan slot based on O1's position, and B4 is the cheapest robot to assign.

**If B4 also fails:** No-progress again. Planner has now had two robots fail to steal. At this point, the planner can still select B1 or B4 again (they may have better angles now that O1 has drifted), or Formation's natural positioning provides containment — shadow and marking roles cut off O1's passing options and narrow the dribbling corridor. O1 will eventually advance (entering a higher-threat area where our positioning is stronger) or pass (changing the game state and triggering a fresh plan), or make a mistake.

**What this validates:** No-progress detection prevents deadlock. The Planner avoids immediately repeating a failed approach. Formation's passive positioning provides a containment strategy without any explicit "press" logic — it falls out of shadow and marking roles constraining the opponent's options.

---

## Known Issues and Risks

### Importance/Redirect Cost Calibration

The relative weighting between role importance and redirect cost is the primary tuning surface. Miscalibration symptoms:
- **Importance too dominant:** Frequent reassignment. Robots redirected across the field for marginal importance gains. The formation looks hyperactive.
- **Redirect cost too dominant:** Sluggish adaptation. Critical defensive roles go unfilled because filling them would require redirecting a robot. The formation looks lazy.

Starting point: 1 importance point ≈ 300-500ms of redirect time. Adjust based on observed behavior in the simulator.

### Shadow Generator Coordination

The shadow generator produces internally-coordinated positions, which is qualitatively different from other generators. This is necessary for goal coverage but should not become a pattern. If future generators need internal coordination, each new instance adds hidden coupling that's difficult to reason about. Prefer independent roles wherever possible and use coordinated generators only where the joint optimization is well-defined and geometrically tractable (as it is for goal coverage angles).

### Spread and Coverage Gaps

The role-based system does not guarantee global field coverage. Pathological opponent formations could cause all high-importance roles to cluster in one area. For the MVP, the residual/coverage roles and the diversity of generators should prevent the worst cases. If testing reveals persistent gaps, add a minimum-spread constraint to the assignment: no two formation robots assigned closer than a threshold unless all available roles are clustered. This is one parameter, not a tuning surface.

### Commit-Abort Oscillation During Passes

If the passing lane flickers open and closed, the Driver might repeatedly commit (claim receiver, start preparation) and abort (release receiver), causing the receiver to switch between receive skill and formation movement. The sustained-viability requirement (lane must be open for ~100-150ms before committing) mitigates this. If it persists, the Planner should detect repeated pass failures and switch strategy (dribble or shoot instead). The architecture supports this through the failure reason: repeated "lane blocked" failures signal a pattern the Planner can respond to.

### Two-Plan-Slot Duration

During Phase 3 of a pass, two robots are in plan slots and Formation is reduced to N-2. If the preparation window extends (Driver can't find a good kick moment), Formation is running thin. The bounded preparation timeout (300-500ms max) prevents this from becoming open-ended. If the timeout is frequently hit, the viability check in Phase 2 may need to be more selective (don't commit unless conditions are very clearly favorable).

### Plan Context Role Lifecycle

The lifecycle of plan context roles (created by the plan, retired when the Driver claims a receiver, re-established on abort, removed on replan) has several transitions that must be managed atomically. A missed transition causes either a phantom role (Formation sends an extra robot to an area) or a missing role (Formation doesn't know the plan needs a receiver). The sequencing rules in the Interaction section prevent this, but the implementation must respect the ordering strictly.

### Receiver Identity Instability

During Phase 1, the Driver doesn't track a specific receiver — it watches the position. If Formation reassigns the robot heading toward the target (because a higher-priority role appeared), a different robot may end up as the eventual receiver. This is correct behavior (Formation is making a globally-informed decision) but it means the Driver cannot pre-compute anything about the receiver's approach trajectory or readiness. The Driver must evaluate the commit decision fresh each tick based on whoever is currently nearest the target.

---

## Glossary

**Role:** A function from world state to a position and importance score. The atomic unit of Formation's positioning system.

**Role generator:** A function that produces zero or more roles based on the current world state. Each generator encapsulates one tactical concern (goal coverage, opponent marking, etc.).

**Plan context role:** A role generated by the plan's requirements. Used to request Formation support (e.g., positioning a receiver for a pass).

**Plan slot:** A special role that reserves a robot for Driver control. The robot assigned to a plan slot uses Driver-commanded skills instead of Formation's movement skill.

**Redirect cost:** The estimated time for a robot, given its current position and velocity, to reach a role's position. Used as the primary cost metric in assignment to produce stability from physical reality.

**The Driver:** The layer between the Planner and the skills. Realizes waypoints by producing skill activations, adapting parameters dynamically, and reporting success/failure.

**Waypoint:** A desired ball-state transition. The fundamental unit of the plan. Types: capture, dribble, pass, shoot.

**Commit (pass):** The moment the Driver decides to execute a pass, claims the receiver as a plan slot, and begins preparation. Irreversible in intent (though abortable before the kick).

**Preparation window:** The bounded time between pass commit and kick, during which both passer and receiver are under Driver control preparing for the pass.
