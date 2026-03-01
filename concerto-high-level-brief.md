# Concerto Strategy System — MVP Project Brief

## Overview

Concerto is a new strategy architecture for our RoboCup SSL team. It replaces monolithic strategy logic with a two-layer system: a continuous positioning layer called Formation, and a plan-execute-replan loop that drives offensive ball movement. The system implements the standard Dies strategy interface — it consumes world state and produces skill activations for each robot.

The core philosophy is "offense is the best defense." The system always has an offensive plan. There is no separate defensive mode. When we don't have the ball, the plan is to take it and score. Formation handles positioning for all robots at all times, and the plan influences Formation's behavior so that robots naturally support whatever offensive sequence is underway.

This document describes the MVP: the minimum implementation needed to validate that the architecture produces intentional, coordinated offensive play. The MVP is deliberately simple. The planner is a small set of hardcoded waypoint sequences. There is no tree search, no world model, no generator composition. Those come later. The MVP exists to prove that the replan loop, the waypoint abstraction, the Formation layer, and the coupling between them all work correctly together.

---

## Design Principles

**Always aggressive.** The planner always produces a plan that ends with a shot on goal. Even when the opponent has the ball, the plan starts with "steal the ball" and proceeds toward their goal. There is no idle state, no "wait and see" mode.

**Minimize discrete switching.** The system avoids binary mode transitions like "offense vs defense." Formation runs continuously and smoothly adjusts to changing conditions. Plan selection should favor continuity — don't abandon a working plan because a slightly better one appeared.

**Waypoints are ball state transitions.** The planner thinks exclusively about moving the ball. It does not think about robot assignments or trajectories. Waypoints describe what should happen to the ball (captured, moved to an area, passed to an area, shot at goal), and the execution and Formation layers figure out how to make that happen with actual robots.

**Replan constantly.** After every completed or failed waypoint, the planner runs again with the current world state. Plans are cheap and disposable. The system never commits to a long sequence — it commits to one waypoint at a time and reconsiders after each.

**Formation is always running.** Every robot is always under Formation control. When a robot is executing a waypoint (the "active robot"), its Formation assignment is overridden by the execution layer, but it conceptually still has a Formation slot. When the waypoint completes or fails, the robot seamlessly returns to Formation behavior with no special handoff logic.

---

## Architecture

### Layer 1: Formation

Formation is responsible for positioning all robots on the field at all times. It runs every tick. It assigns each robot a target position based on a scoring function that evaluates how valuable each position on the field is given the current game state.

#### Scoring Function

The scoring function takes a position on the field and the current world state (including the current plan context, described below) and returns a scalar score. Higher scores mean more valuable positions.

The MVP scoring function considers the following factors, combined as a weighted sum:

**Goal coverage.** Positions between the ball and our goal that block opponent shooting lanes score highly. This is the baseline defensive behavior. Weight these positions by the actual threat level — an opponent near our goal with the ball is a high threat, an opponent in their own half is not. This factor ensures that even while attacking, some robots stay in positions that prevent easy counter-goals.

**Opponent marking.** Positions near opponent robots (but between them and our goal) score well, particularly for opponents in dangerous areas. This doesn't need to be tight man-marking — proximity within a reasonable radius is sufficient. The effect is that Formation naturally distributes robots to cover threats.

**Offensive support.** Positions in the opponent half that have clear lines of sight to the opponent goal score well, especially if they are not near opponent robots. These are "open for a pass" positions. This factor ensures that even the default formation creates passing options.

**Plan context bonus.** When the current plan includes a pass waypoint with a target area, that area gets a large scoring bonus. This is how Formation supports the plan — it drives a robot toward the area where the planner wants to send the ball. This bonus should be strong enough to override other factors for at least one robot, ensuring a receiver is actually placed where the plan needs one.

**Spread.** A penalty for positions that are too close to other assigned teammates. This prevents clumping and ensures field coverage. Can be implemented as a repulsive term based on distance to nearest assigned teammate position.

**Goalkeeper exclusion.** The goalkeeper is not part of Formation. It runs its own dedicated positioning logic (stay on the goal line, track the ball, cover the angle). Formation manages the remaining field players.

#### Position Assignment

Each tick, Formation computes a set of candidate positions (or evaluates a grid/sample of positions across the field) and assigns robots to positions using a one-to-one matching that maximizes total score. The simplest approach: greedily assign each robot to the highest-scoring unoccupied position, with some bias toward each robot's current assignment to prevent unnecessary swapping.

Assignment stability matters. If two positions have similar scores, robots should not constantly swap between them. A robot's current target position should have a small bonus (hysteresis) to prevent jitter.

The active robot (the one currently executing a waypoint) is excluded from Formation assignment. Its position is controlled by the execution layer. Formation assigns positions to all other field players.

#### Formation Output

Formation produces a target position for each non-active field player. These are fed into the movement skill. Robots drive toward their Formation target at all times unless overridden by the execution layer.

---

### Layer 2: Plan-Execute-Replan Loop

This layer is responsible for moving the ball toward the opponent goal. It consists of three components: the Planner, which selects a sequence of waypoints; the Execution layer, which drives the active robot through the current waypoint; and the Replan trigger logic, which decides when to invoke the planner again.

#### Waypoints

A waypoint represents a desired ball state transition. It is the fundamental unit of the plan. The MVP defines four waypoint types:

**Capture.** The ball transitions from not-controlled-by-us to controlled-by-us. This covers three sub-situations: the ball is loose and we pick it up, an opponent has the ball and we steal it, or the ball is in flight (e.g., after a deflection) and we intercept it. The waypoint specifies which robot should attempt the capture (chosen by the planner based on proximity and feasibility). Success condition: the designated robot has possession of the ball. Failure conditions: a timeout expires without gaining possession, or the ball moves far from the expected area (e.g., opponent cleared it), or the designated robot cannot make progress toward the ball (stuck, blocked, or the opponent is faster).

**Dribble.** The ball transitions from its current position to a target area while remaining under our control. The active robot carries the ball toward the target. The waypoint specifies a target area (not an exact point — a region on the field). Success condition: the ball is within the target area and still under our control. Failure conditions: possession is lost, a timeout expires, or the robot is unable to make progress toward the target (blocked by opponents, cornered).

**Pass.** The ball transitions from controlled-by-the-active-robot to a target area where a teammate should receive it. The waypoint specifies only the target area. It does not specify which robot receives — that is determined by whoever Formation places in that area. Before committing to the kick, the execution layer must verify that a teammate is in or near the target area and is in a position to receive. If no receiver is available within a time window, the waypoint fails. After the kick, the ball is in flight. The waypoint tracks the ball until either a teammate receives it in roughly the target area (success) or an opponent intercepts, the ball goes out of bounds, or no teammate receives within a reasonable time (failure). On success, the receiving robot becomes the new active robot.

**Shoot.** The ball transitions from controlled-by-the-active-robot toward the opponent goal. The waypoint specifies a target on the goal (far post, near post, center — or just "goal" for the MVP). The execution layer positions the robot for the shot and kicks. Success condition: goal scored (but for planning purposes, the waypoint "completes" when the shot is taken — the plan resets regardless of whether the goal goes in, since the ball state has fundamentally changed). Failure conditions: the robot cannot find a shooting angle within a time window, or possession is lost before the shot.

#### The Planner

The planner is a function that takes the current world state and returns a sequence of waypoints. In the MVP, this is a simple decision function with a small number of hardcoded plans. There is no search, no optimization, no generator composition. The planner always returns a plan — it never returns "no plan."

The MVP planner selects between plans based on the current possession state and the active robot's position:

**We have possession and a clear shot.** If the active robot (or the robot with the ball) has a reasonably unobstructed line to the goal and is within shooting range, the plan is a single waypoint: shoot. "Clear shot" means no opponent robot is directly in the line between the ball and the goal target within a threshold distance, and the angle to goal is not too extreme. Be generous with this check for the MVP — if there's any reasonable chance, take the shot.

**We have possession but no clear shot.** The plan is: pass to the best available area, then shoot. "Best available area" is a simple heuristic — find the area near the opponent goal that is furthest from any opponent robot and has a reasonable passing lane from the current ball position. The plan is two waypoints: pass to that area, then shoot. If no reasonable passing target exists (all lanes blocked), fall back to: dribble toward the opponent goal (to create a better angle or draw defenders), then shoot.

**Ball is loose (no team has possession).** The plan is: capture with the nearest robot, then shoot (or pass-then-shoot depending on the capture position — apply the same logic as "we have possession" after the capture waypoint).

**Opponent has possession.** The plan is: capture (steal) with the robot best positioned to challenge the ball carrier, then proceed as above. "Best positioned" considers distance to the opponent ball carrier, approach angle (prefer intercepting from the front or side, not chasing from behind), and whether the robot is between the opponent and our goal (prefer robots that are already in a defensive position so the steal attempt doesn't leave us exposed).

The planner also identifies which robot should be the active robot for the first waypoint. For capture waypoints, this is based on proximity and positioning. For plans starting with pass/dribble/shoot, the active robot is whoever currently has the ball.

**Plan continuity.** If the current plan is still in progress (active waypoint is executing and hasn't failed), the planner should not replace it unless the situation has changed substantially. A simple heuristic: if the current plan's first waypoint is the same type and involves the same robot as what the planner would choose fresh, keep the current plan. Only replan when the current plan is stale — different robot should be active, or the possession state has changed.

#### Execution Layer

The execution layer takes the current waypoint and the world state and produces skill activations for the active robot. It is responsible for translating the abstract waypoint into concrete robot behavior.

For each waypoint type, the execution layer activates the corresponding skill:

**Capture execution.** Activate the movement skill to drive toward the ball. If the ball is controlled by an opponent, this involves moving to intercept the ball carrier — ideally getting ahead of them rather than chasing. If the ball is loose, drive directly to it. If the ball is in flight, move to the predicted landing/arrival area. Once close enough, transition to the dribble or ball-control skill to secure possession.

**Dribble execution.** Activate the dribble skill toward the center of the target area. The execution layer should handle obstacle avoidance at a high level — if an opponent is directly in the path, adjust the dribble direction slightly while still heading generally toward the target. The low-level dribble skill handles the ball control; the execution layer handles the strategic direction.

**Pass execution.** First, check whether a teammate is in or near the target area. If no teammate is present, wait briefly (hold the ball, perhaps dribble slowly toward the target area to buy time). If a teammate arrives, activate the kick skill aimed at the teammate's position (not the center of the target area — aim at the actual receiver). If no teammate arrives within a timeout, fail the waypoint. After the kick, track the ball. Identify the intended receiver as the teammate closest to the target area at the moment of the kick. Monitor whether that robot gains possession (success) or the ball is lost (failure).

**Shoot execution.** Position the robot for the best available shot angle (which may require a small positional adjustment before kicking). Activate the kick skill aimed at the goal target. If the robot cannot achieve a shot angle within a timeout (defenders closing in, angle too tight), fail the waypoint.

The execution layer also monitors for global failure conditions that apply to all waypoints: if the ball suddenly moves far away (cleared by an opponent, deflection), any active waypoint fails immediately.

#### Replan Triggers

The planner is invoked (and a new plan produced) when any of the following occur:

- The current waypoint succeeds. The plan advances. After advancing, replan from the new state. (This means even if the plan has more waypoints queued, we replan rather than blindly executing the next one. The queued waypoints informed our initial decision, but the fresh replan might choose differently.)
- The current waypoint fails. Replan from the new state.
- No plan exists yet (game start, after a goal, after a stoppage).

Between replan triggers, the current waypoint continues executing. The planner does not run every tick — only on these events.

---

### Putting It Together: The Main Loop

Each tick, the strategy system does the following:

First, determine the plan state. If a replan trigger has occurred, invoke the planner to get a new waypoint sequence and identify the active robot and current waypoint. If no trigger has occurred, continue with the existing plan and waypoint.

Second, run the execution layer. If there is an active waypoint, produce skill activations for the active robot based on the current waypoint and world state. Also evaluate whether the current waypoint has succeeded or failed (which will trigger a replan next tick).

Third, run Formation. Compute target positions for all field players except the active robot, using the scoring function with plan context (the current plan's pass target area, if any). For the goalkeeper, compute its dedicated positioning.

Fourth, produce the final output. For the active robot, output the skill activation from the execution layer. For all other field players, output movement skill activations toward their Formation-assigned target positions. For the goalkeeper, output its dedicated positioning skill activation.

This is the complete strategy interface: world state in, skill activations for all robots out.

---

## Detailed Case Studies

These case studies trace through the system's behavior in specific game situations. They are intended to clarify how the layers interact and to expose edge cases.

### Case Study 1: Clean Counter-Attack

**Situation.** Our robot B3 intercepts a stray pass at the midfield line. The ball is now loose, rolling slowly near B3. Two opponents are behind the ball (in their own half). Our robots B1 and B5 are in the opponent half, placed there by Formation's offensive support scoring.

**Tick N.** The ball becomes loose near B3. No active plan exists. Replan trigger: no current plan.

**Planner runs.** Possession state: ball is loose. Nearest robot: B3. Plan: capture(B3), then evaluate — B3 will be at midfield with two teammates ahead and few defenders. Planner produces: capture(B3), pass(area near opponent goal where B5 is roughly positioned), shoot.

**Active robot: B3. Current waypoint: capture.**

**Formation runs** with plan context "pass target: area near B5's position." Formation adjusts: B5 gets a bonus for being near the pass target area, so Formation keeps B5 there or nudges it into a better receiving position. B1 and other field players reposition to maintain coverage and offensive support.

**Execution runs capture waypoint.** B3 is very close to the ball. Movement skill drives B3 to the ball. Within a few ticks, B3 has possession.

**Tick N+12.** Capture waypoint succeeds. Replan trigger fires.

**Planner runs.** Possession state: we have possession (B3). B3 is at midfield. Clear shot? No, too far. Best passing target: area near B5, who Formation has positioned in a good receiving spot with an open lane. Plan: pass(B5's area), shoot.

**Active robot: B3. Current waypoint: pass to B5's area.**

**Execution runs pass waypoint.** Checks for receiver — B5 is in the target area. Activates kick skill aimed at B5's actual position. Ball is launched.

**Tick N+15.** Ball is in flight toward B5. B3 is no longer the active robot (the kick happened). The execution layer designates B5 as the intended receiver and monitors the ball. B3 returns to Formation control and repositions.

**Tick N+22.** B5 receives the ball. Pass waypoint succeeds. Replan trigger fires.

**Planner runs.** Possession state: we have possession (B5). B5 is near the opponent goal. Clear shot? Yes — the goalkeeper is covering center but the far post is open. Plan: shoot(far post).

**Active robot: B5. Current waypoint: shoot.**

**Execution runs shoot waypoint.** B5 adjusts angle slightly and kicks toward far post.

**Tick N+25.** Shot is taken. Waypoint completes. Replan trigger fires. Ball is now in flight toward the goal — regardless of whether it goes in, the ball state has changed fundamentally. If it's a goal, play resets. If saved or missed, the planner will see a new ball state (loose ball, or opponent goalkeeper has it) and plan accordingly.

**What this validates.** The full replan loop works end to end. Formation placed a receiver where the plan needed one. The pass waypoint correctly waited for and verified a receiver. The handoff from passer to receiver worked smoothly.

### Case Study 2: Failed Steal and Recovery

**Situation.** Opponent O2 has the ball at midfield, dribbling toward our half. Our robots are in Formation — B1 and B4 are between O2 and our goal, others are spread across the field.

**Planner runs.** Possession state: opponent has ball. Best robot to challenge: B1, who is ahead of O2 and between O2 and our goal. Plan: capture(B1, steal from O2), pass(best open area), shoot.

**Active robot: B1. Current waypoint: capture (steal).**

**Formation runs** with plan context pointing toward a pass target area. Other robots maintain defensive positioning and prepare a passing option.

**Execution runs capture waypoint.** B1 moves to intercept O2. B1 approaches from a good angle (in front, not chasing).

**Tick N+20.** O2 sees B1 approaching and passes to O4, who is on the wing. B1 does not have the ball. The ball is no longer near B1 — it has moved far away to O4's position. Execution detects this: the ball moved significantly away from the capture target. Waypoint fails.

**Replan trigger fires.** Planner runs. Possession state: opponent has ball (O4, on the wing). Best robot to challenge: B4, who is nearest to O4. Plan: capture(B4, steal from O4), pass(area near B1 — who is now in an advanced midfield position from the failed steal attempt), shoot.

**Key observation.** B1's failed steal left B1 in an advanced position. The replanner incorporates this organically — B1 is now a good passing target. The system exploits the consequences of failure without any special logic.

**Active robot: B4. Current waypoint: capture (steal from O4).**

**Execution runs.** B4 challenges O4. This time O4 is on the wing with less space. B4 wins the ball.

**Replan.** B4 has possession on the wing. Clear shot? No, bad angle. Pass to B1 in the center? Good lane, good position. Plan: pass(B1's area), shoot.

**The sequence continues.** B4 passes to B1, B1 shoots.

**What this validates.** Failed waypoints lead to clean replanning. The system doesn't get stuck on a failed approach — it immediately reassesses with fresh information. The consequences of actions (even failed ones) change the game state in ways the replanner can exploit.

### Case Study 3: Pass With No Receiver

**Situation.** B2 has the ball near the opponent's penalty area. The planner wants to pass to the right wing area.

**Planner runs.** Plan: pass(right wing area), shoot.

**Formation runs** with plan context "pass target: right wing." Formation adds a bonus to the right wing for the nearest available robot. However, all available teammates are far from the right wing — B4 is the closest but is on the left side of midfield.

**Active robot: B2. Current waypoint: pass to right wing area.**

**Execution runs pass waypoint.** Before kicking, it checks: is any teammate in or approaching the target area? B4 is heading there (Formation is driving B4 toward the right wing) but is still far away.

**Execution waits.** B2 holds the ball. Perhaps the dribble skill activates gently to maintain possession and shield from opponents. Each tick, execution checks whether a receiver has arrived.

**Tick N+30.** B4 is getting closer but an opponent is now pressuring B2. B2 is at risk of losing possession. The timeout for the pass waypoint expires — too long without a viable receiver.

**Pass waypoint fails.** Replan trigger fires.

**Planner runs.** B2 still has possession but is under pressure. Right wing is still not well-covered. Clear shot? Marginal, but the planner is aggressive. New plan: shoot (take the shot even though it's not ideal — better than losing possession).

**B2 shoots.** Maybe it goes in, maybe it doesn't. Either way, the system made a reasonable decision under pressure: tried to set up a pass, recognized it wasn't coming together, fell back to a direct shot.

**What this validates.** The pass execution layer correctly waits for a receiver and correctly fails when one doesn't arrive. Formation-plan coupling isn't magic — sometimes Formation can't get a robot where the plan needs one fast enough. The system degrades gracefully by replanning into a simpler action.

### Case Study 4: Replan Stability Under Pressure

**Situation.** B3 has the ball at midfield. The planner produces: dribble(toward opponent half), pass(left wing), shoot. B3 starts dribbling.

**Tick N+5.** B3 is dribbling forward. An opponent starts closing from the right. The dribble waypoint is still active — B3 hasn't reached the target area yet and hasn't lost the ball.

**Question: should the planner intervene here?** No. The dribble waypoint is still executing. No replan trigger has fired (no success, no failure). The execution layer is responsible for handling the approaching opponent — it should adjust the dribble path slightly left to avoid the challenge while still heading toward the target area. This is within the execution layer's responsibility (get the ball to the target area safely).

**Tick N+15.** B3 has dribbled past the opponent and reached the target area. Dribble waypoint succeeds. Replan trigger fires.

**Planner runs.** B3 is now in the opponent half. The left wing is open, B5 is there (Formation placed them). Plan: pass(left wing), shoot. This happens to be the same next step as the original plan, but it was re-derived from the current state, not remembered from the old plan.

**Alternative timeline — tick N+10.** The opponent successfully tackles B3. Ball is loose. Dribble waypoint fails (possession lost). Replan trigger fires.

**Planner runs.** Ball is loose near B3 and the opponent. Nearest robot to the ball: B3 is right there, but so is the opponent. B1 is also nearby. Planner picks: capture(B3, contested ball). If B3 wins it back, proceed. If not, replan again.

**What this validates.** Between replan triggers, the execution layer handles tactical adjustments autonomously. The planner isn't micromanaging — it sets the strategic direction (dribble to this area) and execution handles the moment-to-moment challenges. Replanning only happens at waypoint boundaries, which prevents oscillation while still being responsive to major state changes.

### Case Study 5: Ball In Flight Transition

**Situation.** B2 passes to B5 on the left wing. The ball is in the air.

**Tick N.** B2 kicks the ball. The pass waypoint transitions to "monitoring" mode. B2 is no longer the active robot in a meaningful sense — the ball is gone. But the pass waypoint is still active, being monitored.

**Critical question: what does Formation do with B2 now?** B2 should immediately return to Formation control. It has no ball, no active skill to run. Formation assigns it a new position — likely somewhere that provides defensive cover or an offensive option for the next phase.

**Critical question: what does B5 do while the ball is in flight?** B5 is currently a Formation robot positioned in the left wing. It should stay there — that's where the ball is coming. Formation is already keeping it there because of the plan context bonus. B5's movement skill should position it to receive the pass (facing the ball, in a good receiving stance). This is subtle — B5 isn't yet the active robot (the pass waypoint hasn't succeeded yet), but it needs to behave like a receiver. This can be handled by Formation: when a robot is in the target area of an active pass waypoint, its movement target should incorporate facing the incoming ball.

**Tick N+8.** Ball arrives at B5. B5 controls it. Pass waypoint succeeds. Replan trigger.

**Planner runs.** B5 has the ball on the left wing. Plan: shoot if clear, or dribble/pass toward goal.

**New active robot: B5.** B5 transitions from Formation control to execution control seamlessly.

**What this validates.** The handoff during a pass — from passer to ball-in-flight to receiver — works smoothly. Formation handles the passer's repositioning and the receiver's positioning simultaneously. The system doesn't need explicit "claim" or "release" logic — it falls out of the active robot being the one with the ball.

### Case Study 6: The No-Progress Trap

**Situation.** Opponent O1 has the ball near midfield. Our B1 is the designated steal robot. B1 chases O1, but O1 is faster and keeps dribbling sideways, never advancing but never losing the ball. B1 is running alongside O1, never getting closer.

**This is the scenario that must not become a deadlock.**

**Execution layer responsibility.** The capture waypoint tracks whether B1 is making progress. "Progress" means the distance between B1 and the ball is decreasing, or B1 is getting into a better interception angle. If neither is happening over a sustained period (several hundred milliseconds, tunable), the waypoint fails with a "no progress" reason.

**Replan trigger fires.** Planner runs. O1 still has the ball. B1 is still nearby but has proven unable to capture. The planner should not immediately send B1 again — that would create an infinite loop.

**This requires planner awareness of recent failure.** The simplest mechanism: if a capture attempt with robot X against opponent Y just failed due to no-progress, the planner should prefer a different robot for the next capture attempt. B4, who is in a different position, might have a better approach angle. If no other robot has a better angle, the planner can still select B1, but the execution layer will fail quickly again if no progress resumes, and eventually the situation will change (the opponent will advance or pass, which changes the calculus).

**Deeper solution.** Formation can help. If direct stealing isn't working, Formation's positioning can focus on cutting off the opponent's options — blocking passing lanes, narrowing the dribbling corridor — rather than directly challenging. This is an emergent behavior from Formation: if all nearby robots are failing to steal, Formation's defensive positioning naturally creates a "press" that constrains the opponent until they make a mistake.

**What this validates.** The no-progress detection prevents deadlock. The planner avoids repeating failed approaches. Formation's continuous positioning provides a fallback containment strategy when direct stealing fails.

---

## Boundary Conditions and Edge Cases

### Game Stoppages and Restarts

When the referee stops play (foul, goal, out of bounds), the current plan is discarded. On restart, the planner runs fresh based on the restart type: if it's our kickoff or free kick, the planner produces a plan starting from the restart position. If it's the opponent's restart, the plan starts with a capture (anticipating where the ball will go). Formation positions for restarts follow the league rules for robot placement. The system does not need special restart "modes" — the planner simply operates on the world state, which includes the game phase (stopped, running, restart).

### Goalkeeper Integration

The goalkeeper is outside Formation. It runs dedicated logic: stay on the goal line, position between the ball and the goal center, move to cover the shot angle. The goalkeeper never becomes the active robot in a plan — it does not leave the goal area. If the ball comes to the goalkeeper (opponent shot saved), the goalkeeper's "clear the ball" action creates a loose ball or a controlled ball, and the planner picks up from there as a normal capture situation.

If the league rules allow the goalkeeper to participate in play (e.g., passing out from the back), this can be added later as a special case in the planner. For the MVP, the goalkeeper stays home.

### Ball Out of Play

When the ball goes out of bounds, the current waypoint fails (ball moved to an unreachable location). Replan on restart.

### Multiple Robots Contesting the Same Ball

When the ball is loose and multiple of our robots are nearby, the planner's capture waypoint designates exactly one robot. The others remain under Formation control. Formation should not drive multiple robots to the same loose ball — the spread penalty and the fact that one robot is already assigned (active robot) should prevent this. If two robots collide going for the ball, that's a Formation scoring issue (the active robot's area should not attract Formation robots).

### Active Robot Loses Possession Unexpectedly

This can happen during any waypoint. An opponent challenges and takes the ball, or the ball bounces off the robot unpredictably. The execution layer should detect loss of possession within a few ticks (the ball is no longer under our control and is not where we expected it). The current waypoint fails immediately, and a replan trigger fires. This is the default failure mode and should be the fastest path back to replanning — don't wait for a timeout when possession is clearly lost.

---

## Implementation Guidance

### What to Build First

Start with Formation in isolation. Get the scoring function working and watch robots position themselves in the GUI. They should spread out sensibly, cover the goal, stay between opponents and the goal, and offer passing options. Don't worry about the plan context bonus yet — just get the baseline positioning working. This is independently useful and testable.

Then build the waypoint types and the execution layer. Test each waypoint type individually: put a robot on the field with a ball and verify that the capture, dribble, pass, and shoot executions work and that success/failure detection triggers correctly. Use the simplest scenarios — no opponents, just verify the skill activations are correct and the state transitions are detected.

Then build the planner with the four hardcoded plans. Wire it into the replan loop. Watch the full system in the simulator: Formation positions robots, the planner picks a plan, execution drives the active robot, the waypoint completes or fails, the planner runs again.

Finally, add the plan context bonus to Formation's scoring function. This is when passes start working — the planner says "pass to the left wing" and Formation actually drives a robot there. This is the moment the system comes alive.

### What to Watch For

**Formation jitter.** Robots constantly swapping positions or oscillating between two targets. This means the scoring function has ambiguities that the hysteresis isn't smoothing out. Increase the hysteresis or adjust scoring weights.

**Pass timing.** The passer kicks before the receiver arrives, or the passer holds too long and gets dispossessed. The pass execution's receiver-presence check and its timeout are the tuning knobs here. Start with a generous timeout (let the passer hold for a while) and tighten it as you observe behavior.

**Replan cascades.** Rapid cycling between plans without completing any waypoint. This could mean failure detection is too sensitive (declares failure too quickly) or the planner oscillates between two equally-scored plans. Add hysteresis to the planner: slight preference for continuing the current plan over switching to a new one of similar quality.

**Active robot selection thrash.** The planner keeps picking a different robot as the active robot each replan. This means the closest robot keeps changing as robots move. The planner should have a slight bias toward keeping the current active robot if it's still a reasonable choice.

**Defensive vulnerability during attacks.** All robots rush toward the opponent goal and leave our half exposed. This means Formation's goal coverage scoring is too weak relative to the offensive support and plan context bonus. Strengthen the defensive weights — remember, Formation should always keep at least one or two robots in defensive positions even during attacks.

### What Not to Optimize Yet

Don't try to make the planner smart. Four hardcoded plans is enough. The MVP is not about tactical brilliance — it's about proving the replan loop, waypoint abstraction, and Formation coupling work. A smart planner on a broken architecture is useless. A dumb planner on a solid architecture is the foundation for everything that comes after.

Don't try to handle every edge case in execution. If the robot occasionally does something weird during a waypoint, that's fine for the MVP. Get the common path working first.

Don't tune scoring weights extensively. Get them roughly right and move on. Fine-tuning is for after the architecture is validated.

---

## Success Criteria

The MVP is successful when the following behaviors are observable in the simulator:

The team maintains a reasonable defensive shape at all times, even during attacks. At least one field player (besides the goalkeeper) remains between the ball and our goal when we're attacking.

When the ball is loose, the nearest robot drives toward it and gains possession. Other robots maintain formation rather than swarming the ball.

When a robot has the ball and a clear shot, it shoots promptly.

When a robot has the ball without a clear shot, it passes to a teammate who has been positioned by Formation in a useful area. The receiving teammate then shoots or continues the attack.

When a pass fails (intercepted, no receiver available), the system recovers within a few ticks — a new plan is produced and a new robot takes action.

When the opponent has the ball, our robots actively challenge for possession rather than passively holding formation. The system looks aggressive, not reactive.

The team does not exhibit persistent deadlocks, oscillations, or robots standing still with no apparent purpose. Every robot is always either executing a waypoint or moving toward a Formation-assigned position.
