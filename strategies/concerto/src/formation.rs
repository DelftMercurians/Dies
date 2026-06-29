//! Formation — one cost-aware matching that places every field robot (except the
//! keeper and the reserved plan robots) *and* elects the ball-winner.
//!
//! Role generators produce smoothly-varying positions with continuous importance;
//! robots are matched to roles by minimum total cost, where cost blends a
//! momentum-aware redirect time with the role's importance. When we don't hold the
//! ball, "win the ball" enters that same matching as the capture role — so the
//! decision "who leaves their post for the ball" is made by the same optimiser,
//! against the same costs, as every defensive duty. The robot that draws it is the
//! capturer (driven by the Planner/Driver, not positioned here).
//!
//! Stability comes from three physical sources, never a stay-bonus:
//!   1. continuity — role positions are continuous functions of the world;
//!   2. redirect cost — a moving robot continues cheaply, reverses expensively;
//!   3. cadence — assignments are recomputed only on events (cooldown-bounded),
//!      while positions update every tick so motion stays smooth.

use std::collections::HashMap;

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use crate::config;
use crate::geometry;
use crate::matching::assign_min_cost;
use crate::planner::{Engagement, PlanContext};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum RoleKind {
    /// Win the loose/contested ball. The robot matched to this role is the
    /// capturer — driven by the Planner/Driver, not positioned by Formation — so
    /// the decision "who leaves their post for the ball" is made by the same
    /// cost-aware matching as every defensive duty, weighing each robot's value.
    Capture,
    Shadow,
    Mark,
    /// Offensive outlets. Slots `0..SUPPORT_COUNT` are the wide supporters; the
    /// final slot is the central box-runner (cutback target / rebound crasher),
    /// faded in by the attack fraction. Folding the box-runner into Support (rather
    /// than a separate kind) makes central↔wide reassignment a within-kind slot
    /// swap, not a tactical role flip — see the generator.
    Support,
    /// Deep recycle pivot behind the ball, ball-side: the safe outlet the carrier
    /// lays the ball back to when nothing is on forward. Never in the box.
    Pivot,
    /// Rest-defense body kept home while attacking, on the ball→own-goal ray in our
    /// own half. Holds one robot back so a turnover doesn't leave the goal to the
    /// keeper + box anchor alone (it recovers into a wing of the wall). See the
    /// generator (gated on `attacking`).
    Balance,
    Spread,
}

fn role_name(kind: RoleKind) -> &'static str {
    match kind {
        RoleKind::Capture => "capture",
        RoleKind::Shadow => "shadow",
        RoleKind::Mark => "mark",
        RoleKind::Support => "support",
        RoleKind::Pivot => "pivot",
        RoleKind::Balance => "balance",
        RoleKind::Spread => "spread",
    }
}

/// Stable identity for a role across ticks. The same `(kind, slot)` denotes the
/// same tactical concept, so the robot serving it has low redirect cost to its
/// slightly-moved position — continuity, not stickiness.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct RoleId {
    kind: RoleKind,
    slot: u16,
}

#[derive(Clone, Debug)]
struct Role {
    id: RoleId,
    position: Vector2,
    importance: f64,
    face: Vector2,
}

/// A positioning command for one formation-controlled robot.
pub struct FormationCommand {
    pub id: PlayerId,
    pub target: Vector2,
    pub face: Vector2,
    pub role: &'static str,
}

/// Request to include a ball-winner in the matching this tick. Present only when
/// we don't hold the ball and may pursue it. The matcher weighs capturing the
/// ball against each robot's defensive duty; the winner is returned as the
/// capturer (and is not given a positioning command — the Driver controls it).
#[derive(Clone, Debug, PartialEq)]
pub struct CaptureRole {
    /// Where to win the ball (a velocity-led intercept point).
    pub pos: Vector2,
    /// Value of winning the ball, weighed against defensive importances.
    pub importance: f64,
    /// Robots barred from ball duty (double-touch robot, no-progress cooldown).
    /// They still take defensive roles — they just can't draw the capture role.
    pub ineligible: Vec<PlayerId>,
}

/// Result of a formation update: positioning for the non-capturing field robots,
/// plus the capturer if the capture role was filled.
pub struct FormationOutput {
    pub commands: Vec<FormationCommand>,
    pub capturer: Option<PlayerId>,
}

pub struct Formation {
    /// Frozen assignment from the last recalc; positions re-resolved each tick.
    assignment: HashMap<PlayerId, RoleId>,
    last_recalc: f64,
    queued: bool,
    // Change-detection snapshots for recalc triggers.
    last_assignable: Vec<PlayerId>,
    last_reserved: Vec<PlayerId>,
    last_plan_ctx: PlanContext,
    last_capture: Option<CaptureRole>,
    /// Robot that drew the capture role last tick (the incumbent pursuer). Fix D:
    /// the matcher gives it a small commitment discount on the capture role so a
    /// chase isn't re-elected to a marginally-closer robot every recompute as the
    /// (lead) ball point jitters — the cause of most capturing↔unassigned churn.
    last_capturer: Option<PlayerId>,
}

impl Default for Formation {
    fn default() -> Self {
        Self::new()
    }
}

impl Formation {
    pub fn new() -> Self {
        Self {
            assignment: HashMap::new(),
            last_recalc: f64::NEG_INFINITY,
            queued: false,
            last_assignable: Vec::new(),
            last_reserved: Vec::new(),
            last_plan_ctx: PlanContext::default(),
            last_capture: None,
            last_capturer: None,
        }
    }

    /// Match every field robot (except the keeper and the reserved plan robots) to
    /// a role in one cost-aware assignment. When `capture` is given, the ball-winner
    /// is one of those roles, so who pursues the ball is decided jointly with who
    /// holds defensive shape. Returns positioning for the non-capturers plus the
    /// capturer (if the capture role was filled).
    pub fn update(
        &mut self,
        world: &World,
        reserved: &[PlayerId],
        plan_ctx: &PlanContext,
        capture: Option<&CaptureRole>,
        now: f64,
    ) -> FormationOutput {
        let keeper_id = world.our_keeper_id();

        // Assignable = own field robots minus keeper minus reserved plan robots.
        let mut assignable: Vec<PlayerId> = world
            .own_players()
            .iter()
            .map(|p| p.id)
            .filter(|id| Some(*id) != keeper_id && !reserved.contains(id))
            .collect();
        assignable.sort_by_key(|id| id.as_u32());

        // Generate the role set (positions/importance recomputed every tick).
        let roles = self.generate_roles(world, plan_ctx, capture, assignable.len());
        let role_by_id: HashMap<RoleId, &Role> = roles.iter().map(|r| (r.id, r)).collect();

        // ── Decide whether to recompute the assignment (cadence) ────────
        let assignable_changed = assignable != self.last_assignable;
        let reserved_changed = reserved != self.last_reserved.as_slice();
        let plan_ctx_changed = plan_ctx_differs(&self.last_plan_ctx, plan_ctx);
        let capture_changed = capture_differs(self.last_capture.as_ref(), capture);
        let bg_due = now - self.last_recalc >= config::RECALC_BG_PERIOD;
        if assignable_changed || reserved_changed || plan_ctx_changed || capture_changed || bg_due {
            self.queued = true;
        }
        let cooldown_ok = now - self.last_recalc >= config::RECALC_COOLDOWN;
        let stale = self.assignment.is_empty() && !assignable.is_empty();
        // Capture appearing/disappearing must take effect this tick (don't make a
        // pursuit wait out the cooldown, or leave a stale capturer after we score).
        let capture_presence_changed = self.last_capture.is_some() != capture.is_some();
        if (self.queued && cooldown_ok) || stale || capture_presence_changed {
            self.recompute(world, &assignable, &roles, capture);
            self.last_recalc = now;
            self.queued = false;
        }

        self.last_assignable = assignable.clone();
        self.last_reserved = reserved.to_vec();
        self.last_plan_ctx = plan_ctx.clone();
        self.last_capture = capture.cloned();

        // ── Emit commands; the capture-role robot is the capturer (driven by the
        //    Driver, so it gets no positioning command). ─────────────────────────
        let mut capturer = None;
        let mut commands = Vec::with_capacity(assignable.len());
        for id in &assignable {
            let role = self.assignment.get(id).and_then(|rid| role_by_id.get(rid));
            let role = match role {
                Some(r) => *r,
                None => {
                    // Assigned role vanished (count change) or robot unassigned —
                    // fall back to the nearest existing role and force a recalc.
                    self.queued = true;
                    match nearest_role(world, *id, &roles) {
                        Some(r) => r,
                        None => continue,
                    }
                }
            };
            if role.id.kind == RoleKind::Capture {
                capturer = Some(*id);
                continue;
            }
            commands.push(FormationCommand {
                id: *id,
                target: role.position,
                face: role.face,
                role: role_name(role.id.kind),
            });
        }
        self.last_capturer = capturer;
        FormationOutput { commands, capturer }
    }

    /// Run the cost-aware matching and store the new assignment. The capture role
    /// (if any) is gated: a robot barred from ball duty, or unable to reach the
    /// ball within `CAPTURE_TIME_HORIZON`, cannot take it — so the matcher never
    /// sends the freest robot on a hopeless chase, and leaves capture unfilled when
    /// no one can reach it (the "is it worth pursuing" gate).
    fn recompute(
        &mut self,
        world: &World,
        assignable: &[PlayerId],
        roles: &[Role],
        capture: Option<&CaptureRole>,
    ) {
        if assignable.is_empty() || roles.is_empty() {
            self.assignment.clear();
            return;
        }
        // Cost matrix: rows = robots, cols = roles (rows <= cols by over-generation).
        let cost: Vec<Vec<f64>> = assignable
            .iter()
            .map(|id| {
                let p = world.own_player(*id);
                roles
                    .iter()
                    .map(|role| match p {
                        Some(p) => {
                            let t = geometry::redirect_time(
                                p.position,
                                p.velocity,
                                role.position,
                                config::V_MAX,
                                config::A_MAX,
                            );
                            if role.id.kind == RoleKind::Capture {
                                let barred =
                                    capture.map(|c| c.ineligible.contains(id)).unwrap_or(true);
                                if barred || t > config::CAPTURE_TIME_HORIZON {
                                    return f64::INFINITY;
                                }
                                // Fix D: commitment discount for the incumbent
                                // pursuer — a challenger must be decisively faster
                                // (by CAPTURE_COMMIT seconds of redirect) to take
                                // the chase, so the capturer doesn't flip on lead-
                                // point jitter. Applied after the eligibility gate
                                // so it never un-gates an unreachable robot.
                                if Some(*id) == self.last_capturer {
                                    return t
                                        - role.importance * config::SEC_PER_IMPORTANCE
                                        - config::CAPTURE_COMMIT;
                                }
                            }
                            t - role.importance * config::SEC_PER_IMPORTANCE
                        }
                        None => f64::INFINITY,
                    })
                    .collect()
            })
            .collect();

        let matching = assign_min_cost(&cost);
        self.assignment.clear();
        for (row, col) in matching.iter().enumerate() {
            if let Some(c) = col {
                self.assignment.insert(assignable[row], roles[*c].id);
            }
        }
    }

    /// Build the role set for this tick. Generators run in priority order; spread
    /// roles top up to the over-generation target.
    fn generate_roles(
        &self,
        world: &World,
        plan_ctx: &PlanContext,
        capture: Option<&CaptureRole>,
        n: usize,
    ) -> Vec<Role> {
        let mut roles: Vec<Role> = Vec::new();
        let own_goal = world.own_goal_center();
        let ball = world
            .ball_position()
            .unwrap_or_else(|| Vector2::new(0.0, 0.0));
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;

        // 0. Capture — the ball-winner, when we're pursuing a ball we don't hold.
        //    The robot matched to it becomes the capturer (driven by the Driver).
        if let Some(c) = capture {
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Capture,
                    slot: 0,
                },
                position: c.pos,
                importance: c.importance,
                face: ball,
            });
        }

        // Coverage accounting: opponents already pressured by a plan robot need no
        // mark (the plan robot *is* the mark); ball-contest points soft-suppress
        // nearby roles so a second robot doesn't pile onto the contested ball.
        let engaged_opps: Vec<PlayerId> = plan_ctx
            .engagements
            .iter()
            .filter_map(|e| match e {
                Engagement::Engaging { opponent } => Some(*opponent),
                _ => None,
            })
            .collect();
        let contest_pts: Vec<Vector2> = plan_ctx
            .engagements
            .iter()
            .filter_map(|e| match e {
                Engagement::BallContest { at } => Some(*at),
                _ => None,
            })
            .collect();

        let ball_threat = geometry::threat(
            ball,
            own_goal,
            config::THREAT_GOAL_NEAR,
            config::THREAT_GOAL_FAR,
        );

        // Defending an opponent set piece (free kick / kickoff / placement) with
        // the ball deep in our own third: the goal is acutely threatened and we are
        // barred from contesting the ball (must stand off it), so there's no
        // capturer to absorb a body. Commit a full goal wall and pull everyone off
        // offensive `support` (below). Without this the matcher staffs forward
        // support roles while the goal mouth is left to the keeper alone — the
        // corner free-kick collapse (leg-2 fix).
        let setpiece_defense = !world.us_operating()
            && !matches!(
                world.game_state(),
                GameState::Run | GameState::Halt | GameState::Timeout | GameState::Unknown
            )
            && ball.x < -half_len / 3.0;

        // 1. Shadow / goal coverage. Fix A: always emit `SHADOW_MAX` shadows as a
        //    fixed-width wall; importance is staggered center-out (the central
        //    shadow on the ball→goal ray is most valuable, the wings fade) and
        //    scaled by threat, so the *staffed* count emerges continuously — no
        //    integer count step, hence no blink. Fixed cardinality also keeps each
        //    shadow slot's identity stable tick-to-tick, which the redirect-cost
        //    continuity (and Fix B) rely on. Relief for a plan robot contesting in
        //    our zone is handled by the continuous soft-suppression pass below, not
        //    a discrete count cut. On set-piece defense the stagger is flattened so
        //    the full wall is committed (the importance-space form of leg-2's
        //    `k = SHADOW_MAX`); support is zeroed below so nothing forward outranks
        //    the outer wall slots.
        let positions = geometry::shadow_arc(
            ball,
            own_goal,
            config::SHADOW_MAX,
            config::SHADOW_STANDOFF,
            config::SHADOW_SPACING,
        );
        let shadow_center = (config::SHADOW_MAX as f64 - 1.0) / 2.0;
        // The central shadow on the ball→goal ray is promoted to an "anchor": a
        // sticky last-line field defender (distinct from the keeper at the mouth).
        // In open play it is (a) pulled in to the penalty-area edge so a body
        // always hugs the box, and (b) floored in importance so it doesn't
        // collapse with ball_threat the way the wing wall does — otherwise when we
        // commit forward (ball_threat→0) the whole wall drops below the
        // striker/support roles and the box is left to the keeper alone. On
        // set-piece defense the centre stays in the contiguous wall (leg-2 fix).
        let pen_depth = world
            .field()
            .map(|f| f.penalty_area_depth)
            .unwrap_or(1000.0);
        let anchor_standoff = pen_depth + config::ANCHOR_BOX_MARGIN;
        for (i, pos) in positions.into_iter().enumerate() {
            let off_center = (i as f64 - shadow_center).abs();
            let is_anchor = off_center < 1e-9 && !setpiece_defense;
            let stagger = if setpiece_defense {
                1.0
            } else {
                config::SHADOW_STAGGER.powf(off_center)
            };
            let threat_factor = if is_anchor {
                (0.5 + 0.5 * ball_threat).max(config::ANCHOR_THREAT_FLOOR)
            } else {
                0.5 + 0.5 * ball_threat
            };
            let position = if is_anchor {
                geometry::goal_ray_point(ball, own_goal, anchor_standoff)
            } else {
                pos
            };
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Shadow,
                    slot: i as u16,
                },
                position,
                importance: config::IMP_SHADOW_BASE * threat_factor * stagger,
                face: ball,
            });
        }

        // 2. Marking — one role per opponent, slot = stable opponent index.
        let mut opp_ids: Vec<PlayerId> = world.opp_player_ids();
        opp_ids.sort_by_key(|id| id.as_u32());
        for (slot, oid) in opp_ids.iter().enumerate() {
            // A plan robot is already pressuring this opponent — don't double up.
            if engaged_opps.contains(oid) {
                continue;
            }
            if let Some(opp) = world.opp_player(*oid) {
                let opp_threat = geometry::threat(
                    opp.position,
                    own_goal,
                    config::THREAT_GOAL_NEAR,
                    config::THREAT_GOAL_FAR,
                );
                // Fix A: no hard threat cutoff. Mark importance ramps to ~0 with
                // opp_threat·openness, so a far opponent's mark simply loses the
                // match (staffed only if a robot has nothing better) instead of
                // blinking in/out at a threshold — one fewer role-set discontinuity.
                // Stand off the opponent toward our own goal (goal-side marking),
                // and never cross midfield chasing them onto the opponent half.
                let to_goal = own_goal - opp.position;
                let n_tg = to_goal.norm();
                let mut pos = if n_tg > 1e-6 {
                    opp.position + to_goal / n_tg * config::MARK_STANDOFF
                } else {
                    opp.position
                };
                pos.x = pos.x.min(config::MARK_MAX_X);
                // How open is the ball→opponent lane (can this opponent receive a
                // pass and become a threat)? Exclude the opponent itself, which sits
                // at the lane endpoint and would otherwise read as a blocker.
                let others: Vec<PlayerState> = world
                    .opp_players()
                    .iter()
                    .filter(|p| p.id != *oid)
                    .cloned()
                    .collect();
                let openness = geometry::lane_openness(
                    ball,
                    opp.position,
                    &others,
                    config::MARK_LANE_CORRIDOR,
                );
                roles.push(Role {
                    id: RoleId {
                        kind: RoleKind::Mark,
                        slot: slot as u16,
                    },
                    position: pos,
                    importance: config::IMP_MARK_BASE * opp_threat * openness,
                    face: ball,
                });
            }
        }

        // 3. Offensive support — open, forward outlets the carrier can pass into.
        // Positioned by ball→candidate lane openness (not a static flank point),
        // so supporters stop ending up stranded behind opponents.
        let opp_box = world.opp_penalty_area();
        let opp_pen_depth = opp_box.max.x - opp_box.min.x;
        let opp_pen_half_width = opp_box.max.y;
        // Fix A: a continuous attack fraction `af ∈ [0,1]` replaces the `attacking`
        // boolean. It ramps up as the ball advances from SUPPORT_ATTACK_BALL_X
        // toward ..._FULL, and is gated down as our-goal threat rises from
        // ..._MIN_THREAT to ..._MAX_THREAT. Support importance lerps
        // IMP_SUPPORT→IMP_SUPPORT_ATTACK with `af`, and the box-runner's importance
        // scales with `af` (≈0 when not attacking) — so the forward block fades in
        // and out smoothly across midfield rather than snapping at a threshold. The
        // *placement* layout still switches at the midpoint (`attacking`, af ≥ 0.5):
        // a lone position transition, damped by redirect cost + Fix B, while role
        // existence and weight stay continuous.
        let af_pos = ((ball.x - config::SUPPORT_ATTACK_BALL_X)
            / (config::SUPPORT_ATTACK_BALL_X_FULL - config::SUPPORT_ATTACK_BALL_X))
            .clamp(0.0, 1.0);
        let af_threat = ((config::SUPPORT_ATTACK_MAX_THREAT - ball_threat)
            / (config::SUPPORT_ATTACK_MAX_THREAT - config::SUPPORT_ATTACK_MIN_THREAT))
            .clamp(0.0, 1.0);
        let af = af_pos * af_threat;
        let attacking = af >= 0.5; // positioning layout only; existence/weight are continuous
                                   // Always two wide supporters; the central box-runner (below) is the third
                                   // forward body, faded in by `af`. Count is fixed → no role-set step. On
                                   // set-piece defense support importance is zeroed (leg-2: pull every body off
                                   // offence onto the goal wall) — the importance-space form of `support = 0`.
        let support_count = config::SUPPORT_COUNT;
        let support_imp = if setpiece_defense {
            0.0
        } else {
            config::IMP_SUPPORT + (config::IMP_SUPPORT_ATTACK - config::IMP_SUPPORT) * af
        };
        // Placed greedily; each supporter excludes spots already claimed by an
        // earlier one (see `taken` below) so two never converge on the same outlet.
        let mut taken: Vec<Vector2> = Vec::with_capacity(support_count);
        for slot in 0..support_count {
            // 0 → right-flank bias, 1 → left-flank bias, 2 → centre. The sign is only
            // a soft tie-break now; the `taken` exclusion is what structurally keeps
            // supporters from resolving to the same `best_support_pos`.
            let sign = match slot {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            };
            let pos = geometry::best_support_pos(
                ball,
                world.opp_players(),
                sign,
                half_len,
                half_wid,
                config::SUPPORT_LANE_CORRIDOR,
                opp_pen_depth,
                opp_pen_half_width,
                attacking,
                config::SUPPORT_FLANK_Y_FRACS,
                config::SUPPORT_GOAL_LINE_SETBACK,
                &taken,
                config::SUPPORT_MIN_SEPARATION,
            );
            taken.push(pos);
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Support,
                    slot: slot as u16,
                },
                position: pos,
                importance: support_imp,
                face: world.opp_goal_center(),
            });
        }

        // 3b. Central box-runner — a single advanced body just in front of the
        // opponent box, offset to the side away from the ball so it offers a cutback
        // angle across the keeper. This is the central outlet the planner cuts back
        // to (its large lateral separation from a wide carrier trips the final-third
        // cross branch) and the close-range finisher v0's wall+arc-keeper leave open.
        // Fix A/B: it is the *central support slot* (slot 2 of the Support kind), not
        // a separate role kind. Folding it into Support means the former
        // striker↔support reassignment (a forward robot swapping the central body for
        // a wide outlet, ~13% of post-A role churn) is now a within-kind slot swap:
        // the wide/central positions stay spatially fixed by slot, so redirect cost
        // keeps the same robot central, and the change no longer reads as a tactical
        // role flip. Importance scales with `af` (≈0 on set-piece defense, so it is
        // unstaffed there too) so the central body fades in only as we commit forward.
        {
            let front_x =
                world.opp_goal_center().x - (opp_pen_depth + config::BOX_RUNNER_FRONT_MARGIN);
            // Bias to the side away from the ball; default to +y when the ball is
            // dead-central so the choice is still deterministic.
            let away = if ball.y > 0.0 { -1.0 } else { 1.0 };
            let base = Vector2::new(front_x, away * config::BOX_RUNNER_Y_OFFSET);
            // Opponent keeper ≈ the opponent nearest the goal we attack; passed so
            // the pocket search respects the keeper's wider blocking shadow (matches
            // the planner's shot model). Degrades gracefully if none is near.
            let opp_keeper = world
                .opp_players()
                .iter()
                .min_by(|a, b| {
                    let goal = world.opp_goal_center();
                    (a.position - goal)
                        .norm()
                        .partial_cmp(&(b.position - goal).norm())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|p| p.id);
            // Slide the finisher into the open shooting pocket the defence leaves,
            // rather than a static cutback point — converts the cutback the planner
            // already targets into a real shot. Falls back to `base` if no window.
            let pos = geometry::best_finishing_pocket(
                base,
                ball,
                world.opp_goal_center().x,
                world.goal_width(),
                world.opp_players(),
                opp_keeper,
                config::SHOT_ROBOT_RADIUS,
                config::SHOT_KEEPER_RADIUS,
                config::BALL_RADIUS,
                config::SHOT_KEEPER_BIAS,
                half_wid,
                config::BOX_RUNNER_Y_OFFSET,
            );
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Support,
                    slot: config::SUPPORT_COUNT as u16,
                },
                position: pos,
                importance: config::IMP_STRIKER * af,
                face: world.opp_goal_center(),
            });
        }

        // 3c. Deep recycle pivot behind the ball on the ball's flank — the safe
        // outlet the carrier lays the ball back to when nothing is on forward (the
        // planner's recycle branch). Placed deep (never in the box) so it does
        // not invite the defence to collapse the shooting zone the way an advanced
        // central body does. Only while attacking (ball advanced, our goal safe).
        if attacking {
            let y_mag = ball.y.abs().clamp(config::PIVOT_Y_MIN, config::PIVOT_Y_MAX);
            let py = if ball.y >= 0.0 { y_mag } else { -y_mag };
            let px = (ball.x - config::PIVOT_SETBACK).clamp(-half_len + 400.0, half_len - 400.0);
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Pivot,
                    slot: 0,
                },
                position: Vector2::new(px, py),
                importance: config::IMP_PIVOT,
                face: world.opp_goal_center(),
            });
        }

        // 3d. Rest defense — keep ONE body home while we commit forward, so a
        // turnover doesn't leave the goal to the keeper + box anchor alone (the
        // counter-attack that opens the wall's flanks). Placed on the ball→own-goal
        // ray in our own half — the most likely counter lane — at a recoverable
        // depth, so on turnover it is already there to take a wing of the wall while
        // the deep attackers recover. Importance (IMP_BALANCE) sits above the
        // forward block but below the anchor, so it displaces the pivot (the least
        // critical forward outlet), not a supporter — netting a 3-back/3-forward
        // shape instead of 2-back/4-forward. Only while attacking (same gate as the
        // pivot it replaces); when defending, the 3-wide wall already lights up.
        if attacking {
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Balance,
                    slot: 0,
                },
                position: geometry::goal_ray_point(ball, own_goal, config::BALANCE_STANDOFF),
                importance: config::IMP_BALANCE,
                face: ball,
            });
        }

        // (No receiver role: Model B — the planner picks a concrete receiver, it
        // becomes a plan slot, and the PassCoordinator positions it. Formation
        // staffing one too would put a second robot on the receive area.)

        // 4. Spread/residual — Fix A: the fixed catalog above (shadows + marks +
        //    supports + striker) already exceeds the field-robot count in normal
        //    play, so spread is now only a floor guaranteeing every assignable
        //    robot has *some* role; we no longer over-generate a 1.5× cushion.
        let target = n;
        let mut slot = 0u16;
        while roles.len() < target.max(n) {
            let frac = if target.max(n) <= 1 {
                0.5
            } else {
                slot as f64 / (target.max(n) as f64)
            };
            let y = -half_wid * 0.5 + half_wid * frac;
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Spread,
                    slot,
                },
                position: Vector2::new(0.0, y),
                importance: config::IMP_SPREAD,
                face: ball,
            });
            slot += 1;
        }

        // Soft suppression: de-prioritise (don't remove) roles near a plan robot's
        // ball contest, so the matcher avoids stacking a second body on the
        // contested ball but can still place one if nothing better exists. Falloff
        // is linear to zero at SUPPRESS_RADIUS; we take the strongest contest's
        // effect rather than summing, so overlapping contests don't over-penalise.
        if !contest_pts.is_empty() {
            for role in roles.iter_mut() {
                // The capture role sits *on* the contested ball by design — it's the
                // robot we want there — so never suppress it; only the defensive
                // roles that shouldn't pile on.
                if role.id.kind == RoleKind::Capture {
                    continue;
                }
                let prox = contest_pts
                    .iter()
                    .map(|pt| {
                        (1.0 - (role.position - pt).norm() / config::SUPPRESS_RADIUS).max(0.0)
                    })
                    .fold(0.0_f64, f64::max);
                role.importance -= config::IMP_SUPPRESS * prox;
            }
        }

        roles
    }
}

/// Whether the capture spec changed enough to trigger a recalc. The capture point
/// tracks the (lead) ball and so jitters every tick; compare it with an epsilon so
/// normal ball motion doesn't force a recompute every frame (presence flips are
/// handled separately, with an immediate recompute). Importance/eligibility compare
/// exactly.
fn capture_differs(a: Option<&CaptureRole>, b: Option<&CaptureRole>) -> bool {
    match (a, b) {
        (None, None) => false,
        (Some(x), Some(y)) => {
            (x.pos - y.pos).norm() > config::PLAN_CTX_MOVE_EPS
                || x.importance != y.importance
                || x.ineligible != y.ineligible
        }
        _ => true,
    }
}

/// Whether two plan contexts differ enough to count as a recalc trigger. Relies
/// on the driver building engagements in a stable order; positions compare with
/// an epsilon so per-tick ball jitter doesn't force a recompute every frame.
fn plan_ctx_differs(a: &PlanContext, b: &PlanContext) -> bool {
    if a.engagements.len() != b.engagements.len() {
        return true;
    }
    a.engagements
        .iter()
        .zip(&b.engagements)
        .any(|(x, y)| engagement_differs(x, y))
}

fn engagement_differs(a: &Engagement, b: &Engagement) -> bool {
    match (a, b) {
        (Engagement::BallContest { at: p }, Engagement::BallContest { at: q })
        | (Engagement::ReceiveArea { at: p }, Engagement::ReceiveArea { at: q }) => {
            (p - q).norm() > config::PLAN_CTX_MOVE_EPS
        }
        (Engagement::Engaging { opponent: p }, Engagement::Engaging { opponent: q }) => p != q,
        _ => true, // different variant in the same slot
    }
}

/// Nearest existing role position to a robot (fallback when its role vanished).
fn nearest_role<'a>(world: &World, id: PlayerId, roles: &'a [Role]) -> Option<&'a Role> {
    let pos = world.own_player(id)?.position;
    // Positioning roles only — the capturer must come from the gated cost matching,
    // never from a proximity fallback (which would bypass eligibility/horizon and
    // could double-elect the capture role, leaving the matched capturer uncommanded).
    roles
        .iter()
        .filter(|r| r.id.kind != RoleKind::Capture)
        .min_by(|a, b| {
            let da = (a.position - pos).norm();
            let db = (b.position - pos).norm();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_strategy_protocol::{BallState, GameState, WorldSnapshot};

    fn player(id: u32, x: f64, y: f64) -> PlayerState {
        PlayerState::new(
            PlayerId::new(id),
            Vector2::new(x, y),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        )
    }

    fn world_with(own: Vec<PlayerState>, opp: Vec<PlayerState>, keeper: u32) -> World {
        World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: Vector2::new(0.0, 0.0),
                velocity: Vector2::new(0.0, 0.0),
                detected: true,
            }),
            own_players: own,
            opp_players: opp,
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: Some(PlayerId::new(keeper)),
            freekick_kicker: None,
            possession: Possession::Loose,
            possession_stale: false,
            ball_contest: None,
        })
    }

    #[test]
    fn every_assignable_robot_gets_one_command() {
        let own = vec![
            player(1, -4000.0, 0.0), // keeper
            player(2, -2000.0, 1000.0),
            player(3, -2000.0, -1000.0),
            player(4, 0.0, 500.0),
            player(5, 1000.0, -500.0), // active robot (plan slot)
        ];
        let opp = vec![player(11, -1000.0, 0.0), player(12, 2000.0, 1500.0)];
        let world = world_with(own, opp, 1);

        let mut f = Formation::new();
        let reserved = vec![PlayerId::new(5)];
        let out = f.update(&world, &reserved, &PlanContext::default(), None, 0.0);

        // Assignable = ids {2,3,4} (exclude keeper 1 and reserved 5).
        let mut assigned: Vec<u32> = out.commands.iter().map(|c| c.id.as_u32()).collect();
        assigned.sort_unstable();
        assert_eq!(assigned, vec![2, 3, 4]);
        // Each gets exactly one command, all finite.
        assert_eq!(out.commands.len(), 3);
        for c in &out.commands {
            assert!(c.target.x.is_finite() && c.target.y.is_finite());
        }
        assert_eq!(out.capturer, None, "no capture role given → no capturer");
    }

    #[test]
    fn engaging_suppresses_that_opponents_mark() {
        // A plan robot pressuring opp 11 means Formation must not also mark it.
        // Both opponents sit near our goal so they clear the MARK_MIN_THREAT floor
        // (a far opponent on its own half is intentionally not marked).
        let own = vec![
            player(1, -4000.0, 0.0),
            player(2, -2000.0, 1000.0),
            player(3, 0.0, 0.0),
        ];
        let opp = vec![player(11, 500.0, 0.0), player(12, -1000.0, 1000.0)];
        let world = world_with(own, opp, 1);
        let f = Formation::new();

        let marks = |rs: &[Role]| rs.iter().filter(|r| r.id.kind == RoleKind::Mark).count();
        let plain = f.generate_roles(&world, &PlanContext::default(), None, 3);
        let engaged = f.generate_roles(
            &world,
            &PlanContext {
                engagements: vec![Engagement::Engaging {
                    opponent: PlayerId::new(11),
                }],
            },
            None,
            3,
        );
        assert_eq!(marks(&plain), 2);
        assert_eq!(marks(&engaged), 1);
        // The surviving mark is opp 12 (slot 1), not the engaged opp 11 (slot 0).
        assert!(engaged
            .iter()
            .any(|r| r.id.kind == RoleKind::Mark && r.id.slot == 1));
    }

    #[test]
    fn ball_contest_softly_suppresses_nearby_roles() {
        let own = vec![
            player(1, -4000.0, 0.0),
            player(2, -2000.0, 1000.0),
            player(3, 0.0, 0.0),
        ];
        let opp = vec![player(11, 500.0, 0.0), player(12, 2000.0, 1500.0)];
        let world = world_with(own, opp, 1);
        let f = Formation::new();

        let plain = f.generate_roles(&world, &PlanContext::default(), None, 4);
        let supp = f.generate_roles(
            &world,
            &PlanContext {
                engagements: vec![Engagement::BallContest {
                    at: Vector2::new(0.0, 0.0),
                }],
            },
            None,
            4,
        );

        // BallContest only re-weights importance, never the role set.
        assert_eq!(plain.len(), supp.len());
        let mut any_reduced = false;
        for r in &supp {
            let p = plain.iter().find(|q| q.id == r.id).expect("same role ids");
            assert!(
                r.importance <= p.importance + 1e-9,
                "no role gains importance"
            );
            if r.importance < p.importance - 1e-9 {
                any_reduced = true;
            }
        }
        assert!(
            any_reduced,
            "a role near the contest must be de-prioritised"
        );
    }

    #[test]
    fn attacking_stages_a_central_box_runner() {
        // Ball advanced in the opponent half with our goal safe → attacking. A
        // single Striker role must appear, central and just in front of the
        // opponent box, while the wide-supporter count drops by one (the central
        // body replaces a wing, not adds to it).
        let own = vec![
            player(1, -4000.0, 0.0),
            player(2, 1000.0, 800.0),
            player(3, 1500.0, -600.0),
        ];
        let opp = vec![player(11, 3000.0, 200.0)];
        let world = World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: Vector2::new(2200.0, 600.0), // opp half, ball.y > 0
                velocity: Vector2::new(0.0, 0.0),
                detected: true,
            }),
            own_players: own,
            opp_players: opp,
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: Some(PlayerId::new(1)),
            freekick_kicker: None,
            possession: Possession::We(PlayerId::new(2)),
            possession_stale: false,
            ball_contest: None,
        });
        let f = Formation::new();
        let roles = f.generate_roles(&world, &PlanContext::default(), None, 4);

        // The box-runner is the central Support slot (slot == SUPPORT_COUNT).
        let runners: Vec<&Role> = roles
            .iter()
            .filter(|r| {
                r.id.kind == RoleKind::Support && r.id.slot as usize == config::SUPPORT_COUNT
            })
            .collect();
        assert_eq!(runners.len(), 1, "exactly one central box-runner slot");
        let s = runners[0];
        // Just in front of the opponent box, central.
        let box_front = world.opp_penalty_area().min.x;
        assert!(
            s.position.x < box_front && s.position.x > box_front - 600.0,
            "box-runner sits just outside the box front, got x={}",
            s.position.x
        );
        // Ball is at +y, so the runner biases to -y (open cutback side).
        assert!(
            s.position.y < 0.0,
            "runner should bias away from the ball side"
        );
        // Total support bodies = two wide + one central = SUPPORT_ATTACK_COUNT.
        let supports = roles
            .iter()
            .filter(|r| r.id.kind == RoleKind::Support)
            .count();
        assert_eq!(supports, config::SUPPORT_ATTACK_COUNT);
    }

    #[test]
    fn assignment_is_frozen_within_cooldown() {
        let own = vec![
            player(1, -4000.0, 0.0),
            player(2, -2000.0, 1000.0),
            player(3, -2000.0, -1000.0),
            player(4, 0.0, 500.0),
        ];
        let opp = vec![player(11, -1000.0, 0.0)];
        let world = world_with(own, opp, 1);

        let mut f = Formation::new();
        let _ = f.update(&world, &[], &PlanContext::default(), None, 0.0);
        let snapshot = f.assignment.clone();
        // A tick later, within cooldown, no trigger → assignment unchanged.
        let _ = f.update(&world, &[], &PlanContext::default(), None, 0.05);
        assert_eq!(f.assignment, snapshot);
    }

    #[test]
    fn capture_role_spares_the_deepest_defender() {
        // Loose ball in our third; the deepest defender (p2) is also nearest — a
        // greedy min-time pick would yank it and gut the goal cover. The matching
        // weighs its defensive value and elects a freer, near-enough robot, keeping
        // the deepest defender home.
        let own = vec![
            player(1, -4400.0, 0.0),    // keeper
            player(2, -3000.0, 100.0),  // deepest + nearest to the ball
            player(3, -2600.0, -300.0), // midfielder, slightly farther
            player(4, -1500.0, 800.0),
            player(5, 1500.0, 0.0),
        ];
        let opp = vec![player(11, -2000.0, 0.0)];
        // Ball loose deep in our half (world_with places it at origin, so build a
        // world with the ball where we want it via a contest-free snapshot).
        let world = World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: Vector2::new(-2800.0, 0.0),
                velocity: Vector2::new(0.0, 0.0),
                detected: true,
            }),
            own_players: own,
            opp_players: opp,
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: Some(PlayerId::new(1)),
            freekick_kicker: None,
            possession: Possession::Loose,
            possession_stale: false,
            ball_contest: None,
        });

        let mut f = Formation::new();
        let capture = CaptureRole {
            pos: Vector2::new(-2800.0, 0.0),
            importance: config::CAPTURE_IMPORTANCE,
            ineligible: Vec::new(),
        };
        let out = f.update(&world, &[], &PlanContext::default(), Some(&capture), 0.0);

        let capturer = out.capturer.expect("a near robot should take the ball");
        assert_ne!(
            capturer,
            PlayerId::new(2),
            "the deepest defender must be spared, not sent to the ball"
        );
        // And it must not also be emitted a positioning command.
        assert!(!out.commands.iter().any(|c| c.id == capturer));
    }
}
