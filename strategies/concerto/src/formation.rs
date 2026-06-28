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
    Support,
    /// Central box-runner: a single advanced body in front of the opponent box, the
    /// cutback target / rebound crasher. Staffed only while attacking.
    Striker,
    Spread,
}

fn role_name(kind: RoleKind) -> &'static str {
    match kind {
        RoleKind::Capture => "capture",
        RoleKind::Shadow => "shadow",
        RoleKind::Mark => "mark",
        RoleKind::Support => "support",
        RoleKind::Striker => "striker",
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
                                let barred = capture
                                    .map(|c| c.ineligible.contains(id))
                                    .unwrap_or(true);
                                if barred || t > config::CAPTURE_TIME_HORIZON {
                                    return f64::INFINITY;
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

        // 1. Shadow / goal coverage (coordinated set), count scales with threat.
        let span = (config::SHADOW_MAX - config::SHADOW_MIN) as f64;
        let mut k = config::SHADOW_MIN + (span * ball_threat).round() as usize;
        // A plan robot contesting the ball in our defensive zone provides the
        // front-line pressure a shadow would — count it as one, so shadows don't
        // pile up directly behind the snatcher (the reported clustering).
        let contesting_def = contest_pts
            .iter()
            .filter(|p| {
                geometry::threat(
                    **p,
                    own_goal,
                    config::THREAT_GOAL_NEAR,
                    config::THREAT_GOAL_FAR,
                ) > config::SHADOW_RELIEF_THREAT
            })
            .count();
        k = k.saturating_sub(contesting_def).max(config::SHADOW_MIN);
        let positions = geometry::shadow_arc(
            ball,
            own_goal,
            k,
            config::SHADOW_STANDOFF,
            config::SHADOW_SPACING,
        );
        for (i, pos) in positions.into_iter().enumerate() {
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Shadow,
                    slot: i as u16,
                },
                position: pos,
                importance: config::IMP_SHADOW_BASE * (0.5 + 0.5 * ball_threat),
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
                // Skip opponents that pose no real threat (far up their own half);
                // an importance-0 mark role is still assignable and would lure an
                // up-field robot away from defending.
                if opp_threat < config::MARK_MIN_THREAT {
                    continue;
                }
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
        // When the ball is in the opponent half and our own goal is not under
        // threat, commit more bodies forward as supporters (keeper + one shadow
        // stay home) and place them advanced/wide — a deep carrier near the box
        // can't use a strictly-forward outlet, so we flank the goal for crosses.
        let attacking = ball.x > config::SUPPORT_ATTACK_BALL_X
            && ball_threat < config::SUPPORT_ATTACK_MAX_THREAT;
        // When attacking, one forward body becomes the central box-runner (added
        // below), so stage one fewer *wide* supporter to keep the body-count
        // balanced — central presence replaces a wing, it doesn't add a fourth.
        let support_count = if attacking {
            config::SUPPORT_ATTACK_COUNT.saturating_sub(1)
        } else {
            config::SUPPORT_COUNT
        };
        let support_imp = if attacking {
            config::IMP_SUPPORT_ATTACK
        } else {
            config::IMP_SUPPORT
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
        if attacking {
            let front_x = world.opp_goal_center().x - (opp_pen_depth + config::BOX_RUNNER_FRONT_MARGIN);
            // Bias to the side away from the ball; default to +y when the ball is
            // dead-central so the choice is still deterministic.
            let away = if ball.y > 0.0 { -1.0 } else { 1.0 };
            let pos = Vector2::new(front_x, away * config::BOX_RUNNER_Y_OFFSET);
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Striker,
                    slot: 0,
                },
                position: pos,
                importance: config::IMP_STRIKER,
                face: world.opp_goal_center(),
            });
        }

        // (No receiver role: Model B — the planner picks a concrete receiver, it
        // becomes a plan slot, and the PassCoordinator positions it. Formation
        // staffing one too would put a second robot on the receive area.)

        // 4. Spread/residual — top up to the over-generation target.
        let target = (config::OVERGEN_FACTOR * n as f64).ceil() as usize;
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

        let strikers: Vec<&Role> = roles.iter().filter(|r| r.id.kind == RoleKind::Striker).collect();
        assert_eq!(strikers.len(), 1, "exactly one box-runner when attacking");
        let s = strikers[0];
        // Just in front of the opponent box, central.
        let box_front = world.opp_penalty_area().min.x;
        assert!(
            s.position.x < box_front && s.position.x > box_front - 600.0,
            "box-runner sits just outside the box front, got x={}",
            s.position.x
        );
        // Ball is at +y, so the runner biases to -y (open cutback side).
        assert!(s.position.y < 0.0, "runner should bias away from the ball side");
        // Wide supporters reduced from SUPPORT_ATTACK_COUNT to that minus one.
        let supports = roles.iter().filter(|r| r.id.kind == RoleKind::Support).count();
        assert_eq!(supports, config::SUPPORT_ATTACK_COUNT - 1);
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
