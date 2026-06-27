//! Formation — positions all field robots except the keeper and the active
//! (plan-controlled) robots.
//!
//! Role generators produce smoothly-varying positions with continuous importance;
//! robots are matched to roles by minimum total cost, where cost blends a
//! momentum-aware redirect time with the role's importance. Stability comes from
//! three physical sources, never a stay-bonus:
//!   1. continuity — role positions are continuous functions of the world;
//!   2. redirect cost — a moving robot continues cheaply, reverses expensively;
//!   3. cadence — assignments are recomputed only on events (cooldown-bounded),
//!      while positions update every tick so motion stays smooth.
//!
//! v1 excludes plan-controlled robots from the assignable set (equivalent to the
//! plan-slot model for a single active robot, and simpler). The passing milestone
//! switches to plan-slot accounting when two robots become plan-controlled.

use std::collections::HashMap;

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use crate::config;
use crate::geometry;
use crate::matching::assign_min_cost;
use crate::planner::{Engagement, PlanContext};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum RoleKind {
    Shadow,
    Mark,
    Support,
    Spread,
}

fn role_name(kind: RoleKind) -> &'static str {
    match kind {
        RoleKind::Shadow => "shadow",
        RoleKind::Mark => "mark",
        RoleKind::Support => "support",
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

pub struct Formation {
    /// Frozen assignment from the last recalc; positions re-resolved each tick.
    assignment: HashMap<PlayerId, RoleId>,
    last_recalc: f64,
    queued: bool,
    // Change-detection snapshots for recalc triggers.
    last_assignable: Vec<PlayerId>,
    last_plan_slots: Vec<PlayerId>,
    last_plan_ctx: PlanContext,
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
            last_plan_slots: Vec::new(),
            last_plan_ctx: PlanContext::default(),
        }
    }

    /// Compute positioning for all field robots except the keeper and plan slots.
    pub fn update(
        &mut self,
        world: &World,
        plan_slots: &[PlayerId],
        plan_ctx: &PlanContext,
        now: f64,
    ) -> Vec<FormationCommand> {
        let keeper_id = world.our_keeper_id();

        // Assignable = own field robots minus keeper minus plan-controlled robots.
        let mut assignable: Vec<PlayerId> = world
            .own_players()
            .iter()
            .map(|p| p.id)
            .filter(|id| Some(*id) != keeper_id && !plan_slots.contains(id))
            .collect();
        assignable.sort_by_key(|id| id.as_u32());

        // Generate the role set (positions/importance recomputed every tick).
        let roles = self.generate_roles(world, plan_ctx, assignable.len());
        let role_by_id: HashMap<RoleId, &Role> = roles.iter().map(|r| (r.id, r)).collect();

        // ── Decide whether to recompute the assignment (cadence) ────────
        let assignable_changed = assignable != self.last_assignable;
        let plan_slots_changed = plan_slots != self.last_plan_slots.as_slice();
        let plan_ctx_changed = plan_ctx_differs(&self.last_plan_ctx, plan_ctx);
        let bg_due = now - self.last_recalc >= config::RECALC_BG_PERIOD;
        if assignable_changed || plan_slots_changed || plan_ctx_changed || bg_due {
            self.queued = true;
        }
        let cooldown_ok = now - self.last_recalc >= config::RECALC_COOLDOWN;
        let stale = self.assignment.is_empty() && !assignable.is_empty();
        if (self.queued && cooldown_ok) || stale {
            self.recompute(world, &assignable, &roles);
            self.last_recalc = now;
            self.queued = false;
        }

        self.last_assignable = assignable.clone();
        self.last_plan_slots = plan_slots.to_vec();
        self.last_plan_ctx = plan_ctx.clone();

        // ── Emit commands: resolve each robot's RoleId to its current position ──
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
            commands.push(FormationCommand {
                id: *id,
                target: role.position,
                face: role.face,
                role: role_name(role.id.kind),
            });
        }
        commands
    }

    /// Run the cost-aware matching and store the new assignment.
    fn recompute(&mut self, world: &World, assignable: &[PlayerId], roles: &[Role]) {
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
                            geometry::redirect_time(
                                p.position,
                                p.velocity,
                                role.position,
                                config::V_MAX,
                                config::A_MAX,
                            ) - role.importance * config::SEC_PER_IMPORTANCE
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
    fn generate_roles(&self, world: &World, plan_ctx: &PlanContext, n: usize) -> Vec<Role> {
        let mut roles: Vec<Role> = Vec::new();
        let own_goal = world.own_goal_center();
        let ball = world
            .ball_position()
            .unwrap_or_else(|| Vector2::new(0.0, 0.0));
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;
        let half_goal = world.goal_width() / 2.0;

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
        let positions = geometry::shadow_arc(ball, own_goal, k, config::SHADOW_STANDOFF, half_goal);
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
        let support_count = if attacking {
            config::SUPPORT_ATTACK_COUNT
        } else {
            config::SUPPORT_COUNT
        };
        let support_imp = if attacking {
            config::IMP_SUPPORT_ATTACK
        } else {
            config::IMP_SUPPORT
        };
        for slot in 0..support_count {
            // 0 → right flank, 1 → left flank, 2 → central trailer (sign 0 keeps the
            // y candidates on the centre column). Distinct signs stop two supporters
            // resolving to the same `best_support_pos` and stacking.
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
            );
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
    roles.iter().min_by(|a, b| {
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
        let plan_slots = vec![PlayerId::new(5)];
        let cmds = f.update(&world, &plan_slots, &PlanContext::default(), 0.0);

        // Assignable = ids {2,3,4} (exclude keeper 1 and plan slot 5).
        let mut assigned: Vec<u32> = cmds.iter().map(|c| c.id.as_u32()).collect();
        assigned.sort_unstable();
        assert_eq!(assigned, vec![2, 3, 4]);
        // Each gets exactly one command, all finite.
        assert_eq!(cmds.len(), 3);
        for c in &cmds {
            assert!(c.target.x.is_finite() && c.target.y.is_finite());
        }
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
        let plain = f.generate_roles(&world, &PlanContext::default(), 3);
        let engaged = f.generate_roles(
            &world,
            &PlanContext {
                engagements: vec![Engagement::Engaging {
                    opponent: PlayerId::new(11),
                }],
            },
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

        let plain = f.generate_roles(&world, &PlanContext::default(), 4);
        let supp = f.generate_roles(
            &world,
            &PlanContext {
                engagements: vec![Engagement::BallContest {
                    at: Vector2::new(0.0, 0.0),
                }],
            },
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
        let _ = f.update(&world, &[], &PlanContext::default(), 0.0);
        let snapshot = f.assignment.clone();
        // A tick later, within cooldown, no trigger → assignment unchanged.
        let _ = f.update(&world, &[], &PlanContext::default(), 0.05);
        assert_eq!(f.assignment, snapshot);
    }
}
