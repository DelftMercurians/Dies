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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum RoleKind {
    Shadow,
    Mark,
    Support,
    Receiver,
    Spread,
}

fn role_name(kind: RoleKind) -> &'static str {
    match kind {
        RoleKind::Shadow => "shadow",
        RoleKind::Mark => "mark",
        RoleKind::Support => "support",
        RoleKind::Receiver => "receiver",
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
    last_plan_ctx: Option<Vector2>,
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
            last_plan_ctx: None,
        }
    }

    /// Compute positioning for all field robots except the keeper and plan slots.
    pub fn update(
        &mut self,
        world: &World,
        plan_slots: &[PlayerId],
        plan_ctx: Option<Vector2>,
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
        let plan_ctx_changed = plan_ctx_differs(self.last_plan_ctx, plan_ctx);
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
        self.last_plan_ctx = plan_ctx;

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
    fn generate_roles(&self, world: &World, plan_ctx: Option<Vector2>, n: usize) -> Vec<Role> {
        let mut roles: Vec<Role> = Vec::new();
        let own_goal = world.own_goal_center();
        let ball = world
            .ball_position()
            .unwrap_or_else(|| Vector2::new(0.0, 0.0));
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;
        let half_goal = world.goal_width() / 2.0;

        let ball_threat = geometry::threat(
            ball,
            own_goal,
            config::THREAT_GOAL_NEAR,
            config::THREAT_GOAL_FAR,
        );

        // 1. Shadow / goal coverage (coordinated set), count scales with threat.
        let span = (config::SHADOW_MAX - config::SHADOW_MIN) as f64;
        let k = config::SHADOW_MIN + (span * ball_threat).round() as usize;
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
            if let Some(opp) = world.opp_player(*oid) {
                let to_ball = ball - opp.position;
                let n_tg = to_ball.norm();
                let pos = if n_tg > 1e-6 {
                    opp.position + to_ball / n_tg * config::MARK_STANDOFF
                } else {
                    opp.position
                };
                let opp_threat = geometry::threat(
                    opp.position,
                    own_goal,
                    config::THREAT_GOAL_NEAR,
                    config::THREAT_GOAL_FAR,
                );
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

        // 3. Offensive support — flank-split forward positions.
        for slot in 0..config::SUPPORT_COUNT {
            let sign = if slot % 2 == 0 { 1.0 } else { -1.0 };
            let mut pos = Vector2::new(half_len * 0.5, sign * half_wid * 0.3);
            if let Some(opp) = world.closest_opp_player_to(pos) {
                let away = pos - opp.position;
                let d = away.norm();
                if d < config::SUPPORT_AVOID_RANGE && d > 1e-6 {
                    pos += away / d * (config::SUPPORT_AVOID_RANGE - d) * 0.5;
                }
            }
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Support,
                    slot: slot as u16,
                },
                position: pos,
                importance: config::IMP_SUPPORT,
                face: world.opp_goal_center(),
            });
        }

        // 4. Plan-context receiver role (passing milestone). None in v1.
        if let Some(area) = plan_ctx {
            roles.push(Role {
                id: RoleId {
                    kind: RoleKind::Receiver,
                    slot: 0,
                },
                position: area,
                importance: config::IMP_RECEIVER,
                face: ball,
            });
        }

        // 5. Spread/residual — top up to the over-generation target.
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

        roles
    }
}

/// Whether two plan-context areas differ enough to count as a recalc trigger.
fn plan_ctx_differs(a: Option<Vector2>, b: Option<Vector2>) -> bool {
    match (a, b) {
        (None, None) => false,
        (Some(_), None) | (None, Some(_)) => true,
        (Some(x), Some(y)) => (x - y).norm() > config::PLAN_CTX_MOVE_EPS,
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
        let cmds = f.update(&world, &plan_slots, None, 0.0);

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
        let _ = f.update(&world, &[], None, 0.0);
        let snapshot = f.assignment.clone();
        // A tick later, within cooldown, no trigger → assignment unchanged.
        let _ = f.update(&world, &[], None, 0.05);
        assert_eq!(f.assignment, snapshot);
    }
}
