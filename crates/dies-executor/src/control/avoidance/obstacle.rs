//! Shared obstacle model for the avoidance stack.
//!
//! One geometry primitive ([`ObstacleShape`]) with a signed-distance + gradient
//! query ([`ObstacleShape::clearance`]) is read by **both** avoidance layers:
//! the global planner (occupancy / line-of-sight via the distance) and the ORCA
//! solver (static keep-out half-planes via the distance + gradient).
//!
//! The static/dynamic split is structural, not a flag: walls, defense areas and
//! the ball are [`StaticObstacle`]s (plain blockers); robots are
//! [`DynamicAgent`]s (reciprocal agents for ORCA, frozen-as-discs for the
//! planner). An [`ObstacleSet`] is built per robot per tick by
//! [`ObstacleSet::build`], which consolidates what the old iLQR `build_obstacles`
//! and the team controller's ad-hoc ball logic each used to do.
//!
//! Everything is in team-relative coordinates, millimetres.

use dies_core::{
    AvoidanceConfig, BallData, FieldGeometry, GameState, PlayerId, TeamData, Vector2, PLAYER_RADIUS,
};

/// Geometry of a single keep-out / keep-in region.
#[derive(Clone, Copy, Debug)]
pub enum ObstacleShape {
    /// Keep-out disk: clearance is `‖p − center‖ − radius` (safe when ≥ 0).
    Circle { center: Vector2, radius: f64 },
    /// Keep-in half-plane: clearance is `offset − normal·p` (safe when ≥ 0).
    /// `normal` is unit length and points toward the forbidden side.
    HalfPlane { normal: Vector2, offset: f64 },
    /// Axis-aligned keep-out box: clearance is the box signed distance —
    /// positive outside the box, negative inside.
    Box { min: Vector2, max: Vector2 },
}

impl ObstacleShape {
    /// Signed clearance `d` and its gradient `∂d/∂p` at position `p`. Clearance
    /// is positive in the safe region; the gradient points away from the
    /// obstacle (toward increasing clearance) and is unit length except at
    /// degenerate points. Ported from the iLQR obstacle barrier (Hessian
    /// dropped — ORCA only linearises, it doesn't need curvature).
    pub fn clearance(&self, p: Vector2) -> (f64, Vector2) {
        match *self {
            ObstacleShape::Circle { center, radius } => {
                let delta = p - center;
                let dist = delta.norm();
                if dist > 1.0e-9 {
                    (dist - radius, delta / dist)
                } else {
                    // Exactly on the centre: pick an arbitrary outward direction.
                    (-radius, Vector2::new(1.0, 0.0))
                }
            }
            ObstacleShape::HalfPlane { normal, offset } => (offset - normal.dot(&p), -normal),
            ObstacleShape::Box { min, max } => {
                let center = 0.5 * (min + max);
                let half = 0.5 * (max - min);
                let rel = p - center;
                let sgn = Vector2::new(
                    if rel.x >= 0.0 { 1.0 } else { -1.0 },
                    if rel.y >= 0.0 { 1.0 } else { -1.0 },
                );
                // q = |rel| − half, per axis. q > 0 means outside along that axis.
                let q = Vector2::new(rel.x.abs() - half.x, rel.y.abs() - half.y);
                if q.x > 0.0 || q.y > 0.0 {
                    // Outside: Euclidean distance to the nearest face / corner.
                    let qpos = Vector2::new(q.x.max(0.0), q.y.max(0.0));
                    let dist = qpos.norm();
                    if dist > 1.0e-9 {
                        let grad = Vector2::new(qpos.x * sgn.x, qpos.y * sgn.y) / dist;
                        (dist, grad)
                    } else if q.x >= q.y {
                        (q.x, Vector2::new(sgn.x, 0.0))
                    } else {
                        (q.y, Vector2::new(0.0, sgn.y))
                    }
                } else {
                    // Inside: clearance is the (negative) distance to the closest
                    // face; gradient points out along that axis.
                    if q.x >= q.y {
                        (q.x, Vector2::new(sgn.x, 0.0))
                    } else {
                        (q.y, Vector2::new(0.0, sgn.y))
                    }
                }
            }
        }
    }

    /// Signed clearance only (no gradient).
    pub fn distance(&self, p: Vector2) -> f64 {
        self.clearance(p).0
    }
}

/// A non-reciprocal blocker: walls, defense boxes, the ball, the ball-placement
/// corridor. ORCA treats it as a one-sided linearised constraint; the planner
/// treats it as solid geometry.
#[derive(Clone, Copy, Debug)]
pub struct StaticObstacle {
    pub shape: ObstacleShape,
}

impl StaticObstacle {
    fn new(shape: ObstacleShape) -> Self {
        Self { shape }
    }
}

/// A reciprocal agent: every robot other than the ego robot. For ORCA the
/// `radius` is the **single** robot radius — ORCA sums the two radii pairwise,
/// so baking `2·R` in here (as iLQR did) would double-count.
#[derive(Clone, Copy, Debug)]
pub struct DynamicAgent {
    /// Own robot → ORCA reciprocal responsibility 0.5; opponent → 1.0 (we take
    /// the full avoidance burden, since they don't cooperate).
    pub is_own: bool,
    pub position: Vector2,
    pub velocity: Vector2,
    pub radius: f64,
}

/// Everything one robot must avoid this tick.
#[derive(Clone, Debug, Default)]
pub struct ObstacleSet {
    pub statics: Vec<StaticObstacle>,
    pub agents: Vec<DynamicAgent>,
}

/// Per-robot avoidance gates, mirroring the flags the old MTP/iLQR paths used.
///
/// Robot-robot avoidance is always on (own robots are always added); these gates
/// only toggle the *other* obstacle classes.
#[derive(Clone, Copy, Debug, Default)]
pub struct AvoidanceGates {
    pub avoid_defense_area: bool,
    pub avoid_ball: bool,
    pub avoid_ball_care: f64,
    /// When false, opponents are dropped from the agent set (the waller-near-own
    /// goal exception).
    pub avoid_opp_robots: bool,
}

impl ObstacleSet {
    /// Build the obstacle set for one robot in team-relative coordinates.
    ///
    /// Consolidates the per-robot construction the iLQR wrapper did (robots →
    /// discs, walls → half-planes, defense areas → boxes) and the team
    /// controller's game-state-dependent ball logic into one place.
    pub fn build(
        world: &TeamData,
        this_id: PlayerId,
        gates: AvoidanceGates,
        game_state: GameState,
        cfg: &AvoidanceConfig,
    ) -> ObstacleSet {
        let mut set = ObstacleSet::default();

        // Robots → reciprocal agents. Own players (excluding self) are always
        // avoided; opponents only when `avoid_opp_robots` (the waller-near-own
        // goal exception turns them off).
        for p in world.own_players.iter().filter(|p| p.id != this_id) {
            set.agents.push(DynamicAgent {
                is_own: true,
                position: p.position,
                velocity: p.velocity,
                radius: PLAYER_RADIUS,
            });
        }
        if gates.avoid_opp_robots {
            for p in world.opp_players.iter() {
                set.agents.push(DynamicAgent {
                    is_own: false,
                    position: p.position,
                    velocity: p.velocity,
                    radius: PLAYER_RADIUS,
                });
            }
        }

        if let Some(field) = world.field_geom.as_ref() {
            set.push_walls(field, cfg);
            if gates.avoid_defense_area {
                set.push_defense_areas(field, cfg);
            }
        }

        set.push_ball(world.ball.as_ref(), game_state, gates, cfg);

        set
    }

    /// Field walls → keep-in half-planes, inset from the physical boundary by
    /// `wall_margin`.
    fn push_walls(&mut self, field: &FieldGeometry, cfg: &AvoidanceConfig) {
        let x_max = field.field_length / 2.0 + field.boundary_width - cfg.wall_margin;
        let y_max = field.field_width / 2.0 + field.boundary_width - cfg.wall_margin;
        for (normal, offset) in [
            (Vector2::new(1.0, 0.0), x_max),
            (Vector2::new(-1.0, 0.0), x_max),
            (Vector2::new(0.0, 1.0), y_max),
            (Vector2::new(0.0, -1.0), y_max),
        ] {
            self.statics
                .push(StaticObstacle::new(ObstacleShape::HalfPlane {
                    normal,
                    offset,
                }));
        }
    }

    /// Defense areas → keep-out boxes, extended behind the goal line so a robot
    /// near one is always pushed back out toward the field.
    fn push_defense_areas(&mut self, field: &FieldGeometry, cfg: &AvoidanceConfig) {
        let hl = field.field_length / 2.0;
        let depth = field.penalty_area_depth;
        let half_w = field.penalty_area_width / 2.0;
        let m = cfg.defense_margin;
        let back = field.boundary_width + m;
        for (min, max) in [
            (
                Vector2::new(-hl - back, -half_w - m),
                Vector2::new(-hl + depth + m, half_w + m),
            ),
            (
                Vector2::new(hl - depth - m, -half_w - m),
                Vector2::new(hl + back, half_w + m),
            ),
        ] {
            self.statics
                .push(StaticObstacle::new(ObstacleShape::Box { min, max }));
        }
    }

    /// Ball → keep-out disk (game-state-dependent radius), plus the placement
    /// corridor during `BallReplacement` (approximated as discs along the
    /// ball→target segment).
    fn push_ball(
        &mut self,
        ball: Option<&BallData>,
        game_state: GameState,
        gates: AvoidanceGates,
        cfg: &AvoidanceConfig,
    ) {
        let Some(ball) = ball else { return };
        let ball_pos = ball.position.xy();

        let want_ball = gates.avoid_ball
            || game_state == GameState::PreparePenalty
            || game_state == GameState::Stop;
        if want_ball {
            let radius = if game_state == GameState::Stop {
                cfg.ball_stop_radius
            } else {
                cfg.ball_base_radius + cfg.ball_care_scale * gates.avoid_ball_care
            };
            self.statics
                .push(StaticObstacle::new(ObstacleShape::Circle {
                    center: ball_pos,
                    radius,
                }));
        }

        if let GameState::BallReplacement(target) = game_state {
            // Keep-out corridor between the ball and its placement target. Discs
            // along the segment approximate the swept tube faithfully
            let radius = cfg.ball_stop_radius;
            let seg = target - ball_pos;
            let len = seg.norm();
            let step = (radius * 0.5).max(1.0);
            let n = (len / step).ceil() as usize;
            for i in 0..=n {
                let t = if n == 0 { 0.0 } else { i as f64 / n as f64 };
                self.statics
                    .push(StaticObstacle::new(ObstacleShape::Circle {
                        center: ball_pos + seg * t,
                        radius,
                    }));
            }
        }
    }

    /// Frozen view for the planner: dynamic agents become keep-out discs at
    /// their current position (inflated by `clearance`) alongside the static
    /// shapes. The planner treats the whole world as solid geometry and tests
    /// each shape's signed distance against the ego radius, so `clearance` here
    /// is the only agent-specific inflation; the ego radius is applied uniformly
    /// by the caller.
    pub fn as_planner_shapes(&self, clearance: f64) -> impl Iterator<Item = ObstacleShape> + '_ {
        let agent_discs = self.agents.iter().map(move |a| ObstacleShape::Circle {
            center: a.position,
            radius: a.radius + clearance,
        });
        self.statics.iter().map(|s| s.shape).chain(agent_discs)
    }
}
