//! ORCA — Optimal Reciprocal Collision Avoidance (van den Berg et al.).
//!
//! The reactive layer: given each robot's *preferred* velocity (the MTP output
//! toward its planned waypoint), find the velocity closest to it that won't
//! cause a collision within the horizon τ, assuming the other robots also
//! avoid. Each neighbour and each static obstacle contributes one half-plane
//! constraint on the new velocity; a small 2D linear program picks the optimum.
//!
//! Reciprocity between own robots is **motion-gated**: two moving robots split
//! avoidance 50/50 (this is what kills the mutual-dodge dance the old MTP had),
//! but a stationary robot does not yield — the mover takes the full burden, and
//! a robot holding position ignores movers entirely. Opponents always get full
//! responsibility (we take the whole burden, since they don't cooperate).
//! Static obstacles (walls, defense areas, ball) are folded in as one linearised
//! half-plane each, capping the approach speed by the remaining clearance.
//!
//! The LP is the canonical incremental method; on infeasibility (robots already
//! overlapping) it falls back to the 3-D LP that returns the least-penetrating
//! velocity, so `solve` is total and never returns NaN. ORCA may command a
//! velocity jump larger than `max_accel·dt` — that's intentional; the downstream
//! per-tick acceleration clamp in the player controller tracks toward it.

use dies_core::{AvoidanceConfig, PlayerId, Vector2};

use super::obstacle::{DynamicAgent, StaticObstacle};

const RVO_EPSILON: f64 = 1.0e-5;

/// One robot's ORCA inputs for this tick.
pub struct OrcaAgent {
    pub id: PlayerId,
    pub position: Vector2,
    pub velocity: Vector2,
    /// Preferred velocity == MTP output toward the planned waypoint.
    pub pref_velocity: Vector2,
    pub radius: f64,
    pub max_speed: f64,
    pub neighbors: Vec<DynamicAgent>,
    pub statics: Vec<StaticObstacle>,
}

pub struct OrcaSolver {
    tau: f64,
    neighbor_dist: f64,
    max_neighbors: usize,
    clearance: f64,
    stationary_speed: f64,
    prefer_steering: bool,
}

impl OrcaSolver {
    pub fn new(cfg: &AvoidanceConfig) -> Self {
        Self {
            tau: cfg.time_horizon.max(1.0e-3),
            neighbor_dist: cfg.neighbor_dist,
            max_neighbors: cfg.max_neighbors,
            clearance: cfg.robot_clearance,
            stationary_speed: cfg.stationary_speed,
            prefer_steering: cfg.prefer_steering,
        }
    }

    pub fn update_settings(&mut self, cfg: &AvoidanceConfig) {
        self.tau = cfg.time_horizon.max(1.0e-3);
        self.neighbor_dist = cfg.neighbor_dist;
        self.max_neighbors = cfg.max_neighbors;
        self.clearance = cfg.robot_clearance;
        self.stationary_speed = cfg.stationary_speed;
        self.prefer_steering = cfg.prefer_steering;
    }

    /// Collision-free velocity closest to `pref_velocity`, clamped to
    /// `max_speed`. Total: returns the least-penetrating velocity on
    /// infeasibility.
    pub fn solve(&self, agent: &OrcaAgent, dt: f64) -> Vector2 {
        // A robot with no intent holds position — it must not be nudged by ORCA
        // (a stationary robot does not respond to movers). Without this, the
        // responsibility-0 constraint anchored at the robot's noisy measured
        // velocity emits a small nonzero "safe" velocity that the output tracker
        // then integrates into a slow random drift.
        if agent.pref_velocity.norm() < 1.0e-3 {
            return Vector2::zeros();
        }

        let mut lines: Vec<Line> = Vec::new();

        // Static obstacle half-planes first — these are the "hard" lines the
        // 3-D fallback keeps inviolable.
        for s in &agent.statics {
            if let Some(line) = self.static_line(agent, s) {
                lines.push(line);
            }
        }
        let num_obst_lines = lines.len();

        // Nearest neighbours within range, capped.
        let mut neigh: Vec<&DynamicAgent> = agent
            .neighbors
            .iter()
            .filter(|o| (o.position - agent.position).norm() <= self.neighbor_dist)
            .collect();
        neigh.sort_by(|a, b| {
            let da = (a.position - agent.position).norm_squared();
            let db = (b.position - agent.position).norm_squared();
            da.total_cmp(&db)
        });
        neigh.truncate(self.max_neighbors);
        for other in neigh {
            lines.push(self.agent_line(agent, other, dt));
        }

        let pref = self.biased_pref(agent);
        let mut result = Vector2::zeros();
        let line_fail = linear_program2(&lines, agent.max_speed, pref, false, &mut result);
        if line_fail < lines.len() {
            linear_program3(
                &lines,
                num_obst_lines,
                line_fail,
                agent.max_speed,
                &mut result,
            );
        }

        if result.norm() < 1.0e-3 {
            return Vector2::zeros();
        }
        result
    }

    /// Linearised half-plane for a static obstacle: cap the velocity component
    /// toward the obstacle so clearance can't breach `radius + clearance` within
    /// τ. Constraint `grad·v ≥ (R_static − d)/τ`, with `grad` pointing away from
    /// the obstacle.
    fn static_line(&self, agent: &OrcaAgent, s: &StaticObstacle) -> Option<Line> {
        let (d, grad) = s.shape.clearance(agent.position);
        let g_norm = grad.norm();
        if g_norm < 1.0e-9 {
            return None;
        }
        let n = grad / g_norm;
        let r_static = agent.radius + self.clearance;
        let c = (r_static - d) / self.tau;
        // Feasible region {v : n·v ≥ c}. Line direction perpendicular to n with
        // the feasible side on the left; boundary point c·n.
        Some(Line {
            point: n * c,
            direction: Vector2::new(n.y, -n.x),
        })
    }

    /// Reciprocal ORCA half-plane for a neighbouring robot. Own robots split the
    /// avoidance (responsibility 0.5); opponents get full responsibility.
    fn agent_line(&self, agent: &OrcaAgent, other: &DynamicAgent, dt: f64) -> Line {
        let inv_tau = 1.0 / self.tau;
        let rel_position = other.position - agent.position;
        let rel_velocity = agent.velocity - other.velocity;
        let dist_sq = rel_position.norm_squared();
        let combined_radius = agent.radius + other.radius + self.clearance;
        let combined_radius_sq = combined_radius * combined_radius;

        let direction;
        let u;
        if dist_sq > combined_radius_sq {
            // No collision: project relative velocity onto the velocity-obstacle
            // boundary (cut-off circle or one of the legs).
            let w = rel_velocity - inv_tau * rel_position;
            let w_len_sq = w.norm_squared();
            let dot1 = w.dot(&rel_position);
            // The cut-off-circle projection is the "slow down and wait" branch;
            // the leg projection is the "steer around" branch. `prefer_steering`
            // suppresses the former so a constrained robot deflects at speed
            // instead of crawling. Genuinely boxed-in cases still resolve via the
            // 3-D LP fallback.
            let use_cutoff =
                !self.prefer_steering && dot1 < 0.0 && dot1 * dot1 > combined_radius_sq * w_len_sq;
            if use_cutoff {
                // Project on cut-off circle.
                let w_len = w_len_sq.sqrt();
                let unit_w = if w_len > 1.0e-9 {
                    w / w_len
                } else {
                    Vector2::new(1.0, 0.0)
                };
                direction = Vector2::new(unit_w.y, -unit_w.x);
                u = (combined_radius * inv_tau - w_len) * unit_w;
            } else {
                // Project on a leg of the VO cone. Standard ORCA picks the leg
                // nearest the current relative velocity (least deflection). When
                // steering is preferred that choice is unstable for a near-head-on
                // approach (the relative velocity points almost at the obstacle,
                // so the nearest leg flips frame to frame and the robot orbits /
                // stalls). Pick a stable side instead: the side the goal is on,
                // breaking an exact tie by robot-id parity.
                let leg = (dist_sq - combined_radius_sq).sqrt();
                let go_left = if self.prefer_steering {
                    let side = det(rel_position, agent.pref_velocity);
                    if side.abs() > 1.0 {
                        side > 0.0
                    } else {
                        agent.id.as_u32() % 2 == 0
                    }
                } else {
                    det(rel_position, w) > 0.0
                };
                if go_left {
                    direction = Vector2::new(
                        rel_position.x * leg - rel_position.y * combined_radius,
                        rel_position.x * combined_radius + rel_position.y * leg,
                    ) / dist_sq;
                } else {
                    direction = -Vector2::new(
                        rel_position.x * leg + rel_position.y * combined_radius,
                        -rel_position.x * combined_radius + rel_position.y * leg,
                    ) / dist_sq;
                }
                let dot2 = rel_velocity.dot(&direction);
                u = dot2 * direction - rel_velocity;
            }
        } else {
            // Already colliding: resolve over a single timestep instead of τ.
            let inv_dt = 1.0 / dt.max(1.0e-3);
            let w = rel_velocity - inv_dt * rel_position;
            let w_len = w.norm();
            let unit_w = if w_len > 1.0e-9 {
                w / w_len
            } else {
                Vector2::new(1.0, 0.0)
            };
            direction = Vector2::new(unit_w.y, -unit_w.x);
            u = (combined_radius * inv_dt - w_len) * unit_w;
        }

        let responsibility = self.responsibility(agent, other);
        Line {
            point: agent.velocity + responsibility * u,
            direction,
        }
    }

    /// How much of the avoidance the ego takes for this neighbour.
    ///
    /// - Opponents: always 1.0 — they don't cooperate, we take the whole burden.
    /// - Own robots: reciprocity is **motion-gated**. A stationary robot does
    ///   not yield to a moving one, so the mover takes the full burden:
    ///   - ego holding (pref ≈ 0), other moving → 0.0 (don't respond)
    ///   - ego moving, other stationary → 1.0 (we do all the work)
    ///   - both moving → 0.5 (classic reciprocity)
    ///   - both holding → 0.5 (no real interaction either way)
    fn responsibility(&self, agent: &OrcaAgent, other: &DynamicAgent) -> f64 {
        if !other.is_own {
            return 1.0;
        }
        // Ego intent from its preferred velocity; neighbour motion from its
        // observed velocity (we can't see its intent).
        let ego_moving = agent.pref_velocity.norm() > self.stationary_speed;
        let other_moving = other.velocity.norm() > self.stationary_speed;
        match (ego_moving, other_moving) {
            (false, true) => 0.0,
            (true, false) => 1.0,
            _ => 0.5,
        }
    }

    /// Tiny perpendicular bias on the preferred velocity, keyed on robot id
    /// parity, to break head-on symmetry that would otherwise deadlock two
    /// robots oscillating into each other. Negligible (~1°) for tracking.
    fn biased_pref(&self, agent: &OrcaAgent) -> Vector2 {
        let p = agent.pref_velocity;
        if p.norm() < 1.0e-6 || agent.neighbors.is_empty() {
            return p;
        }
        let sign: f64 = if agent.id.as_u32() % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let ang = 0.02 * sign;
        let (c, s) = (ang.cos(), ang.sin());
        Vector2::new(c * p.x - s * p.y, s * p.x + c * p.y)
    }
}

/// A directed constraint line; feasible region is to the left of `direction`.
#[derive(Clone, Copy, Default)]
struct Line {
    point: Vector2,
    direction: Vector2,
}

fn det(a: Vector2, b: Vector2) -> f64 {
    a.x * b.y - a.y * b.x
}

/// 1-D LP along line `line_no`, intersected with the speed circle and all prior
/// constraints. Returns false if infeasible.
fn linear_program1(
    lines: &[Line],
    line_no: usize,
    radius: f64,
    opt_velocity: Vector2,
    direction_opt: bool,
    result: &mut Vector2,
) -> bool {
    let line = lines[line_no];
    let dot_product = line.point.dot(&line.direction);
    let discriminant = dot_product * dot_product + radius * radius - line.point.norm_squared();
    if discriminant < 0.0 {
        // Max-speed circle fully invalidates this line.
        return false;
    }
    let sqrt_disc = discriminant.sqrt();
    let mut t_left = -dot_product - sqrt_disc;
    let mut t_right = -dot_product + sqrt_disc;

    for i in 0..line_no {
        let denominator = det(line.direction, lines[i].direction);
        let numerator = det(lines[i].direction, line.point - lines[i].point);
        if denominator.abs() <= RVO_EPSILON {
            // Nearly parallel.
            if numerator < 0.0 {
                return false;
            }
            continue;
        }
        let t = numerator / denominator;
        if denominator >= 0.0 {
            t_right = t_right.min(t);
        } else {
            t_left = t_left.max(t);
        }
        if t_left > t_right {
            return false;
        }
    }

    if direction_opt {
        if opt_velocity.dot(&line.direction) > 0.0 {
            *result = line.point + t_right * line.direction;
        } else {
            *result = line.point + t_left * line.direction;
        }
    } else {
        let t = line.direction.dot(&(opt_velocity - line.point));
        if t < t_left {
            *result = line.point + t_left * line.direction;
        } else if t > t_right {
            *result = line.point + t_right * line.direction;
        } else {
            *result = line.point + t * line.direction;
        }
    }
    true
}

/// 2-D LP: velocity closest to `opt_velocity` satisfying all `lines`, within the
/// speed circle. Returns `lines.len()` on success or the index of the first
/// infeasible line.
fn linear_program2(
    lines: &[Line],
    radius: f64,
    opt_velocity: Vector2,
    direction_opt: bool,
    result: &mut Vector2,
) -> usize {
    if direction_opt {
        *result = opt_velocity * radius;
    } else if opt_velocity.norm_squared() > radius * radius {
        *result = opt_velocity.normalize() * radius;
    } else {
        *result = opt_velocity;
    }

    for i in 0..lines.len() {
        if det(lines[i].direction, lines[i].point - *result) > 0.0 {
            let temp = *result;
            if !linear_program1(lines, i, radius, opt_velocity, direction_opt, result) {
                *result = temp;
                return i;
            }
        }
    }
    lines.len()
}

/// 3-D LP fallback: on infeasibility, return the velocity minimising the maximum
/// penetration, keeping the first `num_obst_lines` (static obstacles)
/// inviolable.
fn linear_program3(
    lines: &[Line],
    num_obst_lines: usize,
    begin_line: usize,
    radius: f64,
    result: &mut Vector2,
) {
    let mut distance = 0.0;
    for i in begin_line..lines.len() {
        if det(lines[i].direction, lines[i].point - *result) > distance {
            let mut proj_lines: Vec<Line> = lines[0..num_obst_lines].to_vec();
            for j in num_obst_lines..i {
                let mut line = Line::default();
                let determinant = det(lines[i].direction, lines[j].direction);
                if determinant.abs() <= RVO_EPSILON {
                    if lines[i].direction.dot(&lines[j].direction) > 0.0 {
                        continue; // Same direction.
                    }
                    line.point = 0.5 * (lines[i].point + lines[j].point);
                } else {
                    line.point = lines[i].point
                        + (det(lines[j].direction, lines[i].point - lines[j].point) / determinant)
                            * lines[i].direction;
                }
                let dir = lines[j].direction - lines[i].direction;
                line.direction = if dir.norm() > 1.0e-9 {
                    dir.normalize()
                } else {
                    dir
                };
                proj_lines.push(line);
            }
            let temp = *result;
            let opt = Vector2::new(-lines[i].direction.y, lines[i].direction.x);
            if linear_program2(&proj_lines, radius, opt, true, result) < proj_lines.len() {
                // Should not happen; keep the safest velocity found so far.
                *result = temp;
            }
            distance = det(lines[i].direction, lines[i].point - *result);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::obstacle::{DynamicAgent, ObstacleKind, ObstacleShape, StaticObstacle};
    use super::{OrcaAgent, OrcaSolver};
    use dies_core::{AvoidanceConfig, PlayerId, Vector2};

    fn solver() -> OrcaSolver {
        OrcaSolver::new(&AvoidanceConfig::default())
    }

    fn agent(pos: Vector2, vel: Vector2, pref: Vector2) -> OrcaAgent {
        OrcaAgent {
            id: PlayerId::new(0),
            position: pos,
            velocity: vel,
            pref_velocity: pref,
            radius: 90.0,
            max_speed: 3000.0,
            neighbors: Vec::new(),
            statics: Vec::new(),
        }
    }

    fn own(pos: Vector2, vel: Vector2) -> DynamicAgent {
        DynamicAgent {
            is_own: true,
            position: pos,
            velocity: vel,
            radius: 90.0,
        }
    }

    #[test]
    fn no_constraints_passes_pref_through() {
        let a = agent(
            Vector2::zeros(),
            Vector2::zeros(),
            Vector2::new(1000.0, 0.0),
        );
        let v = solver().solve(&a, 0.02);
        assert!((v - a.pref_velocity).norm() < 1.0, "got {:?}", v);
    }

    #[test]
    fn stationary_ego_does_not_respond() {
        // Ego holding position (pref ≈ 0) while a teammate drives at it: the ego
        // must not dodge — the mover takes the full burden.
        let mut a = agent(Vector2::zeros(), Vector2::zeros(), Vector2::zeros());
        a.neighbors
            .push(own(Vector2::new(300.0, 0.0), Vector2::new(-1500.0, 0.0)));
        let v = solver().solve(&a, 0.02);
        assert!(
            v.norm() < 1.0,
            "stationary robot should stay put, got {:?}",
            v
        );
    }

    #[test]
    fn mover_takes_full_burden_vs_stationary() {
        // Ego driving toward a stationary teammate must deflect (the teammate
        // won't yield).
        let mut a = agent(
            Vector2::new(0.0, 0.0),
            Vector2::new(1000.0, 0.0),
            Vector2::new(1000.0, 0.0),
        );
        a.neighbors
            .push(own(Vector2::new(1000.0, 0.0), Vector2::zeros()));
        let v = solver().solve(&a, 0.02);
        assert!(
            (v - a.pref_velocity).norm() > 1.0,
            "expected deflection around stationary robot, got {:?}",
            v
        );
    }

    #[test]
    fn head_on_pair_deflects() {
        // Two own robots driving straight at each other: ORCA must steer the ego
        // away from a pure head-on velocity.
        let mut a = agent(
            Vector2::new(0.0, 0.0),
            Vector2::new(1000.0, 0.0),
            Vector2::new(1000.0, 0.0),
        );
        a.neighbors
            .push(own(Vector2::new(1000.0, 0.0), Vector2::new(-1000.0, 0.0)));
        let v = solver().solve(&a, 0.02);
        assert!(
            (v - a.pref_velocity).norm() > 1.0,
            "expected deflection, got {:?}",
            v
        );
        assert!(v.x.is_finite() && v.y.is_finite());
    }

    /// A right-hand keep-in wall at `x = offset` (normal points to the forbidden
    /// +x side, so its clearance decreases as the robot approaches).
    fn wall(offset: f64) -> StaticObstacle {
        StaticObstacle {
            shape: ObstacleShape::HalfPlane {
                normal: Vector2::new(1.0, 0.0),
                offset,
            },
            kind: ObstacleKind::Wall,
        }
    }

    #[test]
    fn wall_parallel_motion_passes_through() {
        // Well inside the field, driving parallel to the wall → non-binding.
        let mut a = agent(
            Vector2::zeros(),
            Vector2::zeros(),
            Vector2::new(0.0, 1000.0),
        );
        a.statics.push(wall(1000.0));
        let v = solver().solve(&a, 0.02);
        assert!((v - a.pref_velocity).norm() < 1.0, "got {:?}", v);
    }

    #[test]
    fn wall_caps_inward_velocity() {
        // Robot edge already past the keep-in plane (d < radius): inward (+x)
        // velocity toward the wall must be forced non-positive.
        let mut a = agent(
            Vector2::new(950.0, 0.0),
            Vector2::zeros(),
            Vector2::new(2000.0, 0.0),
        );
        a.statics.push(wall(1000.0));
        let v = solver().solve(&a, 0.02);
        assert!(v.x < 0.0, "should be pushed back off the wall, got {:?}", v);
        assert!(v.x.is_finite() && v.y.is_finite());
    }

    #[test]
    fn wall_recovers_when_outside() {
        // Robot penetrated past the plane, commanded inward → strong inward, finite.
        let mut a = agent(
            Vector2::new(1050.0, 0.0),
            Vector2::zeros(),
            Vector2::new(-500.0, 0.0),
        );
        a.statics.push(wall(1000.0));
        let v = solver().solve(&a, 0.02);
        assert!(v.x < 0.0, "should recover inward, got {:?}", v);
        assert!(v.x.is_finite() && v.y.is_finite());
    }

    #[test]
    fn overlapping_robots_finite() {
        // Robots interpenetrating (distance < combined radius): the 3-D fallback
        // must still produce a finite velocity.
        let mut a = agent(
            Vector2::new(0.0, 0.0),
            Vector2::zeros(),
            Vector2::new(1000.0, 0.0),
        );
        a.neighbors
            .push(own(Vector2::new(50.0, 0.0), Vector2::zeros()));
        let v = solver().solve(&a, 0.02);
        assert!(v.x.is_finite() && v.y.is_finite(), "got {:?}", v);
        assert!(v.norm() <= a.max_speed + 1.0);
    }
}
