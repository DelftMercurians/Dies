//! Path-following trajectory controller — replaces the old MTP.
//!
//! Consumes the *full* planned path (a polyline) plus the robot state and
//! produces a dynamically-shaped preferred velocity in one place: pure-pursuit
//! steering for the direction, and a speed envelope (cruise / cornering /
//! braking-to-goal) for the magnitude. Acceleration limiting is applied by the
//! caller (`PlayerController`) as a first-order asymmetric clamp — there is no
//! separate jerk/S-curve stage.
//!
//! Because it owns the whole speed profile, the corner/pass-through/decel
//! behaviour is intrinsic: no `is_final`/`speed_frac`/`waypoint_tolerance`
//! flags. Coordinates are team-relative, millimetres.

use dies_core::{ControllerSettings, Vector2};

/// Within this distance of the final goal the follower commands a full stop.
const ARRIVE_DEADBAND: f64 = 15.0;
/// Floor on the commanded speed outside the arrive deadband, so the terminal
/// proportional profile can't decay into the drivetrain's stiction band and
/// park the robot short of its target. Deliberately tiny — at the default
/// `approach_kp` (2.0) the profile only drops this low inside the deadband, so
/// this is a guard for low-gain configurations, not the primary anti-stiction
/// measure (skills that must arrive precisely add their own floor).
const STICTION_FLOOR: f64 = 30.0;
/// Floor on the cornering radius so a near-reversal still keeps a little speed.
const R_MIN: f64 = 90.0;
/// Allowed cornering deviation from a path vertex (junction-deviation model) [mm].
const CORNER_DEVIATION: f64 = 60.0;
/// Upper bound on the pure-pursuit lookahead [mm].
const LOOKAHEAD_MAX: f64 = 800.0;

pub struct PathFollower {
    v_max: f64,
    a_lat: f64,
    /// Baseline position approach gain from settings; the per-tick effective gain
    /// (scaled by a robot's aggressiveness) is passed into [`PathFollower::follow`].
    approach_kp: f64,
    lookahead_min: f64,
    lookahead_time: f64,
}

/// Output of the path follower: the preferred velocity plus a flag requesting
/// the caller bypass its acceleration slew clamp. `hard_brake` is set only during
/// the terminal active-braking maneuver, where the command may jump (or reverse)
/// in one tick so the firmware reverse-thrusts to a crisp stop.
#[derive(Debug, Clone, Copy)]
pub struct FollowCmd {
    pub velocity: Vector2,
    pub hard_brake: bool,
}

impl FollowCmd {
    fn cruise(velocity: Vector2) -> Self {
        Self {
            velocity,
            hard_brake: false,
        }
    }
}

impl PathFollower {
    pub fn new(s: &ControllerSettings) -> Self {
        let mut f = Self {
            v_max: 0.0,
            a_lat: 0.0,
            approach_kp: 0.0,
            lookahead_min: 0.0,
            lookahead_time: 0.0,
        };
        f.update_settings(s);
        f
    }

    pub fn update_settings(&mut self, s: &ControllerSettings) {
        self.v_max = s.max_velocity;
        self.a_lat = s.lateral_acceleration.max(1.0);
        self.approach_kp = s.approach_kp.max(0.1);
        self.lookahead_min = s.lookahead_min.max(1.0);
        self.lookahead_time = s.lookahead_time.max(0.0);
    }

    /// Baseline (un-scaled) position approach gain from settings.
    pub fn base_approach_kp(&self) -> f64 {
        self.approach_kp
    }

    /// Preferred path-following velocity \[mm/s\]: pure-pursuit direction × a
    /// speed envelope. A pure target — the caller applies acceleration limiting.
    /// `speed` is the robot's current command speed, used to scale the lookahead;
    /// `velocity` is the robot's measured velocity, used by terminal active
    /// braking; `brake_gain` is the effective active-braking gain for this robot;
    /// `approach_kp` is the effective position approach gain (baseline scaled by
    /// the robot's aggressiveness).
    pub fn follow(
        &self,
        path: &[Vector2],
        pos: Vector2,
        velocity: Vector2,
        speed: f64,
        brake_gain: f64,
        approach_kp: f64,
    ) -> FollowCmd {
        match path.len() {
            0 => return FollowCmd::cruise(Vector2::zeros()),
            1 => return self.seek_point(path[0], pos, velocity, brake_gain, approach_kp),
            _ => {}
        }

        let (seg, foot) = project_onto_path(pos, path);
        let s_goal = arc_to_goal(foot, seg, path);
        if s_goal < ARRIVE_DEADBAND {
            return FollowCmd::cruise(Vector2::zeros());
        }

        // Final approach: once close to the goal, seek it directly. Pure pursuit
        // toward a lookahead point can orbit a nearby goal, so hand off to a
        // straight proportional approach that converges cleanly.
        let goal = path[path.len() - 1];
        if s_goal < self.lookahead_min {
            return self.seek_point(goal, pos, velocity, brake_gain, approach_kp);
        }

        // Pure-pursuit steering direction.
        let lookahead = (self.lookahead_time * speed).clamp(self.lookahead_min, LOOKAHEAD_MAX);
        let look = point_ahead(foot, seg, path, lookahead);
        let dir = (look - pos)
            .try_normalize(1.0e-6)
            .or_else(|| (goal - pos).try_normalize(1.0e-6))
            .unwrap_or_else(Vector2::zeros);

        let v = self
            .speed_envelope(foot, seg, path, s_goal, approach_kp)
            .max(STICTION_FLOOR);
        FollowCmd::cruise(dir * v)
    }

    /// Straight proportional approach to a point: velocity ∝ distance (capped at
    /// cruise), over-damped so it converges without overshoot.
    ///
    /// Active braking: if the robot is overspeeding relative to that proportional
    /// profile, push the commanded speed *below* the profile by `brake_gain ×
    /// overspeed` (allowed to go negative → reverse) and flag `hard_brake` so the
    /// caller bypasses its slew clamp and the firmware reverse-thrusts to a crisp
    /// stop. As speed bleeds off, overspeed → 0 and it eases back into the plain
    /// proportional settle. `brake_gain = 0` disables this entirely.
    fn seek_point(
        &self,
        target: Vector2,
        pos: Vector2,
        velocity: Vector2,
        brake_gain: f64,
        approach_kp: f64,
    ) -> FollowCmd {
        let to = target - pos;
        let dist = to.norm();
        if dist < ARRIVE_DEADBAND {
            return FollowCmd::cruise(Vector2::zeros());
        }
        let dir = to / dist;
        let v_profile = self.v_max.min(approach_kp * dist);

        let overspeed = velocity.dot(&dir) - v_profile;
        if brake_gain > 0.0 && overspeed > 0.0 {
            let mag = (v_profile - brake_gain * overspeed).max(-self.v_max);
            return FollowCmd {
                velocity: dir * mag,
                hard_brake: true,
            };
        }

        FollowCmd::cruise(dir * v_profile.max(STICTION_FLOOR))
    }

    /// Speed cap at the current position: the min of cruise, the proportional
    /// approach to the goal, and the proportional approach to each upcoming
    /// corner (whose speed is set by the junction-deviation radius). Proportional
    /// (velocity ∝ remaining distance) is over-damped, so the robot eases into
    /// corners and the goal without overshooting them.
    fn speed_envelope(
        &self,
        foot: Vector2,
        seg: usize,
        path: &[Vector2],
        s_goal: f64,
        approach_kp: f64,
    ) -> f64 {
        let mut v = self.v_max.min(approach_kp * s_goal);

        let mut dist = (path[seg + 1] - foot).norm();
        let mut i = seg + 1;
        while i + 1 < path.len() {
            let vc = corner_speed(path[i] - path[i - 1], path[i + 1] - path[i], self.a_lat);
            v = v.min(vc + approach_kp * dist);
            if dist > self.v_max / approach_kp {
                break; // beyond the proportional braking range of v_max
            }
            dist += (path[i + 1] - path[i]).norm();
            i += 1;
        }
        v
    }
}

/// Cornering speed at a vertex via the junction-deviation model: the robot may
/// stray `CORNER_DEVIATION` from the vertex, giving an inscribed radius `R`, and
/// the speed is the lateral-acceleration-limited `sqrt(a_lat·R)`. Straight ⇒
/// unbounded; reverse ⇒ floored at `sqrt(a_lat·R_MIN)`.
fn corner_speed(incoming: Vector2, outgoing: Vector2, a_lat: f64) -> f64 {
    let (i, o) = match (
        incoming.try_normalize(1.0e-9),
        outgoing.try_normalize(1.0e-9),
    ) {
        (Some(i), Some(o)) => (i, o),
        _ => return f64::INFINITY,
    };
    let cos_phi = i.dot(&o).clamp(-1.0, 1.0); // 1 = straight, -1 = reversal
    let cos_half = ((1.0 + cos_phi) / 2.0).sqrt(); // cos(phi/2)
    if cos_half >= 1.0 - 1.0e-9 {
        return f64::INFINITY; // straight-through: no corner limit
    }
    let r = (CORNER_DEVIATION * cos_half / (1.0 - cos_half)).max(R_MIN);
    (a_lat * r).sqrt()
}

/// Closest segment index and foot point of `pos` on the polyline.
fn project_onto_path(pos: Vector2, path: &[Vector2]) -> (usize, Vector2) {
    let mut best = (0usize, path[0], f64::INFINITY);
    for k in 0..path.len() - 1 {
        let a = path[k];
        let ab = path[k + 1] - a;
        let len_sq = ab.norm_squared();
        let t = if len_sq > 1.0e-12 {
            ((pos - a).dot(&ab) / len_sq).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let foot = a + ab * t;
        let d = (pos - foot).norm_squared();
        if d < best.2 {
            best = (k, foot, d);
        }
    }
    (best.0, best.1)
}

/// Arc length from `foot` (on segment `seg`) to the end of the path.
fn arc_to_goal(foot: Vector2, seg: usize, path: &[Vector2]) -> f64 {
    let mut s = (path[seg + 1] - foot).norm();
    for k in (seg + 1)..(path.len() - 1) {
        s += (path[k + 1] - path[k]).norm();
    }
    s
}

/// Point `dist` further along the path from `foot` (clamped to the goal).
fn point_ahead(foot: Vector2, seg: usize, path: &[Vector2], mut dist: f64) -> Vector2 {
    let mut cur = foot;
    let mut k = seg;
    while k + 1 < path.len() {
        let next = path[k + 1];
        let d = (next - cur).norm();
        if d >= dist {
            let t = if d > 1.0e-9 { dist / d } else { 0.0 };
            return cur + (next - cur) * t;
        }
        dist -= d;
        cur = next;
        k += 1;
    }
    path[path.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn follower() -> PathFollower {
        let mut s = ControllerSettings::default();
        s.max_velocity = 3000.0;
        s.max_deceleration = 6000.0;
        s.lateral_acceleration = 4000.0;
        s.lookahead_min = 200.0;
        s.lookahead_time = 0.2;
        PathFollower::new(&s)
    }

    // Convenience: follow with zero measured velocity and braking disabled — the
    // pre-active-braking behaviour, for the path-shape assertions. Uses the
    // follower's baseline approach gain.
    fn vel(f: &PathFollower, path: &[Vector2], pos: Vector2, speed: f64) -> Vector2 {
        f.follow(
            path,
            pos,
            Vector2::zeros(),
            speed,
            0.0,
            f.base_approach_kp(),
        )
        .velocity
    }

    #[test]
    fn cruises_on_a_straight_then_brakes_to_goal() {
        let f = follower();
        let path = [Vector2::new(0.0, 0.0), Vector2::new(5000.0, 0.0)];
        // Far from goal at cruise speed → full speed, +x.
        let v = vel(&f, &path, Vector2::new(1000.0, 0.0), 3000.0);
        assert!(v.x > 2500.0 && v.y.abs() < 1.0, "got {:?}", v);
        // Near goal → braking envelope reduces speed.
        let v_near = vel(&f, &path, Vector2::new(4950.0, 0.0), 3000.0);
        assert!(v_near.x < v.x, "should brake near goal: {:?}", v_near);
    }

    #[test]
    fn stops_within_deadband() {
        let f = follower();
        let path = [Vector2::new(0.0, 0.0), Vector2::new(1000.0, 0.0)];
        let v = vel(&f, &path, Vector2::new(995.0, 0.0), 1000.0);
        assert!(v.norm() < 1.0e-6, "should stop at goal: {:?}", v);
    }

    #[test]
    fn slows_for_a_right_angle_corner() {
        let f = follower();
        // L-shaped path; robot on the first leg approaching the corner.
        let path = [
            Vector2::new(0.0, 0.0),
            Vector2::new(2000.0, 0.0),
            Vector2::new(2000.0, 2000.0),
        ];
        let near_corner = vel(&f, &path, Vector2::new(1950.0, 0.0), 3000.0);
        let on_straight = vel(&f, &path, Vector2::new(200.0, 0.0), 3000.0);
        assert!(
            near_corner.norm() < on_straight.norm(),
            "should slow into the corner: near={:?} straight={:?}",
            near_corner,
            on_straight
        );
        assert!(near_corner.norm() > 1.0, "but not stop at the corner");
    }

    #[test]
    fn converges_from_off_path() {
        let f = follower();
        let path = [Vector2::new(0.0, 0.0), Vector2::new(4000.0, 0.0)];
        // Off the path in +y; velocity should have a -y component to converge.
        let v = vel(&f, &path, Vector2::new(1000.0, 400.0), 2000.0);
        assert!(v.y < 0.0, "should steer back toward the path: {:?}", v);
    }

    #[test]
    fn active_braking_reverses_when_overspeeding_into_goal() {
        let f = follower();
        let path = [Vector2::new(0.0, 0.0), Vector2::new(1000.0, 0.0)];
        // Near the goal (within proportional range) but barreling in at full tilt.
        let pos = Vector2::new(950.0, 0.0);
        let fast = Vector2::new(3000.0, 0.0);
        // gain 0 → plain proportional, forward, no clamp bypass.
        let kp = f.base_approach_kp();
        let gentle = f.follow(&path, pos, fast, 3000.0, 0.0, kp);
        assert!(
            gentle.velocity.x > 0.0 && !gentle.hard_brake,
            "{:?}",
            gentle
        );
        // gain 1 → command reverses (−x) and requests the slew-clamp bypass.
        let brake = f.follow(&path, pos, fast, 3000.0, 1.0, kp);
        assert!(
            brake.velocity.x < 0.0 && brake.hard_brake,
            "should reverse-thrust: {:?}",
            brake
        );
    }

    #[test]
    fn stiction_floor_binds_only_below_it() {
        // With a low approach gain the terminal proportional profile would decay
        // into the stiction band; the floor keeps it at STICTION_FLOOR outside
        // the arrive deadband, and the deadband still stops the robot.
        let f = follower();
        let path = [Vector2::new(1000.0, 0.0)];
        let slow = f.follow(
            &path,
            Vector2::new(900.0, 0.0),
            Vector2::zeros(),
            0.0,
            0.0,
            0.1,
        );
        assert!(
            (slow.velocity.norm() - STICTION_FLOOR).abs() < 1e-9,
            "{slow:?}"
        );
        let stopped = f.follow(
            &path,
            Vector2::new(995.0, 0.0),
            Vector2::zeros(),
            0.0,
            0.0,
            0.1,
        );
        assert!(stopped.velocity.norm() < 1e-9, "{stopped:?}");
    }

    #[test]
    fn no_active_braking_when_below_profile() {
        let f = follower();
        let path = [Vector2::new(0.0, 0.0), Vector2::new(1000.0, 0.0)];
        // Near goal but already slow → no overspeed, plain proportional.
        let cmd = f.follow(
            &path,
            Vector2::new(950.0, 0.0),
            Vector2::new(50.0, 0.0),
            50.0,
            1.0,
            f.base_approach_kp(),
        );
        assert!(cmd.velocity.x > 0.0 && !cmd.hard_brake, "{:?}", cmd);
    }
}
