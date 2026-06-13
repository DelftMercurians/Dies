use dies_core::Vector2;

use crate::ControlParameters;

/// MTP — minimum-time-path velocity controller.
///
/// A pure accel/decel velocity profile toward a single setpoint: proportional
/// velocity magnitude with a near-target threshold, PI tracking with antiwindup,
/// and a hard per-tick acceleration cap on the commanded setpoint. Collision
/// avoidance lives entirely in the separate `avoidance` modules (global planner
/// + ORCA); this controller just answers "how fast toward the given waypoint".
///
/// The setpoint is the waypoint chosen by the global planner — when the path is
/// clear that is simply the final target.
pub struct TwoStepMTP {
    setpoint: Option<Vector2>,
    kp: f64,
    thresh: f64,
    cutoff_distance: f64,
    last_vel: Option<Vector2>,
    integral: f64,
}

impl TwoStepMTP {
    pub fn new() -> Self {
        Self {
            setpoint: None,
            kp: 1.5,
            cutoff_distance: 10.0,
            thresh: 100.0,
            last_vel: None,
            integral: 0.0,
        }
    }

    pub fn set_setpoint(&mut self, setpoint: Vector2) {
        self.setpoint = Some(setpoint);
    }

    pub fn update_settings(&mut self, kp: f64, cutoff_distance: f64, thresh: f64) {
        self.kp = kp;
        self.cutoff_distance = cutoff_distance;
        self.thresh = thresh;
    }

    /// Compute the desired global-frame velocity \[mm/s\] toward the setpoint.
    ///
    /// `is_final` controls arrival behaviour: when true (the planner's final
    /// target) the controller decelerates to a stop near the setpoint; when
    /// false (an intermediate planner waypoint) it decelerates only to
    /// `arrival_speed_frac · max_speed` — the corner speed — so it flows through
    /// the corner without stopping but slow enough not to overshoot it.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        current: Vector2,
        velocity: Vector2,
        dt: f64,
        max_accel: f64,
        max_speed: f64,
        _max_decel: f64,
        carefullness: f64,
        aggressiveness: f64,
        control_parameters: Option<ControlParameters>,
        is_final: bool,
        arrival_speed_frac: f64,
    ) -> Vector2 {
        let setpoint = match self.setpoint {
            Some(s) => s,
            None => return Vector2::zeros(),
        };

        let displacement = setpoint - current;
        let distance = displacement.magnitude();

        // Stop only at the final target; pass through intermediate waypoints.
        if is_final && distance < self.cutoff_distance {
            self.integral = 0.0;
            return Vector2::zeros();
        }

        let direction = if distance > f64::EPSILON {
            displacement * (1.0 / distance)
        } else {
            Vector2::zeros()
        };

        let kp = control_parameters.as_ref().map(|p| p.kp).unwrap_or(self.kp);
        let thresh = control_parameters
            .as_ref()
            .map(|p| p.thresh)
            .unwrap_or(self.thresh);
        let antiwindup = control_parameters
            .as_ref()
            .map(|p| p.antiwindup)
            .unwrap_or(40.0);
        let aggressiveness = 1.0 + aggressiveness;
        // Proportional ramp toward an arrival speed: 0 at the final target (with
        // the near-target threshold), or the corner speed at an intermediate
        // waypoint so the robot slows just enough to make the turn.
        let proportional_velocity_magnitude = if is_final {
            (f64::max(distance - self.cutoff_distance, 0.0) * (kp * aggressiveness)
                + (thresh - 70.0 * carefullness))
                .clamp(0.0, max_speed)
        } else {
            let arrival_speed = (arrival_speed_frac * max_speed).clamp(0.0, max_speed);
            (f64::max(distance - self.cutoff_distance, 0.0) * (kp * aggressiveness) + arrival_speed)
                .clamp(arrival_speed, max_speed)
        };

        let measured_vel = velocity.magnitude();
        let target_vel = direction * proportional_velocity_magnitude;
        let error = target_vel.magnitude() - measured_vel;
        self.integral += error * dt;
        self.integral = self.integral.clamp(-antiwindup, antiwindup);

        let current_vel = self.last_vel.get_or_insert(Vector2::zeros()).clone();
        let dv = if target_vel.magnitude() > current_vel.magnitude() {
            cap_vec(target_vel - current_vel, max_accel * dt)
        } else {
            target_vel - current_vel
        };
        let new_vel = current_vel + dv;
        self.last_vel = Some(new_vel);
        new_vel
    }
}

fn cap_vec(v: Vector2, cap: f64) -> Vector2 {
    if v.magnitude() < f64::EPSILON {
        return v;
    }
    v.normalize() * v.magnitude().min(cap)
}
