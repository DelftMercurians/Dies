use std::time::Duration;

use dies_core::{Obstacle, PlayerData, TeamData, Vector2};

use super::team_context::PlayerContext;

/// Two-Step MTP Controller
/// Samples intermediate points on a circle around the robot to find collision-free paths
pub struct TwoStepMTP {
    setpoint: Option<Vector2>,
    kp: f64,
    proportional_time_window: Duration,
    cutoff_distance: f64,
    sample_count: usize,
    last_vel: Option<Vector2>,
}

impl TwoStepMTP {
    pub fn new() -> Self {
        Self {
            setpoint: None,
            kp: 1.5,
            proportional_time_window: Duration::from_millis(400),
            cutoff_distance: 10.0,
            sample_count: 8,
            last_vel: None,
        }
    }

    pub fn set_setpoint(&mut self, setpoint: Vector2) {
        self.setpoint = Some(setpoint);
    }

    pub fn update_settings(
        &mut self,
        kp: f64,
        proportional_time_window: Duration,
        cutoff_distance: f64,
    ) {
        self.kp = kp;
        self.proportional_time_window = proportional_time_window;
        self.cutoff_distance = cutoff_distance;
    }

    pub fn update(
        &mut self,
        current: Vector2,
        velocity: Vector2,
        dt: f64,
        max_accel: f64,
        max_speed: f64,
        max_decel: f64,
        carefullness: f64,
        aggressiveness: f64,
        player_context: &PlayerContext,
        world_data: &TeamData,
        current_player: &PlayerData,
        avoid_goal_area: bool,
        obstacles: Vec<Obstacle>,
    ) -> Vector2 {
        let setpoint = match self.setpoint {
            Some(s) => s,
            None => return Vector2::zeros(),
        };

        let displacement = setpoint - current;
        let distance = displacement.magnitude();

        if distance < self.cutoff_distance {
            return Vector2::zeros();
        }

        // Calculate intermediate target using two-step sampling
        let intermediate_target = self.find_best_intermediate_point(
            current,
            setpoint,
            world_data,
            current_player,
            player_context,
            avoid_goal_area,
            obstacles,
        );

        // Use the intermediate target as the new setpoint for MTP control
        let intermediate_displacement = intermediate_target - current;
        let intermediate_distance = intermediate_displacement.magnitude();

        if intermediate_distance < self.cutoff_distance {
            return Vector2::zeros();
        }

        let direction = if intermediate_distance > f64::EPSILON {
            intermediate_displacement * (1.0 / intermediate_distance)
        } else {
            Vector2::zeros()
        };

        let current_speed = velocity.magnitude();
        let time_to_target = intermediate_distance / max_speed;

        // player_context.debug_string("TwoStepMTPTimeToTarget", time_to_target.to_string());
        // player_context.debug_string(
        //     "TwoStepMTPProportionalTimeWindow",
        //     self.proportional_time_window.as_secs_f64().to_string(),
        // );
        // player_context.debug_string("TwoStepMTPKp", self.kp.to_string());
        // player_context.debug_string(
        //     "TwoStepMTPcutoff_distance",
        //     self.cutoff_distance.to_string(),
        // );
        // player_context.debug_string("TwoStepMTPCurrentSpeed", current_speed.to_string());
        // player_context.debug_string("TwoStepMTPMaxSpeed", max_speed.to_string());
        // player_context.debug_string("TwoStepMTPMaxAccel", max_accel.to_string());
        // player_context.debug_string("TwoStepMTPDt", dt.to_string());
        // player_context.debug_string("TwoStepMTPMaxDecel", max_decel.to_string());
        // player_context.debug_string("TwoStepMTPCarefullness", carefullness.to_string());

        if time_to_target <= self.proportional_time_window.as_secs_f64() {
            // Proportional control
            let aggressiveness = 1.0 + aggressiveness;
            let proportional_velocity_magnitude =
                (f64::max(intermediate_distance - self.cutoff_distance, 0.0)
                    * (self.kp * aggressiveness)
                    + (150.0 - 70.0 * carefullness))
                    .clamp(0.0, max_speed);
            let current_vel = self.last_vel.get_or_insert(velocity).clone();
            if false {
                let target_vel = proportional_velocity_magnitude * direction;
                let dv = cap_vec(target_vel - current_vel, 2500.0 * dt);
                current_vel + dv
            } else {
                direction * proportional_velocity_magnitude
            }
        } else {
            // Cruise phase
            // player_context.debug_string("TwoStepMTPMode", "Cruise");
            direction * max_speed
        }
    }

    fn find_best_intermediate_point(
        &self,
        current: Vector2,
        target: Vector2,
        world_data: &TeamData,
        current_player: &PlayerData,
        player_context: &PlayerContext,
        avoid_goal_area: bool,
        obstacles: Vec<Obstacle>,
    ) -> Vector2 {
        // Debug visualization for boundary rectangles
        let to_target = target - current;
        let distance_to_target = to_target.magnitude();

        // Circle radius is half the distance to target + some random number
        // no worries, when we are close, the proportial control is triggered
        // (proportional_time_window) - which generally means we just go directly to the target
        let circle_radius = distance_to_target * 0.5 + 40.0;

        // Sample points uniformly around the circle
        // First, sample the direct trajectory as a starting point
        let mut best_point = current + to_target.normalize() * circle_radius;
        let mut best_cost = self.calculate_path_cost(
            current,
            best_point,
            target,
            world_data,
            current_player,
            avoid_goal_area,
            obstacles.clone(),
        );

        for i in 0..self.sample_count {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / self.sample_count as f64;
            let sample_point =
                current + Vector2::new(circle_radius * angle.cos(), circle_radius * angle.sin());

            // Calculate cost for this sample point
            let cost = self.calculate_path_cost(
                current,
                sample_point,
                target,
                world_data,
                current_player,
                avoid_goal_area,
                obstacles.clone(),
            );

            if cost < best_cost {
                best_cost = cost;
                best_point = sample_point;
            }
        }

        player_context.debug_string("tdist", format!("{:.0}", distance_to_target));

        player_context.debug_string("tmid_x", format!("{:.0}", best_point.x));
        player_context.debug_string("tmid_y", format!("{:.0}", best_point.y));
        // Visualize the best chosen midpoint as a larger cross
        player_context.debug_circle_fill_colored(
            "two_step_best",
            best_point,
            100.0,
            dies_core::DebugColor::Gray,
        );

        best_point
    }

    fn calculate_path_cost(
        &self,
        start: Vector2,
        mid: Vector2,
        end: Vector2,
        world_data: &TeamData,
        current_player: &PlayerData,
        avoid_goal_area: bool,
        obstacles: Vec<Obstacle>,
    ) -> f64 {
        // total cost is multiplied by magic coeff -> lower implies we care more about
        // avoiding shit, less means we are straighter (less gay)
        let mut total_cost = 0.1 * ((start - mid).magnitude() + (mid - end).magnitude());
        let robot_scare = 190.0; // mm - 2xrobot radius + some margin
        let ball_scare = 100.0; // mm robot_radius + ball_radius
                                //
        let pstart = start;
        let pmid = mid;
        let pend = end;

        // Calculate intersection cost with other robots
        for robot in world_data
            .own_players
            .iter()
            .chain(world_data.opp_players.iter())
        {
            if robot.id == current_player.id {
                continue;
            }

            let intersection_length =
                self.line_circle_intersection_length(
                    start,
                    mid,
                    robot.position,
                    robot_scare, // Robot radius
                ) + self.line_circle_intersection_length(mid, end, robot.position, robot_scare)
                    * 0.1;
            if intersection_length > 0.0 {
                total_cost += 500.0
            }
            total_cost += intersection_length; // Weight for robot avoidance
        }

        // Calculate intersection cost with field boundaries
        if let Some(field) = &world_data.field_geom {
            let intersection_length =
                self.calculate_field_intersection_cost(start, mid, field, avoid_goal_area)
                    + self.calculate_field_intersection_cost(mid, end, field, avoid_goal_area)
                        * 0.1;
            if intersection_length > 0.0 {
                total_cost += 500.0
            }
            total_cost += intersection_length; // High weight for field boundaries
        }

        // calculate instersection cost with all of the obstacles
        for obstacle in obstacles {
            let intersection_length = match obstacle {
                Obstacle::Circle { center, radius } => {
                    self.line_circle_intersection_length(pstart, pmid, center, radius)
                        + self.line_circle_intersection_length(pmid, pend, center, radius) * 0.1
                }
                Obstacle::Rectangle { min, max } => {
                    self.line_rectangle_intersection_length(pstart, pmid, min, max)
                        + self.line_rectangle_intersection_length(pmid, pend, min, max) * 0.1
                }
                Obstacle::Line { start, end } => {
                    self.line_line_collision_avoidance(pstart, pmid, start, end)
                        + self.line_line_collision_avoidance(pmid, pend, start, end) * 0.1
                }
            };
            if intersection_length > 0.0 {
                total_cost += 1500.0; // High penalty for obstacle collision
            }
            total_cost += intersection_length * 2.0; // Weight for obstacle avoidance
        }

        total_cost
    }

    fn line_circle_intersection_length(
        &self,
        line_start: Vector2,
        line_end: Vector2,
        circle_center: Vector2,
        circle_radius: f64,
    ) -> f64 {
        let line_vec = line_end - line_start;
        let line_length = line_vec.magnitude();

        if line_length < f64::EPSILON {
            return 0.0;
        }

        let start_dist = (line_start - circle_center).magnitude();
        let end_dist = (line_end - circle_center).magnitude();
        let start_inside = start_dist <= circle_radius;
        let end_inside = end_dist <= circle_radius;

        match (start_inside, end_inside) {
            (true, true) => {
                // Both endpoints inside circle - return full line length
                line_length
            }
            (true, false) => {
                // Start inside, end outside - find exit point
                let line_dir = line_vec / line_length;
                let to_center = circle_center - line_start;
                let projection = to_center.dot(&line_dir);
                let dist_to_center =
                    (circle_center - (line_start + line_dir * projection)).magnitude();

                if dist_to_center >= circle_radius {
                    return 0.0;
                }

                let half_chord = (circle_radius.powi(2) - dist_to_center.powi(2)).sqrt();
                let exit_t = (projection + half_chord).min(line_length);
                exit_t
            }
            (false, true) => {
                // Start outside, end inside - find entry point
                let line_dir = line_vec / line_length;
                let to_center = circle_center - line_start;
                let projection = to_center.dot(&line_dir);
                let dist_to_center =
                    (circle_center - (line_start + line_dir * projection)).magnitude();

                if dist_to_center >= circle_radius {
                    return 0.0;
                }

                let half_chord = (circle_radius.powi(2) - dist_to_center.powi(2)).sqrt();
                let entry_t = (projection - half_chord).max(0.0);
                line_length - entry_t
            }
            (false, false) => {
                // Both outside - find intersection segment
                let line_dir = line_vec / line_length;
                let to_center = circle_center - line_start;
                let projection = to_center.dot(&line_dir);
                let dist_to_center =
                    (circle_center - (line_start + line_dir * projection)).magnitude();

                if dist_to_center >= circle_radius {
                    return 0.0;
                }

                let half_chord = (circle_radius.powi(2) - dist_to_center.powi(2)).sqrt();
                let entry_t = (projection - half_chord).max(0.0);
                let exit_t = (projection + half_chord).min(line_length);

                (exit_t - entry_t).max(0.0)
            }
        }
    }

    fn is_point_inside_rectangle(
        &self,
        point: Vector2,
        rect_min: Vector2,
        rect_max: Vector2,
    ) -> bool {
        point.x >= rect_min.x
            && point.x <= rect_max.x
            && point.y >= rect_min.y
            && point.y <= rect_max.y
    }

    fn line_rectangle_intersection_length(
        &self,
        line_start: Vector2,
        line_end: Vector2,
        rect_min: Vector2,
        rect_max: Vector2,
    ) -> f64 {
        let line_vec = line_end - line_start;
        let line_length = line_vec.magnitude();

        if line_length < f64::EPSILON {
            return 0.0;
        }

        let start_inside = self.is_point_inside_rectangle(line_start, rect_min, rect_max);
        let end_inside = self.is_point_inside_rectangle(line_end, rect_min, rect_max);

        match (start_inside, end_inside) {
            (true, true) => {
                // Both endpoints inside - return full line length
                line_length
            }
            (true, false) => {
                // Start inside, end outside - find exit point
                if let Some(exit_point) =
                    self.find_line_rectangle_exit_point(line_start, line_end, rect_min, rect_max)
                {
                    (exit_point - line_start).magnitude()
                } else {
                    0.0
                }
            }
            (false, true) => {
                // Start outside, end inside - find entry point
                if let Some(entry_point) =
                    self.find_line_rectangle_entry_point(line_start, line_end, rect_min, rect_max)
                {
                    (line_end - entry_point).magnitude()
                } else {
                    0.0
                }
            }
            (false, false) => {
                // Both outside - find entry and exit points
                if let Some(entry_point) =
                    self.find_line_rectangle_entry_point(line_start, line_end, rect_min, rect_max)
                {
                    if let Some(exit_point) = self.find_line_rectangle_exit_point(
                        entry_point,
                        line_end,
                        rect_min,
                        rect_max,
                    ) {
                        (exit_point - entry_point).magnitude()
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
        }
    }

    fn find_line_rectangle_entry_point(
        &self,
        line_start: Vector2,
        line_end: Vector2,
        rect_min: Vector2,
        rect_max: Vector2,
    ) -> Option<Vector2> {
        let line_vec = line_end - line_start;
        let line_length = line_vec.magnitude();

        if line_length < f64::EPSILON {
            return None;
        }

        let line_dir = line_vec / line_length;
        let mut t_entry = f64::NEG_INFINITY;

        // Check intersection with each rectangle edge
        for axis in 0..2 {
            let ray_origin = if axis == 0 {
                line_start.x
            } else {
                line_start.y
            };
            let ray_dir = if axis == 0 { line_dir.x } else { line_dir.y };
            let rect_min_axis = if axis == 0 { rect_min.x } else { rect_min.y };
            let rect_max_axis = if axis == 0 { rect_max.x } else { rect_max.y };

            if ray_dir.abs() > f64::EPSILON {
                let t1 = (rect_min_axis - ray_origin) / ray_dir;
                let t2 = (rect_max_axis - ray_origin) / ray_dir;

                let t_near = t1.min(t2);
                t_entry = t_entry.max(t_near);
            }
        }

        if t_entry >= 0.0 && t_entry <= line_length {
            Some(line_start + line_dir * t_entry)
        } else {
            None
        }
    }

    fn find_line_rectangle_exit_point(
        &self,
        line_start: Vector2,
        line_end: Vector2,
        rect_min: Vector2,
        rect_max: Vector2,
    ) -> Option<Vector2> {
        let line_vec = line_end - line_start;
        let line_length = line_vec.magnitude();

        if line_length < f64::EPSILON {
            return None;
        }

        let line_dir = line_vec / line_length;
        let mut t_exit = f64::INFINITY;

        // Check intersection with each rectangle edge
        for axis in 0..2 {
            let ray_origin = if axis == 0 {
                line_start.x
            } else {
                line_start.y
            };
            let ray_dir = if axis == 0 { line_dir.x } else { line_dir.y };
            let rect_min_axis = if axis == 0 { rect_min.x } else { rect_min.y };
            let rect_max_axis = if axis == 0 { rect_max.x } else { rect_max.y };

            if ray_dir.abs() > f64::EPSILON {
                let t1 = (rect_min_axis - ray_origin) / ray_dir;
                let t2 = (rect_max_axis - ray_origin) / ray_dir;

                let t_far = t1.max(t2);
                t_exit = t_exit.min(t_far);
            }
        }

        if t_exit >= 0.0 && t_exit <= line_length {
            Some(line_start + line_dir * t_exit)
        } else {
            None
        }
    }

    fn calculate_field_intersection_cost(
        &self,
        line_start: Vector2,
        line_end: Vector2,
        field: &dies_core::FieldGeometry,
        avoid_goal_area: bool,
    ) -> f64 {
        let mut cost = 0.0;
        let margin = 40.0;

        // Field boundaries
        let fmargin = 0.0;
        let half_width = field.boundary_width + field.field_width / 2.0 - fmargin;
        let half_length = field.boundary_width + field.field_length / 2.0 - fmargin;

        // Check intersection with field boundaries
        let field_lines = [
            // Top boundary
            (
                Vector2::new(-half_length, half_width),
                Vector2::new(half_length, half_width),
            ),
            // Bottom boundary
            (
                Vector2::new(-half_length, -half_width),
                Vector2::new(half_length, -half_width),
            ),
            // Left boundary
            (
                Vector2::new(-half_length, -half_width),
                Vector2::new(-half_length, half_width),
            ),
            // Right boundary
            (
                Vector2::new(half_length, -half_width),
                Vector2::new(half_length, half_width),
            ),
        ];

        for (boundary_start, boundary_end) in field_lines.iter() {
            if let Some(intersection) =
                self.line_line_intersection(line_start, line_end, *boundary_start, *boundary_end)
            {
                // Add penalty for crossing field boundaries
                cost += 10000.0;
            }
        }

        let penalty_depth = field.penalty_area_depth - fmargin + margin; // no need for margin here: already shifted
                                                                         // the field boundary
        let penalty_width = field.penalty_area_width + 2.0 * margin;

        if avoid_goal_area {
            let left_rect_min = Vector2::new(-half_length, -penalty_width / 2.0);
            let left_rect_max = Vector2::new(-half_length + penalty_depth, penalty_width / 2.0);
            let left_intersection_length = self.line_rectangle_intersection_length(
                line_start,
                line_end,
                left_rect_min,
                left_rect_max,
            );
            cost += left_intersection_length * 10.0; // Weight for left boundary area
            debug_rect(left_rect_min, left_rect_max);

            // Right side boundary rectangle (outside field)
            let right_rect_min = Vector2::new(half_length - penalty_depth, -penalty_width / 2.0);
            let right_rect_max = Vector2::new(half_length, penalty_width / 2.0);
            let right_intersection_length = self.line_rectangle_intersection_length(
                line_start,
                line_end,
                right_rect_min,
                right_rect_max,
            );
            cost += right_intersection_length * 10.0; // Weight for right boundary area
            debug_rect(right_rect_min, right_rect_max);
        }

        cost
    }

    fn line_line_collision_avoidance(
        &self,
        path_start: Vector2,
        path_end: Vector2,
        line_start: Vector2,
        line_end: Vector2,
    ) -> f64 {
        let path_vec = path_end - path_start;
        let path_length = path_vec.magnitude();

        if path_length < f64::EPSILON {
            return 0.0;
        }

        let path_dir = path_vec / path_length;
        let step_size = 50.0;
        let mut total_cost = 0.0;

        // Sample circles along the path, including endpoints
        let num_steps = (path_length / step_size).ceil() as usize;

        for i in 0..=(num_steps + 1) {
            let t = if num_steps == 0 {
                0.0
            } else {
                (i as f64 / num_steps as f64) * path_length
            };
            let sample_point = path_start + path_dir * t;

            // Calculate distance from sample point to the line obstacle
            let distance = self.point_to_line_distance(sample_point, line_start, line_end);

            // If within collision radius, add cost
            let collision_radius = 700.0; // Similar to robot collision radius
            if distance < collision_radius {
                let penetration = collision_radius - distance;
                total_cost += penetration * step_size / 500.0; // Cost proportional to penetration and step size
            }
        }

        total_cost
    }

    fn point_to_line_distance(
        &self,
        point: Vector2,
        line_start: Vector2,
        line_end: Vector2,
    ) -> f64 {
        let line_vec = line_end - line_start;
        let line_length = line_vec.magnitude();

        if line_length < f64::EPSILON {
            // Line is a point
            return (point - line_start).magnitude();
        }

        let line_dir = line_vec / line_length;
        let to_point = point - line_start;
        let projection = to_point.dot(&line_dir);

        // Clamp projection to line segment
        let clamped_projection = projection.clamp(0.0, line_length);
        let closest_point = line_start + line_dir * clamped_projection;

        (point - closest_point).magnitude()
    }

    fn line_line_intersection(
        &self,
        line1_start: Vector2,
        line1_end: Vector2,
        line2_start: Vector2,
        line2_end: Vector2,
    ) -> Option<Vector2> {
        let line1_vec = line1_end - line1_start;
        let line2_vec = line2_end - line2_start;

        let denominator = line1_vec.x * line2_vec.y - line1_vec.y * line2_vec.x;

        if denominator.abs() < f64::EPSILON {
            return None; // Lines are parallel
        }

        let start_diff = line2_start - line1_start;
        let t = (start_diff.x * line2_vec.y - start_diff.y * line2_vec.x) / denominator;
        let u = (start_diff.x * line1_vec.y - start_diff.y * line1_vec.x) / denominator;

        if t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0 {
            Some(line1_start + line1_vec * t)
        } else {
            None
        }
    }
}

fn cap_vec(v: Vector2, cap: f64) -> Vector2 {
    v.normalize() * v.magnitude().min(cap)
}

fn cap_magnitude(v: f64, max: f64) -> f64 {
    if v > max {
        max
    } else if v < -max {
        -max
    } else {
        v
    }
}

fn debug_rect(low: Vector2, high: Vector2) {
    return;
    dies_core::debug_line(
        &format!("__{}{}_seg", low.x, high.y),
        low.xy(),
        Vector2::new(low.x, high.y),
        dies_core::DebugColor::Blue,
    );

    dies_core::debug_line(
        &format!("__{}{}_seg", low.y, high.x),
        low.xy(),
        Vector2::new(high.x, low.y),
        dies_core::DebugColor::Blue,
    );

    dies_core::debug_line(
        &format!("__{}{}_seg", high.x, high.y),
        high.xy(),
        Vector2::new(high.x, low.y),
        dies_core::DebugColor::Blue,
    );

    dies_core::debug_line(
        &format!("__{}{}_seg", high.x, low.x),
        high.xy(),
        Vector2::new(low.x, high.y),
        dies_core::DebugColor::Blue,
    );
}
