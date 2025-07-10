use std::time::Duration;

use dies_core::{Vector2, TeamData, PlayerData, Obstacle};

use super::team_context::PlayerContext;


/// Two-Step MTP Controller
/// Samples intermediate points on a circle around the robot to find collision-free paths
pub struct TwoStepMTP {
    setpoint: Option<Vector2>,
    kp: f64,
    proportional_time_window: Duration,
    cutoff_distance: f64,
    sample_count: usize,
}

impl TwoStepMTP {
    pub fn new() -> Self {
        Self {
            setpoint: None,
            kp: 1.0,
            proportional_time_window: Duration::from_millis(700),
            cutoff_distance: 30.0,
            sample_count: 20,
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
        &self,
        current: Vector2,
        velocity: Vector2,
        dt: f64,
        max_accel: f64,
        max_speed: f64,
        max_decel: f64,
        carefullness: f64,
        player_context: &PlayerContext,
        world_data: &TeamData,
        current_player: &PlayerData,
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
        let care_factor = carefullness * 0.2;
        let time_to_target = intermediate_distance / max_speed;

        if time_to_target <= self.proportional_time_window.as_secs_f64() {
            // Proportional control
            let proportional_velocity_magnitude = f64::max(intermediate_distance - self.cutoff_distance, 0.0)
                * (self.kp * (1.0 - care_factor))
                + 100.0 * care_factor;
            let dv_magnitude = proportional_velocity_magnitude - velocity.magnitude();

            let mut v_control = dv_magnitude;
            if v_control < 0.0 {
                v_control *= 5.0 * (1.0 + care_factor);
            }

            player_context.debug_string("TwoStepMTPMode", "Proportional");
            let new_speed = current_speed + cap_magnitude(v_control, max_decel * dt);
            direction * new_speed
        } else if current_speed < max_speed {
            // Acceleration phase
            player_context.debug_string("TwoStepMTPMode", "Acceleration");
            let new_speed = current_speed + max_accel * dt;
            direction * new_speed
        } else {
            // Cruise phase
            player_context.debug_string("TwoStepMTPMode", "Cruise");
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
    ) -> Vector2 {
        // Debug visualization for boundary rectangles
        let to_target = target - current;
        let distance_to_target = to_target.magnitude();

        // Circle radius is half the distance to target + some random number
        // no worries, when we are close, the proportial control is triggered
        // (proportional_time_window) - which generally means we just go directly to the target
        let circle_radius = distance_to_target * 0.5 + 60.0;

        // Sample points uniformly around the circle
        let mut best_point = target;
        let mut best_cost = f64::INFINITY;

        // First, sample the direct trajectory as a starting point
        let direct_cost = self.calculate_path_cost(
            current,
            target,
            target,
            world_data,
            current_player,
        );

        // Visualize the direct trajectory sample
        player_context.debug_circle_fill_colored(
            "two_step_direct",
            target,
            5.0,
            dies_core::DebugColor::Blue,
        );

        if direct_cost < best_cost {
            best_cost = direct_cost;
            best_point = target;
        }

        for i in 0..self.sample_count {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / self.sample_count as f64;
            let sample_point = current + Vector2::new(
                circle_radius * angle.cos(),
                circle_radius * angle.sin(),
            );

            // Calculate cost for this sample point
            let cost = self.calculate_path_cost(
                current,
                sample_point,
                target,
                world_data,
                current_player,
            );

            // Visualize all sample points as very small crosses
            player_context.debug_circle_fill_colored(
                &format!("two_step_sample_{}", i),
                sample_point,
                5.0,
                dies_core::DebugColor::Gray,
            );

            if cost < best_cost {
                best_cost = cost;
                best_point = sample_point;
            }
        }

        // Visualize the best chosen midpoint as a larger cross
        player_context.debug_cross_colored(
            "two_step_best",
            best_point,
            dies_core::DebugColor::Green,
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
    ) -> f64 {
        // total cost is multiplied by magic coeff -> lower implies we care more about
        // avoiding shit, less means we are straighter (less gay)
        let mut total_cost = 0.01 * (start - mid).magnitude() + (mid - end).magnitude();
        let robot_scare = 190.0; // mm - 2xrobot radius + some margin
        let ball_scare = 90.0 + 90.0; // mm robot_radius + ball_radius

        // Calculate intersection cost with other robots
        for robot in world_data.own_players.iter().chain(world_data.opp_players.iter()) {
            if robot.id == current_player.id {
                continue;
            }

            let intersection_length = self.line_circle_intersection_length(
                start,
                mid,
                robot.position,
                robot_scare, // Robot radius
            ) + self.line_circle_intersection_length(mid, end, robot.position, robot_scare) * 0.1;
            if intersection_length > 0.0 {
                total_cost += 500.0
            }
            total_cost += intersection_length; // Weight for robot avoidance
        }

        // Calculate intersection cost with ball
        if let Some(ball) = &world_data.ball {
            let intersection_length = self.line_circle_intersection_length(
                start,
                mid,
                ball.position.xy(),
                ball_scare, // Ball radius
            ) + self.line_circle_intersection_length(mid, end, ball.position.xy(), ball_scare) * 0.1;
            if intersection_length > 0.0 {
                total_cost += 1000.0
            }
            total_cost += intersection_length; // Weight for ball avoidance
        }

        // Calculate intersection cost with field boundaries
        if let Some(field) = &world_data.field_geom {
            let intersection_length = self.calculate_field_intersection_cost(
                start,
                mid,
                field,
            ) + self.calculate_field_intersection_cost(mid, end, field) * 0.1;
            if intersection_length > 0.0 {
                total_cost += 500.0
            }
            total_cost += intersection_length; // High weight for field boundaries
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
                let dist_to_center = (circle_center - (line_start + line_dir * projection)).magnitude();

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
                let dist_to_center = (circle_center - (line_start + line_dir * projection)).magnitude();

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
                let dist_to_center = (circle_center - (line_start + line_dir * projection)).magnitude();

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
        point.x >= rect_min.x && point.x <= rect_max.x &&
        point.y >= rect_min.y && point.y <= rect_max.y
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
                if let Some(exit_point) = self.find_line_rectangle_exit_point(line_start, line_end, rect_min, rect_max) {
                    (exit_point - line_start).magnitude()
                } else {
                    0.0
                }
            }
            (false, true) => {
                // Start outside, end inside - find entry point
                if let Some(entry_point) = self.find_line_rectangle_entry_point(line_start, line_end, rect_min, rect_max) {
                    (line_end - entry_point).magnitude()
                } else {
                    0.0
                }
            }
            (false, false) => {
                // Both outside - find entry and exit points
                if let Some(entry_point) = self.find_line_rectangle_entry_point(line_start, line_end, rect_min, rect_max) {
                    if let Some(exit_point) = self.find_line_rectangle_exit_point(entry_point, line_end, rect_min, rect_max) {
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
            let ray_origin = if axis == 0 { line_start.x } else { line_start.y };
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
            let ray_origin = if axis == 0 { line_start.x } else { line_start.y };
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
    ) -> f64 {
        let mut cost = 0.0;

        // Field boundaries
        let half_width = field.field_width / 2.0;
        let half_length = field.field_length / 2.0;
        let boundary_width = field.boundary_width;

        // Check intersection with field boundaries
        let field_lines = [
            // Top boundary
            (Vector2::new(-half_length, half_width), Vector2::new(half_length, half_width)),
            // Bottom boundary
            (Vector2::new(-half_length, -half_width), Vector2::new(half_length, -half_width)),
            // Left boundary
            (Vector2::new(-half_length, -half_width), Vector2::new(-half_length, half_width)),
            // Right boundary
            (Vector2::new(half_length, -half_width), Vector2::new(half_length, half_width)),
        ];

        for (boundary_start, boundary_end) in field_lines.iter() {
            if let Some(intersection) = self.line_line_intersection(
                line_start,
                line_end,
                *boundary_start,
                *boundary_end,
            ) {
                // Add penalty for crossing field boundaries
                cost += 1000.0;
            }
        }


        let left_rect_min = Vector2::new(-half_length, -field.penalty_area_width / 2.0);
        let left_rect_max = Vector2::new(-half_length + field.penalty_area_depth, field.penalty_area_width / 2.0);
        let left_intersection_length = self.line_rectangle_intersection_length(
            line_start,
            line_end,
            left_rect_min,
            left_rect_max,
        );
        cost += left_intersection_length * 10.0; // Weight for left boundary area

        // Right side boundary rectangle (outside field)
        let right_rect_min = Vector2::new(half_length - field.penalty_area_depth, -field.penalty_area_width / 2.0);
        let right_rect_max = Vector2::new(half_length, field.penalty_area_width / 2.0);
        let right_intersection_length = self.line_rectangle_intersection_length(
            line_start,
            line_end,
            right_rect_min,
            right_rect_max,
        );
        cost += right_intersection_length * 10.0; // Weight for right boundary area

        cost
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

fn cap_magnitude(v: f64, max: f64) -> f64 {
    if v > max {
        max
    } else if v < -max {
        -max
    } else {
        v
    }
}
