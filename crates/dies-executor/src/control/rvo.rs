use std::f64::{consts::PI, EPSILON};

use dies_core::{Angle, Obstacle, PlayerData, Vector2};

// Constants
const PLAYER_MARGIN: f64 = 30.0;
const OVER_APPROX_C2S: f64 = 1.5;
const PLAYER_RADIUS: f64 = 90.0;

/// The type of VO algorithm to use
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum VelocityObstacleType {
    VO,
    RVO,
    HRVO,
}

// Function to compute Euclidean distance between two points
fn distance(p1: &Vector2, p2: &Vector2) -> f64 {
    (p1 - p2).norm() + EPSILON // Adding small constant to avoid division by zero
}

// Main function to update velocities based on RVO
pub fn velocity_obstacle_update(
    player: &PlayerData,
    desired_velocity: &Vector2,
    players: &[&PlayerData],
    obstacles: &[Obstacle],
    vo_type: VelocityObstacleType,
    avoid_robots: bool,
) -> Vector2 {
    let player_radius = PLAYER_RADIUS + PLAYER_MARGIN;
    let mut rvo_ba_all = Vec::new();

    // Consider other agents
    for other_player in players.iter() {
        if other_player.position != player.position && !avoid_robots {
            let rvo_ba =
                compute_velocity_obstacle(player, other_player, &vo_type, player_radius * 1.0);
            rvo_ba_all.push(rvo_ba);
        }
    }

    // Consider static obstacles
    for obstacle in obstacles {
        let rvo_ba = match obstacle {
            Obstacle::Circle { center, radius } => {
                compute_obstacle_velocity_obstacle(player, center, radius, player_radius)
            }
            Obstacle::Rectangle { min, max } => {
                // strange stuff i do to convert types?
                let lower = Vector2::from(min + min) / 2.0;
                let upper = Vector2::from(max + max) / 2.0;
                compute_obstacle_velocity_obstacle_rect(player, &lower, &upper, player_radius)
            }
            Obstacle::Line { start, end } => {
                continue;
            }
        };
        rvo_ba_all.push(rvo_ba);
    }

    // Compute optimal velocity
    intersect(player, desired_velocity, &rvo_ba_all)
}

// Compute velocity obstacle for agent-obstacle interaction
fn compute_obstacle_velocity_obstacle_rect(
    agent: &PlayerData,
    obstacle_min: &Vector2,
    obstacle_max: &Vector2,
    robot_radius: f64,
) -> VelocityObstacle {
    let pa = agent.position;
    let lower = *obstacle_min;
    let upper = *obstacle_max;
    let corner_lu = Vector2::new(lower.x, upper.y);
    let corner_ul = Vector2::new(upper.x, lower.y);

    // angles to each of the box corners
    let theta_1 = (lower - pa).y.atan2((lower - pa).x);
    let theta_2 = (corner_lu - pa).y.atan2((corner_lu - pa).x);
    let theta_3 = (corner_ul - pa).y.atan2((corner_ul - pa).x);
    let theta_4 = (upper - pa).y.atan2((upper - pa).x);

    // find the minimal angle
    let mut angles = vec![theta_1, theta_2, theta_3, theta_4];
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut lowest_angle = Angle::from_radians(0.0);
    let mut uppest_angle = Angle::from_radians(0.0);
    let mut prev_max = 0.0;
    for lower_angle_rad in &angles {
        for upper_angle_rad in &angles {
            let lower_angle = Angle::from_radians(*lower_angle_rad);
            let upper_angle = Angle::from_radians(*upper_angle_rad);
            if (lower_angle - upper_angle).abs() > prev_max {
                prev_max = (lower_angle - upper_angle).abs();
                lowest_angle = lower_angle;
                uppest_angle = upper_angle;
            }
        }
    }

    let pb = Vector2::from(lower + upper) / 2.0;
    let translated_center = pa;
    let radius = (upper.x - lower.x).min(upper.y - lower.y) / 2.0;

    let dist_ba = distance(&pa, &pb);
    let mut theta_low: f64 = lowest_angle.radians();
    let mut theta_high: f64 = uppest_angle.radians();

    // ok, idfk how to fix the issue with getting stuck inside of
    // the objects, so i will just hardcode ignoring the obstacle
    // if we are inside of it :(

    if (lower.x <= pa.x) && (pa.x <= upper.x) && (lower.y <= pa.y) && (pa.y <= upper.y) {
        theta_low = 0.0;
        theta_high = 0.0;
    }

    VelocityObstacle {
        translated_center,
        left_bound: Vector2::new(theta_low.cos(), theta_low.sin()),
        right_bound: Vector2::new(theta_high.cos(), theta_high.sin()),
        dist: dist_ba,
        radius: radius + robot_radius,
    }
}

// Compute velocity obstacle for agent-agent interaction
fn compute_velocity_obstacle(
    agent_a: &PlayerData,
    agent_b: &PlayerData,
    vo_type: &VelocityObstacleType,
    robot_radius: f64,
) -> VelocityObstacle {
    let pa = agent_a.position;
    let pb = agent_b.position;
    let va = agent_a.velocity;
    let vb = agent_b.velocity;

    let dist_ba = distance(&pa, &pb);
    let dist_ba = dist_ba.max(2.0 * robot_radius);
    let theta_ba = Angle::between_points(pa, pb);
    let theta_ba_ort = Angle::from_radians((2.0 * robot_radius / dist_ba).asin());
    let theta_ort_left = theta_ba + theta_ba_ort;
    let theta_ort_right = theta_ba - theta_ba_ort;

    let translated_center = match vo_type {
        VelocityObstacleType::VO => pa + vb,
        VelocityObstacleType::RVO => pa + (va + vb) / 2.0,
        VelocityObstacleType::HRVO => {
            let dist_dif = ((vb - va) / 2.0).norm();
            pa + vb + theta_ort_left.to_vector() * dist_dif
        }
    };

    VelocityObstacle {
        translated_center,
        left_bound: theta_ort_left.to_vector(),
        right_bound: theta_ort_right.to_vector(),
        dist: dist_ba,
        radius: 2.0 * robot_radius,
    }
}

// Compute velocity obstacle for agent-obstacle interaction
fn compute_obstacle_velocity_obstacle(
    agent: &PlayerData,
    obstacle_center: &Vector2,
    obstacle_radius: &f64,
    robot_radius: f64,
) -> VelocityObstacle {
    let pa = agent.position;
    let pb = *obstacle_center;
    let translated_center = pa;

    let dist_ba = distance(&pa, &pb);
    let theta_ba = (pb - pa).y.atan2((pb - pa).x);

    let rad = obstacle_radius * OVER_APPROX_C2S;
    let dist_ba = dist_ba.max(rad + robot_radius);
    let theta_ba_ort = ((rad + robot_radius) / dist_ba).asin();

    let theta_ort_left = theta_ba + theta_ba_ort;
    let theta_ort_right = theta_ba - theta_ba_ort;

    VelocityObstacle {
        translated_center,
        left_bound: Vector2::new(theta_ort_left.cos(), theta_ort_left.sin()),
        right_bound: Vector2::new(theta_ort_right.cos(), theta_ort_right.sin()),
        dist: dist_ba,
        radius: rad + robot_radius,
    }
}

// Struct to represent a velocity obstacle
struct VelocityObstacle {
    translated_center: Vector2,
    left_bound: Vector2,
    right_bound: Vector2,
    dist: f64,
    radius: f64,
}

// Function to find the optimal velocity that satisfies all constraints
fn intersect(
    player: &PlayerData,
    desired_velocity: &Vector2,
    velocity_obstacles: &[VelocityObstacle],
) -> Vector2 {
    let position = player.position;
    let norm_v = desired_velocity.norm();
    let mut suitable_v = Vec::new();
    let mut unsuitable_v = Vec::new();

    // Check if the desired velocity satisfies all constraints
    if is_velocity_suitable(&position, desired_velocity, velocity_obstacles) {
        return *desired_velocity;
    } else {
        unsuitable_v.push(*desired_velocity);
    }

    // Parameters for sampling
    let radial_samples = 20;
    let angular_samples = 60;
    let max_radius = norm_v * 1.5; // Sample up to 1.5 times the desired velocity magnitude

    // Sample velocities and check if they satisfy all constraints
    for theta in (0..angular_samples).map(|t| (t as f64 / angular_samples as f64) * 2.0 * PI - PI) {
        for rad in (1..radial_samples).map(|r| (r as f64 / radial_samples as f64) * max_radius) {
            let new_v = Vector2::new(rad * theta.cos(), rad * theta.sin());
            if is_velocity_suitable(&player.position, &new_v, velocity_obstacles) {
                suitable_v.push(new_v);
            } else {
                unsuitable_v.push(new_v);
            }
        }
    }

    // If suitable velocities exist, choose the one closest to the desired velocity, that also has
    // a similar magnitude (e.g. even if we have to move to the side, move with decent speed)
    // technically, we want to choose the velocity that is going to bring as the fastest to the
    // target, but we don't know what the target point is, since only the 'desired velocity' is
    // available. So, this heuristic seems to be alright
    if !suitable_v.is_empty() {
        let best_v = suitable_v
            .into_iter()
            .min_by(|v1, v2| {
                let score1 = (v1 - desired_velocity).norm()
                    + (v1.norm() - desired_velocity.norm()).abs() * 2.0;
                let score2 = (v2 - desired_velocity).norm()
                    + (v2.norm() - desired_velocity.norm()).abs() * 2.0;

                score1.partial_cmp(&score2).unwrap()
            })
            .unwrap();

        return best_v;
    }

    // If no suitable velocity found, choose the "least bad" option

    unsuitable_v
        .into_iter()
        .min_by(|v1, v2| {
            let tc1 = time_to_collision(&position, v1, velocity_obstacles);
            let tc2 = time_to_collision(&position, v2, velocity_obstacles);
            let cost1 = 0.2 / (tc1 + 1e-6) + (v1 - desired_velocity).norm();
            let cost2 = 0.2 / (tc2 + 1e-6) + (v2 - desired_velocity).norm();
            cost1.partial_cmp(&cost2).unwrap()
        })
        .unwrap()
}

// Function to check if a velocity satisfies all constraints
fn is_velocity_suitable(
    position: &Vector2,
    velocity: &Vector2,
    velocity_obstacles: &[VelocityObstacle],
) -> bool {
    velocity_obstacles.iter().all(|vo| {
        let dif = velocity + position - vo.translated_center;
        let theta_dif = dif.y.atan2(dif.x);
        let theta_right = vo.right_bound.y.atan2(vo.right_bound.x);
        let theta_left = vo.left_bound.y.atan2(vo.left_bound.x);
        !in_between(theta_right, theta_dif, theta_left)
    })
}

// Function to compute time to collision for a given velocity
fn time_to_collision(
    position: &Vector2,
    velocity: &Vector2,
    velocity_obstacles: &[VelocityObstacle],
) -> f64 {
    let out = velocity_obstacles
        .iter()
        .filter_map(|vo| {
            let dif = velocity + position - vo.translated_center;
            let theta_dif = dif.y.atan2(dif.x);
            let theta_right = vo.right_bound.y.atan2(vo.right_bound.x);
            let theta_left = vo.left_bound.y.atan2(vo.left_bound.x);

            if in_between(theta_right, theta_dif, theta_left) {
                let small_theta = (theta_dif - 0.5 * (theta_left + theta_right)).abs();
                let rad = vo.radius.max((vo.dist * small_theta.sin()).abs());
                let big_theta = (vo.dist * small_theta.sin() / rad).abs().asin();
                let dist_tg = (vo.dist * small_theta.cos()).abs() - (rad * big_theta.cos()).abs();
                let dist_tg = dist_tg.max(0.0);
                Some(dist_tg / dif.norm())
            } else {
                None
            }
        })
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(f64::INFINITY);

    if out.is_nan() {
        f64::INFINITY
    } else {
        out
    }
}

// Function to check if an angle is between two other angles
fn in_between(theta_right: f64, theta_dif: f64, theta_left: f64) -> bool {
    if (theta_right - theta_left).abs() <= PI {
        (theta_right <= theta_dif && theta_dif <= theta_left)
            || (theta_right >= theta_dif && theta_dif >= theta_left)
    } else {
        let (mut theta_left, mut theta_right, mut theta_dif) = (theta_left, theta_right, theta_dif);
        if theta_left < 0.0 && theta_right > 0.0 {
            theta_left += 2.0 * PI;
            if theta_dif < 0.0 {
                theta_dif += 2.0 * PI;
            }
            (theta_right <= theta_dif && theta_dif <= theta_left)
                || (theta_right >= theta_dif && theta_dif >= theta_left)
        } else if theta_left > 0.0 && theta_right < 0.0 {
            theta_right += 2.0 * PI;
            if theta_dif < 0.0 {
                theta_dif += 2.0 * PI;
            }
            (theta_left <= theta_dif && theta_dif <= theta_right)
                || (theta_left >= theta_dif && theta_dif >= theta_right)
        } else {
            false
        }
    }
}
