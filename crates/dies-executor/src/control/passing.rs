use dies_core::{Angle, FieldGeometry, PlayerData, PlayerId, TeamData, Vector2};
use std::{sync::Arc, time::Duration};

use crate::behavior_tree::RobotSituation;

#[derive(Clone, Debug)]
pub enum ShootTarget {
    Goal(Vector2),
    Player {
        id: PlayerId,
        position: Option<Vector2>,
    },
}

impl ShootTarget {
    pub fn position(&self) -> Option<Vector2> {
        match self {
            ShootTarget::Goal(position) => Some(*position),
            ShootTarget::Player { position, .. } => *position,
        }
    }
}

#[derive(Clone)]
pub struct PassingStore {
    pub player_id: PlayerId,
    pub world: Arc<TeamData>,
}

impl PassingStore {
    pub fn new(player_id: PlayerId, world: Arc<TeamData>) -> Self {
        Self { player_id, world }
    }

    pub fn player_id_hash(&self) -> f64 {
        (self.player_id.as_u32() as f64 * 0.6180339887498949) % 1.0
    }

    pub fn player_data(&self) -> &PlayerData {
        self.world
            .own_players
            .iter()
            .find(|&p| p.id == self.player_id)
            .unwrap()
    }

    pub fn get_teammate_by_id(&self, teammate_id: PlayerId) -> Option<&PlayerData> {
        self.world.own_players.iter().find(|p| p.id == teammate_id)
    }

    pub fn field(&self) -> FieldGeometry {
        self.world.field_geom.clone().unwrap_or_default()
    }

    pub fn get_own_goal_position(&self) -> Vector2 {
        self.world
            .field_geom
            .as_ref()
            .map(|f| Vector2::new(-f.field_length / 2.0, 0.0))
            .unwrap_or_else(|| Vector2::new(-4500.0, 0.0))
    }

    pub fn get_opp_goal_position(&self) -> Vector2 {
        self.world
            .field_geom
            .as_ref()
            .map(|f| Vector2::new(f.field_length / 2.0, 0.0))
            .unwrap_or_else(|| Vector2::new(4500.0, 0.0))
    }

    pub fn force_self_position(&self, pos: Vector2) -> PassingStore {
        let mut temp = self.clone();
        let mut world_copy = (*temp.world).clone();
        if let Some(p) = world_copy
            .own_players
            .iter_mut()
            .find(|p| p.id == self.player_id)
        {
            p.position = pos;
        }
        temp.world = world_copy.into();
        temp
    }

    pub fn change_situation_player(&self, other_id: PlayerId) -> PassingStore {
        let mut copy = self.clone();
        copy.player_id = other_id;
        copy
    }
}

impl From<RobotSituation> for PassingStore {
    fn from(value: RobotSituation) -> Self {
        PassingStore {
            player_id: value.player_id,
            world: value.world,
        }
    }
}

impl<'a> From<&'a RobotSituation> for PassingStore {
    fn from(value: &'a RobotSituation) -> Self {
        PassingStore {
            player_id: value.player_id,
            world: value.world.clone(),
        }
    }
}

impl<'a> From<&PassingStore> for PassingStore {
    fn from(value: &PassingStore) -> Self {
        PassingStore {
            player_id: value.player_id,
            world: value.world.clone(),
        }
    }
}

fn erf_approx(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

fn would_hit_goal_planks(
    robot_pos: Vector2,
    target_pos: Vector2,
    goal_pos: Vector2,
    goal_width: f64,
    goal_depth: f64,
) -> bool {
    let left_plank_start = Vector2::new(goal_pos.x + goal_depth, goal_pos.y - goal_width / 2.0);
    let left_plank_end = Vector2::new(goal_pos.x, goal_pos.y - goal_width / 2.0);

    let right_plank_start = Vector2::new(goal_pos.x + goal_depth, goal_pos.y + goal_width / 2.0);
    let right_plank_end = Vector2::new(goal_pos.x, goal_pos.y + goal_width / 2.0);

    // Check if the shot line intersects with either plank
    line_segments_intersect(robot_pos, target_pos, left_plank_start, left_plank_end)
        || line_segments_intersect(robot_pos, target_pos, right_plank_start, right_plank_end)
}

fn line_segments_intersect(p1: Vector2, p2: Vector2, p3: Vector2, p4: Vector2) -> bool {
    // Check if line segment p1-p2 intersects with line segment p3-p4
    let d1 = direction(p3, p4, p1);
    let d2 = direction(p3, p4, p2);
    let d3 = direction(p1, p2, p3);
    let d4 = direction(p1, p2, p4);

    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0))
        && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0))
    {
        return true;
    }

    // Check for collinear cases
    if d1.abs() < 1e-9 && on_segment(p3, p1, p4) {
        return true;
    }
    if d2.abs() < 1e-9 && on_segment(p3, p2, p4) {
        return true;
    }
    if d3.abs() < 1e-9 && on_segment(p1, p3, p2) {
        return true;
    }
    if d4.abs() < 1e-9 && on_segment(p1, p4, p2) {
        return true;
    }

    false
}

fn direction(a: Vector2, b: Vector2, c: Vector2) -> f64 {
    // Returns > 0 for counterclockwise, < 0 for clockwise, 0 for collinear
    (c.x - a.x) * (b.y - a.y) - (b.x - a.x) * (c.y - a.y)
}

fn on_segment(p: Vector2, q: Vector2, r: Vector2) -> bool {
    // Check if point q lies on segment pr
    q.x <= p.x.max(r.x) && q.x >= p.x.min(r.x) && q.y <= p.y.max(r.y) && q.y >= p.y.min(r.y)
}

pub fn best_goal_shoot(s: &PassingStore) -> (Vector2, f64) {
    let robot_pos = s.player_data().position;
    let goal_pos = s.get_opp_goal_position();

    // Get goal geometry
    let geom = s.world.field_geom.clone().unwrap();
    let goal_width = geom.goal_width;
    let goal_depth = geom.goal_depth;

    // Calculate goal boundaries
    let goal_left = Vector2::new(goal_pos.x, goal_pos.y - goal_width / 2.0);
    let goal_right = Vector2::new(goal_pos.x, goal_pos.y + goal_width / 2.0);

    // Sample angles between goal boundaries
    let mut best_prob: f64 = 0.0;
    let mut best_pos = goal_pos;

    let num_samples = 40;
    for i in 0..num_samples {
        let t = i as f64 / (num_samples - 1) as f64;

        let pos = goal_left * (1.0 - t) + goal_right * t;

        // Check if shot would hit the goal planks
        if would_hit_goal_planks(robot_pos, pos, goal_pos, goal_width, goal_depth) {
            continue; // Skip this target if it would hit the planks
        }

        let prob = goal_shoot_success_probability(s, pos);

        if prob > best_prob {
            best_prob = prob;
            best_pos = pos;
        }
    }

    (best_pos, best_prob)
}

fn goal_shoot_success_probability(s: &PassingStore, target_pos: Vector2) -> f64 {
    let mut prob: f64 = 1.0;

    let player_pos = s.player_data().position;
    let goal_pos = s.get_opp_goal_position();
    let direction = Angle::between_points(player_pos, target_pos);

    // backward discounting
    let backward_dist = (player_pos.y - target_pos.y).max(0.0); // + when bad, - when good
    prob *= (1.2 - (5000.0 / backward_dist).min(1.0));

    let nearest_opponent_distance = find_nearest_opponent_distance_along_direction(s, direction);

    // Factor 1: Distance to nearest opponent (larger is better, smooth curve)
    let opp_distance_factor = if nearest_opponent_distance < 100.0 {
        0.03
    } else if nearest_opponent_distance >= 1000.0 {
        1.0
    } else {
        let normalized_distance = (nearest_opponent_distance - 100.0) / 900.0;
        let smooth_factor = normalized_distance * normalized_distance; // smoothstep
        0.03 + smooth_factor * 0.97
    };
    prob *= opp_distance_factor;

    // Factor 2: Angle to middle of goal (closer to center is better)
    let center_angle = Angle::between_points(goal_pos, player_pos);
    let angle_diff = (direction.radians() - center_angle.radians()).abs();
    let angle_factor = if angle_diff <= 0.1 {
        1.0
    } else if angle_diff <= 0.3 {
        0.7
    } else {
        0.4
    };
    prob *= angle_factor;

    // Factor 2': Angle of the goal visibility
    // this one can actually be computed analytically
    let shooting_noise_std = 0.14; // in rad
    let goal_left = Vector2::new(
        goal_pos.x,
        goal_pos.y - s.world.field_geom.as_ref().unwrap().goal_width / 2.0,
    );
    let goal_right = Vector2::new(
        goal_pos.x,
        goal_pos.y + s.world.field_geom.as_ref().unwrap().goal_width / 2.0,
    );
    let left_angle = Angle::between_points(player_pos, goal_left);
    let right_angle = Angle::between_points(player_pos, goal_right);

    // Compute probability mass of normal distribution falling into goal interval
    let center_angle = direction.radians();
    let left_bound = left_angle.radians();
    let right_bound = right_angle.radians();

    // Normalize angles to [-π, π] and handle wrapping
    let mut left_diff = left_bound - center_angle;
    let mut right_diff = right_bound - center_angle;

    // Handle angle wrapping
    if left_diff > std::f64::consts::PI {
        left_diff -= 2.0 * std::f64::consts::PI;
    } else if left_diff < -std::f64::consts::PI {
        left_diff += 2.0 * std::f64::consts::PI;
    }

    if right_diff > std::f64::consts::PI {
        right_diff -= 2.0 * std::f64::consts::PI;
    } else if right_diff < -std::f64::consts::PI {
        right_diff += 2.0 * std::f64::consts::PI;
    }

    // Ensure left_diff <= right_diff for proper integration bounds
    if left_diff > right_diff {
        std::mem::swap(&mut left_diff, &mut right_diff);
    }

    // Calculate CDF values using error function approximation
    let sqrt_2 = std::f64::consts::SQRT_2;
    let left_z = left_diff / (shooting_noise_std * sqrt_2);
    let right_z = right_diff / (shooting_noise_std * sqrt_2);

    // Approximate error function using built-in methods
    let left_cdf = 0.5 * (1.0 + erf_approx(left_z));
    let right_cdf = 0.5 * (1.0 + erf_approx(right_z));

    let visibility_factor = (right_cdf - left_cdf).max(0.0);

    prob *= visibility_factor;

    // Factor 3: Distance preference (closer intersection with goal line)
    let direction_vector = direction.to_vector();
    let t = (goal_pos.x - player_pos.x) / direction_vector.x;
    let intersection_y = player_pos.y + t * direction_vector.y;
    let intersection = Vector2::new(goal_pos.x, intersection_y);
    let distance_to_intersection = (player_pos - intersection).norm();
    let goal_distance_factor = if distance_to_intersection < 1000.0 {
        1.0
    } else if distance_to_intersection < 1500.0 {
        0.97
    } else if distance_to_intersection < 2000.0 {
        0.95
    } else if distance_to_intersection < 2500.0 {
        0.93
    } else if distance_to_intersection < 3000.0 {
        0.9
    } else if distance_to_intersection < 4000.0 {
        0.8
    } else {
        0.7
    };
    prob *= goal_distance_factor;

    prob
}

pub fn pass_success_probability(s: &PassingStore, teammate_pos: Vector2) -> f64 {
    let mut prob: f64 = 0.4;
    let player_pos = s.player_data().position;
    // score based on how far is the robot: not too close, not too far
    let robot_dist = (player_pos - teammate_pos).norm();
    let no_miss_probability = if robot_dist < 400.0 {
        0.5 // not a super good idea to do short shoots
    } else if robot_dist < 800.0 {
        0.8 // this one is doable, but maybe a bit further would be nice
    } else if robot_dist < 1200.0 {
        0.9 // very doable
    } else if robot_dist < 2000.0 {
        0.95 // bearable
    } else if robot_dist < 3500.0 {
        1.0 // good
    } else if robot_dist < 5000.0 {
        0.8 // this is mid
    } else {
        0.4 // meh
    };
    prob *= no_miss_probability;

    // score based on how bad is the trajectory: are there opponents on the shoot line?
    let angle = Angle::between_points(teammate_pos, player_pos);
    let nearest_opponent_distance =
        find_nearest_opponent_distance_along_direction(s, angle).clamp(0.0, 1000.0);
    let no_intercept_prob = if nearest_opponent_distance < 200.0 {
        0.1
    } else if nearest_opponent_distance < 250.0 {
        0.3
    } else if nearest_opponent_distance < 300.0 {
        0.6
    } else if nearest_opponent_distance < 600.0 {
        0.8
    } else {
        1.0
    };
    prob *= no_intercept_prob;

    // println!("{} passing {}: {:.2}; clean: {:.2}", s.player_id, teammate.id, score, clean_shoot_score);

    prob
}

pub fn best_teammate_pass_or_shoot(s: &PassingStore, allow_passing: bool) -> (ShootTarget, f64) {
    let teammates = &s.world.own_players;

    let (best_target_direct, best_prob_direct) = best_goal_shoot(s);
    let best_target_direct = ShootTarget::Goal(best_target_direct);
    let mut best_prob_pass = 0.0;
    let mut best_target_pass = best_target_direct.clone();

    for teammate in teammates {
        if teammate.id == s.player_id {
            continue;
        }
        if teammate.position.x < 0.0 {
            continue;
        }

        // score is a combination of clean shoot from the teammate and passing discount
        let (t, goal_shoot_prob) = best_goal_shoot(&s.change_situation_player(teammate.id));
        let prob = goal_shoot_prob * pass_success_probability(s, teammate.position);

        if prob > best_prob_pass {
            best_prob_pass = prob;
            best_target_pass = ShootTarget::Player {
                id: teammate.id,
                position: teammate.position.into(),
            };
        }
    }

    // Non-deterministic choice between passing and shooting
    // We use a probabilistic approach that considers both the absolute probabilities
    // and the relative difference between them

    // Apply a bias factor to make the choice more interesting
    let bias_factor = 1.5; // higher -> more favor
    let min_prob_threshold = 0.0; // Minimum probability to consider an option

    // Only consider options above minimum threshold
    let direct_viable = best_prob_direct >= min_prob_threshold;
    let pass_viable = best_prob_pass >= min_prob_threshold;
    let best_prob = best_prob_direct.max(best_prob_pass);
    if !allow_passing {
        // println!("no passing, direct: {:.3}", best_prob_direct);
        return (best_target_direct, best_prob_direct);
    }

    // return (best_target_pass, best_prob_pass);
    let best_target = if !direct_viable && !pass_viable {
        // println!("no direct or pass, direct: {:.3}", best_prob_direct);
        // If neither option is viable, default to direct shooting
        best_target_direct
    } else if !pass_viable {
        // Only direct shooting is viable
        // println!("no pass, direct: {:.3}", best_prob_direct);
        best_target_direct
    } else if !direct_viable {
        // Only passing is viable
        // println!("no direct, pass: {:.3}", best_prob_pass);
        best_target_pass
    } else {
        // Both options are viable - make probabilistic choice
        let total_prob = best_prob_direct + best_prob_pass;

        // Normalize probabilities and apply bias toward the better option
        let direct_weight = best_prob_direct / total_prob;
        let pass_weight = best_prob_pass / total_prob;

        // Apply bias factor to make the better option more likely
        let biased_direct = direct_weight.powf(bias_factor);
        let biased_pass = pass_weight.powf(bias_factor);
        let biased_total = biased_direct + biased_pass;

        let final_direct_prob = biased_direct / biased_total;
        // println!(
        //     "direct: {:.3}, pass: {:.3}",
        //     final_direct_prob,
        //     1.0 - final_direct_prob
        // );

        // Use a simple random number generator based on robot position and time-like hash
        // This provides deterministic but seemingly random behavior
        let random_value = 0.25 + s.player_id_hash() * 0.5;
        // println!(

        if random_value < final_direct_prob {
            // println!("chosen direct shoot with p={:.3}", final_direct_prob);
            best_target_direct
        } else {
            // println!("chosen pass with p={:.3}", 1.0 - final_direct_prob);
            best_target_pass
        }
    };

    (best_target, best_prob)
}

pub fn best_receiver_target_score(s: &PassingStore) -> f64 {
    // for each teammate, check how far the ball is, and take average of their probabilityies
    // weighted by the distance to the ball (the closer our teammate to the ball, the more
    // important it is for us to support him).
    // PS: we don't have to explicitly try to get far away from other robots because we
    // already account for this in probability of passing.
    let ball = match s.world.ball.as_ref() {
        Some(b) => b,
        None => return 0.0, // early return if ball is not found
    };
    let ball_pos = ball.position.xy();

    let (_, goal_shoot_prob) = best_goal_shoot(s);

    let hypothetical = s.force_self_position(ball_pos);
    pass_success_probability(&hypothetical, s.player_data().position) * goal_shoot_prob
}

pub fn combination_discounting_for_receivers(s: &PassingStore) -> f64 {
    // value between 0 and 1 that totally screws up our pure bayesian shit
    // and tries to allocate the robots such that the distance between robots are large and
    // that we are not blocking line of sight for other robots
    // TODO: add blocking based on how much we block of the goal (we don't want to stand between
    // the goal and the ball)

    let ball = match s.world.ball.as_ref() {
        Some(b) => b,
        None => return 0.0, // early return if ball is not found
    };
    let ball_pos = ball.position.xy();
    let mut discount = 1.0;
    let player_pos = s.player_data().position;
    let teammates = &s.world.own_players;

    // Factor 1: Distance penalty - penalize being too close to teammates
    for teammate in teammates {
        if teammate.id == s.player_id {
            continue;
        }
        if teammate.position.x < 0.0 {
            continue;
        }

        let distance = (player_pos - teammate.position).norm();
        if distance < 2000.0 {
            let proximity_penalty = (2000.0 - distance) / 2000.0;
            discount *= 1.0 - proximity_penalty * 0.5;
        }
    }

    // Factor 2: Line of sight blocking penalty to goal
    let goal_pos = s.get_opp_goal_position();
    let blocking_penalty = calculate_line_blocking_penalty(player_pos, ball_pos, goal_pos);
    discount *= blocking_penalty;

    discount.max(0.001) // Minimum discount to avoid completely zeroing out
}

fn calculate_line_blocking_penalty(
    blocker_pos: Vector2,
    shooter_pos: Vector2,
    target_pos: Vector2,
) -> f64 {
    let shooter_to_target = target_pos - shooter_pos;
    let shooter_to_blocker = blocker_pos - shooter_pos;

    let line_length = shooter_to_target.norm();
    if line_length < 100.0 {
        return 1.0; // Too close to matter
    }

    // Project blocker onto the line from shooter to target
    let projection = shooter_to_blocker.dot(&shooter_to_target) / line_length;

    // Only care if blocker is between shooter and target
    if projection < 0.0 || projection > line_length {
        return 1.0; // Not blocking
    }

    // Calculate distance from blocker to the line
    let projected_point = shooter_pos + (projection / line_length) * shooter_to_target;
    let distance_to_line = (blocker_pos - projected_point).norm();

    // Apply penalty if too close to the line
    if distance_to_line < 200.0 {
        return 0.4; // Significant penalty for blocking
    } else if distance_to_line < 350.0 {
        return 0.7; // Moderate penalty
    } else if distance_to_line < 500.0 {
        return 0.9;
    }

    1.0 // No penalty
}

pub fn find_best_receiver_target(
    s: &PassingStore,
    last_position: Option<Vector2>,
) -> (Vector2, f64) {
    // we want to sample positions all around the enemy half;
    // however, this will be slow, so we limit the number to "merely" a 100 positions to consider
    // sample points (x,y) on the opponents half of the field, choose the best one

    let Some(field_geom) = &s.world.field_geom else {
        return (s.player_data().position, 0.0); // fallback if no field geometry
    };

    let half_length = field_geom.field_length / 2.0;
    let half_width = field_geom.field_width / 2.0;

    // Sample in opponent's half (positive x direction)
    let x_min = 0.0; // center line
    let x_max = half_length - 200.0; // stay away from goal line
    let y_min = -half_width + 200.0; // stay in bounds
    let y_max = half_width - 200.0;

    let opp_goal_pos = s.get_opp_goal_position();
    let penalty_depth = field_geom.penalty_area_depth;
    let penalty_width = field_geom.penalty_area_width;

    let mut best_position = s.player_data().position;
    let mut best_score = 0.0;

    let num_samples = 150;
    let samples_per_axis = (num_samples as f64).sqrt() as i32;

    for i in 0..samples_per_axis {
        for j in 0..samples_per_axis {
            let t_x = i as f64 / (samples_per_axis - 1) as f64;
            let t_y = j as f64 / (samples_per_axis - 1) as f64;

            let x = x_min + t_x * (x_max - x_min);
            let y = y_min + t_y * (y_max - y_min);

            let candidate_pos = Vector2::new(x, y);

            // if candidate position within goalie area -> skip
            if candidate_pos.x >= opp_goal_pos.x - penalty_depth
                && candidate_pos.y >= opp_goal_pos.y - penalty_width / 2.0
                && candidate_pos.y <= opp_goal_pos.y + penalty_width / 2.0
            {
                continue;
            }

            // Create a temporary situation with this position
            let temp_situation = s.force_self_position(candidate_pos);

            let dumb_heuristic = combination_discounting_for_receivers(&temp_situation);
            let mut score = best_receiver_target_score(&temp_situation) * dumb_heuristic;

            // Add stability bonus if we have a last position
            if let Some(last_pos) = last_position {
                let distance_to_last = (candidate_pos - last_pos).norm();
                if distance_to_last < 50.0 {
                    // Within 5cm
                    score *= 1.3;
                }
            }

            if score > best_score {
                best_score = score;
                best_position = candidate_pos;
            }
        }
    }

    (best_position, best_score)
}

pub fn score_shoot_target(s: &PassingStore, target: ShootTarget) -> f64 {
    match target {
        ShootTarget::Goal(pos) => goal_shoot_success_probability(s, pos),
        ShootTarget::Player { id, position } => {
            let (t, goal_shoot_prob) = best_goal_shoot(&s.change_situation_player(id));
            let pos = if let Some(p) = s.get_teammate_by_id(id) {
                p.position
            } else {
                position.unwrap_or(Vector2::zeros())
            };
            let prob = goal_shoot_prob * pass_success_probability(s, pos);
            prob * goal_shoot_prob
        }
    }
}

pub fn find_best_shoot_target(s: impl Into<PassingStore>, allow_pasing: bool) -> ShootTarget {
    let s: PassingStore = s.into();
    let (t, _) = best_teammate_pass_or_shoot(&s, allow_pasing);
    t
}

pub fn find_best_preshoot(
    s: &PassingStore,
    last_target: Option<ShootTarget>,
    allow_passing: bool,
) -> ShootTarget {
    let ball = match s.world.ball.as_ref() {
        Some(b) => b,
        None => return ShootTarget::Goal(Vector2::zeros()), // early return if ball is not found
    };
    let player_pos = s.player_data().position;
    let ball_pos = ball.position.xy();
    let hypothetical = s.force_self_position(ball_pos);

    let (cur_target, cur_score) = best_teammate_pass_or_shoot(&hypothetical, allow_passing);

    if let Some(last_target) = last_target {
        let last_score = score_shoot_target(s, last_target.clone());
        if cur_score > last_score * 1.1 {
            return cur_target;
        } else {
            return last_target;
        }
    }
    cur_target
}

pub fn find_best_preshoot_target(s: &PassingStore, allow_pasing: bool) -> Vector2 {
    let ball = match s.world.ball.as_ref() {
        Some(b) => b,
        None => return Vector2::zeros(), // early return if ball is not found
    };
    let player_pos = s.player_data().position;
    let ball_pos = ball.position.xy();
    let hypothetical = s.force_self_position(ball_pos);

    let goal_pos = s.get_opp_goal_position();
    let target = find_best_shoot_target(&hypothetical, allow_pasing)
        .position()
        .unwrap_or(goal_pos);
    let to_target = target - ball_pos;

    ball_pos - to_target.normalize() * 150.0
}

pub fn find_best_preshoot_heading(s: &PassingStore, allow_pasing: bool) -> Angle {
    let ball = match s.world.ball.as_ref() {
        Some(b) => b,
        None => return Angle::from_radians(0.0), // early return if ball is not found
    };
    let player_pos = s.player_data().position;
    let ball_pos = ball.position.xy();
    Angle::between_points(find_best_preshoot_target(s, allow_pasing), ball_pos)
}

pub fn find_nearest_opponent_distance_along_direction(s: &PassingStore, direction: Angle) -> f64 {
    let player_pos = s.player_data().position;
    let direction_vector = direction.to_vector();

    let mut min_distance = f64::INFINITY;

    // Check all opponent robots
    for player in s.world.opp_players.iter() {
        if player.id != s.player_data().id {
            let opp_pos = player.position;
            let to_opponent = opp_pos - player_pos;

            // Project opponent position onto the shooting direction
            let projection = to_opponent.dot(&direction_vector);

            // Only consider opponents in front of us
            if projection > 0.0 {
                let perpendicular_distance = (to_opponent - projection * direction_vector).norm();

                // Consider robot radius (approximate as 90mm)
                let effective_distance = perpendicular_distance - 90.0;
                min_distance = min_distance.min(effective_distance.max(0.0));
            }
        }
    }

    min_distance
}
