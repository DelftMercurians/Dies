use dies_core::{
    find_intersection, perp, BallData, FieldGeometry, GameState, PlayerId, Vector2, WorldData,
};

use super::{attacker::AttackerSection, RoleCtx};
use crate::{
    invoke_skill,
    roles::{
        skills::{ApproachBall, Face, FetchBallWithHeading, Kick},
        Role, SkillResult,
    },
    skill, PlayerControlInput,
};

const MARGIN: f64 = 200.0;
const CORNER_RADIUS: f64 = 200.0;

enum State {
    Walling,
    Shooting(PlayerId),
}

/// A role that moves the player to the intersection of the ball's path with the goal
/// line, acting as a wall to block the ball from reaching the goal.
pub struct Waller {
    state: State,
    base_offset: f64,
    offset: f64,
}

impl Waller {
    /// Create a new Waller role with the given offset from the intersection point.
    pub fn new(offset: f64) -> Self {
        Self {
            state: State::Walling,
            offset,
            base_offset: offset,
        }
    }

    pub fn fetch(&mut self) {
        // self.state = State::Shooting(())
    }

    pub fn goalie_shooting(&mut self, shooting: bool) {
        if shooting {
            self.offset = self.base_offset + 100.0 * self.base_offset.signum();
        } else {
            self.offset = self.base_offset;
        }
    }

    /// Find the intersection point of the ball's path with the goal line and return the
    /// target position for the player.
    fn find_intersection(
        &self,
        ball: &BallData,
        world: &FieldGeometry,
        player_id: PlayerId,
    ) -> Vector2 {
        let ball_pos = ball.position.xy() + ball.velocity.xy() * 0.3; // fake it til you make it
                                                                      // compensate for delay
        let half_length = world.field_length / 2.0;
        let goal_center = Vector2::new(-half_length, 0.0);

        // Compute the direction vector from the ball to the goal center
        let direction = (goal_center - ball_pos).normalize();

        // Points redebuesenting the boundary of the goalkeeper area
        let area_top_y = world.penalty_area_width / 2.0 + MARGIN; // top boundary y-coordinate
        let area_bottom_y = -world.penalty_area_width / 2.0 - MARGIN; // bottom boundary y-coordinate
        let area_right_x = -half_length + world.penalty_area_depth + MARGIN; // right boundary x-coordinate

        let top_right = Vector2::new(area_right_x, area_top_y);
        let bottom_right = Vector2::new(area_right_x, area_bottom_y);

        // Intersect with the right boundary
        if let Some(raw_intersection) =
            find_intersection(ball_pos, direction, top_right, Vector2::y())
        {
            let shift = Vector2::new(0.0, self.offset);
            let intersection = raw_intersection - shift;
            if intersection.y <= area_top_y && intersection.y >= area_bottom_y {
                // Check if it is a corner
                let corner = if (intersection - top_right + shift).norm() < CORNER_RADIUS / 2.0 {
                    Some(top_right - Vector2::new(0.6, 0.6) * CORNER_RADIUS)
                } else if (intersection - bottom_right + shift).norm() < CORNER_RADIUS / 2.0 {
                    Some(bottom_right - Vector2::new(0.6, -0.6) * CORNER_RADIUS)
                } else {
                    None
                };
                if let Some(c) = corner {
                    let center = c + (intersection - c).normalize();
                    return center;
                }
                return intersection;
            }
        }

        if direction.y != 0.0 {
            // Intersect with the bottom boundary
            if let Some(raw_intersection) =
                find_intersection(ball_pos, direction, bottom_right, Vector2::x())
            {
                let shift = Vector2::new(self.offset, 0.0);
                let intersection = raw_intersection - shift;
                if intersection.x <= area_right_x && ball_pos.y < 0.0 {
                    let new_x = f64::max(intersection.x, -half_length);
                    return Vector2::new(new_x, intersection.y);
                }
            }

            // Intersect with the top boundary
            if let Some(raw_intersection) =
                find_intersection(ball_pos, direction, top_right, Vector2::x())
            {
                let shift = Vector2::new(-self.offset, 0.0);
                let intersection = raw_intersection - shift;
                if intersection.x <= area_right_x && ball_pos.y > 0.0 {
                    let new_x = f64::max(intersection.x, -half_length);
                    return Vector2::new(new_x, intersection.y);
                }
            }
        }

        // Default fallback to ball position (should not happen in normal cases)
        Vector2::new(area_right_x,
                     f64::min(
                        f64::max(
                            ball_pos.y, -world.penalty_area_width / 2.0 - MARGIN * 2.0
                        ),
                        world.penalty_area_width / 2.0 + MARGIN * 2.0)
                     )
    }
}

impl Role for Waller {
    fn update(&mut self, mut ctx: RoleCtx<'_>) -> PlayerControlInput {
        if let (Some(ball), Some(geom)) = (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref()) {
            if ctx.world.current_game_state.game_state == GameState::Stop {
                ctx.reset_skills();
                let mut input = PlayerControlInput::new();

                let mut target_pos = self.find_intersection(ball, geom, ctx.player.id);
                let mut input = PlayerControlInput::new();
                target_pos.y = target_pos
                    .y
                    .max(-geom.penalty_area_width / 2.0 - MARGIN)
                    .min(geom.penalty_area_width / 2.0 + MARGIN);
                input.with_position(target_pos);

                input.with_speed_limit(1300.0);
                input.avoid_ball = true;
                if let Some(ball) = ctx.world.ball.as_ref() {
                    let ball_pos = ball.position.xy();
                    let dist = (ball_pos - ctx.player.position.xy()).norm();
                    if dist < 560.0 {
                        // Move away from the ball
                        let target = ball_pos.xy()
                            + (ctx.player.position - ball_pos.xy()).normalize() * 650.0;
                        input.with_position(target);
                    }
                }

                return input;
            }

            match self.state {
                State::Walling => {
                    let mut target_pos = self.find_intersection(ball, geom, ctx.player.id);
                    let mut input = PlayerControlInput::new();
                    target_pos.y = target_pos
                        .y
                        .max(-geom.penalty_area_width / 2.0 - MARGIN)
                        .min(geom.penalty_area_width / 2.0 + MARGIN);
                    input.with_position(target_pos);
                    // input.with_yaw(Angle::between_points(
                    //     ctx.player.position,
                    //     ball.position.xy(),
                    // ));
                    input
                }
                State::Shooting(receiver_id) => {
                    skill!(ctx, FetchBallWithHeading::towards_own_player(receiver_id));

                    loop {
                        skill!(ctx, ApproachBall::new());
                        match invoke_skill!(ctx, Face::towards_own_player(receiver_id)) {
                            crate::roles::SkillProgress::Continue(mut input) => {
                                input.with_dribbling(1.0);
                                return input;
                            }
                            _ => {}
                        }
                        if let SkillResult::Success = skill!(ctx, Kick::new()) {
                            break PlayerControlInput::new();
                        }
                    }
                }
            }
        } else {
            PlayerControlInput::new()
        }
    }
}

/// Calculate the distance between two points.
fn distance(a: Vector2, b: Vector2) -> f64 {
    (a - b).norm()
}

// /// Find a position with the best line of sight to the goal withing the given section.
// fn find_best_striker_position(world: &WorldData, field: &FieldGeometry) -> Vector2 {
//     let (min_y, max_y) = section.y_bounds(field);
//     let min_x = 100.0;
//     let max_x = field.field_length / 2.0 - 100.0;

//     let mut best_position = Vector2::new(0.0, 0.0);
//     let mut best_score = 0.0;

//     for x in (min_x as i32..max_x as i32).step_by(20) {
//         for y in (min_y as i32..max_y as i32).step_by(20) {
//             let position = Vector2::new(x as f64, y as f64);
//             if !is_pos_valid(position, field) {
//                 continue;
//             }

//             let ball_score = score_line_of_sight(
//                 world,
//                 position,
//                 world
//                     .ball
//                     .as_ref()
//                     .map(|b| b.position.xy())
//                     .unwrap_or_default(),
//                 field,
//             );
//             let goal_score = score_line_of_sight(
//                 world,
//                 position,
//                 Vector2::new(field.field_length / 2.0, 0.0),
//                 field,
//             );
//             let goal_dist_score =
//                 1.0 / (position - Vector2::new(field.field_length / 2.0, 0.0)).norm();
//             let score = ball_score + goal_score + goal_dist_score;
//             if score > best_score {
//                 best_score = score;
//                 best_position = position;
//             }
//         }
//     }

//     best_position
// }

// fn find_best_passer_position(
//     starting_pos: Vector2,
//     max_radius: f64,
//     world: &WorldData,
//     field: &FieldGeometry,
// ) -> (Vector2, Option<PlayerId>, f64) {
//     let mut best_position = Vector2::new(0.0, 0.0);
//     let mut best_score = 0.0;
//     let mut best_striker = None;

//     let attackers = world
//         .own_players
//         .iter()
//         .filter(|p| p.position.x > 0.0)
//         .collect::<Vec<_>>();

//     let min_theta = -FRAC_PI_2;
//     let max_theta = FRAC_PI_2;
//     for theta in (min_theta as i32..max_theta as i32).step_by(10) {
//         let theta = theta as f64;
//         for radius in (0..max_radius as i32).step_by(20) {
//             let x = starting_pos.x + (radius as f64) * theta.cos();
//             let y = starting_pos.y + (radius as f64) * theta.sin();
//             let position = Vector2::new(x, y);
//             if !is_pos_valid(position, field) {
//                 continue;
//             }

//             let striker_score = attackers
//                 .iter()
//                 .map(|p| {
//                     (
//                         p.id,
//                         score_line_of_sight(world, position, p.position, field),
//                     )
//                 })
//                 .max_by_key(|&x| x.1 as i64);
//             let goal_score = score_line_of_sight(
//                 world,
//                 position,
//                 Vector2::new(field.field_length / 2.0, 0.0),
//                 field,
//             );
//             if let Some((striker_id, score)) = striker_score {
//                 let score = score + goal_score;
//                 if score > best_score {
//                     best_score = score;
//                     best_position = position;
//                     best_striker = Some(striker_id);
//                 }
//             }
//         }
//     }

//     (best_position, best_striker, best_score)
// }

fn is_pos_valid(pos: Vector2, field: &FieldGeometry) -> bool {
    const MARGIN: f64 = 100.0;
    // check if pos inside penalty area
    if pos.x.abs() > field.field_length / 2.0 - field.penalty_area_depth - MARGIN
        && pos.y.abs() < field.penalty_area_width / 2.0 + MARGIN
    {
        return false;
    }
    true
}

/// Compute a "badness" score for a line of sight between two points based on the minumum
/// distance to the line of sight from the closest enemy player.
///
/// The score is higher if the line of sight is further from the enemy players.
fn score_line_of_sight(
    world: &WorldData,
    from: Vector2,
    to: Vector2,
    field: &FieldGeometry,
) -> f64 {
    let mut min_distance = f64::MAX;
    for player in world.opp_players.iter() {
        let distance = distance_to_line(from, to, player.position);
        if distance < min_distance {
            min_distance = distance;
        }
    }
    min_distance
}

fn distance_to_line(a: Vector2, b: Vector2, p: Vector2) -> f64 {
    let n = (b - a).normalize();
    let ap = p - a;
    let proj = ap.dot(&n);
    let proj = proj.max(0.0).min((b - a).norm());
    (ap - proj * n).norm()
}
