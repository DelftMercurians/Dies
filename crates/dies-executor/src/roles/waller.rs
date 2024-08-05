use dies_core::{find_intersection, Angle, BallData, FieldGeometry, PlayerId, Vector2, WorldData};

use super::RoleCtx;
use crate::{
    invoke_skill,
    roles::{
        skills::{ApproachBall, Face, FetchBallWithHeading, Kick},
        Role, SkillResult,
    },
    skill, PlayerControlInput,
};

const MARGIN: f64 = 200.0;

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
    pub fn new_with_index(index: usize) -> Self {
        let spacing: f64 = 100.0;
        let offset = spacing * ((index / 2) + 1) as f64 * if index % 2 == 0 { 1.0 } else { -1.0 };
        dbg!(offset);
        Self {
            state: State::Walling,
            offset,
            base_offset: offset,
        }
    }

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
        Vector2::new(
            area_right_x,
            f64::min(
                f64::max(ball_pos.y, -world.penalty_area_width / 2.0 - MARGIN * 2.0),
                world.penalty_area_width / 2.0 + MARGIN * 2.0,
            ),
        )
    }
}

impl Role for Waller {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        if let (Some(ball), Some(geom)) = (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref()) {
            match self.state {
                State::Walling => {
                    dies_core::debug_string(format!("p{}.waller_mode", ctx.player.id), "walling");
                    let mut target_pos = self.find_intersection(ball, geom, ctx.player.id);
                    let mut input = PlayerControlInput::new();
                    target_pos.y = target_pos
                        .y
                        .max(-geom.penalty_area_width / 2.0 - MARGIN)
                        .min(geom.penalty_area_width / 2.0 + MARGIN);
                    input.with_position(target_pos);
                    input.ignore_robots();
                    input.with_yaw(Angle::between_points(
                        ctx.player.position,
                        ball.position.xy(),
                    ));
                    input
                }
                State::Shooting(receiver_id) => {
                    dies_core::debug_string(format!("p{}.waller_mode", ctx.player.id), "shooting");
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
