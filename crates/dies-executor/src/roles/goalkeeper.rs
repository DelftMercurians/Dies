use dies_core::{find_intersection, score_line_of_sight, Angle, PlayerId, Vector2, WorldData};

use crate::{
    invoke_skill,
    roles::{
        skills::{ApproachBall, Face, FetchBallWithHeading, Kick},
        Role, RoleCtx, SkillResult,
    },
    skill, PlayerControlInput,
};

const KEEPER_X_OFFSET: f64 = 250.0;

pub struct Goalkeeper {
    pub kicking_to: Option<PlayerId>,
    pub defenders: Vec<PlayerId>,
    pub penalty: bool,
}

impl Goalkeeper {
    pub fn new() -> Self {
        Self {
            kicking_to: None,
            defenders: Vec::new(),
            penalty: false,
        }
    }

    pub fn set_defenders(&mut self, defenders: Vec<PlayerId>) {
        self.defenders = defenders;
    }

    pub fn with_penalty(mut self) -> Self {
        self.penalty = true;
        self
    }
}

impl Role for Goalkeeper {
    fn update(&mut self, mut ctx: RoleCtx<'_>) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();
        if let (Some(ball), Some(field_geom)) =
            (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref())
        {
            let defence_width = 1.3 * (field_geom.goal_width / 2.0);
            let ball_pos = ball.position.xy();
            let mut goalkeeper_x = -field_geom.field_length / 2.0 + KEEPER_X_OFFSET;
            let fallback_y = ball_pos.y.min(defence_width).max(-defence_width);

            // if matches!(
            //     ctx.world.current_game_state.game_state,
            //     GameState::Stop | GameState::BallReplacement(_)
            // ) {
            //     ctx.reset_skills();
            //     let mut input = PlayerControlInput::new();
            //     input.with_speed_limit(1300.0);

            //     if let Some(ball) = ctx.world.ball.as_ref() {
            //         let ball_pos = ball.position.xy();
            //         let dist = (ball_pos - ctx.player.position.xy()).norm();
            //         if dist < 560.0 {
            //             // Move away from the ball
            //             let mut target = ball_pos.xy()
            //                 + (ctx.player.position - ball_pos.xy()).normalize() * 650.0;
            //             let field = field_geom;
            //             if !is_pos_in_field(target, field) {
            //                 let min_distance = 650.0;
            //                 let min_theta = -120;
            //                 let max_theta = 120;
            //                 let max_radius = 1000;

            //                 target = dies_core::nearest_safe_pos(
            //                     ball_pos,
            //                     min_distance,
            //                     ctx.player.position.xy(),
            //                     min_theta,
            //                     max_theta,
            //                     max_radius,
            //                     field,
            //                 );
            //             }
            //             input.with_position(target);
            //         }
            //     }

            //     let mut target = if ball_pos.x < 0.0 && ball.velocity.x < 0.0 {
            //         let mut out = find_intersection(
            //             Vector2::new(goalkeeper_x, 0.0),
            //             Vector2::y(),
            //             ball_pos,
            //             ball.velocity.xy(),
            //         )
            //         .unwrap_or(Vector2::new(goalkeeper_x, ball_pos.y));
            //         // limit the delta by how much the ball is gonna move in a second or so
            //         let limit = ball.velocity.y.abs() * 2.0;
            //         out.y = out
            //             .y
            //             .min(ball.position.y + limit)
            //             .max(ball.position.y - limit);
            //         out
            //     } else {
            //         Vector2::new(goalkeeper_x, fallback_y)
            //     };
            //     target.y = target.y.max(-defence_width).min(defence_width);
            //     // we want to bring the point closer to center based on the ball x position
            //     let factor = 1.0 - f64::min(f64::max((ball_pos.x + 2000.0) / 5000.0, 0.0), 1.0);
            //     target.y = target.y * factor;
            //     input.with_position(target);

            //     return input;
            // }

            let ball_pos = ball.position.xy();
            let ball_in_defence = ball_pos.x
                < (-field_geom.field_length / 2.0 + field_geom.penalty_area_depth)
                && ball_pos.y.abs() < field_geom.penalty_area_width / 2.0;
            let ball_vel = ball.velocity.xy().norm();

            if self.kicking_to.is_none() && ball_in_defence && ball_vel < 100.0 {
                self.kicking_to = Some(
                    find_best_angle(
                        ctx.player.id,
                        ctx.player.position,
                        ctx.world,
                        &self.defenders,
                    )
                    .unwrap_or(ctx.world.own_players[0].id),
                );
                ctx.reset_skills();
            } else if self.kicking_to.is_some() && !ball_in_defence {
                self.kicking_to = None;
                ctx.reset_skills();
            }

            if self.kicking_to.is_none() {
                let ball_pos = ball.position.xy() + ball.velocity.xy() * 0.3;
                let delta = f64::max(f64::min(ctx.player.position.y.abs() / 5.0, 150.0), 0.0);
                goalkeeper_x = if !self.penalty {
                    goalkeeper_x - delta
                } else {
                    -field_geom.field_length / 2.0
                };

                let target_angle = Angle::between_points(ctx.player.position, ball_pos);
                input.with_yaw(target_angle);

                let defence_width = 1.3 * (field_geom.goal_width / 2.0);
                let mut target = if ball_pos.x < 0.0 && ball.velocity.x < 0.0 {
                    let mut out = find_intersection(
                        Vector2::new(goalkeeper_x, 0.0),
                        Vector2::y(),
                        ball_pos,
                        ball.velocity.xy(),
                    )
                    .unwrap_or(Vector2::new(goalkeeper_x, ball_pos.y));
                    // limit the delta by how much the ball is gonna move in a second or so
                    let limit = ball.velocity.y.abs() * 2.0;
                    out.y = out
                        .y
                        .min(ball.position.y + limit)
                        .max(ball.position.y - limit);
                    out
                } else {
                    Vector2::new(goalkeeper_x, fallback_y)
                };
                target.y = target.y.max(-defence_width).min(defence_width);
                // we want to bring the point closer to center based on the ball x position
                let factor = 1.0 - f64::min(f64::max((ball_pos.x + 2000.0) / 5000.0, 0.0), 1.0);
                target.y *= factor;
                input.with_position(target);
            } else if let Some(kicking_to) = self.kicking_to {
                skill!(ctx, FetchBallWithHeading::towards_own_player(kicking_to));
                loop {
                    skill!(ctx, ApproachBall::new());
                    match invoke_skill!(ctx, Face::towards_own_player(kicking_to).with_ball()) {
                        crate::roles::SkillProgress::Continue(mut input) => {
                            input.with_dribbling(1.0);
                            return input;
                        }
                        _ => {}
                    }
                    if let SkillResult::Success = skill!(ctx, Kick::new()) {
                        break;
                    }
                }
            }
        }

        input
    }

    fn role_type(&self) -> dies_core::RoleType {
        dies_core::RoleType::Goalkeeper
    }
}

fn find_best_angle(
    id: PlayerId,
    starting_pos: Vector2,
    world: &WorldData,
    exclude: &Vec<PlayerId>,
) -> Option<PlayerId> {
    let targets = world
        .own_players
        .iter()
        .filter(|p| p.position.x < 0.0 && p.id != id && !exclude.contains(&p.id))
        .collect::<Vec<_>>();

    let best_target = targets
        .iter()
        .map(|p| (p.id, score_line_of_sight(world, starting_pos, p.position)))
        .max_by_key(|&x| x.1 as i64);

    if let Some((striker_id, _)) = best_target {
        Some(striker_id)
    } else {
        None
    }
}
