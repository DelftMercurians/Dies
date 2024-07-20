use super::RoleCtx;
use dies_core::Angle;
use dies_core::BallData;
use dies_core::FieldGeometry;
use dies_core::Vector2;
use dies_core::WorldData;

use crate::invoke_skill;
use crate::roles::skills::ApproachBall;
use crate::roles::skills::Face;
use crate::roles::skills::FetchBallWithHeading;
use crate::roles::skills::Kick;
use crate::roles::SkillResult;
use crate::skill;
use crate::{roles::Role, PlayerControlInput};

enum State {
    Harassing,
    Shooting,
}

pub struct Harasser {
    state: State,
    distance_behind_ball: f64,
    shooting_target: Vector2,
}

impl Harasser {
    pub fn new(distance_behind_ball: f64) -> Self {
        Self {
            state: State::Harassing,
            distance_behind_ball,
            shooting_target: Vector2::new(4500.0, 0.0),
        }
    }

    fn find_intersection(&self, ball: &BallData, world: &FieldGeometry) -> Vector2 {
        let ball_pos = ball.position.xy();
        let half_length = world.field_length / 2.0;
        let goal_center = Vector2::new(-half_length, 0.0);

        // Compute the direction vector from the ball to the goal center
        let direction = (goal_center - ball_pos).normalize();

        let target_pos = ball_pos + direction * self.distance_behind_ball;

        // Ensuring the target position is infront of the line between the ball and the goal center
        if (target_pos - goal_center).dot(&direction) < 0.0 {
            return target_pos;
        }

        target_pos
    }

    pub fn set_shooting_target(&mut self, target: Vector2) {
        self.shooting_target = target;
    }
}

impl Role for Harasser {
    fn update(&mut self, mut ctx: RoleCtx<'_>) -> PlayerControlInput {
        // if matches!(
        //     ctx.world.current_game_state.game_state,
        //     GameState::Stop | GameState::BallReplacement(_)
        // ) {
        //     ctx.reset_skills();
        //     let mut input = PlayerControlInput::new();
        //     input.with_speed_limit(1300.0);
        //     input.avoid_ball = true;
        //     if let (Some(ball), Some(field)) = (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref()) {
        //         let ball_pos = ball.position.xy();
        //         let dist = (ball_pos - ctx.player.position.xy()).norm();
        //         let min_distance = 600.0;
        //         if dist < min_distance {
        //             // Move away from the ball
        //             // let target =
        //             //     ball_pos.xy() + (ctx.player.position - ball_pos.xy()).normalize() * 650.0;

        //             // use the function to take the field limits into account
        //             let min_theta = -120;
        //             let max_theta = 120;
        //             let max_radius = 1000;
        //             // let field = ctx.world.field_geom.as_ref().unwrap();

        //             let target = dies_core::nearest_safe_pos(ball_pos, min_distance, ctx.player.position.xy(), min_theta, max_theta, max_radius, field);
        //             input.with_position(target);
        //         }
        //     }
        //     return input;
        // }

        match self.state {
            State::Harassing => {
                if let (Some(ball), Some(geom)) =
                    (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref())
                {
                    if ball_dist_to_closest_enemy(ctx.world) > 400.0 && ball.position.x < 0.0 {
                        ctx.reset_skills();
                        self.state = State::Shooting;
                    }

                    let mut target_pos = self.find_intersection(ball, geom);
                    target_pos.x = target_pos.x.min(-500.0);
                    let mut input = PlayerControlInput::new();
                    input.with_position(target_pos);
                    if (ball.position.xy() - ctx.player.position).norm() > 80.0 {
                        input.with_yaw(Angle::between_points(
                            ctx.player.position,
                            ball.position.xy(),
                        ));
                    }
                    input
                } else {
                    PlayerControlInput::new()
                }
            }
            State::Shooting => {
                if ctx.player.position.x > -500.0 {
                    ctx.reset_skills();
                    self.state = State::Harassing;
                    return PlayerControlInput::new();
                }

                skill!(
                    ctx,
                    FetchBallWithHeading::towards_position(self.shooting_target)
                );
                loop {
                    skill!(ctx, ApproachBall::new());
                    match invoke_skill!(
                        ctx,
                        Face::towards_position(self.shooting_target).with_ball()
                    ) {
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
                ctx.reset_skills();
                self.state = State::Harassing;
                PlayerControlInput::new()
            }
        }
    }
}

fn ball_dist_to_closest_enemy(world: &WorldData) -> f64 {
    let ball_pos = world.ball.as_ref().unwrap().position.xy();
    world
        .opp_players
        .iter()
        .map(|p| (p.position - ball_pos).norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(f64::MAX)
}
