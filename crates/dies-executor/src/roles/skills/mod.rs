use std::{fmt::format, time::Duration};

use dies_core::{find_intersection, Angle, Vector2};

use crate::PlayerControlInput;

use super::{Skill, SkillCtx, SkillProgress, SkillResult};

const DEFAULT_POS_TOLERANCE: f64 = 70.0;
const DEFAULT_VEL_TOLERANCE: f64 = 10.0;
const DEFAULT_BALL_VEL_TOLERANCE: f64 = 10.0;
const BALL_VEL_CORRECTION: f64 = 0.5;

/// A skill that makes the player go to a specific position
pub struct GoToPosition {
    target_pos: Vector2,
    target_heading: Option<Angle>,
    target_velocity: Vector2,
    pos_tolerance: f64,
    velocity_tolerance: f64,
    with_ball: bool,
}

impl GoToPosition {
    pub fn new(target: Vector2) -> Self {
        Self {
            target_pos: target,
            target_heading: None,
            target_velocity: Vector2::zeros(),
            pos_tolerance: DEFAULT_POS_TOLERANCE,
            velocity_tolerance: DEFAULT_VEL_TOLERANCE,
            with_ball: false,
        }
    }

    pub fn with_heading(mut self, heading: Angle) -> Self {
        self.target_heading = Some(heading);
        self
    }

    /// Drive with the ball.
    ///
    /// This activates the dribbler and makes sure the relative velocity between the
    /// player and the ball is below a certain threshold.
    pub fn with_ball(mut self) -> Self {
        self.with_ball = true;
        self
    }
}

impl Skill for GoToPosition {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let position = ctx.player.position;
        let distance = (self.target_pos - position).norm();
        let dv = (self.target_velocity - ctx.player.velocity).norm();
        if distance < self.pos_tolerance && dv < self.velocity_tolerance {
            return SkillProgress::success();
        }

        let mut input = PlayerControlInput::new();
        input.with_position(self.target_pos);
        if let Some(heading) = self.target_heading {
            input.with_yaw(heading);
        }
        if let (true, Some(ball)) = (self.with_ball, ctx.world.ball.as_ref()) {
            let ball_vel = ball.velocity.xy();
            let relative_velocity = ball_vel - ctx.player.velocity;
            if relative_velocity.norm() > DEFAULT_BALL_VEL_TOLERANCE {
                let correction = relative_velocity * BALL_VEL_CORRECTION;
                input.add_global_velocity(correction);
            }
        }

        SkillProgress::Continue(input)
    }
}

/// A skill that just waits for a certain amount of time
pub struct Wait {
    amount: f64,
    until: Option<f64>,
}

impl Wait {
    /// Creates a new `WaitSkill` that waits for the given amount of time (starting
    /// from the next frame)
    pub fn new(amount: Duration) -> Self {
        Self {
            amount: amount.as_secs_f64(),
            until: None,
        }
    }

    /// Creates a new `WaitSkill` that waits for the given amount of time in seconds
    /// (starting from the next frame)
    pub fn new_secs_f64(amount: f64) -> Self {
        Self {
            amount,
            until: None,
        }
    }
}

impl Skill for Wait {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let until = *self.until.get_or_insert(ctx.world.t_received + self.amount);
        if ctx.world.t_received >= until {
            SkillProgress::success()
        } else {
            SkillProgress::Continue(PlayerControlInput::new())
        }
    }
}

/// A skill that fetches the ball
pub struct FetchBall {
    max_relative_speed: f64,
    initial_ball_direction: Option<Vector2>,
}

impl FetchBall {
    pub fn new() -> Self {
        Self {
            max_relative_speed: 1000.0,
            initial_ball_direction: None,
        }
    }
}

impl Skill for FetchBall {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();
            let ball_pos = ball.position.xy();
            let player_pos = ctx.player.position;
            let inital_ball_direction = self
                .initial_ball_direction
                .get_or_insert((ball_pos - player_pos).normalize());
            let ball_normal = Vector2::new(inital_ball_direction.y, -inital_ball_direction.x);
            let distance = (ball_pos - player_pos).norm();
            if distance < 100.0 {
                return SkillProgress::success();
            }

            let heading = Angle::between_points(player_pos, ball_pos);
            input.with_yaw(heading);
            if ball.velocity.norm() < 1000.0 {
                // If the ball is too slow, just go to the ball
                let target_pos = ball_pos + heading * Vector2::new(-100.0, 0.0);
                dies_core::debug_string(format!("p{}.BallState", ctx.player.id), "STOPPED");
                input.with_position(target_pos);
            } else {
                // If the ball is moving, try to intercept it
                let intersection = find_intersection(
                    player_pos,
                    ball_normal,
                    ball_pos,
                    ball.velocity.xy().normalize(),
                );
                if let Some(intersection) = intersection {
                    dies_core::debug_string(format!("p{}.BallState", ctx.player.id), format!("INTERCEPT, ballvel_norm is: {}", ball.velocity.norm()));

                    input.with_position(intersection);
                } else {
                    dies_core::debug_string(format!("p{}.BallState", ctx.player.id), "cannot intercept, go to ball");

                    input.with_position(ball_pos);
                }
                //return the intersection if it is valid, otherwise return the ball position

                let target_pos = intersection.unwrap_or(ball_pos);
                input.with_position(target_pos);
            }

            let relative_velocity = ball.velocity.xy() - ctx.player.velocity;
            if relative_velocity.norm() > self.max_relative_speed {
                input.add_global_velocity(relative_velocity.normalize() * self.max_relative_speed);
            }

            SkillProgress::Continue(input)
        } else {
            SkillProgress::failure()
        }
    }
}
