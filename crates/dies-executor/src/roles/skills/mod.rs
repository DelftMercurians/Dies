use dies_core::{
    find_intersection, get_tangent_line_direction, perp, which_side_of_robot, Angle, Vector2,
};

use crate::{control::Velocity, roles::SkillResult, PlayerControlInput};

use super::{Skill, SkillCtx, SkillProgress};

const DEFAULT_POS_TOLERANCE: f64 = 70.0;
const DEFAULT_VEL_TOLERANCE: f64 = 30.0;
const DEFAULT_BALL_VEL_TOLERANCE: f64 = 30.0;
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
        if let (true, Some(_)) = (self.with_ball, ctx.world.ball.as_ref()) {
            if !ctx.player.breakbeam_ball_detected {
                return SkillProgress::failure();
            }

            input.with_dribbling(1.0);
            input.with_acceleration_limit(500.0);
            input.with_angular_acceleration_limit(360.0f64.to_radians());

            // let ball_vel = ball.velocity.xy();
            // let relative_velocity = ball_vel - ctx.player.velocity;
            // if relative_velocity.norm() > DEFAULT_BALL_VEL_TOLERANCE {
            //     let correction = relative_velocity * BALL_VEL_CORRECTION;
            //     input.add_global_velocity(correction);
            // }
        }

        SkillProgress::Continue(input)
    }
}

pub struct Kick {
    has_kicked: bool,
}

impl Kick {
    pub fn new() -> Self {
        Self { has_kicked: false }
    }
}

impl Skill for Kick {
    fn update(&mut self, _ctx: SkillCtx<'_>) -> SkillProgress {
        if self.has_kicked {
            return SkillProgress::success();
        }

        let mut input = PlayerControlInput::new();
        input.with_kicker(crate::KickerControlInput::Kick);
        self.has_kicked = true;
        SkillProgress::Continue(input)
    }
}

/// A skill that just waits for a certain amount of time
pub struct Wait {
    amount: f64,
    until: Option<f64>,
}

impl Wait {
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
    dribbling_distance: f64,
    dribbling_speed: f64,
    stop_distance: f64,
    max_relative_speed: f64,
    initial_ball_direction: Option<Vector2>,
    breakbeam_ball_detected: f64,
    last_good_heading: Option<Angle>,
}

impl FetchBall {
    pub fn new() -> Self {
        Self {
            dribbling_distance: 1000.0,
            dribbling_speed: 1.0,
            stop_distance: 200.0,
            max_relative_speed: 1500.0,
            initial_ball_direction: None,
            breakbeam_ball_detected: 0.0,
            last_good_heading: None,
        }
    }
}

impl Skill for FetchBall {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();
            let ball_pos = ball.position.xy();
            let ball_speed = ball.velocity.xy().norm();
            let player_pos = ctx.player.position;
            let inital_ball_direction = self
                .initial_ball_direction
                .get_or_insert((ball_pos - player_pos).normalize());
            let ball_normal = Vector2::new(inital_ball_direction.y, -inital_ball_direction.x);
            let distance = (ball_pos - player_pos).norm();

            let ball_angle = {
                let angle = Angle::between_points(player_pos, ball_pos);
                if distance > 50.0 {
                    self.last_good_heading = Some(angle);
                    angle
                } else {
                    self.last_good_heading.unwrap_or(angle)
                }
            };
            input.with_yaw(ball_angle);

            if ctx.player.breakbeam_ball_detected {
                self.breakbeam_ball_detected += ctx.world.dt;
                if self.breakbeam_ball_detected > 0.2 {
                    return SkillProgress::success();
                }
            } else {
                self.breakbeam_ball_detected = 0.0;
            }

            if ball_speed < 500.0 {
                // If the ball is too slow, just go to the ball
                let target_pos = ball_pos + ball_angle * Vector2::new(-self.stop_distance, 0.0);
                input.with_position(target_pos);
            } else if ball_speed > 0.0 {
                // If the ball is moving, try to intercept it
                let intersection = find_intersection(
                    player_pos,
                    ball_normal,
                    ball_pos,
                    ball.velocity.xy().normalize(),
                );

                // Move to the intersection point if it exists, otherwise go to the ball
                if let Some(intersection) = intersection {
                    input.with_position(intersection);
                } else {
                    input.with_position(ball_pos);
                }
            }

            if distance < self.dribbling_distance {
                input.with_dribbling(self.dribbling_speed);
            }

            // Once we're close enough, use a proptional control to approach the ball
            if distance < self.dribbling_distance && ball_speed < 500.0 {
                input.position = None;
                input.velocity = Velocity::global(
                    distance
                        * (self.max_relative_speed / self.dribbling_distance)
                        * (ball_pos - player_pos).normalize(),
                );
            }

            SkillProgress::Continue(input)
        } else {
            // Wait for the ball to appear
            SkillProgress::Continue(PlayerControlInput::default())
        }
    }
}

pub struct FetchBallWithHeading {
    init_ball_pos: Option<Vector2>,
    target_heading: Angle,
}

impl FetchBallWithHeading {
    pub fn new(target_heading: Angle) -> Self {
        Self {
            init_ball_pos: None,
            target_heading,
        }
    }
}

impl Skill for FetchBallWithHeading {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let player_data = ctx.player;
        let world_data = ctx.world;
        let ball_radius = world_data.field_geom.as_ref().unwrap().ball_radius * 10.0;

        let ball_pos = if let Some(ball) = world_data.ball.as_ref() {
            ball.position.xy()
        } else {
            return SkillProgress::Continue(PlayerControlInput::default());
        };
        let init_ball_pos = self.init_ball_pos.get_or_insert(ball_pos);

        let target_pos = *init_ball_pos - Angle::to_vector(&self.target_heading) * ball_radius;

        if (player_data.position - target_pos).norm() < 100.0
            && (player_data.yaw - self.target_heading).abs()
                < ctx.world.player_model.dribbler_angle.radians() / 2.0
        {
            return SkillProgress::Done(SkillResult::Success);
        }
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();
            let ball_distance = (ball.position.xy() - *init_ball_pos).norm();
            if ball_distance >= 100.0 {
                //ball movement
                return SkillProgress::Done(SkillResult::Failure);
            }

            let direct_target =
                which_side_of_robot(self.target_heading, target_pos, player_data.position);
            if direct_target {
                input
                    .with_position(target_pos)
                    .with_yaw(self.target_heading);
                SkillProgress::Continue(input)
            } else {
                let (dir_a, dir_b) = get_tangent_line_direction(
                    ball.position.xy(),
                    ball_radius - 20.0,
                    player_data.position,
                );
                let target_pos_a = find_intersection(
                    player_data.position,
                    Angle::to_vector(&dir_a),
                    target_pos,
                    perp(self.target_heading.to_vector()),
                );
                let target_pos_b = find_intersection(
                    player_data.position,
                    Angle::to_vector(&dir_b),
                    target_pos,
                    perp(self.target_heading.to_vector()),
                );

                // pick the nearest point as the target
                let mut indir_target_pos = if (target_pos_a.unwrap() - *init_ball_pos).norm()
                    < (target_pos_b.unwrap() - *init_ball_pos).norm()
                {
                    target_pos_a.unwrap()
                } else {
                    target_pos_b.unwrap()
                };

                if indir_target_pos.x.is_nan() || indir_target_pos.y.is_nan() {
                    let dir_c = perp(player_data.position - ball.position.xy());
                    indir_target_pos = find_intersection(
                        player_data.position,
                        dir_c,
                        target_pos,
                        perp(self.target_heading.to_vector()),
                    )
                    .unwrap();
                }
                input
                    .with_position(indir_target_pos)
                    .with_yaw(self.target_heading);
                SkillProgress::Continue(input)
            }
        } else {
            SkillProgress::Continue(PlayerControlInput::default())
        }
    }
}
