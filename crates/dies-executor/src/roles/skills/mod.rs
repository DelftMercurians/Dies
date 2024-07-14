use std::{ time::Duration};

use dies_core::{find_intersection, Angle, Vector2, which_side_of_robot, get_tangent_line_direction, perp};

use crate::{control::Velocity, PlayerControlInput};

use super::{Skill, SkillCtx, SkillProgress, SkillResult};

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
    dribbling_distance: f64,
    dribbling_speed: f64,
    stop_distance: f64,
    max_relative_speed: f64,
    initial_ball_direction: Option<Vector2>,
    last_good_heading: Option<Angle>,
    initial_ball_velocity: Option<Angle>,
}

impl FetchBall {
    pub fn new() -> Self {
        Self {
            dribbling_distance: 1000.0,
            dribbling_speed: 1.0,
            stop_distance: 200.0,
            max_relative_speed: 1000.0,
            initial_ball_direction: None,
            last_good_heading: None,
            initial_ball_velocity: None,
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
            let initial_ball_velocity = self.initial_ball_velocity.or_else(|| {
                if ball.velocity.norm() > 10.0 {
                    Some(Angle::from_vector(ball.velocity.xy()))
                } else {
                    None
                }
            });
            let ball_normal = Vector2::new(inital_ball_direction.y, -inital_ball_direction.x);
            let distance = (ball_pos - player_pos).norm();

            if ctx.player.breakbeam_ball_detected {
                return SkillProgress::success();
            }

            // let ball_angle = {
            //     let angle: Angle = Angle::between_points(player_pos, ball_pos);
            //     if distance > 50.0 {
            //         self.last_good_heading = Some(angle);
            //         angle
            //     } else {
            //         self.last_good_heading.unwrap_or(angle)
            //     }
            // };
            let ball_angle = Angle::between_points(player_pos, ball_pos);
            input.with_yaw(ball_angle);

            if ball_speed < 100.0 {
                // If the ball is too slow, just go to the ball
                let target_pos = ball_pos + ball_angle * Vector2::new(-self.stop_distance, 0.0);
                dies_core::debug_string(format!("p{}.BallState", ctx.player.id), "STOPPED");
                input.with_position(target_pos);
            } else if ball_speed > 0.0 {
                // If the ball is moving, try to intercept it
                let intersection = find_intersection(
                    player_pos,
                    ball_normal,
                    ball_pos,
                    ball.velocity.xy().normalize(),
                );

                //return the intersection if it is valid, otherwise return the ball position
                if let Some(intersection) = intersection {
                    dies_core::debug_cross(
                        format!("p{}.BallIntersection", ctx.player.id),
                        intersection,
                        dies_core::DebugColor::Purple,
                    );
                    dies_core::debug_string(
                        format!("p{}.BallState", ctx.player.id),
                        format!("INTERCEPT, ballvel_norm is: {}", ball.velocity.norm()),
                    );

                    input.with_position(intersection);
                } else {
                    dies_core::debug_string(
                        format!("p{}.BallState", ctx.player.id),
                        "cannot intercept, go to ball",
                    );

                    input.with_position(ball_pos);
                }
                // input.with_yaw(
                //     initial_ball_velocity
                //         .unwrap_or(Angle::from_vector(inital_ball_direction.clone())),
                // );
            }

            // if we're close enough, use a PID to override the velocity
            // commands of the position controller

            // this is meant to avoid touching the ball at high speeds and losing control
            // maybe this issue is not present in real life, it happens in the sim
            if distance < self.dribbling_distance && ball_speed < 200.0 {
                input.with_dribbling(self.dribbling_speed);
                // override the velocity inverse to the distance
                // at distance 0, the velocity is 0
                // at distance dribbling_distance, the velocity max_relative_speed
                // dies_core::debug_string(
                //     format!("p{}.BallState", ctx.player.id),
                //     "overriding velocity",
                // );

                input.position = None;
                input.velocity = Velocity::global(
                    distance
                        * (self.max_relative_speed / self.dribbling_distance)
                        * (ball_pos - player_pos).normalize(),
                );
            }

            SkillProgress::Continue(input)
        } else {
            // wait for the ball to appear
            SkillProgress::Continue(PlayerControlInput::default())
        }
    }
}


pub struct FetchBallWithHeading {
    init_ball_pos: Vector2,
    target_heading: Angle,
}

impl FetchBallWithHeading {
    pub fn new(init_ball_pos: Vector2, target_heading: Angle) -> Self {
        Self {
            init_ball_pos,
            target_heading,
        }
    }
}

impl Skill for FetchBallWithHeading {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let player_data = ctx.player;
        log::info!("player_data.position: {:?}", player_data.position);
        let world_data = ctx.world;
        let ball_radius = world_data.field_geom.as_ref().unwrap().ball_radius * 10.0;
        let target_pos:Vector2 = self.init_ball_pos - Angle::to_vector(&self.target_heading) * ball_radius;
        dies_core::debug_value("p0.target_pos_x", target_pos.x);
        dies_core::debug_value("p0.target_pos_y", target_pos.y);
        if (player_data.position - target_pos).norm() < 100.0 && (player_data.yaw - self.target_heading).abs() < 0.1{
            return SkillProgress::Done(SkillResult::Success);
        }
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();
            let ball_distance = (ball.position.xy() - self.init_ball_pos).norm();
            if(ball_distance >= 100.0) {   //ball movement
                return SkillProgress::Done(SkillResult::Failure);
            }
            
            let direct_target = which_side_of_robot(self.target_heading, target_pos, player_data.position);
            if direct_target == true {
                println!("direct target");
                dies_core::debug_value("p0.target_pos_x", target_pos.x);
                dies_core::debug_value("p0.target_pos_y", target_pos.y);
                input.with_position(target_pos).with_yaw(self.target_heading);
                return SkillProgress::Continue(input);
            }
            else {
                println!("indirect target");
                let (dirA, dirB) = get_tangent_line_direction(  ball.position.xy(), ball_radius-20.0, player_data.position);
                let target_posA = find_intersection(player_data.position, Angle::to_vector(&dirA), target_pos, perp(self.target_heading.to_vector()));
                let target_posB = find_intersection(player_data.position, Angle::to_vector(&dirB), target_pos, perp(self.target_heading.to_vector()));
                log::info!("target_posA: {:?}", target_posA);
                log::info!("target_posB: {:?}", target_posB);
                // pick the nearest point as the target
                let mut indir_target_pos = if (target_posA.unwrap() - self.init_ball_pos).norm() < (target_posB.unwrap() -self.init_ball_pos).norm() {
                    target_posA.unwrap()
                } else {
                    target_posB.unwrap()
                };
                
                if indir_target_pos.x.is_nan() || indir_target_pos.y.is_nan() {
                    println!("fixing ball inside circle");
                    let dirC = perp(player_data.position - ball.position.xy());
                    indir_target_pos = find_intersection(player_data.position, dirC, target_pos, perp(self.target_heading.to_vector())).unwrap();
                }
                dies_core::debug_value("p0.target_pos_x", indir_target_pos.x);
                dies_core::debug_value("p0.target_pos_y", indir_target_pos.y);
                input.with_position(indir_target_pos).with_yaw(self.target_heading);
                return SkillProgress::Continue(input);
            }

        } else {
            SkillProgress::Continue(PlayerControlInput::default())
        }
    }
}