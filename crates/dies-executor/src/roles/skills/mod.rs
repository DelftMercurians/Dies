use std::time::Instant;

use dies_core::{
    find_intersection, get_tangent_line_direction, perp, which_side_of_robot, Angle, PlayerId,
    SysStatus, Vector2,
};

use crate::{control::Velocity, roles::SkillResult, KickerControlInput, PlayerControlInput};

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
    avoid_ball: bool,
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
            avoid_ball: false,
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

    pub fn avoid_ball(mut self) -> Self {
        self.avoid_ball = true;
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

        if let Some(_) = ctx.world.ball.as_ref() {
            if self.avoid_ball {
                input.avoid_ball = true;
            } else if self.with_ball {
                if !ctx.player.breakbeam_ball_detected {
                    return SkillProgress::failure();
                }

                input.with_dribbling(1.0);
                input.with_acceleration_limit(700.0);
                input.with_angular_acceleration_limit(180.0f64.to_radians());
                input.with_angular_speed_limit(180.0f64.to_radians());
            }

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

pub struct Face {
    heading: HeadingTarget,
    with_ball: bool,
}

impl Face {
    pub fn new(heading: Angle) -> Self {
        Self {
            with_ball: false,
            heading: HeadingTarget::Angle(heading),
        }
    }

    pub fn towards_position(pos: Vector2) -> Self {
        Self {
            with_ball: false,
            heading: HeadingTarget::Position(pos),
        }
    }

    pub fn towards_own_player(id: PlayerId) -> Self {
        Self {
            with_ball: false,
            heading: HeadingTarget::OwnPlayer(id),
        }
    }

    pub fn towards_ball() -> Self {
        Self {
            with_ball: false,
            heading: HeadingTarget::Ball,
        }
    }

    pub fn with_ball(mut self) -> Self {
        self.with_ball = true;
        self
    }
}

impl Skill for Face {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let mut input = PlayerControlInput::new();
        if let Some(ball) = ctx.world.ball.as_ref() {
            let balldist = (ball.position.xy() - ctx.player.position).magnitude();
            if self.with_ball && balldist > 250.0 {
                return SkillProgress::failure();
            }
        }
        let heading = if let Some(heading) = self.heading.heading(&ctx) {
            heading
        } else {
            return SkillProgress::failure();
        };

        input.with_yaw(heading);
        input.with_care(0.7); // this compensates for flakiness of with_ball
        if self.with_ball {
            input.with_dribbling(1.0);
            input.with_angular_acceleration_limit(180.0f64.to_radians());
            input.with_angular_speed_limit(180.0f64.to_radians());
        }

        if (ctx.player.yaw - heading).abs() < 3f64.to_radians() {
            return SkillProgress::success();
        }
        SkillProgress::Continue(input)
    }
}

pub struct Kick {
    has_kicked: usize,
    timer: Option<Instant>,
}

impl Kick {
    pub fn new() -> Self {
        Self {
            has_kicked: 1,
            timer: None,
        }
    }
}

impl Skill for Kick {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let mut input = PlayerControlInput::new();
        input.with_dribbling(1.0);

        if self.has_kicked == 0 {
            let timer = self.timer.get_or_insert(Instant::now());
            if timer.elapsed().as_secs_f64() > 0.1 {
                if ctx
                    .world
                    .ball
                    .as_ref()
                    .map(|b| (b.position.xy() - ctx.player.position).norm() > 200.0)
                    .unwrap_or(false)
                {
                    return SkillProgress::success();
                } else {
                    return SkillProgress::failure();
                }
            } else {
                return SkillProgress::Continue(input);
            }
        }

        let ready = match ctx.player.kicker_status.as_ref() {
            Some(SysStatus::Ready) => true,
            _ => false,
        };

        if ready {
            println!("Kicking");
            if !ctx.player.breakbeam_ball_detected {
                return SkillProgress::failure();
            }
            input.with_kicker(crate::KickerControlInput::Kick);
            self.has_kicked -= 1;
            SkillProgress::Continue(input)
        } else {
            input.kicker = KickerControlInput::Arm;
            SkillProgress::Continue(input)
        }
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
            breakbeam_ball_detected: 0.0,
            last_good_heading: None,
        }
    }
}

impl Skill for FetchBall {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();
            input.with_dribbling(self.dribbling_speed);

            let ball_pos = ball.position.xy();
            let ball_speed = ball.velocity.xy().norm();
            let player_pos = ctx.player.position;
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

            if ctx.player.breakbeam_ball_detected && distance < 200.0 {
                return SkillProgress::success();
            }

            if (!ball.detected && distance < 300.0)
                || (distance < self.dribbling_distance && ball_speed < 500.0)
            {
                input.velocity = Velocity::global(
                    distance
                        * (self.max_relative_speed / self.dribbling_distance)
                        * (ball_pos - player_pos).normalize(),
                );
            } else if ball_speed < 250.0 {
                // If the ball is too slow, just go to the ball
                let target_pos = ball_pos + ball_angle * Vector2::new(-self.stop_distance, 0.0);
                input.with_position(target_pos);
            } else {
                // if ball is fast and we are far away
                // sample bunch of points on the ball ray, and see which 'segment'
                // we are capable to reach in time. Then go to this segment.
                // time to reach segment is simple: distance / normal speed - small discount
                // time for ball to reach segment is technically the same formula, wow

                // schedule of points: from 0 seconds to 2 seconds
                let points_schedule = vec![
                    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0,
                    1.2, 1.5, 2.0,
                ];

                let friction_factor = 3.0;

                let ball_points: Vec<Vector2> = points_schedule
                    .iter()
                    .map(|t| {
                        ball_pos
                            + ball.velocity.xy()
                                * (*t)
                                * (1.0 - f64::min(1.0, (*t) / friction_factor))
                    })
                    .collect();

                let mut intersection = ball_points[ball_points.len() - 1].clone();
                for i in 0..ball_points.len() - 1 {
                    let a = ball_points[i].clone();
                    let b = ball_points[i + 1].clone();
                    let must_be_reached_before = points_schedule[i];
                    // now we have both segment points available, lets compute time_to_reach
                    let mut time_to_reach = f64::min(
                        ctx.world.time_to_reach_point(ctx.player, a),
                        ctx.world.time_to_reach_point(ctx.player, b),
                    );
                    // but this time is also kinda useless, mostly, so add 0.3 seconds
                    // to compensate
                    time_to_reach = time_to_reach * 1.2 + 0.1;

                    if time_to_reach < must_be_reached_before {
                        intersection = b;
                        break;
                    }
                }
                input.with_position(intersection);

                if distance < self.dribbling_distance {
                    input.with_dribbling(self.dribbling_speed);
                }

                // Once we're close enough, use a proptional control to approach the ball
                if distance < self.dribbling_distance {
                    input.position = None;
                    // velocity is ball velocity + control
                    input.velocity = Velocity::global(
                        (ball_speed * 0.8
                            + distance * (self.max_relative_speed / self.dribbling_distance))
                            * (ball_pos - player_pos).normalize(),
                    );
                }
            }

            SkillProgress::Continue(input)
        } else {
            // Wait for the ball to appear
            SkillProgress::Continue(PlayerControlInput::default())
        }
    }
}

pub struct InterceptBall {
    intercept_line: Option<(Vector2, Vector2)>,
}

impl InterceptBall {
    pub fn new() -> Self {
        Self {
            intercept_line: None,
        }
    }
}

impl Skill for InterceptBall {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let intercept_line = self
                .intercept_line
                .get_or_insert((ctx.player.position, perp(ball.velocity.xy())));

            if let Some(intersection) = find_intersection(
                intercept_line.0,
                intercept_line.1,
                ball.position.xy(),
                ball.velocity.xy(),
            ) {
                let mut input = PlayerControlInput::new();

                input.with_position(intersection);
                input.with_yaw(Angle::between_points(
                    ctx.player.position,
                    ball.position.xy(),
                ));

                SkillProgress::Continue(input)
            } else {
                SkillProgress::failure()
            }
        } else {
            SkillProgress::failure()
        }
    }
}

pub struct ApproachBall {}

impl ApproachBall {
    pub fn new() -> Self {
        Self {}
    }
}

impl Skill for ApproachBall {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let ball_pos = ball.position.xy();
            let player_pos = ctx.player.position;

            if ctx.player.breakbeam_ball_detected && (ball_pos - player_pos).norm() < 150.0 {
                return SkillProgress::success();
            }

            let ball_speed = ball.velocity.xy().norm();
            let distance = (ball_pos - player_pos).norm();

            let mut input = PlayerControlInput::new();
            if (ball_pos - player_pos).norm() > 50.0 && ball.detected {
                input.with_yaw(Angle::between_points(player_pos, ball_pos));
            }

            let max_relative_speed = 800.0;
            let dribbling_distance = 1000.0;

            input.velocity = Velocity::global(
                (ball_speed * 0.8 + distance * (max_relative_speed / dribbling_distance))
                    * (ball_pos - player_pos).normalize(),
            );
            input.with_dribbling(1.0);

            SkillProgress::Continue(input)
        } else {
            SkillProgress::failure()
        }
    }
}

enum HeadingTarget {
    Angle(Angle),
    Ball,
    Position(Vector2),
    OwnPlayer(PlayerId),
}

impl HeadingTarget {
    fn heading(&self, ctx: &SkillCtx) -> Option<Angle> {
        match self {
            HeadingTarget::Angle(angle) => Some(*angle),
            HeadingTarget::Ball => {
                if let Some(ball) = ctx.world.ball.as_ref() {
                    Some(Angle::between_points(
                        ctx.player.position,
                        ball.position.xy(),
                    ))
                } else {
                    None
                }
            }
            HeadingTarget::Position(pos) => Some(Angle::between_points(ctx.player.position, *pos)),
            HeadingTarget::OwnPlayer(id) => {
                if let Some(player) = ctx.world.get_player(*id) {
                    Some(Angle::between_points(ctx.player.position, player.position))
                } else {
                    None
                }
            }
        }
    }
}

pub struct FetchBallWithHeading {
    init_ball_pos: Option<Vector2>,
    target_heading: HeadingTarget,
}

impl FetchBallWithHeading {
    pub fn new(target_heading: Angle) -> Self {
        Self {
            init_ball_pos: None,
            target_heading: HeadingTarget::Angle(target_heading),
        }
    }

    pub fn towards_position(pos: Vector2) -> Self {
        Self {
            init_ball_pos: None,
            target_heading: HeadingTarget::Position(pos),
        }
    }

    pub fn towards_own_player(id: PlayerId) -> Self {
        Self {
            init_ball_pos: None,
            target_heading: HeadingTarget::OwnPlayer(id),
        }
    }
}

impl Skill for FetchBallWithHeading {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let player_data = ctx.player;
        let world_data = ctx.world;
        let ball_radius = world_data.field_geom.as_ref().unwrap().ball_radius * 8.0;

        let ball_pos = if let Some(ball) = world_data.ball.as_ref() {
            ball.position.xy()
        } else {
            return SkillProgress::Continue(PlayerControlInput::default());
        };
        let init_ball_pos = self.init_ball_pos.get_or_insert(ball_pos);
        let init_ball_pos = *init_ball_pos;

        let target_heading = if let Some(heading) = self.target_heading.heading(&ctx) {
            // println!("Heading: {:?}", heading);
            heading
        } else {
            return SkillProgress::failure();
        };

        let target_pos = init_ball_pos - Angle::to_vector(&target_heading) * ball_radius;

        if (player_data.position - target_pos).norm() < 100.0
            && (player_data.yaw - target_heading).abs()
                < 0.2 // this threshold should be changed only with real robots in sight
        {
            return SkillProgress::Done(SkillResult::Success);
        }
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();
            let ball_distance = (ball.position.xy() - init_ball_pos).norm();
            if ball_distance >= 120.0 {
                //ball movement
                return SkillProgress::Done(SkillResult::Failure);
            }

            input.with_position(target_pos).with_yaw(target_heading);
            input.avoid_ball = true;
            SkillProgress::Continue(input)
        } else {
            SkillProgress::Continue(PlayerControlInput::default())
        }
    }
}
