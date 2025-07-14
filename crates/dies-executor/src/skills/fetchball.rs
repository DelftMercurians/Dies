use dies_core::{Angle, Vector2, BALL_RADIUS, PLAYER_RADIUS};

use super::{SkillCtx, SkillProgress};
use crate::{control::Velocity, PlayerControlInput};

/// A skill that fetches the ball
#[derive(Clone)]
pub struct FetchBall {
    dribbling_distance: f64,
    dribbling_speed: f64,
    stop_distance: f64,
    max_relative_speed: f64,
    last_good_heading: Option<Angle>,
    starting_position: Option<Vector2>,
}

impl FetchBall {
    pub fn new() -> Self {
        Self {
            dribbling_distance: 1000.0,
            dribbling_speed: 1.0,
            stop_distance: PLAYER_RADIUS + BALL_RADIUS + 50.0,
            max_relative_speed: 1500.0,
            last_good_heading: None,
            starting_position: None,
        }
    }
}

impl FetchBall {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();
            input.with_dribbling(self.dribbling_speed);

            let ball_pos = ball.position.xy();
            let ball_speed = ball.velocity.xy().norm();
            let player_pos = ctx.player.position;
            let distance = 80.0 + (ball_pos - player_pos).norm();

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

            dies_core::debug_string(
                format!("fetchball_state_{}", ctx.player.id),
                format!(
                    "bb: {}; dist: {:.1}; speed: {:.1}",
                    ctx.player.breakbeam_ball_detected, distance, ball_speed
                ),
            );
            if ctx.player.breakbeam_ball_detected && distance < 400.0 {
                dies_core::debug_string("fetchball_state", "done");
                return SkillProgress::success();
            }

            if ball_speed < 250.0 {
                // Ball is stationary/slow, so we move to get it
                if distance > self.stop_distance {
                    // move close to the ball
                    let target_pos = ball_pos + ball_angle * Vector2::new(-self.stop_distance, 0.0);
                    input.with_position(target_pos);
                } else if (!ball.detected && distance < 300.0)
                    || (distance < self.dribbling_distance && ball_speed < 500.0)
                {
                    let start_pos = self.starting_position.unwrap_or(player_pos);
                    let distance = (player_pos - start_pos).norm();
                    if distance > self.stop_distance * 1.2 {
                        // if we moved too far we probably got stuck, so fail
                        return SkillProgress::failure();
                    }
                    input.velocity = Velocity::global(
                        distance
                            * (self.max_relative_speed / self.dribbling_distance)
                            * (ball_pos - player_pos).normalize(),
                    );
                }
            } else {
                // if ball is fast and we are far away
                // sample bunch of points on the ball ray, and see which 'segment'
                // we are capable to reach in time. Then go to this segment.
                // time to reach segment is simple: distance / normal speed - small discount
                // time for ball to reach segment is technically the same formula, wow

                // schedule of points: from 0 seconds to 2 seconds
                let points_schedule = [
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

                let mut intersection = ball_points[ball_points.len() - 1];
                for i in 0..ball_points.len() - 1 {
                    let a = ball_points[i];
                    let b = ball_points[i + 1];
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
