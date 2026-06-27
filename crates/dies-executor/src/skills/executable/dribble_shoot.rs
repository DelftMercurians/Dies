//! DribbleShoot skill - aim by orbiting the ball, then kick.
//!
//! Assumes the ball is already captured. Slides the robot tangentially around
//! the ball (keeping the dribbler pressed against it) until the shoot axis lines
//! up with `target_heading`, then kicks.

use std::time::{Duration, Instant};

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{KickerControlInput, PlayerControlInput, Velocity};

const BALL_TO_ROBOT_DISTANCE: f64 = 111.0;
const ORBIT_SPEED: f64 = 800.0;
const ORBIT_GAIN: f64 = 600.0;
const MIN_ORBIT_SPEED: f64 = 40.0;
const CREEP_IN: f64 = 0.0;
const YAW_TOLERANCE: f64 = 5.0 * std::f64::consts::PI / 180.0;
const DRIBBLER_SPEED: f64 = 0.6;
const KICK_SPEED: f64 = 4000.0;
const TIMEOUT: Duration = Duration::from_secs(4);
const LANE_HALF_WIDTH: f64 = 120.0;
const LANE_RANGE: f64 = 3000.0;

enum AimState {
    Aiming,
    Kicking,
    Kicked,
}

pub struct DribbleShootSkill {
    target_heading: Angle,
    status: SkillStatus,
    state: AimState,
    start: Option<Instant>,
}

impl DribbleShootSkill {
    pub fn new(target_heading: Angle) -> Self {
        Self {
            target_heading,
            status: SkillStatus::Running,
            state: AimState::Aiming,
            start: None,
        }
    }
}

impl ExecutableSkill for DribbleShootSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::DribbleShoot { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::DribbleShoot { target_heading } = command {
            self.target_heading = *target_heading;
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        let Some(ball) = ctx.world.ball.as_ref() else {
            self.status = SkillStatus::Failed;
            return SkillProgress::failure();
        };
        // if !ctx.player.has_ball {
        //     log::warn!("dribble_shoot: lost ball");
        //     self.status = SkillStatus::Failed;
        //     return SkillProgress::failure();
        // }

        let start = *self.start.get_or_insert_with(Instant::now);
        if start.elapsed() > TIMEOUT {
            log::warn!("dribble_shoot: timeout");
            self.status = SkillStatus::Failed;
            return SkillProgress::failure();
        }

        let ball_pos = ball.position.xy();
        let player_pos = ctx.player.position;
        if lane_blocked(&ctx, ball_pos, self.target_heading) {
            log::warn!("dribble_shoot: lane blocked");
            self.status = SkillStatus::Failed;
            return SkillProgress::failure();
        }

        let err = self.target_heading - ctx.player.yaw;
        log::debug!("dribble_shoot tick: err={:.1}deg", err.degrees());
        let mut input = PlayerControlInput::new();
        input.avoid_robots = false;
        input.with_dribbling(DRIBBLER_SPEED);
        input.with_angular_speed_limit(1000.0);

        match self.state {
            AimState::Aiming => {
                if err.abs() < YAW_TOLERANCE {
                    self.state = AimState::Kicking;
                }

                const RADIUS_KP: f64 = 2.0;
                let speed = (ORBIT_GAIN * err.abs()).clamp(MIN_ORBIT_SPEED, ORBIT_SPEED);
                let r = player_pos - ball_pos;
                let r_hat = r.normalize();
                let tangent = Vector2::new(-r_hat.y, r_hat.x); // CCW
                let v_tan = err.signum() * speed * tangent; // sign to confirm
                let v_rad = -RADIUS_KP * (r.norm() - BALL_TO_ROBOT_DISTANCE) * r_hat;
                input.velocity = Velocity::global(v_tan + v_rad);
                input.with_yaw(Angle::from_vector(-r)); // face ball

                // let lateral = -err.signum() * speed;
                // input.velocity = Velocity::local(Vector2::new(CREEP_IN, lateral));
                // input.angular_velocity = Some(-lateral / BALL_TO_ROBOT_DISTANCE);
                // input.with_yaw(Angle::from_vector(ball_pos - player_pos));
            }
            AimState::Kicking => {
                input.with_yaw(self.target_heading);
                input.with_kicker(KickerControlInput::Kick);
                input.kick_speed = Some(KICK_SPEED);
                input.with_dribbling(0.0);
                self.state = AimState::Kicked;
            }
            AimState::Kicked => {
                self.status = SkillStatus::Succeeded;
                return SkillProgress::success();
            }
        }

        self.status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "DribbleShoot"
    }

    fn description(&self) -> String {
        let phase = match self.state {
            AimState::Aiming => "orbiting to aim",
            AimState::Kicking => "kicking",
            AimState::Kicked => "kicked",
        };
        format!("{phase} @ {:.0}°", self.target_heading.degrees())
    }
}

/// Whether another robot sits in the shoot corridor (ray from the ball along
/// `heading`, half-width `LANE_HALF_WIDTH`, length `LANE_RANGE`).
fn lane_blocked(ctx: &SkillContext<'_>, ball_pos: Vector2, heading: Angle) -> bool {
    let dir = heading.to_vector();
    ctx.world
        .own_players
        .iter()
        .filter(|p| p.id != ctx.player.id)
        .chain(ctx.world.opp_players.iter())
        .any(|p| {
            let rel = p.position - ball_pos;
            let proj = rel.dot(&dir);
            let perp = (rel - dir * proj).norm();
            proj > 0.0 && proj < LANE_RANGE && perp < LANE_HALF_WIDTH
        })
}
