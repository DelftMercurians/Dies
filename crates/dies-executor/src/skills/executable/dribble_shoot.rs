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
/// Tangential orbit speed cap. Kept low enough that the dribbler can retain the
/// ball through the slide — a faster orbit squeezes it out of the dribbler mouth.
const ORBIT_SPEED: f64 = 400.0;
const ORBIT_GAIN: f64 = 600.0;
const MIN_ORBIT_SPEED: f64 = 40.0;
/// Robot→ball distance beyond which the ball is considered lost (well past the
/// `BALL_TO_ROBOT_DISTANCE` hold radius). Distance-based so a brief breakbeam
/// flicker during the slide doesn't abort the orbit.
const LOST_BALL_DISTANCE: f64 = 220.0;
const YAW_TOLERANCE: f64 = 5.0 * std::f64::consts::PI / 180.0;
const DRIBBLER_SPEED: f64 = 0.6;
const KICK_SPEED: f64 = 4000.0;
const TIMEOUT: Duration = Duration::from_secs(4);
const LANE_HALF_WIDTH: f64 = 120.0;
const LANE_RANGE: f64 = 3000.0;
/// How far the ball must move from its at-kick position to count as "the kick
/// connected". Primary signal — position is not subject to the filter lag.
const KICK_DEPART_DIST: f64 = 100.0;
/// Ball speed that also counts as departure. Secondary signal: the ball velocity
/// estimate is Kalman-filtered and lags, so it confirms but rarely fires first.
const KICK_DEPART_SPEED: f64 = 1000.0;
/// How long to wait for the ball to leave after commanding a kick before
/// declaring the kick a whiff and failing.
const VERIFY_WINDOW: Duration = Duration::from_millis(200);

enum AimState {
    Aiming,
    Kicking,
    /// Kick commanded; waiting to confirm the ball actually left the dribbler.
    Verifying,
}

pub struct DribbleShootSkill {
    target_heading: Angle,
    status: SkillStatus,
    state: AimState,
    start: Option<Instant>,
    /// Ball position recorded the instant the kick was commanded.
    kick_ball_pos: Option<Vector2>,
    /// When the kick was commanded (start of the verification window).
    kick_time: Option<Instant>,
}

impl DribbleShootSkill {
    pub fn new(target_heading: Angle) -> Self {
        Self {
            target_heading,
            status: SkillStatus::Running,
            state: AimState::Aiming,
            start: None,
            kick_ball_pos: None,
            kick_time: None,
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

        let start = *self.start.get_or_insert_with(Instant::now);
        if start.elapsed() > TIMEOUT {
            log::warn!("dribble_shoot: timeout");
            self.status = SkillStatus::Failed;
            return SkillProgress::failure();
        }

        let ball_pos = ball.position.xy();
        let player_pos = ctx.player.position;
        // A robot in the shoot corridor must not be kicked into — but this is a
        // transient, positional condition (the opponent keeper is always in front
        // of the goal we aim at), so it gates the *kick*, not the whole skill. We
        // keep aiming/holding while blocked instead of failing and freezing.
        let blocked = lane_blocked(&ctx, ball_pos, self.target_heading);

        let err = self.target_heading - ctx.player.yaw;
        log::debug!(
            "dribble_shoot tick: err={:.1}deg blocked={}",
            err.degrees(),
            blocked
        );
        let mut input = PlayerControlInput::new();
        input.avoid_robots = false;
        input.with_dribbling(DRIBBLER_SPEED);
        input.with_angular_speed_limit(1000.0);

        match self.state {
            AimState::Aiming => {
                // Lost the ball outright → fail fast so the strategy re-captures
                // (the executor re-creates this skill on the next command). Use a
                // distance gate, not breakbeam, so a brief flicker mid-slide
                // doesn't abort the orbit.
                let r = player_pos - ball_pos;
                if r.norm() > LOST_BALL_DISTANCE {
                    log::warn!("dribble_shoot: lost ball");
                    self.status = SkillStatus::Failed;
                    return SkillProgress::failure();
                }

                // Commit to the kick only when aligned, the lane is clear, and the
                // ball is actually seated at the kicker — never kick into a robot
                // or at a ball that isn't there.
                if err.abs() < YAW_TOLERANCE && !blocked && ctx.player.has_ball {
                    self.state = AimState::Kicking;
                }

                const RADIUS_KP: f64 = 2.0;
                let r_hat = r.normalize();
                let tangent = Vector2::new(-r_hat.y, r_hat.x); // CCW
                let v_rad = -RADIUS_KP * (r.norm() - BALL_TO_ROBOT_DISTANCE) * r_hat;
                // Slide tangentially to aim; when blocked, hold (radius only) so
                // the robot keeps the ball aimed and waits for the lane to clear.
                let v_tan = if blocked {
                    Vector2::zeros()
                } else {
                    let speed = (ORBIT_GAIN * err.abs()).clamp(MIN_ORBIT_SPEED, ORBIT_SPEED);
                    err.signum() * speed * tangent // sign to confirm
                };
                input.velocity = Velocity::global(v_tan + v_rad);
                input.with_yaw(Angle::from_vector(-r)); // face ball
            }
            AimState::Kicking => {
                input.with_yaw(self.target_heading);
                input.with_kicker(KickerControlInput::Kick);
                input.kick_speed = Some(KICK_SPEED);
                input.with_dribbling(0.0);
                // Snapshot the ball so the verify gate can tell if it actually left.
                self.kick_ball_pos = Some(ball_pos);
                self.kick_time = Some(Instant::now());
                self.state = AimState::Verifying;
            }
            AimState::Verifying => {
                // Only succeed once the ball has demonstrably left the dribbler;
                // a commanded kick that doesn't connect is a whiff, not a success.
                let departed = self
                    .kick_ball_pos
                    .map(|p0| (ball_pos - p0).norm() > KICK_DEPART_DIST)
                    .unwrap_or(false)
                    || ball.velocity.norm() > KICK_DEPART_SPEED;
                if departed {
                    self.status = SkillStatus::Succeeded;
                    return SkillProgress::success();
                }
                if self
                    .kick_time
                    .map(|t| t.elapsed() > VERIFY_WINDOW)
                    .unwrap_or(true)
                {
                    log::warn!("dribble_shoot: kick did not connect");
                    self.status = SkillStatus::Failed;
                    return SkillProgress::failure();
                }
                // Hold still and watch — dribbler off, no re-kick.
                input.with_yaw(self.target_heading);
                input.with_dribbling(0.0);
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

    fn is_oneshot(&self) -> bool {
        true
    }

    fn description(&self) -> String {
        let phase = match self.state {
            AimState::Aiming => "orbiting to aim",
            AimState::Kicking => "kicking",
            AimState::Verifying => "verifying kick",
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
