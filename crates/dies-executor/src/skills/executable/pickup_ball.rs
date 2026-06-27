use std::time::{Duration, Instant};

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{KickerControlInput, PlayerControlInput};

const DRIBBLER_SPEED: f64 = 0.6;
const APPROACH_DISTANCE: f64 = 200.0;
const COMMIT_DISTANCE: f64 = 280.0;
const COMMIT_PERP: f64 = 30.0;
const GATE_PERP: f64 = 80.0;
const APPROACH_GAIN: f64 = 3.0;
const APPROACH_MIN_SPEED: f64 = 200.0;
const LATERAL_GAIN: f64 = 4.0;
const APPROACH_CARE: f64 = -0.6;
const BALL_MOVED_FAIL: f64 = 100.0;
const DRIVEN_FAIL: f64 = 250.0;

// ── instant-kick (reflex strike-through) tuning ──────────────────────────────
/// Ball displacement along the kick axis that counts as "the reflex connected".
const KICK_DEPART_DIST: f64 = 100.0;
/// Ball speed that also counts as departure (filtered estimate; confirms).
const KICK_DEPART_SPEED: f64 = 1000.0;
/// How long to wait after arming the reflex for the ball to depart before
/// declaring a whiff and failing (so a dribbler-on hold can't linger and
/// accumulate double-touch contact).
const REFLEX_TIMEOUT: Duration = Duration::from_millis(600);

pub struct PickupBallSkill {
    target_heading: Angle,
    status: SkillStatus,
    commit_pos: Option<Vector2>,
    commit_ball: Option<Vector2>,
    /// Strike-through release: arm a reflex kick instead of capturing.
    instant_kick: bool,
    /// Ball position when the reflex was first armed (for the departure check).
    kick_ball_pos: Option<Vector2>,
    /// When the reflex was first armed (for the whiff timeout).
    armed_at: Option<Instant>,
}

impl PickupBallSkill {
    pub fn new(target_heading: Angle, instant_kick: bool) -> Self {
        Self {
            target_heading,
            status: SkillStatus::Running,
            commit_pos: None,
            commit_ball: None,
            instant_kick,
            kick_ball_pos: None,
            armed_at: None,
        }
    }

    pub fn set_target_heading(&mut self, target_heading: Angle) {
        self.target_heading = target_heading;
    }
}

impl ExecutableSkill for PickupBallSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::PickupBall { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::PickupBall {
            target_heading,
            instant_kick,
        } = command
        {
            self.target_heading = *target_heading;
            self.instant_kick = *instant_kick;
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        // Capture mode completes on possession. Strike-through mode never holds
        // the ball, so it must NOT succeed on breakbeam — it succeeds on the ball
        // departing after the reflex fires (handled below).
        if !self.instant_kick && ctx.player.has_ball {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }

        let Some(ball) = ctx.world.ball.as_ref() else {
            return SkillProgress::Continue(PlayerControlInput::default());
        };

        let ball_pos = ball.position.xy();
        let player_pos = ctx.player.position;
        let dir = self.target_heading.to_vector();

        let rel = player_pos - ball_pos;
        let along = rel.dot(&dir);
        let perp_vec = rel - dir * along;
        let perp = perp_vec.norm();

        let mut input = PlayerControlInput::new();
        input.with_yaw(self.target_heading);
        input.with_dribbling(DRIBBLER_SPEED);

        let committed = along < 0.0 && -along < COMMIT_DISTANCE && perp < COMMIT_PERP;
        if committed {
            input.avoid_ball = false;
            let gate = (1.0 - perp / GATE_PERP).clamp(0.0, 1.0);
            let speed = (-along) * APPROACH_GAIN + APPROACH_MIN_SPEED;
            input.add_global_velocity(dir * speed * gate - perp_vec * LATERAL_GAIN);

            if self.instant_kick {
                // Arm a firmware reflex kick: it fires the instant the ball hits
                // the breakbeam during the drive-through (dribbler-on pulls it
                // onto the beam), so the ball is struck on contact, never held.
                input.with_kicker(KickerControlInput::ReflexKick);
                let kick_ball = *self.kick_ball_pos.get_or_insert(ball_pos);
                let now = *self.armed_at.get_or_insert_with(Instant::now);

                // Success = ball departed along the kick axis (the reflex fired).
                // Checked BEFORE any ball-moved failure, and gated to the kick
                // direction so a lateral knock doesn't read as a clean release.
                let along_depart = (ball_pos - kick_ball).dot(&dir);
                if along_depart > KICK_DEPART_DIST || ball.velocity.norm() > KICK_DEPART_SPEED {
                    self.status = SkillStatus::Succeeded;
                    return SkillProgress::success();
                }
                // Whiff: armed but the ball never left → fail so the driver
                // re-stages a fresh approach instead of lingering in contact.
                if now.elapsed() > REFLEX_TIMEOUT {
                    log::warn!("pickup_ball: reflex kick did not connect");
                    self.status = SkillStatus::Failed;
                    return SkillProgress::failure();
                }
            } else {
                let commit_ball = *self.commit_ball.get_or_insert(ball_pos);
                let commit_pos = *self.commit_pos.get_or_insert(player_pos);
                if (ball_pos - commit_ball).norm() > BALL_MOVED_FAIL
                    || (player_pos - commit_pos).norm() > DRIVEN_FAIL
                {
                    self.status = SkillStatus::Failed;
                    return SkillProgress::failure();
                }
            }
        } else {
            input.with_position(ball_pos - dir * APPROACH_DISTANCE);
            input.avoid_ball = true;
            input.avoid_ball_care = APPROACH_CARE;
            self.commit_pos = Some(player_pos);
            self.commit_ball = Some(ball_pos);
        }

        self.status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "PickupBall"
    }

    fn description(&self) -> String {
        if self.commit_ball.is_some() {
            "committing to ball".to_string()
        } else {
            "approaching ball".to_string()
        }
    }
}
