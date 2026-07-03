//! `Disengage` — deliberately release a held ball so the kick can be re-aimed.
//!
//! The disengage half of the re-aim flag (see the `"HandleBall re-aim"`
//! tunables): instead of orbiting the held ball onto the shot axis, release it
//! and re-strike it with the IRL-proven drive-through. Two sub-phases, both
//! timed off `stage_entered` (world clock):
//!
//! 1. **Settle** ([`DISENGAGE_SETTLE`]): stand still, dribbler cut *instantly*
//!    (`dribbler_brake` bypasses the controller's seam-protection decay ramp —
//!    a decaying dribbler would drag the ball along the back-off), kicker
//!    idle. The ball is released dead in place.
//! 2. **Break contact**: back straight away from the ball until it is
//!    [`BREAK_CONTACT_DIST`] clear of the mouth AND the raw breakbeam is low,
//!    then hand over to the plain drive-through strike — whose staging point
//!    (~200 mm behind the live ball on the strike axis) is the real
//!    destination, so the total ball-free displacement is minimal.
//!
//! The reflex is NEVER armed here: the firmware fires the instant the beam is
//! high, so an armed reflex during a deliberate release would kick away the
//! ball we mean to keep placing.

use super::*;
use crate::control::Velocity;

impl HandleBallSkill {
    /// Enter the deliberate-release stage.
    pub(super) fn enter_disengage(&mut self, now: f64) {
        self.stage = Stage::Disengage;
        self.stage_entered = now;
        // The capture-settle glide and any commit state must not fight the
        // release.
        self.settle_until = None;
        self.commit_latched = false;
        self.committed_since = None;
    }

    /// Drive one `Disengage` frame. On release, hands over to the drive-through
    /// strike on the same tick (no dead frame).
    pub(super) fn tick_disengage(&mut self, ctx: &SkillContext<'_>, now: f64) -> SkillProgress {
        let elapsed = now - self.stage_entered;
        let ball_pos = ctx.world.ball.as_ref().map(|b| b.position.xy());
        let player_pos = ctx.player.position;
        let tc = ctx.team_context;

        let mut input = PlayerControlInput::new();
        input.with_dribbling(0.0);
        input.dribbler_brake = true;
        // Hold the current heading — spinning against a resting ball nudges it.
        input.with_yaw(ctx.player.yaw);
        // We are leaving our own released ball; don't route around it.
        input.avoid_ball = false;

        if elapsed < DISENGAGE_SETTLE() {
            self.detail = "disengage: settle".into();
            tc.debug_string(dkey(ctx, "reaim"), "settle");
            self.status = SkillStatus::Running;
            return SkillProgress::Continue(input);
        }

        // Break contact: released when the ball is clear of the mouth and the
        // raw beam (the signal the firmware fires on) is low.
        let clear_dist = BALL_TO_ROBOT_DISTANCE() + BREAK_CONTACT_DIST();
        let beam_low = !ctx.player.breakbeam_ball_detected;
        let backed_off = match ball_pos {
            Some(bp) => (player_pos - bp).norm() >= clear_dist,
            // Ball occluded/unseen: dead-reckon the back-off, then trust the
            // low beam.
            None => elapsed - DISENGAGE_SETTLE() > clear_dist / DISENGAGE_BACKOFF_SPEED,
        };
        if beam_low && backed_off {
            self.disengaged = true;
            // Sentinel; all further routing goes through `disengaged`.
            self.stage = Stage::Acquire;
            self.stage_entered = now;
            tc.debug_string(dkey(ctx, "reaim"), "released");
            if let Some(target) = self.kick_target() {
                return self.tick_strike(ctx, target, now);
            }
            self.status = SkillStatus::Running;
            return SkillProgress::Continue(input);
        }

        if elapsed - DISENGAGE_SETTLE() > DISENGAGE_TIMEOUT {
            log::warn!("handle_ball: disengage could not release the ball");
            return self.fail();
        }

        self.detail = "disengage: back off".into();
        tc.debug_string(dkey(ctx, "reaim"), "back_off");
        // Straight away from the ball (robust to a slightly off-center rest);
        // straight back in the robot frame when the ball is unseen.
        input.velocity = match ball_pos.and_then(|bp| (player_pos - bp).try_normalize(1e-6)) {
            Some(away) => Velocity::Global(away * DISENGAGE_BACKOFF_SPEED),
            None => Velocity::Local(Vector2::new(-DISENGAGE_BACKOFF_SPEED, 0.0)),
        };
        self.status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }

    /// Whether a held-ball kick at `target` needs a re-aim: yaw error to the
    /// kick axis beyond [`REAIM_YAW_DEG`]. Within the cone, the aligned paths
    /// fire directly.
    pub(super) fn reaim_needed(&self, ctx: &SkillContext<'_>, target: Vector2) -> bool {
        let origin = ctx
            .world
            .ball
            .as_ref()
            .map(|b| b.position.xy())
            .unwrap_or(ctx.player.position);
        let axis = Angle::from_vector(target - origin);
        (axis - ctx.player.yaw).degrees().abs() > REAIM_YAW_DEG()
    }

    /// The kick target of the current action, if it is a kick.
    pub(super) fn kick_target(&self) -> Option<Vector2> {
        match self.action {
            BallAction::Shoot { target } | BallAction::Strike { target, .. } => Some(target),
            _ => None,
        }
    }
}
