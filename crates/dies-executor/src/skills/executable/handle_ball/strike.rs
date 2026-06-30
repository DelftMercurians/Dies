//! `Strike` — a one-motion reflex strike-through.
//!
//! The double-touch-safe attacking-restart release: approach the ball on the
//! strike axis and arm a firmware reflex kick during the drive-through, so the
//! ball is struck the instant it reaches the breakbeam — **never held**. A whiff
//! fails (rather than falling back to a hold) so the driver re-stages a clean
//! approach. Reuses the capture-corridor geometry from [`super::acquire`].

use super::acquire::{commit_velocity, committed, perp_target, stage_point};
use super::*;
use crate::control::KickerControlInput;

impl HandleBallSkill {
    /// `Strike`: a one-motion reflex strike-through. Never holds, never
    /// re-acquires — a whiff fails so the driver re-stages a clean approach.
    pub(super) fn tick_strike(
        &mut self,
        ctx: &SkillContext<'_>,
        target: Vector2,
        now: f64,
    ) -> SkillProgress {
        let Some(ball) = ctx.world.ball.as_ref() else {
            return SkillProgress::Continue(PlayerControlInput::default());
        };
        let ball_pos = ball.position.xy();
        let ball_vel = ball.velocity.xy();
        let player_pos = ctx.player.position;
        let heading = self
            .approach
            .unwrap_or_else(|| Angle::from_vector(target - ball_pos));
        let dir = heading.to_vector();

        let rel = player_pos - ball_pos;
        let along = rel.dot(&dir);
        let perp_vec = rel - dir * along;
        let perp = perp_vec.norm();
        let pt = perp_target(heading, dir);

        // Debug: the fixed strike axis and corridor geometry.
        let tc = ctx.team_context;
        self.emit_common(ctx, "strike");
        tc.debug_line_colored(dkey(ctx, "strike_axis"), ball_pos, target, DebugColor::Red);
        tc.debug_value(dkey(ctx, "along"), along);
        tc.debug_value(dkey(ctx, "perp"), perp);

        let mut input = PlayerControlInput::new();
        input.with_yaw(heading);
        input.with_dribbling(DRIBBLER_SPEED());

        let is_committed = committed(along, perp);
        // Drive-and-reflex-kick: while committed and the ToF sees the ball, hand the
        // final centimetres to firmware magnet capture with the reflex armed
        // (ARM_REFLEX_KICK_MAGNET) so the strike fires the instant the ball reaches
        // the breakbeam. Velocity is ignored while engaged.
        let magnet = self.magnet_engaged(ctx, is_committed);
        input.magnet = magnet;
        tc.debug_value(dkey(ctx, "magnet"), if magnet { 1.0 } else { 0.0 });

        if is_committed {
            tc.debug_value(dkey(ctx, "committed"), 1.0);
            input.avoid_ball = false;
            input.add_global_velocity(commit_velocity(dir, along, perp_vec, ball_vel, pt));

            input.with_kicker(KickerControlInput::ReflexKick);
            let kick_ball = *self.kick_ball_pos.get_or_insert(ball_pos);
            let armed_at = *self.armed_at.get_or_insert(now);
            let along_depart = (ball_pos - kick_ball).dot(&dir);
            let armed_ms = (now - armed_at) * 1000.0;
            tc.debug_value(dkey(ctx, "depart_along"), along_depart);
            tc.debug_value(dkey(ctx, "armed_ms"), armed_ms);
            self.detail = format!("striking d{along_depart:.0} {armed_ms:.0}ms");
            if along_depart > KICK_DEPART_DIST() || ball.velocity.norm() > KICK_DEPART_SPEED() {
                self.status = SkillStatus::Succeeded;
                return SkillProgress::success();
            }
            if now - armed_at > REFLEX_TIMEOUT.as_secs_f64() {
                log::warn!("handle_ball: reflex strike did not connect");
                return self.fail();
            }
        } else {
            let staging = stage_point(ball_pos, dir, pt, ctx.world.field_geom.as_ref());
            tc.debug_value(dkey(ctx, "committed"), 0.0);
            tc.debug_cross_colored(dkey(ctx, "staging"), staging, DebugColor::Red);
            self.detail = "staging".into();
            input.with_position(staging);
            input.avoid_ball = true;
            input.avoid_ball_care = APPROACH_CARE();
        }

        self.status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }
}
