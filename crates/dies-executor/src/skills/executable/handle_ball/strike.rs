//! `Strike` — a one-motion reflex strike-through.
//!
//! The double-touch-safe attacking-restart release: approach the ball on the
//! strike axis and arm a firmware reflex kick during the drive-through, so the
//! ball is struck the instant it reaches the breakbeam — **never held**. A whiff
//! fails (rather than falling back to a hold) so the driver re-stages a clean
//! approach. Reuses the capture-corridor geometry from [`super::acquire`].

use super::acquire::{commit_velocity, committed, latched, perp_target, stage_point, staging_feed};
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
        // The strike axis must point at the target: an explicit `Heading` overrides,
        // otherwise it is derived from the target (a `Fastest`/`Default` no-op here).
        let heading = self.acquire_heading(ball_pos);
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

        // Departure / whiff verify — armed-state gated, NOT latch gated. The kicked
        // ball outruns the ball filter, so the estimate often jumps PAST the latch
        // release band on the very frame it finally moves; a verify inside the
        // committed branch misses its own kick and the robot re-stages after the
        // departed ball (observed: "success" only declared 4 m downfield once it
        // re-committed behind the arrived ball). Once the reflex has been armed,
        // check departure every tick until success/timeout. Directional on purpose
        // (displacement + speed along the strike axis) so someone else's kick or
        // the sim ball's vertical micro-bounce can't count as our departure.
        if let (Some(kick_ball), Some(armed_at)) = (self.kick_ball_pos, self.armed_at) {
            let along_depart = (ball_pos - kick_ball).dot(&dir);
            let vel_along = ball_vel.dot(&dir);
            let armed_ms = (now - armed_at) * 1000.0;
            tc.debug_value(dkey(ctx, "depart_along"), along_depart);
            tc.debug_value(dkey(ctx, "armed_ms"), armed_ms);
            if along_depart > KICK_DEPART_DIST() || vel_along > KICK_DEPART_SPEED() {
                log::debug!("strike departed: d{along_depart:.0} v{vel_along:.0}");
                self.status = SkillStatus::Succeeded;
                return SkillProgress::success();
            }
            if now - armed_at > REFLEX_TIMEOUT.as_secs_f64() {
                log::warn!("handle_ball: reflex strike did not connect");
                return self.fail();
            }
        }

        let mut input = PlayerControlInput::new();
        input.with_yaw(heading);
        // No dribbler while hold-fired: a gated strike loiters at the staging
        // point ~200 mm from the ball for however long the barrier takes, and a
        // spinning dribbler there grabs-and-drags the ball on the slightest
        // brush (in sim the dribbler claims within ~190 mm inside the mouth
        // cone; observed dragging the ball 500+ mm during a hinted pass Stage).
        // The ungated drive-through keeps the tuned always-on dribbler.
        input.with_dribbling(if self.strike_gated {
            0.0
        } else {
            DRIBBLER_SPEED()
        });

        // Schmitt-latched commit (mirrors `drive_acquire`): hold through the release
        // bands (perp *and* along) so a transient nudge — or the ball estimate
        // collapsing into the hull at contact — doesn't disarm the reflex
        // mid-strike. A genuine blow-out clears the latch and re-stages;
        // REFLEX_TIMEOUT bounds the attempt.
        //
        // The external hold-fire gate (pass coordinator) suppresses the latch
        // entirely: the staging point is already inside the commit corridor, so a
        // gated strike waits there lined up and commits the tick after ungating.
        if self.strike_gated {
            self.commit_latched = false;
        } else if committed(along, perp) {
            self.commit_latched = true;
        }
        tc.debug_value(
            dkey(ctx, "gated"),
            if self.strike_gated { 1.0 } else { 0.0 },
        );
        let is_committed = self.commit_latched && latched(along, perp);
        // Drive-and-reflex-kick: while committed and the ToF sees the ball, hand the
        // final centimetres to firmware magnet capture with the reflex armed
        // (ARM_REFLEX_KICK_MAGNET) so the strike fires the instant the ball reaches
        // the breakbeam. Velocity is ignored while engaged.
        let magnet = self.magnet_engaged(ctx, is_committed);
        input.magnet = magnet;
        tc.debug_value(dkey(ctx, "magnet"), if magnet { 1.0 } else { 0.0 });

        if is_committed {
            // Commit drops robot + defense-box ORCA — bail rather than illegally
            // strike a ball inside (or wander into) a defense area (see `acquire`).
            // The keeper is exempt from *our own* box (it may legally play there);
            // the opponent's box is off-limits to everyone, keeper included.
            if let Some(field) = ctx.world.field_geom.as_ref() {
                let is_keeper = ctx.world.current_game_state.our_keeper_id == Some(ctx.player.id);
                let banned = |p, m| {
                    if is_keeper {
                        field.in_opp_defense_area(p, m)
                    } else {
                        field.in_defense_area(p, m)
                    }
                };
                if banned(player_pos, DEFENSE_BAIL_MARGIN())
                    || banned(ball_pos, BALL_IN_BOX_MARGIN())
                {
                    self.detail = "bail: defense area".into();
                    return self.fail();
                }
            }

            tc.debug_value(dkey(ctx, "committed"), 1.0);
            input.avoid_ball = false;
            input.avoid_robots = false; // drive through the contesting opponent
            input.avoid_wall = true; // …but stay on the field (tight margin)
            input.wall_care = COMMIT_WALL_CARE();
            input.add_global_velocity(commit_velocity(dir, along, perp_vec, ball_vel, pt));

            input.with_kicker(KickerControlInput::ReflexKick);
            // Arm-state bookkeeping only — the departure/whiff verify runs above,
            // independent of the latch.
            self.kick_ball_pos.get_or_insert(ball_pos);
            self.armed_at.get_or_insert(now);
            self.detail = format!("striking a{along:.0} p{perp:.0}");
        } else {
            self.commit_latched = false;
            let staging = stage_point(ball_pos, dir, pt, ctx.world.field_geom.as_ref());
            tc.debug_value(dkey(ctx, "committed"), 0.0);
            tc.debug_cross_colored(dkey(ctx, "staging"), staging, DebugColor::Red);
            self.detail = "staging".into();
            input.with_position(staging);
            input.add_global_velocity(staging_feed(player_pos, staging));
            input.avoid_ball = true;
            input.avoid_ball_care = APPROACH_CARE();
        }

        self.status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }
}
