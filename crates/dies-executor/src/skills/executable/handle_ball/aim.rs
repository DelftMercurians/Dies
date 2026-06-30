//! Aim the held ball at a target point and kick (`Shoot`).
//!
//! From the target *point* the skill first chooses where to launch the ball from:
//! if the current spot already has a clear, reachable kicking pose toward the
//! target it aims in place (orbiting the ball, dribbler pressed); if the ball is
//! jammed (e.g. against a boundary) it first dribbles the ball a short distance to
//! a reachable launch point, then aims. Orbit / short-carry / turn-with-ball all
//! fall out of *where the launch point is* relative to the current ball.

use dies_core::FieldGeometry;

use super::*;
use crate::control::{KickerControlInput, Velocity};

impl HandleBallSkill {
    /// Aim (orbit) toward the shot target, repositioning first if the ball is
    /// jammed.
    pub(super) fn drive_aim(
        &mut self,
        ctx: &SkillContext<'_>,
        ball_pos: Vector2,
        player_pos: Vector2,
        now: f64,
    ) -> SkillProgress {
        let BallAction::Shoot { target } = self.action else {
            return self.drive_hold(ctx, self.hold_heading());
        };
        if now - self.stage_entered > AIM_BACKSTOP {
            log::warn!("handle_ball: aim timed out");
            return self.fail();
        }

        let launch = *self
            .launch
            .get_or_insert_with(|| choose_launch(ctx, ball_pos, target));

        // Debug: the chosen launch point and the shot line to the target.
        let tc = ctx.team_context;
        tc.debug_cross_colored(dkey(ctx, "launch"), launch, DebugColor::Green);
        tc.debug_line_colored(dkey(ctx, "shot"), ball_pos, target, DebugColor::Green);

        let mut input = PlayerControlInput::new();
        input.with_dribbling(DRIBBLER_SPEED);

        // Carry the ball to the launch point first if it's away from it.
        let ball_to_launch = (ball_pos - launch).norm();
        tc.debug_value(dkey(ctx, "ball_to_launch"), ball_to_launch);
        if ball_to_launch > REPOSITION_ARRIVE {
            tc.debug_string(dkey(ctx, "aim_phase"), "reposition");
            self.detail = format!("reposition {ball_to_launch:.0}mm");
            let axis = (target - launch)
                .try_normalize(1e-6)
                .unwrap_or_else(|| Vector2::new(1.0, 0.0));
            let pose = launch - axis * BALL_TO_ROBOT_DISTANCE;
            input.with_position(pose);
            input.with_yaw(Angle::from_vector(target - launch));
            input.with_acceleration_limit(CARRY_ACCEL_LIMIT);
            input.with_angular_speed_limit(CARRY_ANGULAR_LIMIT);
            return SkillProgress::Continue(input);
        }

        // Orbit the ball to align the shot axis, then commit.
        input.avoid_robots = false;
        input.with_angular_speed_limit(1000.0);

        let to_target = target - ball_pos;
        let target_heading = Angle::from_vector(to_target);
        let blocked = lane_blocked(ctx, ball_pos, target_heading);
        let err = target_heading - ctx.player.yaw;
        let err_deg = err.degrees().abs();
        tc.debug_string(dkey(ctx, "aim_phase"), "orbit");
        tc.debug_value(dkey(ctx, "yaw_err_deg"), err_deg);
        tc.debug_value(dkey(ctx, "lane_blocked"), if blocked { 1.0 } else { 0.0 });
        self.detail = format!(
            "orbit err{err_deg:.0}°{}",
            if blocked { " blocked" } else { "" }
        );
        if err.abs() < YAW_TOLERANCE && !blocked && ctx.player.has_ball {
            self.stage = Stage::Kicking;
            self.stage_entered = now;
        }

        let r = player_pos - ball_pos;
        let r_hat = r
            .try_normalize(1e-6)
            .unwrap_or_else(|| Vector2::new(1.0, 0.0));
        let tangent = Vector2::new(-r_hat.y, r_hat.x); // CCW
        let v_rad = -RADIUS_KP * (r.norm() - BALL_TO_ROBOT_DISTANCE) * r_hat;
        let v_tan = if blocked {
            Vector2::zeros()
        } else {
            let speed = (ORBIT_GAIN * err.abs()).clamp(MIN_ORBIT_SPEED, ORBIT_SPEED);
            err.signum() * speed * tangent
        };
        input.velocity = Velocity::global(v_tan + v_rad);
        input.with_yaw(Angle::from_vector(-r));
        SkillProgress::Continue(input)
    }

    pub(super) fn drive_kick(
        &mut self,
        ctx: &SkillContext<'_>,
        ball_pos: Vector2,
        now: f64,
    ) -> SkillProgress {
        let BallAction::Shoot { target } = self.action else {
            return self.fail();
        };
        let target_heading = Angle::from_vector(target - ball_pos);
        ctx.team_context
            .debug_line_colored(dkey(ctx, "shot"), ball_pos, target, DebugColor::Green);
        self.detail = format!("fire {KICK_SPEED:.0}");
        let mut input = PlayerControlInput::new();
        input.with_yaw(target_heading);
        input.with_kicker(KickerControlInput::Kick);
        input.kick_speed = Some(KICK_SPEED);
        input.with_dribbling(0.0);
        self.kick_ball_pos = Some(ball_pos);
        self.kick_time = Some(now);
        self.stage = Stage::Verifying;
        self.stage_entered = now;
        SkillProgress::Continue(input)
    }

    pub(super) fn drive_verify(
        &mut self,
        ctx: &SkillContext<'_>,
        ball_pos: Vector2,
        ball_vel_norm: f64,
        now: f64,
    ) -> SkillProgress {
        let depart_dist = self
            .kick_ball_pos
            .map(|p0| (ball_pos - p0).norm())
            .unwrap_or(0.0);
        let elapsed = self.kick_time.map(|t| now - t).unwrap_or(0.0);
        let tc = ctx.team_context;
        tc.debug_value(dkey(ctx, "depart_dist"), depart_dist);
        tc.debug_value(dkey(ctx, "verify_ms"), elapsed * 1000.0);
        self.detail = format!("verify {depart_dist:.0}mm {:.0}ms", elapsed * 1000.0);

        let departed = depart_dist > KICK_DEPART_DIST || ball_vel_norm > KICK_DEPART_SPEED;
        if departed {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }
        if self
            .kick_time
            .map(|t| now - t > VERIFY_WINDOW.as_secs_f64())
            .unwrap_or(true)
        {
            log::warn!("handle_ball: kick did not connect");
            return self.fail();
        }
        let mut input = PlayerControlInput::new();
        if let BallAction::Shoot { target } = self.action {
            input.with_yaw(Angle::from_vector(target - ball_pos));
        }
        input.with_dribbling(0.0);
        SkillProgress::Continue(input)
    }
}

/// Choose where to launch the ball from. Prefer the ball's current spot (aim in
/// place); if it's jammed against a boundary or has no on-surface, obstacle-free
/// kicking pose toward the target, search short carries along the inward
/// direction (fanned laterally) for the nearest spot that does.
fn choose_launch(ctx: &SkillContext<'_>, ball: Vector2, target: Vector2) -> Vector2 {
    let field = ctx.world.field_geom.as_ref();
    if !ball_near_boundary(ball, field) && pose_ok(ctx, ball, target, field) {
        return ball;
    }

    let inward = inward_dir(ball, field);
    for &step in &CARRY_STEPS {
        for &rot in &CARRY_FAN {
            let dir = rotate(inward, rot);
            let cand = ball + dir * step;
            if !ball_near_boundary(cand, field) && pose_ok(ctx, cand, target, field) {
                return cand;
            }
        }
    }
    ball // nothing better found — aim in place
}

/// Whether the kicking pose for launching `launch` at `target` is on the playing
/// surface and free of obstacles (so the robot can actually get behind the ball).
fn pose_ok(
    ctx: &SkillContext<'_>,
    launch: Vector2,
    target: Vector2,
    field: Option<&FieldGeometry>,
) -> bool {
    let axis = match (target - launch).try_normalize(1e-6) {
        Some(a) => a,
        None => return false,
    };
    let pose = launch - axis * BALL_TO_ROBOT_DISTANCE;
    on_surface(pose, field) && ctx.obstacles.point_clear(pose, LAUNCH_POSE_EGO)
}

/// True if `p` is on the physical surface, inset by [`LAUNCH_SURFACE_MARGIN`].
fn on_surface(p: Vector2, field: Option<&FieldGeometry>) -> bool {
    let Some(field) = field else {
        return true;
    };
    let max_x = field.field_length / 2.0 + field.boundary_width - LAUNCH_SURFACE_MARGIN;
    let max_y = field.field_width / 2.0 + field.boundary_width - LAUNCH_SURFACE_MARGIN;
    p.x.abs() <= max_x && p.y.abs() <= max_y
}

/// True if the ball is within [`LAUNCH_BOUNDARY_MARGIN`] of any field line.
fn ball_near_boundary(ball: Vector2, field: Option<&FieldGeometry>) -> bool {
    let Some(field) = field else {
        return false;
    };
    let hl = field.field_length / 2.0;
    let hw = field.field_width / 2.0;
    (hl - ball.x.abs()) < LAUNCH_BOUNDARY_MARGIN || (hw - ball.y.abs()) < LAUNCH_BOUNDARY_MARGIN
}

/// Inward direction to nudge the ball: away from the nearest field line, or
/// toward the field centre when not near any line.
fn inward_dir(ball: Vector2, field: Option<&FieldGeometry>) -> Vector2 {
    let Some(field) = field else {
        return (-ball)
            .try_normalize(1e-6)
            .unwrap_or(Vector2::new(1.0, 0.0));
    };
    let hl = field.field_length / 2.0;
    let hw = field.field_width / 2.0;
    let cands = [
        (hw - ball.y, Vector2::new(0.0, -1.0)),
        (hw + ball.y, Vector2::new(0.0, 1.0)),
        (hl - ball.x, Vector2::new(-1.0, 0.0)),
        (hl + ball.x, Vector2::new(1.0, 0.0)),
    ];
    let (dist, normal) = cands
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    if dist < LAUNCH_BOUNDARY_MARGIN {
        normal
    } else {
        (-ball)
            .try_normalize(1e-6)
            .unwrap_or(Vector2::new(1.0, 0.0))
    }
}

/// Rotate a unit vector by `rot` radians.
fn rotate(v: Vector2, rot: f64) -> Vector2 {
    let (s, c) = rot.sin_cos();
    Vector2::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

/// Whether another robot sits in the shoot corridor (ray from the ball along
/// `heading`, half-width [`LANE_HALF_WIDTH`], length [`LANE_RANGE`]).
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
