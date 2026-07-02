//! Acquire (capture) front-end and the shared ball-geometry helpers.
//!
//! The skill picks which side of the ball to take it from (fastest reachable,
//! obstacle-free, biased toward an exit heading), stages behind the ball, then
//! drives through it with the dribbler on. A fast, non-head-on ball is instead
//! tail-caught along its velocity. The geometry helpers (`committed`,
//! `commit_velocity`, `perp_target`, `stage_point`) are reused by [`super::strike`].

use dies_core::FieldGeometry;

use super::*;

impl HandleBallSkill {
    /// Acquire (capture) front-end — drives one tick toward securing the ball.
    pub(super) fn drive_acquire(
        &mut self,
        ctx: &SkillContext<'_>,
        ball_pos: Vector2,
        ball_vel: Vector2,
        player_pos: Vector2,
        now: f64,
    ) -> SkillProgress {
        let exit = self.acquire_heading(ball_pos);
        let exit_bias = self.exit_bias_weight();
        let axis = capture_axis(
            ctx,
            ball_pos,
            ball_vel,
            player_pos,
            exit,
            exit_bias,
            &mut self.chosen_dir,
        );

        let rel = player_pos - ball_pos;
        let along = rel.dot(&axis.dir);
        let perp_vec = rel - axis.dir * along;
        let perp = perp_vec.norm();
        let pt = perp_target(axis.heading, axis.dir);

        let mut input = PlayerControlInput::new();
        let ball_heading = (ball_pos - player_pos)
            .try_normalize(1e-6)
            .map(Angle::from_vector)
            .unwrap_or(axis.heading);
        input.with_yaw(ball_heading);

        // Debug: the chosen approach axis (ball → staging side), the corridor
        // geometry, and whether the velocity-aware tail-catch is engaged.
        let tc = ctx.team_context;
        tc.debug_line_colored(
            dkey(ctx, "approach_axis"),
            axis.aim_point,
            axis.aim_point - axis.dir * APPROACH_DISTANCE(),
            DebugColor::Orange,
        );
        tc.debug_value(dkey(ctx, "along"), along);
        tc.debug_value(dkey(ctx, "perp"), perp);
        tc.debug_value(dkey(ctx, "tail_catch"), if axis.moving { 1.0 } else { 0.0 });

        // Schmitt-latched commit with hysteresis on BOTH axes: enter on the tight
        // corridor (`committed`, perp < COMMIT_PERP, strictly behind); hold while
        // inside the wider release bands (`latched`). A transient nudge — or the
        // ball estimate collapsing into the hull during the final press — no
        // longer silently drops to staging; a genuine blow-out past a release
        // band is a *real* bail (`reacquire`, counts toward MAX_REACQUIRE).
        if committed(along, perp) && !self.commit_latched {
            self.commit_latched = true;
            // Arm the pickup fast-bail from the moment the final drive commits.
            self.committed_since = Some(now);
        }
        let is_committed = self.commit_latched && latched(along, perp);
        let perp_blowout = self.commit_latched && !is_committed;
        // Hand off the final centimetres to firmware magnet capture once the ToF
        // actually sees the ball (the velocity output is ignored while engaged; the
        // firmware keeps the kicker charging via ARM_COUNTER). Capture-and-hold —
        // the reflex coupling lives in `Strike`.
        let magnet = self.magnet_engaged(ctx, is_committed);
        input.magnet = magnet;
        tc.debug_value(dkey(ctx, "magnet"), if magnet { 1.0 } else { 0.0 });

        if is_committed {
            // Pickup fast-bail: committed to the final drive but the breakbeam never
            // latched (ball pinned/wedged against us) — fail so the caller re-decides
            // quickly instead of grinding on an uncatchable ball.
            if let Some(since) = self.committed_since {
                if now - since > PICKUP_TIMEOUT() {
                    self.detail = "pickup timeout".into();
                    return self.fail();
                }
            }

            // The commit drive drops robot + defense-box ORCA, so nothing deflects
            // us out of a defense area. Bail (fail) rather than illegally enter one:
            // our own box is the keeper's, and touching the ball in either box is a
            // foul. Staging (above) keeps `avoid_robots = true` and is deflected
            // out of boxes normally, so only the commit needs this guard.
            if let Some(field) = ctx.world.field_geom.as_ref() {
                if field.in_defense_area(player_pos, DEFENSE_BAIL_MARGIN())
                    || field.in_defense_area(ball_pos, BALL_IN_BOX_MARGIN())
                {
                    self.detail = "bail: defense area".into();
                    return self.fail();
                }
            }

            tc.debug_value(dkey(ctx, "committed"), 1.0);
            self.detail = format!(
                "{}{} a{along:.0} p{perp:.0}",
                if axis.moving { "tail-catch" } else { "commit" },
                if magnet { "+magnet" } else { "" }
            );
            input.avoid_ball = false;
            // Drive straight through a contesting opponent: robot ORCA off for the
            // commit drive (mirrors `snatch`/`aim`). Otherwise reciprocal avoidance
            // of an opponent parked on the ball deflects us off the commit axis and
            // the capture never latches (perp can't fall under COMMIT_PERP). Walls
            // stay on (tight margin) so we can reach a ball against the boundary
            // without driving off the field.
            input.avoid_robots = false;
            input.avoid_wall = true;
            input.wall_care = COMMIT_WALL_CARE();
            input.add_global_velocity(commit_velocity(axis.dir, along, perp_vec, ball_vel, pt));

            let commit_ball = *self.commit_ball.get_or_insert(ball_pos);
            let commit_pos = *self.commit_pos.get_or_insert(player_pos);
            if commit_strayed(
                axis.moving,
                axis.dir,
                ball_pos,
                commit_ball,
                player_pos,
                commit_pos,
            ) {
                // Ball squirted out of the corridor during the commit drive —
                // re-stage internally (bounded by the re-acquire budget).
                self.reacquire(now);
            }
        } else {
            // Approach timeout: still traversing (not committed) — fail if we can't
            // reach the ball in time so the caller re-decides. Re-armed on re-acquire.
            if now - self.stage_entered > APPROACH_TIMEOUT() {
                self.detail = "approach timeout".into();
                return self.fail();
            }
            if perp_blowout {
                // Lateral blow-out during the commit drive — a real loss; surface it
                // instead of silently re-staging in an unbounded loop.
                self.reacquire(now);
            }
            let staging = stage_point(axis.aim_point, axis.dir, pt, ctx.world.field_geom.as_ref());
            tc.debug_value(dkey(ctx, "committed"), 0.0);
            tc.debug_value(dkey(ctx, "staging_dist"), (staging - player_pos).norm());
            tc.debug_cross_colored(dkey(ctx, "staging"), staging, DebugColor::Orange);
            self.detail = if axis.moving {
                "tail-catch: staging".into()
            } else {
                "staging".into()
            };
            input.with_dribbling(DRIBBLER_SPEED());
            input.with_position(staging);
            input.add_global_velocity(staging_feed(player_pos, staging));
            input.avoid_ball = true;
            input.avoid_ball_care = APPROACH_CARE();
            self.commit_pos = Some(player_pos);
            self.commit_ball = Some(ball_pos);
        }

        SkillProgress::Continue(input)
    }
}

/// Pick the push direction (unit) for a capture. Samples candidate sides around
/// the ball, scores each by how quickly the robot reaches the staging point minus
/// a reward for matching `exit_heading`, rejects sides that are blocked or would
/// push the ball out, and keeps `chosen_dir` unless clearly beaten (hysteresis).
/// The robot stages behind the ball along `-dir` and drives through it, so `dir`
/// is also the direction the ball is pushed. `chosen_dir` is updated in place.
fn select_approach_dir(
    ctx: &SkillContext<'_>,
    ball_pos: Vector2,
    player_pos: Vector2,
    exit_heading: Angle,
    exit_bias: f64,
    chosen_dir: &mut Option<Vector2>,
) -> Vector2 {
    let exit = exit_heading.to_vector();
    let field = ctx.world.field_geom.as_ref();
    let obstacles = &ctx.obstacles;

    // (tier, cost) for a push direction `u`; lower is better, tier dominates.
    // Tier ladder relaxes the soft constraints so a side is always returned:
    // 0 = clear staging + clear commit corridor + keeps ball in;
    // 1 = keeps ball in + clear staging (corridor blocked, e.g. a steal);
    // 2 = keeps ball in only; 3 = nothing satisfied (last resort).
    let eval = |u: Vector2| -> (u8, f64) {
        let staging = clamp_into_field(ball_pos - u * APPROACH_DISTANCE(), field);
        let clear = obstacles.point_clear(staging, APPROACH_EGO_RADIUS);
        let commit = obstacles.segment_clear(
            staging,
            ball_pos,
            APPROACH_COMMIT_RADIUS,
            APPROACH_CLEAR_STEP(),
        );
        let keeps_in = push_keeps_ball_in(ball_pos, u, field);
        let tier = match (keeps_in, clear, commit) {
            (true, true, true) => 0,
            (true, true, false) => 1,
            (true, false, _) => 2,
            (false, _, _) => 3,
        };
        let cost = (staging - player_pos).norm() - exit_bias * u.dot(&exit);
        (tier, cost)
    };

    let mut best: Option<(u8, f64, Vector2)> = None;
    for k in 0..N_APPROACH_SAMPLES {
        let theta = std::f64::consts::TAU * (k as f64) / (N_APPROACH_SAMPLES as f64);
        let u = Vector2::new(theta.cos(), theta.sin());
        let (tier, cost) = eval(u);
        let better = match best {
            None => true,
            Some((bt, bc, _)) => (tier, cost) < (bt, bc),
        };
        if better {
            best = Some((tier, cost, u));
        }
    }
    let (best_tier, best_cost, best_u) = best.expect("at least one sample");

    // Hysteresis: stick with the current side unless the new best is a better
    // tier or beats it by more than the margin.
    let dir = match *chosen_dir {
        Some(prev) if prev.norm() > 1e-6 => {
            let (pt, pc) = eval(prev);
            if best_tier < pt || (best_tier == pt && best_cost + APPROACH_HYSTERESIS() < pc) {
                best_u
            } else {
                prev.normalize()
            }
        }
        _ => best_u,
    };
    *chosen_dir = Some(dir);
    dir
}

/// Whether pushing the ball along `u` keeps it inside the field lines: for any
/// line the ball is within [`BALL_KEEPIN_MARGIN()`] of, the push must not have an
/// outward component (we never dribble the ball out — only the ball is "out"
/// past a line, so a robot in the run-off is fine, but the ball must stay in).
fn push_keeps_ball_in(ball: Vector2, u: Vector2, field: Option<&FieldGeometry>) -> bool {
    let Some(field) = field else {
        return true;
    };
    let hl = field.field_length / 2.0;
    let hw = field.field_width / 2.0;
    let lines = [
        (hw - ball.y, Vector2::new(0.0, -1.0)), // top touchline, inward = -y
        (hw + ball.y, Vector2::new(0.0, 1.0)),  // bottom touchline
        (hl - ball.x, Vector2::new(-1.0, 0.0)), // +x goal line
        (hl + ball.x, Vector2::new(1.0, 0.0)),  // -x goal line
    ];
    for (dist, inward) in lines {
        if dist < BALL_KEEPIN_MARGIN() && u.dot(&inward) < MIN_INWARD_PUSH() {
            return false;
        }
    }
    true
}

/// The capture axis for a tick: which direction to drive through the ball, the
/// point to stage behind, the heading to hold, and whether the velocity-aware
/// tail-catch is engaged. A fast, non-head-on ball leads the intercept and faces
/// travel; otherwise the static side-selector picks the side and the exit heading
/// is held.
struct CaptureAxis {
    dir: Vector2,
    aim_point: Vector2,
    heading: Angle,
    moving: bool,
}

/// Choose the capture axis for this tick. A ball rolling faster than
/// [`MOVING_BALL_SPEED()`] and not head-on is tail-caught: drive along its velocity,
/// stage behind the *predicted* intercept, and face travel. Otherwise fall back to
/// [`select_approach_dir`].
fn capture_axis(
    ctx: &SkillContext<'_>,
    ball_pos: Vector2,
    ball_vel: Vector2,
    player_pos: Vector2,
    exit_heading: Angle,
    exit_bias: f64,
    chosen_dir: &mut Option<Vector2>,
) -> CaptureAxis {
    if ball_vel.norm() > MOVING_BALL_SPEED() {
        if let Some(v_hat) = ball_vel.try_normalize(1e-6) {
            let head_on = (player_pos - ball_pos)
                .try_normalize(1e-6)
                .map(|to_robot| v_hat.dot(&to_robot) > HEAD_ON_COS())
                .unwrap_or(false);
            if !head_on {
                let t = intercept_time(ball_pos, ball_vel, player_pos);
                // Keep the side memory consistent if we later drop back to static.
                *chosen_dir = Some(v_hat);
                return CaptureAxis {
                    dir: v_hat,
                    aim_point: ball_pos + ball_vel * t,
                    heading: Angle::from_vector(v_hat),
                    moving: true,
                };
            }
        }
    }
    let dir = select_approach_dir(
        ctx,
        ball_pos,
        player_pos,
        exit_heading,
        exit_bias,
        chosen_dir,
    );
    CaptureAxis {
        dir,
        aim_point: ball_pos,
        heading: exit_heading,
        moving: false,
    }
}

/// Fixed-point estimate of the time (s) to intercept a constant-velocity ball.
fn intercept_time(ball: Vector2, vel: Vector2, robot: Vector2) -> f64 {
    let mut t = 0.0;
    for _ in 0..4 {
        let p = ball + vel * t;
        t = ((p - robot).norm() / INTERCEPT_ROBOT_SPEED()).min(MAX_INTERCEPT_TIME());
    }
    t
}

/// Lateral target (mm, in the plane perpendicular to `dir`) that places the ball
/// [`PICKUP_LATERAL_OFFSET()`] to the robot's left of dribbler center — i.e. the
/// robot center sits that far to the right of the ball line. `heading` is where
/// the dribbler/gear physically face (for a moving catch `heading` == `dir`).
pub(super) fn perp_target(heading: Angle, dir: Vector2) -> Vector2 {
    let h = heading.to_vector();
    let left = Vector2::new(-h.y, h.x);
    let target = -left * PICKUP_LATERAL_OFFSET();
    target - dir * target.dot(&dir)
}

/// Whether the robot is in the commit corridor (behind the ball, close, centered).
pub(super) fn committed(along: f64, perp: f64) -> bool {
    along < 0.0 && -along < COMMIT_DISTANCE() && perp < COMMIT_PERP()
}

/// Whether an engaged commit latch is held: the release-side Schmitt bands.
/// Wider than [`committed`] on every axis — perp up to the release band, along
/// tolerating both overshoot past the ball (the estimate collapses into the
/// hull during the press) and drift past the commit distance (no flap when
/// hovering right at the boundary).
pub(super) fn latched(along: f64, perp: f64) -> bool {
    along < COMMIT_ALONG_OVERSHOOT()
        && -along < COMMIT_DISTANCE() + COMMIT_ALONG_RELEASE()
        && perp < COMMIT_PERP_RELEASE()
}

/// Global velocity for the commit drive-through. Feeds forward the ball velocity
/// so the closing speed is *relative* (a moving ball no longer outruns the speed
/// law and the robot doesn't decelerate to a crawl beside it), drives the
/// remaining along-axis gap, and centers the ball on the offset contact point.
/// Only the proportional term is perp-gated; the MIN_SPEED floor always applies,
/// so the drive can neither stall in the drivetrain's stiction band nor reach
/// the ball below the speed that seats it on the breakbeam in one clean edge.
pub(super) fn commit_velocity(
    dir: Vector2,
    along: f64,
    perp_vec: Vector2,
    ball_vel: Vector2,
    pt: Vector2,
) -> Vector2 {
    let gate = (1.0 - perp_vec.norm() / GATE_PERP()).clamp(0.0, 1.0);
    let close = (-along) * APPROACH_GAIN() * gate + APPROACH_MIN_SPEED();
    ball_vel + dir * close - (perp_vec - pt) * LATERAL_GAIN()
}

/// Anti-stiction feed for the staging drive: a constant-speed pull toward the
/// staging point added on top of the position controller, cut off inside the
/// settle band. The terminal proportional profile alone settles into the
/// drivetrain's stiction band and parks the robot outside the commit gate; the
/// additive feed keeps the total command above breakaway until the robot is
/// close enough that resting anywhere in the band already satisfies the gate.
pub(super) fn staging_feed(player_pos: Vector2, staging: Vector2) -> Vector2 {
    let to = staging - player_pos;
    match to.try_normalize(STAGING_SETTLE_DIST) {
        Some(dir) => dir * STAGING_MIN_SPEED(),
        None => Vector2::zeros(),
    }
}

/// Staging point a fixed distance behind `aim_point` along `-dir`, shifted by the
/// lateral contact offset and clamped onto the playing surface.
pub(super) fn stage_point(
    aim_point: Vector2,
    dir: Vector2,
    pt: Vector2,
    field: Option<&FieldGeometry>,
) -> Vector2 {
    clamp_into_field(aim_point - dir * APPROACH_DISTANCE() + pt, field)
}

/// Whether the ball has escaped the commit corridor and the capture should bail.
/// A moving ball is allowed to roll *along* the axis (only lateral stray counts);
/// a static capture fails on any large ball move or an over-long drive.
fn commit_strayed(
    moving: bool,
    dir: Vector2,
    ball_pos: Vector2,
    commit_ball: Vector2,
    player_pos: Vector2,
    commit_pos: Vector2,
) -> bool {
    if moving {
        let d = ball_pos - commit_ball;
        (d - dir * d.dot(&dir)).norm() > BALL_STRAY_FAIL()
    } else {
        (ball_pos - commit_ball).norm() > BALL_MOVED_FAIL()
            || (player_pos - commit_pos).norm() > DRIVEN_FAIL()
    }
}

/// Clamp a point to the playing area, inset by [`STAGING_FIELD_MARGIN()`] from every
/// boundary. With no field geometry available the point is returned unchanged.
fn clamp_into_field(p: Vector2, field: Option<&FieldGeometry>) -> Vector2 {
    let Some(field) = field else {
        return p;
    };
    // Physical surface = field lines + run-off; robots may use the run-off.
    let max_x = (field.field_length / 2.0 + field.boundary_width - STAGING_FIELD_MARGIN()).max(0.0);
    let max_y = (field.field_width / 2.0 + field.boundary_width - STAGING_FIELD_MARGIN()).max(0.0);
    Vector2::new(p.x.clamp(-max_x, max_x), p.y.clamp(-max_y, max_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field() -> FieldGeometry {
        FieldGeometry::default() // 9000 × 6000
    }

    #[test]
    fn staging_in_runoff_past_the_line_is_allowed() {
        // The touchline is at y=3000 but the surface extends to 3000+boundary
        // (3300). A staging point in the run-off (y=3100) is legal — the robot
        // may stand there to get behind a ball pinned on the line — so it is NOT
        // pulled back inside the line.
        let p = Vector2::new(-2800.0, 3100.0);
        assert_eq!(clamp_into_field(p, Some(&field())), p);
    }

    #[test]
    fn staging_past_physical_edge_is_clamped_to_the_surface() {
        // Past the physical edge (3300) the staging point is pulled back onto the
        // surface, inset by the margin, so the robot doesn't drive off it.
        let clamped = clamp_into_field(Vector2::new(-2800.0, 3400.0), Some(&field()));
        assert!((clamped.y - (3000.0 + 300.0 - STAGING_FIELD_MARGIN())).abs() < 1e-6);
        assert!((clamped.x - (-2800.0)).abs() < 1e-6); // x already inside, untouched
    }

    #[test]
    fn staging_inside_field_is_unchanged() {
        let p = Vector2::new(1000.0, -500.0);
        assert_eq!(clamp_into_field(p, Some(&field())), p);
    }

    #[test]
    fn commit_drive_floor_survives_a_closed_gate() {
        // At perp == GATE_PERP the proportional term is fully gated, but the
        // MIN_SPEED floor must still drive the robot forward (the old fully-gated
        // law stalled here and the ball never seated on the breakbeam).
        let dir = Vector2::new(1.0, 0.0);
        let perp_vec = Vector2::new(0.0, GATE_PERP());
        let v = commit_velocity(dir, -150.0, perp_vec, Vector2::zeros(), Vector2::zeros());
        assert!((v.dot(&dir) - APPROACH_MIN_SPEED()).abs() < 1e-9, "{v:?}");
    }

    #[test]
    fn latch_release_has_hysteresis_on_both_axes() {
        // Entry stays strict…
        assert!(committed(-200.0, COMMIT_PERP() - 1.0));
        assert!(!committed(10.0, 0.0)); // never enter past the ball
        assert!(!committed(-(COMMIT_DISTANCE() + 1.0), 0.0));
        // …while an engaged latch tolerates overshoot past the ball (estimate
        // collapse), drift past the commit distance, and the wider perp band.
        assert!(latched(COMMIT_ALONG_OVERSHOOT() - 1.0, 0.0));
        assert!(!latched(COMMIT_ALONG_OVERSHOOT() + 1.0, 0.0));
        assert!(latched(-(COMMIT_DISTANCE() + COMMIT_ALONG_RELEASE() - 1.0), 0.0));
        assert!(!latched(-(COMMIT_DISTANCE() + COMMIT_ALONG_RELEASE() + 1.0), 0.0));
        assert!(latched(-200.0, COMMIT_PERP_RELEASE() - 1.0));
        assert!(!latched(-200.0, COMMIT_PERP_RELEASE() + 1.0));
    }

    #[test]
    fn staging_feed_pulls_until_the_settle_band() {
        let staging = Vector2::new(1000.0, 0.0);
        // Outside the settle band: constant-speed pull toward staging.
        let feed = staging_feed(Vector2::new(900.0, 0.0), staging);
        assert!((feed - Vector2::new(STAGING_MIN_SPEED(), 0.0)).norm() < 1e-9);
        // Inside it: off, so the robot can settle (and the band is inside the
        // commit gate, so resting here is already latch-able).
        let inside = staging_feed(Vector2::new(1000.0 - STAGING_SETTLE_DIST + 1.0, 0.0), staging);
        assert_eq!(inside, Vector2::zeros());
    }
}
