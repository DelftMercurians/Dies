use std::time::Duration;

use dies_core::{Angle, FieldGeometry, Vector2, PLAYER_RADIUS};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{KickerControlInput, PlayerControlInput};

pub(crate) const DRIBBLER_SPEED: f64 = 0.6;
pub(crate) const APPROACH_DISTANCE: f64 = 200.0;
pub(crate) const COMMIT_DISTANCE: f64 = 280.0;
pub(crate) const COMMIT_PERP: f64 = 30.0;
pub(crate) const GATE_PERP: f64 = 80.0;
pub(crate) const APPROACH_GAIN: f64 = 3.0;
pub(crate) const APPROACH_MIN_SPEED: f64 = 200.0;
pub(crate) const LATERAL_GAIN: f64 = 4.0;
pub(crate) const APPROACH_CARE: f64 = -0.6;
pub(crate) const BALL_MOVED_FAIL: f64 = 100.0;
pub(crate) const DRIVEN_FAIL: f64 = 250.0;
/// Inset (mm) from the *physical* field edge (touchline/goal line **plus** the
/// run-off `boundary_width`) that the staging point is kept inside. Robots may
/// legally stand in the run-off, so the staging point only has to stay on the
/// playing surface — clamping to the line itself would forbid getting *behind* a
/// ball pinned on the line and the robot would stall short of it (only the ball
/// is "out" past the line; the rescue-inward heading keeps the ball in). Covers
/// the robot radius plus a buffer so the body stays on the surface.
pub(crate) const STAGING_FIELD_MARGIN: f64 = 130.0;

// ── heading-free approach-side selection ─────────────────────────────────────
// A normal capture is not given a fixed approach heading: the skill picks which
// side of the ball to take it from, choosing the fastest reachable, obstacle-free
// approach point, biased toward a desired *exit* direction (so the capture also
// roughly sets up the next action). `target_heading` carries that soft exit bias
// for a capture (and remains the hard strike axis for an instant-kick release).
/// Number of candidate push directions sampled around the ball.
pub(crate) const N_APPROACH_SAMPLES: usize = 24;
/// Ego clearance required at a candidate staging point (robot radius + buffer).
pub(crate) const APPROACH_EGO_RADIUS: f64 = PLAYER_RADIUS + 20.0;
/// Clearance required along the commit corridor (staging point → ball). Slightly
/// tighter so only a real blocker between us and the ball rejects an approach.
pub(crate) const APPROACH_COMMIT_RADIUS: f64 = PLAYER_RADIUS;
/// Sampling step (mm) for the commit-corridor clearance check.
pub(crate) const APPROACH_CLEAR_STEP: f64 = 60.0;
/// Value (mm of approach distance) of a perfectly exit-aligned push. Trades off
/// against "closest staging point": a well-aimed capture is worth approaching
/// from a bit farther away. Soft — boundary safety is a hard constraint below.
pub(crate) const EXIT_BIAS_WEIGHT: f64 = 600.0;
/// The chosen side is kept unless a candidate beats it by more than this (mm),
/// so the approach side doesn't flip-flop on noise / orbiting opponents.
pub(crate) const APPROACH_HYSTERESIS: f64 = 200.0;
/// A ball within this distance (mm) of a field line forbids any approach whose
/// push has an outward component there — never dribble the ball out (only the
/// ball is "out" past the line; this is the touchline/goal-line rescue, now
/// folded into the generic selector instead of a special heading).
pub(crate) const BALL_KEEPIN_MARGIN: f64 = 350.0;
/// Tolerance on the inward-push constraint (slightly negative so a near-parallel
/// push along the line is still allowed).
pub(crate) const MIN_INWARD_PUSH: f64 = -0.05;

// ── instant-kick (reflex strike-through) tuning ──────────────────────────────
/// Ball displacement along the kick axis that counts as "the reflex connected".
pub(crate) const KICK_DEPART_DIST: f64 = 100.0;
/// Ball speed that also counts as departure (filtered estimate; confirms).
pub(crate) const KICK_DEPART_SPEED: f64 = 1000.0;
/// How long to wait after arming the reflex for the ball to depart before
/// declaring a whiff and failing (so a dribbler-on hold can't linger and
/// accumulate double-touch contact).
pub(crate) const REFLEX_TIMEOUT: Duration = Duration::from_millis(600);

// ── moving-ball tail-catch + contact offset ──────────────────────────────────
/// Lateral contact offset (mm): the ball is centered this far to the robot's
/// LEFT of dribbler center so it rests clear of the dribbler-drive gear on the
/// right. Applied in the robot's heading frame (for a moving catch the heading is
/// the drive direction, so heading-left and axis-left coincide).
pub(crate) const PICKUP_LATERAL_OFFSET: f64 = 2.5;
/// Ball speed (mm/s) above which a capture switches from static side-selection to
/// a velocity-aware tail-catch: lead the intercept and feed-forward the ball
/// speed so the robot matches pace instead of decelerating into a stern chase.
/// Below this the ball is effectively static and the side-selector is better.
pub(crate) const MOVING_BALL_SPEED: f64 = 300.0;
/// Cosine of the half-cone within which a ball rolling *toward* the robot is
/// treated as head-on and left to `Receive` (a tail-catch would loop around).
pub(crate) const HEAD_ON_COS: f64 = 0.6; // ~53°
/// Nominal robot speed (mm/s) for the intercept-time estimate. Conservative
/// (below true max) so the aim point lands slightly short rather than past the
/// ball.
pub(crate) const INTERCEPT_ROBOT_SPEED: f64 = 2500.0;
/// Cap (s) on the intercept look-ahead.
pub(crate) const MAX_INTERCEPT_TIME: f64 = 1.5;
/// Lateral deviation (mm) of a moving ball from the commit axis that counts as
/// the ball squirting out of the corridor. Replaces the static `BALL_MOVED_FAIL`
/// while the ball is legitimately rolling *along* the axis.
pub(crate) const BALL_STRAY_FAIL: f64 = 150.0;

pub struct PickupBallSkill {
    /// Capture: a soft *exit-direction* bias for which side to take the ball from.
    /// Instant-kick: the hard strike axis.
    target_heading: Angle,
    status: SkillStatus,
    commit_pos: Option<Vector2>,
    commit_ball: Option<Vector2>,
    /// The currently committed push direction (unit), for approach-side
    /// hysteresis. `None` until the first selection.
    chosen_dir: Option<Vector2>,
    /// Strike-through release: arm a reflex kick instead of capturing.
    instant_kick: bool,
    /// Ball position when the reflex was first armed (for the departure check).
    kick_ball_pos: Option<Vector2>,
    /// World time (`t_received`, seconds) the reflex was first armed (for the
    /// whiff timeout). Sim-clock based so the timeout is deterministic and works
    /// under faster-than-realtime sim — wall-clock would be wrong. See CLAUDE.md.
    armed_at: Option<f64>,
}

impl PickupBallSkill {
    pub fn new(target_heading: Angle, instant_kick: bool) -> Self {
        Self {
            target_heading,
            status: SkillStatus::Running,
            commit_pos: None,
            commit_ball: None,
            chosen_dir: None,
            instant_kick,
            kick_ball_pos: None,
            armed_at: None,
        }
    }

    pub fn set_target_heading(&mut self, target_heading: Angle) {
        self.target_heading = target_heading;
    }
}

/// Pick the push direction (unit) for a capture (shared by `PickupBallSkill` and
/// the unified `HandleBallSkill`). Samples candidate sides around the ball, scores
/// each by how quickly the robot reaches the staging point minus a reward for
/// matching `exit_heading`, rejects sides that are blocked or would push the ball
/// out, and keeps `chosen_dir` unless clearly beaten (hysteresis). The robot
/// stages behind the ball along `-dir` and drives through it, so `dir` is also the
/// direction the ball is pushed. `chosen_dir` is updated in place for hysteresis.
pub(crate) fn select_approach_dir(
    ctx: &SkillContext<'_>,
    ball_pos: Vector2,
    player_pos: Vector2,
    exit_heading: Angle,
    chosen_dir: &mut Option<Vector2>,
) -> Vector2 {
    {
        let exit = exit_heading.to_vector();
        let field = ctx.world.field_geom.as_ref();
        let obstacles = &ctx.obstacles;

        // (tier, cost) for a push direction `u`; lower is better, tier dominates.
        // Tier ladder relaxes the soft constraints so a side is always returned:
        // 0 = clear staging + clear commit corridor + keeps ball in;
        // 1 = keeps ball in + clear staging (corridor blocked, e.g. a steal);
        // 2 = keeps ball in only; 3 = nothing satisfied (last resort).
        let eval = |u: Vector2| -> (u8, f64) {
            let staging = clamp_into_field(ball_pos - u * APPROACH_DISTANCE, field);
            let clear = obstacles.point_clear(staging, APPROACH_EGO_RADIUS);
            let commit = obstacles.segment_clear(
                staging,
                ball_pos,
                APPROACH_COMMIT_RADIUS,
                APPROACH_CLEAR_STEP,
            );
            let keeps_in = push_keeps_ball_in(ball_pos, u, field);
            let tier = match (keeps_in, clear, commit) {
                (true, true, true) => 0,
                (true, true, false) => 1,
                (true, false, _) => 2,
                (false, _, _) => 3,
            };
            let cost = (staging - player_pos).norm() - EXIT_BIAS_WEIGHT * u.dot(&exit);
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
                if best_tier < pt || (best_tier == pt && best_cost + APPROACH_HYSTERESIS < pc) {
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
}

/// Whether pushing the ball along `u` keeps it inside the field lines: for any
/// line the ball is within [`BALL_KEEPIN_MARGIN`] of, the push must not have an
/// outward component (we never dribble the ball out — only the ball is "out"
/// past a line, so a robot in the run-off is fine, but the ball must stay in).
pub(crate) fn push_keeps_ball_in(ball: Vector2, u: Vector2, field: Option<&FieldGeometry>) -> bool {
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
        if dist < BALL_KEEPIN_MARGIN && u.dot(&inward) < MIN_INWARD_PUSH {
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
pub(crate) struct CaptureAxis {
    pub dir: Vector2,
    pub aim_point: Vector2,
    pub heading: Angle,
    pub moving: bool,
}

/// Choose the capture axis for this tick (shared by `PickupBallSkill` and
/// `HandleBallSkill`). A ball rolling faster than [`MOVING_BALL_SPEED`] and not
/// head-on is tail-caught: drive along its velocity, stage behind the *predicted*
/// intercept, and face travel. Otherwise fall back to [`select_approach_dir`].
pub(crate) fn capture_axis(
    ctx: &SkillContext<'_>,
    ball_pos: Vector2,
    ball_vel: Vector2,
    player_pos: Vector2,
    exit_heading: Angle,
    chosen_dir: &mut Option<Vector2>,
) -> CaptureAxis {
    if ball_vel.norm() > MOVING_BALL_SPEED {
        if let Some(v_hat) = ball_vel.try_normalize(1e-6) {
            let head_on = (player_pos - ball_pos)
                .try_normalize(1e-6)
                .map(|to_robot| v_hat.dot(&to_robot) > HEAD_ON_COS)
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
    let dir = select_approach_dir(ctx, ball_pos, player_pos, exit_heading, chosen_dir);
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
        t = ((p - robot).norm() / INTERCEPT_ROBOT_SPEED).min(MAX_INTERCEPT_TIME);
    }
    t
}

/// Lateral target (mm, in the plane perpendicular to `dir`) that places the ball
/// [`PICKUP_LATERAL_OFFSET`] to the robot's left of dribbler center — i.e. the
/// robot center sits that far to the right of the ball line. `heading` is where
/// the dribbler/gear physically face (for a moving catch `heading` == `dir`).
pub(crate) fn perp_target(heading: Angle, dir: Vector2) -> Vector2 {
    let h = heading.to_vector();
    let left = Vector2::new(-h.y, h.x);
    let target = -left * PICKUP_LATERAL_OFFSET;
    target - dir * target.dot(&dir)
}

/// Whether the robot is in the commit corridor (behind the ball, close, centered).
pub(crate) fn committed(along: f64, perp: f64) -> bool {
    along < 0.0 && -along < COMMIT_DISTANCE && perp < COMMIT_PERP
}

/// Global velocity for the commit drive-through. Feeds forward the ball velocity
/// so the closing speed is *relative* (a moving ball no longer outruns the speed
/// law and the robot doesn't decelerate to a crawl beside it), drives the
/// remaining along-axis gap, and centers the ball on the offset contact point.
pub(crate) fn commit_velocity(
    dir: Vector2,
    along: f64,
    perp_vec: Vector2,
    ball_vel: Vector2,
    pt: Vector2,
) -> Vector2 {
    let gate = (1.0 - perp_vec.norm() / GATE_PERP).clamp(0.0, 1.0);
    let close = (-along) * APPROACH_GAIN + APPROACH_MIN_SPEED;
    ball_vel + dir * close * gate - (perp_vec - pt) * LATERAL_GAIN
}

/// Staging point a fixed distance behind `aim_point` along `-dir`, shifted by the
/// lateral contact offset and clamped onto the playing surface.
pub(crate) fn stage_point(
    aim_point: Vector2,
    dir: Vector2,
    pt: Vector2,
    field: Option<&FieldGeometry>,
) -> Vector2 {
    clamp_into_field(aim_point - dir * APPROACH_DISTANCE + pt, field)
}

/// Whether the ball has escaped the commit corridor and the capture should bail.
/// A moving ball is allowed to roll *along* the axis (only lateral stray counts);
/// a static capture fails on any large ball move or an over-long drive.
pub(crate) fn commit_strayed(
    moving: bool,
    dir: Vector2,
    ball_pos: Vector2,
    commit_ball: Vector2,
    player_pos: Vector2,
    commit_pos: Vector2,
) -> bool {
    if moving {
        let d = ball_pos - commit_ball;
        (d - dir * d.dot(&dir)).norm() > BALL_STRAY_FAIL
    } else {
        (ball_pos - commit_ball).norm() > BALL_MOVED_FAIL
            || (player_pos - commit_pos).norm() > DRIVEN_FAIL
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
        let ball_vel = ball.velocity.xy();
        let player_pos = ctx.player.position;
        // Instant-kick keeps the commanded strike axis (no leading). A capture
        // chooses its axis: a fast, non-head-on ball is tail-caught with a led
        // intercept and faces travel; otherwise the side-selector picks the
        // approach side (fastest reachable, biased toward the exit direction).
        let axis = if self.instant_kick {
            CaptureAxis {
                dir: self.target_heading.to_vector(),
                aim_point: ball_pos,
                heading: self.target_heading,
                moving: false,
            }
        } else {
            capture_axis(
                &ctx,
                ball_pos,
                ball_vel,
                player_pos,
                self.target_heading,
                &mut self.chosen_dir,
            )
        };

        let rel = player_pos - ball_pos;
        let along = rel.dot(&axis.dir);
        let perp_vec = rel - axis.dir * along;
        let perp = perp_vec.norm();
        let pt = perp_target(axis.heading, axis.dir);

        let mut input = PlayerControlInput::new();
        input.with_yaw(axis.heading);
        input.with_dribbling(DRIBBLER_SPEED);

        if committed(along, perp) {
            input.avoid_ball = false;
            input.add_global_velocity(commit_velocity(axis.dir, along, perp_vec, ball_vel, pt));

            if self.instant_kick {
                // Arm a firmware reflex kick: it fires the instant the ball hits
                // the breakbeam during the drive-through (dribbler-on pulls it
                // onto the beam), so the ball is struck on contact, never held.
                input.with_kicker(KickerControlInput::ReflexKick);
                let kick_ball = *self.kick_ball_pos.get_or_insert(ball_pos);
                let armed_at = *self.armed_at.get_or_insert(ctx.world.t_received);

                // Success = ball departed along the kick axis (the reflex fired).
                // Checked BEFORE any ball-moved failure, and gated to the kick
                // direction so a lateral knock doesn't read as a clean release.
                let along_depart = (ball_pos - kick_ball).dot(&axis.dir);
                if along_depart > KICK_DEPART_DIST || ball.velocity.norm() > KICK_DEPART_SPEED {
                    self.status = SkillStatus::Succeeded;
                    return SkillProgress::success();
                }
                // Whiff: armed but the ball never left → fail so the driver
                // re-stages a fresh approach instead of lingering in contact.
                if ctx.world.t_received - armed_at > REFLEX_TIMEOUT.as_secs_f64() {
                    log::warn!("pickup_ball: reflex kick did not connect");
                    self.status = SkillStatus::Failed;
                    return SkillProgress::failure();
                }
            } else {
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
                    self.status = SkillStatus::Failed;
                    return SkillProgress::failure();
                }
            }
        } else {
            // Stage a fixed distance *behind* the (predicted) ball along the
            // approach axis, but never outside the field: a ball pinned against a
            // boundary would put the staging point off the pitch, so the robot
            // would drive out (and carry the ball over the line). Clamp it in.
            input.with_position(stage_point(
                axis.aim_point,
                axis.dir,
                pt,
                ctx.world.field_geom.as_ref(),
            ));
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

/// Clamp a point to the playing area, inset by [`STAGING_FIELD_MARGIN`] from every
/// boundary. With no field geometry available the point is returned unchanged.
pub(crate) fn clamp_into_field(p: Vector2, field: Option<&FieldGeometry>) -> Vector2 {
    let Some(field) = field else {
        return p;
    };
    // Physical surface = field lines + run-off; robots may use the run-off.
    let max_x = (field.field_length / 2.0 + field.boundary_width - STAGING_FIELD_MARGIN).max(0.0);
    let max_y = (field.field_width / 2.0 + field.boundary_width - STAGING_FIELD_MARGIN).max(0.0);
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
        assert!((clamped.y - (3000.0 + 300.0 - STAGING_FIELD_MARGIN)).abs() < 1e-6);
        assert!((clamped.x - (-2800.0)).abs() < 1e-6); // x already inside, untouched
    }

    #[test]
    fn staging_inside_field_is_unchanged() {
        let p = Vector2::new(1000.0, -500.0);
        assert_eq!(clamp_into_field(p, Some(&field())), p);
    }
}
