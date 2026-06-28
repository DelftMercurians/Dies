use std::time::{Duration, Instant};

use dies_core::{Angle, FieldGeometry, Vector2, PLAYER_RADIUS};
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
/// Inset (mm) from the *physical* field edge (touchline/goal line **plus** the
/// run-off `boundary_width`) that the staging point is kept inside. Robots may
/// legally stand in the run-off, so the staging point only has to stay on the
/// playing surface — clamping to the line itself would forbid getting *behind* a
/// ball pinned on the line and the robot would stall short of it (only the ball
/// is "out" past the line; the rescue-inward heading keeps the ball in). Covers
/// the robot radius plus a buffer so the body stays on the surface.
const STAGING_FIELD_MARGIN: f64 = 130.0;

// ── heading-free approach-side selection ─────────────────────────────────────
// A normal capture is not given a fixed approach heading: the skill picks which
// side of the ball to take it from, choosing the fastest reachable, obstacle-free
// approach point, biased toward a desired *exit* direction (so the capture also
// roughly sets up the next action). `target_heading` carries that soft exit bias
// for a capture (and remains the hard strike axis for an instant-kick release).
/// Number of candidate push directions sampled around the ball.
const N_APPROACH_SAMPLES: usize = 24;
/// Ego clearance required at a candidate staging point (robot radius + buffer).
const APPROACH_EGO_RADIUS: f64 = PLAYER_RADIUS + 20.0;
/// Clearance required along the commit corridor (staging point → ball). Slightly
/// tighter so only a real blocker between us and the ball rejects an approach.
const APPROACH_COMMIT_RADIUS: f64 = PLAYER_RADIUS;
/// Sampling step (mm) for the commit-corridor clearance check.
const APPROACH_CLEAR_STEP: f64 = 60.0;
/// Value (mm of approach distance) of a perfectly exit-aligned push. Trades off
/// against "closest staging point": a well-aimed capture is worth approaching
/// from a bit farther away. Soft — boundary safety is a hard constraint below.
const EXIT_BIAS_WEIGHT: f64 = 600.0;
/// The chosen side is kept unless a candidate beats it by more than this (mm),
/// so the approach side doesn't flip-flop on noise / orbiting opponents.
const APPROACH_HYSTERESIS: f64 = 200.0;
/// A ball within this distance (mm) of a field line forbids any approach whose
/// push has an outward component there — never dribble the ball out (only the
/// ball is "out" past the line; this is the touchline/goal-line rescue, now
/// folded into the generic selector instead of a special heading).
const BALL_KEEPIN_MARGIN: f64 = 350.0;
/// Tolerance on the inward-push constraint (slightly negative so a near-parallel
/// push along the line is still allowed).
const MIN_INWARD_PUSH: f64 = -0.05;

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
            chosen_dir: None,
            instant_kick,
            kick_ball_pos: None,
            armed_at: None,
        }
    }

    pub fn set_target_heading(&mut self, target_heading: Angle) {
        self.target_heading = target_heading;
    }

    /// Pick the push direction (unit) for a capture: sample candidate sides around
    /// the ball, score each by how quickly the robot can reach the staging point
    /// minus a reward for matching the desired exit direction, reject sides that
    /// are blocked or would push the ball out, and keep the previous choice unless
    /// clearly beaten (hysteresis). The robot stages behind the ball along `-dir`
    /// and drives through it, so `dir` is also the direction the ball is pushed.
    fn select_approach_dir(
        &mut self,
        ctx: &SkillContext<'_>,
        ball_pos: Vector2,
        player_pos: Vector2,
    ) -> Vector2 {
        let exit = self.target_heading.to_vector();
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
        let dir = match self.chosen_dir {
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
        self.chosen_dir = Some(dir);
        dir
    }
}

/// Whether pushing the ball along `u` keeps it inside the field lines: for any
/// line the ball is within [`BALL_KEEPIN_MARGIN`] of, the push must not have an
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
        if dist < BALL_KEEPIN_MARGIN && u.dot(&inward) < MIN_INWARD_PUSH {
            return false;
        }
    }
    true
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
        // Instant-kick uses the commanded strike axis directly. A capture instead
        // chooses its own approach side (fastest reachable & obstacle-free, biased
        // toward the commanded exit direction), re-evaluated each tick.
        let dir = if self.instant_kick {
            self.target_heading.to_vector()
        } else {
            self.select_approach_dir(&ctx, ball_pos, player_pos)
        };

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
            // Stage a fixed distance *behind* the ball along the approach axis, but
            // never outside the field: a ball pinned against a boundary would put
            // the staging point off the pitch, so the robot would drive out (and
            // carry the ball over the line). Clamp it into the playing area.
            let stage = ball_pos - dir * APPROACH_DISTANCE;
            input.with_position(clamp_into_field(stage, ctx.world.field_geom.as_ref()));
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
fn clamp_into_field(p: Vector2, field: Option<&FieldGeometry>) -> Vector2 {
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
