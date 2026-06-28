//! DribbleShoot skill — aim a captured ball at a target point, then kick.
//!
//! Given the target *point* the skill first chooses where to launch the ball
//! from: if the current spot already has a clear, reachable kicking pose toward
//! the target it aims in place (orbiting the ball, dribbler pressed); if the ball
//! is jammed (e.g. against a boundary, or no on-surface pose behind it) it first
//! dribbles the ball a short distance to a reachable launch point, then aims.
//! Orbit / short-carry / turn-with-ball all fall out of *where the launch point
//! is* relative to the current ball — one "drive to kicking pose" primitive, not
//! a mode selector.

use std::time::{Duration, Instant};

use dies_core::{Angle, FieldGeometry, Vector2, PLAYER_RADIUS};
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
/// flicker during the slide doesn't abort.
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

// ── launch-point selection ───────────────────────────────────────────────────
/// Clearance required at a candidate kicking pose (robot behind the launch point).
const LAUNCH_POSE_EGO: f64 = PLAYER_RADIUS;
/// Inset from the physical field edge a kicking pose must keep (robot stays on
/// the surface). Matches the pickup staging clamp.
const LAUNCH_SURFACE_MARGIN: f64 = 130.0;
/// A ball this close to a field line is "jammed" — repositioned inward before
/// aiming, so we never try to aim/kick with the ball pinned on the boundary.
const LAUNCH_BOUNDARY_MARGIN: f64 = 350.0;
/// Carry distances (mm) tried when searching for a reachable launch point. Short
/// — repositioning is a small nudge off a bad spot, not a dribble across field.
const CARRY_STEPS: [f64; 3] = [300.0, 500.0, 700.0];
/// Lateral fan (radians) tried around the inward direction at each carry step.
const CARRY_FAN: [f64; 5] = [0.0, 0.5, -0.5, 1.0, -1.0];
/// Ball within this distance of the chosen launch point → done repositioning.
const REPOSITION_ARRIVE: f64 = 90.0;
/// Carry motion limits (reuse the proven `Dribble` law: hold the ball, drive to
/// the kicking pose under an acceleration cap so it isn't shaken loose).
const CARRY_ACCEL_LIMIT: f64 = 500.0;
const CARRY_ANGULAR_LIMIT: f64 = 1.5;

#[derive(Clone, Copy, PartialEq, Eq)]
enum AimState {
    /// Dribble the ball to the chosen launch point.
    Reposition,
    /// Orbit the (now well-placed) ball to align the shot, then commit.
    Aiming,
    Kicking,
    /// Kick commanded; waiting to confirm the ball actually left the dribbler.
    Verifying,
}

pub struct DribbleShootSkill {
    target: Vector2,
    status: SkillStatus,
    state: AimState,
    /// Where to launch the ball from. Chosen once on the first tick and held, so
    /// the aim phase doesn't oscillate between repositioning and orbiting.
    launch: Option<Vector2>,
    start: Option<Instant>,
    /// Ball position recorded the instant the kick was commanded.
    kick_ball_pos: Option<Vector2>,
    /// When the kick was commanded (start of the verification window).
    kick_time: Option<Instant>,
}

impl DribbleShootSkill {
    pub fn new(target: Vector2) -> Self {
        Self {
            target,
            status: SkillStatus::Running,
            state: AimState::Aiming,
            launch: None,
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
        if let SkillCommand::DribbleShoot { target } = command {
            self.target = *target;
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

        // Choose the launch point once, then decide whether we still need to carry
        // the ball there or can start aiming.
        let launch = *self
            .launch
            .get_or_insert_with(|| choose_launch(&ctx, ball_pos, self.target));
        if self.state == AimState::Aiming && (ball_pos - launch).norm() > REPOSITION_ARRIVE {
            // Launch point is away from the ball → carry it there first.
            self.state = AimState::Reposition;
        }

        // Shot heading from the *current* ball toward the target.
        let to_target = self.target - ball_pos;
        let target_heading = Angle::from_vector(to_target);

        let mut input = PlayerControlInput::new();
        input.with_dribbling(DRIBBLER_SPEED);

        match self.state {
            // ── Reposition: dribble the ball to the launch point ────────────
            AimState::Reposition => {
                if !ctx.player.has_ball && (player_pos - ball_pos).norm() > LOST_BALL_DISTANCE {
                    log::warn!("dribble_shoot: lost ball while repositioning");
                    self.status = SkillStatus::Failed;
                    return SkillProgress::failure();
                }
                // Done once the ball sits at the launch point — hand off to aim.
                if (ball_pos - launch).norm() < REPOSITION_ARRIVE {
                    self.state = AimState::Aiming;
                    return SkillProgress::Continue(input);
                }
                // Carry: face the target and drive to the kicking pose behind the
                // launch point, holding the ball under an acceleration cap.
                let axis = (self.target - launch)
                    .try_normalize(1e-6)
                    .unwrap_or_else(|| Vector2::new(1.0, 0.0));
                let pose = launch - axis * BALL_TO_ROBOT_DISTANCE;
                input.with_position(pose);
                input.with_yaw(target_heading);
                input.with_acceleration_limit(CARRY_ACCEL_LIMIT);
                input.with_angular_speed_limit(CARRY_ANGULAR_LIMIT);
            }

            // ── Aiming: orbit the ball to the shot axis, then kick ──────────
            AimState::Aiming => {
                input.avoid_robots = false;
                input.with_angular_speed_limit(1000.0);

                let r = player_pos - ball_pos;
                if r.norm() > LOST_BALL_DISTANCE {
                    log::warn!("dribble_shoot: lost ball");
                    self.status = SkillStatus::Failed;
                    return SkillProgress::failure();
                }

                // A robot in the shoot corridor must not be kicked into — but this
                // is a transient positional condition (the keeper is always in
                // front of the goal), so it gates the *kick*, not the skill: keep
                // aiming/holding while blocked and wait for the lane to clear.
                let blocked = lane_blocked(&ctx, ball_pos, target_heading);
                let err = target_heading - ctx.player.yaw;
                if err.abs() < YAW_TOLERANCE && !blocked && ctx.player.has_ball {
                    self.state = AimState::Kicking;
                }

                const RADIUS_KP: f64 = 2.0;
                let r_hat = r.normalize();
                let tangent = Vector2::new(-r_hat.y, r_hat.x); // CCW
                let v_rad = -RADIUS_KP * (r.norm() - BALL_TO_ROBOT_DISTANCE) * r_hat;
                let v_tan = if blocked {
                    Vector2::zeros()
                } else {
                    let speed = (ORBIT_GAIN * err.abs()).clamp(MIN_ORBIT_SPEED, ORBIT_SPEED);
                    err.signum() * speed * tangent
                };
                input.velocity = Velocity::global(v_tan + v_rad);
                input.with_yaw(Angle::from_vector(-r)); // face ball
            }

            AimState::Kicking => {
                input.with_yaw(target_heading);
                input.with_kicker(KickerControlInput::Kick);
                input.kick_speed = Some(KICK_SPEED);
                input.with_dribbling(0.0);
                self.kick_ball_pos = Some(ball_pos);
                self.kick_time = Some(Instant::now());
                self.state = AimState::Verifying;
            }

            AimState::Verifying => {
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
                input.with_yaw(target_heading);
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
            AimState::Reposition => "repositioning",
            AimState::Aiming => "orbiting to aim",
            AimState::Kicking => "kicking",
            AimState::Verifying => "verifying kick",
        };
        format!("{phase} → ({:.0}, {:.0})", self.target.x, self.target.y)
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
