//! GoToBounded skill — aggressive direct-velocity positioning inside a region.
//!
//! Built for the goalkeeper's guard arc. Unlike `GoToPos` (which feeds a target
//! position through the planner + path follower), this computes the translational
//! velocity itself — a snappy proportional drive toward the target — and bypasses
//! the planner/follower. Containment is delegated to the player controller's
//! no-overshoot velocity envelope via the `bounds` it sets on the control input;
//! the aggressive control profile (gains, speed/accel caps, ORCA-off) lives here,
//! so no per-frame control overrides cross the strategy IPC.

use dies_core::{Angle, MotionBounds, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{PlayerControlInput, Velocity};

/// See [`GoToPosSkill`](super::GoToPosSkill); must clear the path-follower deadband.
const DEFAULT_POS_TOLERANCE: f64 = 30.0;
const DEFAULT_VEL_TOLERANCE: f64 = 50.0;

/// Proportional approach gain [1/s]: commanded speed = `KP × distance`, capped at
/// the speed limit. ~`base_approach_kp (4.0) × (1 + aggressiveness 1.5)` — the old
/// keeper snappiness. Velocity ∝ distance is over-damped, so it settles on the
/// target without overshooting it; the envelope handles the region edges.
const KEEPER_APPROACH_KP: f64 = 10.0;
/// Keeper speed cap [mm/s]. Tight, predictable line motion.
const KEEPER_SPEED_LIMIT: f64 = 2500.0;
/// Keeper acceleration cap [mm/s²] — deliberately harder than the global default
/// (4000 accel / 6000 decel) so the keeper reacts more sharply. Caps both ramp
/// directions and drives the bounded-region brake envelope.
const KEEPER_ACCEL: f64 = 8000.0;

/// Aggressive bounded-region positioning skill (goalkeeper guard).
pub struct GoToBoundedSkill {
    target_pos: Vector2,
    target_heading: Option<Angle>,
    bounds: MotionBounds,
    pos_tolerance: f64,
    vel_tolerance: f64,
    status: SkillStatus,
}

impl GoToBoundedSkill {
    pub fn new(target_pos: Vector2, target_heading: Option<Angle>, bounds: MotionBounds) -> Self {
        Self {
            target_pos,
            target_heading,
            bounds,
            pos_tolerance: DEFAULT_POS_TOLERANCE,
            vel_tolerance: DEFAULT_VEL_TOLERANCE,
            status: SkillStatus::Running,
        }
    }
}

impl ExecutableSkill for GoToBoundedSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::GoToBounded { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::GoToBounded {
            position,
            heading,
            bounds,
        } = command
        {
            self.target_pos = *position;
            self.target_heading = *heading;
            self.bounds = *bounds;
            if matches!(self.status, SkillStatus::Succeeded | SkillStatus::Failed) {
                self.status = SkillStatus::Running;
            }
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        let position = ctx.player.position;
        let distance = (self.target_pos - position).norm();
        let speed = ctx.player.velocity.norm();

        if distance < self.pos_tolerance && speed < self.vel_tolerance {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }
        self.status = SkillStatus::Running;

        // Direct proportional velocity toward the target (bypasses path follower).
        let to = self.target_pos - position;
        let v = if distance > 1.0e-6 {
            (to / distance) * (KEEPER_APPROACH_KP * distance).min(KEEPER_SPEED_LIMIT)
        } else {
            Vector2::zeros()
        };

        let mut input = PlayerControlInput::new();
        // No position target → the planner/path-follower are skipped; the velocity
        // below is the primary command. The envelope (via `bounds`) keeps it in.
        input.velocity = Velocity::Global(v);
        input.bounds = Some(self.bounds);
        input.speed_limit = Some(KEEPER_SPEED_LIMIT);
        input.acceleration_limit = Some(KEEPER_ACCEL);
        input.avoid_robots = false;
        if let Some(heading) = self.target_heading {
            input.with_yaw(heading);
        }

        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "GoToBounded"
    }

    fn description(&self) -> String {
        format!("⤳ ({:.0}, {:.0})", self.target_pos.x, self.target_pos.y)
    }
}
