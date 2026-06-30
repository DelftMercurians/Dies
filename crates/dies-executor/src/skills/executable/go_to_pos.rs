//! GoToPos skill - move to a target position.

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};
use dies_tunables_macro::tunables;

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::PlayerControlInput;

tunables! {
    section "GoToPos";

    /// Arrival position tolerance. Must be ≥ the path-follower's `ARRIVE_DEADBAND`
    /// (15 mm): the controller commands a full stop once within the deadband, so the
    /// robot settles a little short of the exact goal. A tolerance below the deadband
    /// is unsatisfiable in steady state — the skill would never report success and
    /// any `await` on it hangs forever.
    #[tunable(unit = "mm", min = 15.0, max = 200.0, step = 5.0)]
    DEFAULT_POS_TOLERANCE: f64 = 30.0;
    /// Arrival speed tolerance — the robot must be this slow to count as arrived.
    #[tunable(unit = "mm/s", min = 0.0, max = 300.0, step = 10.0)]
    DEFAULT_VEL_TOLERANCE: f64 = 50.0;
}

/// A skill that moves the robot to a target position.
///
/// This is a continuous skill - calling it repeatedly with different positions
/// will smoothly update the trajectory.
pub struct GoToPosSkill {
    target_pos: Vector2,
    target_heading: Option<Angle>,
    pos_tolerance: f64,
    vel_tolerance: f64,
    status: SkillStatus,
}

impl GoToPosSkill {
    /// Create a new GoToPos skill.
    pub fn new(target_pos: Vector2, target_heading: Option<Angle>) -> Self {
        Self {
            target_pos,
            target_heading,
            pos_tolerance: DEFAULT_POS_TOLERANCE(),
            vel_tolerance: DEFAULT_VEL_TOLERANCE(),
            status: SkillStatus::Running,
        }
    }

    /// Set the position tolerance.
    pub fn with_pos_tolerance(mut self, tolerance: f64) -> Self {
        self.pos_tolerance = tolerance;
        self
    }

    /// Set the velocity tolerance.
    pub fn with_vel_tolerance(mut self, tolerance: f64) -> Self {
        self.vel_tolerance = tolerance;
        self
    }
}

impl ExecutableSkill for GoToPosSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::GoToPos { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::GoToPos { position, heading } = command {
            self.target_pos = *position;
            self.target_heading = *heading;
            // Reset status to Running if we were completed
            if matches!(self.status, SkillStatus::Succeeded | SkillStatus::Failed) {
                self.status = SkillStatus::Running;
            }
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        let position = ctx.player.position;
        let velocity = ctx.player.velocity;

        let distance = (self.target_pos - position).norm();
        let speed = velocity.norm();

        // Check if we've arrived
        if distance < self.pos_tolerance && speed < self.vel_tolerance {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }

        self.status = SkillStatus::Running;

        let mut input = PlayerControlInput::new();
        input.with_position(self.target_pos);

        if let Some(heading) = self.target_heading {
            input.with_yaw(heading);
        }

        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "GoToPos"
    }

    fn description(&self) -> String {
        format!("→ ({:.0}, {:.0})", self.target_pos.x, self.target_pos.y)
    }
}
