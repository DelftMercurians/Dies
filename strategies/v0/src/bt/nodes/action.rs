//! Action leaves — where the old legacy skills are *substituted* with the new IPC
//! skills.
//!
//! The original engine ran in-process skills that emitted `PlayerControlInput`.
//! Here each action emits a [`SkillCommand`] and reads the robot's
//! [`SkillStatus`](dies_strategy_protocol::SkillStatus) (carried on the
//! `RobotSituation`) to know when a discrete skill has finished. The mapping:
//!
//! | legacy skill              | new IPC skill(s)                       |
//! |---------------------------|----------------------------------------|
//! | `GoToPosition`            | `GoToPos` / `Dribble` (with ball)      |
//! | `Face`                    | `GoToPos` holding position + heading   |
//! | `FetchBall`               | `PickupBall`                           |
//! | `FetchBallWithPreshoot`   | `PickupBall` → `DribbleShoot`          |
//! | `Shoot`                   | `Shoot` (reflex)                       |
//! | `Kick`                    | `DribbleShoot`                         |
//! | `Wait`                    | tree-side timer (emits `Stop`)         |
//! | `TryReceive`              | passing-target gate (else fail)        |

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use super::super::argument::Argument;
use super::super::situation::{BehaviorStatus, RobotSituation, ShootTarget};
use super::{BehaviorNode, TickResult};

/// Map a discrete skill's reported status onto a tree status.
fn status_of(skill_status: SkillStatus) -> BehaviorStatus {
    match skill_status {
        SkillStatus::Succeeded => BehaviorStatus::Success,
        SkillStatus::Failed => BehaviorStatus::Failure,
        SkillStatus::Idle | SkillStatus::Running => BehaviorStatus::Running,
    }
}

/// Resolve a `ShootTarget` into an aim point.
fn resolve_target(situation: &RobotSituation, target: &ShootTarget) -> Vector2 {
    match target {
        ShootTarget::Goal(pos) => *pos,
        ShootTarget::Player { id, position } => position
            .or_else(|| {
                situation
                    .world
                    .own_players
                    .iter()
                    .find(|p| p.id == *id)
                    .map(|p| p.position)
            })
            .unwrap_or_else(|| situation.get_opp_goal_position()),
    }
}

/// Heading from `from` toward `to`.
fn heading_to(from: Vector2, to: Vector2) -> Angle {
    Angle::between_points(from, to)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FetchPhase {
    Picking,
    Shooting,
}

pub enum SkillDef {
    GoToPosition {
        target_pos: Argument<Vector2>,
        target_heading: Option<Argument<Angle>>,
        with_ball: Argument<bool>,
    },
    Face {
        heading: Argument<Angle>,
        with_ball: Argument<bool>,
    },
    Kick,
    Wait {
        duration_secs: Argument<f64>,
    },
    FetchBall,
    FetchBallWithPreshoot {
        override_target: Argument<Option<ShootTarget>>,
    },
    Shoot {
        target: Argument<ShootTarget>,
    },
    TryReceive,
}

pub struct ActionNode {
    skill_def: SkillDef,
    // ── per-activation runtime state ───────────────────────────────────
    fetch_phase: FetchPhase,
    ticks_in_phase: u32,
    wait_start: Option<f64>,
}

impl ActionNode {
    pub fn new(skill_def: SkillDef) -> Self {
        Self {
            skill_def,
            fetch_phase: FetchPhase::Picking,
            ticks_in_phase: 0,
            wait_start: None,
        }
    }

    fn reset(&mut self) {
        self.fetch_phase = FetchPhase::Picking;
        self.ticks_in_phase = 0;
        self.wait_start = None;
    }

    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        match &self.skill_def {
            SkillDef::GoToPosition {
                target_pos,
                target_heading,
                with_ball,
            } => {
                let pos = target_pos.resolve(situation);
                let heading = target_heading.as_ref().map(|h| h.resolve(situation));
                let cmd = if with_ball.resolve(situation) {
                    SkillCommand::Dribble {
                        target_pos: pos,
                        target_heading: heading.unwrap_or_else(|| situation.heading()),
                    }
                } else {
                    SkillCommand::GoToPos {
                        position: pos,
                        heading,
                    }
                };
                let status = status_of(situation.skill_status);
                if status != BehaviorStatus::Running {
                    self.reset();
                }
                (status, Some(cmd))
            }

            SkillDef::Face { heading, with_ball } => {
                let h = heading.resolve(situation);
                let cmd = if with_ball.resolve(situation) {
                    SkillCommand::Dribble {
                        target_pos: situation.position(),
                        target_heading: h,
                    }
                } else {
                    SkillCommand::GoToPos {
                        position: situation.position(),
                        heading: Some(h),
                    }
                };
                let status = status_of(situation.skill_status);
                if status != BehaviorStatus::Running {
                    self.reset();
                }
                (status, Some(cmd))
            }

            SkillDef::Kick => {
                let cmd = SkillCommand::DribbleShoot {
                    target_heading: situation.heading(),
                };
                let status = status_of(situation.skill_status);
                if status != BehaviorStatus::Running {
                    self.reset();
                }
                (status, Some(cmd))
            }

            SkillDef::FetchBall => {
                let heading = heading_to(situation.position(), situation.ball_position());
                let cmd = SkillCommand::PickupBall {
                    target_heading: heading,
                    instant_kick: false,
                };
                let status = if situation.has_ball() {
                    BehaviorStatus::Success
                } else {
                    status_of(situation.skill_status)
                };
                if status != BehaviorStatus::Running {
                    self.reset();
                }
                (status, Some(cmd))
            }

            SkillDef::Shoot { target } => {
                let target = target.resolve(situation);
                let aim = resolve_target(situation, &target);
                let cmd = SkillCommand::Shoot { target: aim };
                let status = status_of(situation.skill_status);
                if status != BehaviorStatus::Running {
                    self.reset();
                }
                (status, Some(cmd))
            }

            SkillDef::Wait { duration_secs } => {
                let dur = duration_secs.resolve(situation);
                let now = situation.world.timestamp;
                let start = *self.wait_start.get_or_insert(now);
                if now - start >= dur {
                    self.reset();
                    (BehaviorStatus::Success, None)
                } else {
                    (BehaviorStatus::Running, Some(SkillCommand::Stop))
                }
            }

            SkillDef::TryReceive => {
                // Passing is not coordinated in this port (shoot targets are goals),
                // so unless this robot was explicitly nominated as a passing target,
                // there is nothing to receive — fail so the surrounding select moves
                // on to positioning.
                if situation.is_passing_target() {
                    let heading = heading_to(situation.position(), situation.ball_position());
                    if situation.has_ball() {
                        situation.accept_passing_target();
                        self.reset();
                        return (BehaviorStatus::Success, None);
                    }
                    (
                        BehaviorStatus::Running,
                        Some(SkillCommand::PickupBall {
                            target_heading: heading,
                            instant_kick: false,
                        }),
                    )
                } else {
                    (BehaviorStatus::Failure, None)
                }
            }

            SkillDef::FetchBallWithPreshoot { override_target } => {
                let target = override_target
                    .resolve(situation)
                    .unwrap_or_else(|| ShootTarget::Goal(situation.get_opp_goal_position()));
                let aim = resolve_target(situation, &target);
                let ball = situation.ball_position();
                let approach_heading = heading_to(ball, aim);

                self.ticks_in_phase += 1;
                match self.fetch_phase {
                    FetchPhase::Picking => {
                        if situation.has_ball() {
                            self.fetch_phase = FetchPhase::Shooting;
                            self.ticks_in_phase = 0;
                        } else if self.ticks_in_phase > 3
                            && situation.skill_status == SkillStatus::Failed
                        {
                            self.reset();
                            return (BehaviorStatus::Failure, None);
                        }
                        (
                            BehaviorStatus::Running,
                            Some(SkillCommand::PickupBall {
                                target_heading: approach_heading,
                                instant_kick: false,
                            }),
                        )
                    }
                    FetchPhase::Shooting => {
                        // Ignore the carried-over Succeeded from the pickup for the
                        // first couple of ticks after issuing the shoot.
                        if self.ticks_in_phase > 2 {
                            match situation.skill_status {
                                SkillStatus::Succeeded => {
                                    self.reset();
                                    return (BehaviorStatus::Success, None);
                                }
                                SkillStatus::Failed => {
                                    self.reset();
                                    return (BehaviorStatus::Failure, None);
                                }
                                _ => {}
                            }
                        }
                        // Lost the ball without a kick report → re-stage the pickup.
                        if !situation.has_ball() && self.ticks_in_phase > 6 {
                            self.fetch_phase = FetchPhase::Picking;
                            self.ticks_in_phase = 0;
                        }
                        let heading = heading_to(situation.position(), aim);
                        (
                            BehaviorStatus::Running,
                            Some(SkillCommand::DribbleShoot {
                                target_heading: heading,
                            }),
                        )
                    }
                }
            }
        }
    }
}

impl From<ActionNode> for BehaviorNode {
    fn from(node: ActionNode) -> Self {
        BehaviorNode::Action(node)
    }
}

// ── Builders / convenience constructors ──────────────────────────────────────

pub struct GoToPositionBuilder {
    target_pos: Argument<Vector2>,
    target_heading: Option<Argument<Angle>>,
    with_ball: Argument<bool>,
}

impl GoToPositionBuilder {
    pub fn new(target_pos: impl Into<Argument<Vector2>>) -> Self {
        Self {
            target_pos: target_pos.into(),
            target_heading: None,
            with_ball: Argument::Static(false),
        }
    }

    pub fn with_heading(mut self, heading: impl Into<Argument<Angle>>) -> Self {
        self.target_heading = Some(heading.into());
        self
    }

    pub fn with_velocity(self, _velocity: impl Into<Argument<Vector2>>) -> Self {
        self
    }

    pub fn with_pos_tolerance(self, _tolerance: impl Into<Argument<f64>>) -> Self {
        self
    }

    pub fn with_velocity_tolerance(self, _tolerance: impl Into<Argument<f64>>) -> Self {
        self
    }

    pub fn with_ball(mut self) -> Self {
        self.with_ball = Argument::Static(true);
        self
    }

    /// The IPC planner handles ball avoidance itself; kept as a no-op for source
    /// compatibility.
    pub fn avoid_ball(self) -> Self {
        self
    }

    pub fn description(self, _desc: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(SkillDef::GoToPosition {
            target_pos: self.target_pos,
            target_heading: self.target_heading,
            with_ball: self.with_ball,
        })
    }
}

pub fn go_to_position(target_pos: impl Into<Argument<Vector2>>) -> GoToPositionBuilder {
    GoToPositionBuilder::new(target_pos)
}

pub struct FaceBuilder {
    heading: Argument<Angle>,
    with_ball: Argument<bool>,
}

impl FaceBuilder {
    pub fn angle(angle: impl Into<Argument<Angle>>) -> Self {
        Self {
            heading: angle.into(),
            with_ball: Argument::Static(false),
        }
    }

    pub fn position(pos: impl Into<Argument<Vector2>>) -> Self {
        let pos = pos.into();
        let heading = Argument::Callback(std::sync::Arc::new(move |s: &RobotSituation| {
            heading_to(s.position(), pos.resolve(s))
        }));
        Self {
            heading,
            with_ball: Argument::Static(false),
        }
    }

    pub fn own_player(id: impl Into<Argument<u32>>) -> Self {
        let id = id.into();
        let heading = Argument::Callback(std::sync::Arc::new(move |s: &RobotSituation| {
            let pid = dies_strategy_protocol::PlayerId::new(id.resolve(s));
            let target = s
                .world
                .own_players
                .iter()
                .find(|p| p.id == pid)
                .map(|p| p.position)
                .unwrap_or_else(|| s.position());
            heading_to(s.position(), target)
        }));
        Self {
            heading,
            with_ball: Argument::Static(false),
        }
    }

    pub fn with_ball(mut self) -> Self {
        self.with_ball = Argument::Static(true);
        self
    }

    pub fn description(self, _desc: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(SkillDef::Face {
            heading: self.heading,
            with_ball: self.with_ball,
        })
    }
}

pub fn face_angle(angle: impl Into<Argument<Angle>>) -> FaceBuilder {
    FaceBuilder::angle(angle)
}

pub fn face_position(pos: impl Into<Argument<Vector2>>) -> FaceBuilder {
    FaceBuilder::position(pos)
}

pub fn face_own_player(id: impl Into<Argument<u32>>) -> FaceBuilder {
    FaceBuilder::own_player(id)
}

pub struct KickBuilder;

impl KickBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn description(self, _desc: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(SkillDef::Kick)
    }
}

impl Default for KickBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn kick() -> KickBuilder {
    KickBuilder::new()
}

pub struct WaitBuilder {
    duration_secs: Argument<f64>,
}

impl WaitBuilder {
    pub fn new(duration_secs: impl Into<Argument<f64>>) -> Self {
        Self {
            duration_secs: duration_secs.into(),
        }
    }

    pub fn description(self, _desc: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(SkillDef::Wait {
            duration_secs: self.duration_secs,
        })
    }
}

pub fn wait(duration_secs: impl Into<Argument<f64>>) -> WaitBuilder {
    WaitBuilder::new(duration_secs)
}

pub struct FetchBallBuilder;

impl FetchBallBuilder {
    pub fn new() -> Self {
        Self
    }

    pub fn description(self, _desc: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(SkillDef::FetchBall)
    }
}

impl Default for FetchBallBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn fetch_ball() -> FetchBallBuilder {
    FetchBallBuilder::new()
}

pub struct FetchBallWithPreshootBuilder {
    override_target: Option<Argument<ShootTarget>>,
}

impl FetchBallWithPreshootBuilder {
    pub fn new() -> Self {
        Self {
            override_target: None,
        }
    }

    pub fn description(self, _desc: impl AsRef<str>) -> Self {
        self
    }

    /// No-op: the IPC `PickupBall` skill owns the approach distance.
    pub fn with_distance_limit(self, _distance_limit: f64) -> Self {
        self
    }

    /// No-op: ball-avoidance care is handled by the executor planner.
    pub fn with_avoid_ball_care(self, _avoid_ball_care: f64) -> Self {
        self
    }

    /// Passing is not coordinated in this port; accepted but ignored.
    pub fn with_can_pass(self, _can_pass: bool) -> Self {
        self
    }

    pub fn with_override_target(mut self, target: impl Into<Argument<ShootTarget>>) -> Self {
        self.override_target = Some(target.into());
        self
    }

    pub fn build(self) -> ActionNode {
        let override_target = match self.override_target {
            Some(t) => t.map(Some),
            None => Argument::Static(None),
        };
        ActionNode::new(SkillDef::FetchBallWithPreshoot { override_target })
    }
}

impl Default for FetchBallWithPreshootBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn fetch_ball_with_preshoot() -> FetchBallWithPreshootBuilder {
    FetchBallWithPreshootBuilder::new()
}

pub fn shoot(target: impl Into<Argument<ShootTarget>>) -> ActionNode {
    ActionNode::new(SkillDef::Shoot {
        target: target.into(),
    })
}

pub fn at_goal(pos: impl Into<Argument<Vector2>>) -> Argument<ShootTarget> {
    pos.into().map(ShootTarget::Goal)
}

pub fn to_player(
    id: impl Into<Argument<dies_strategy_protocol::PlayerId>>,
) -> Argument<ShootTarget> {
    id.into().map(|id| ShootTarget::Player { id, position: None })
}

pub fn to_player_at_pos(
    target: Argument<(dies_strategy_protocol::PlayerId, Vector2)>,
) -> Argument<ShootTarget> {
    target.map(|(id, pos)| ShootTarget::Player {
        id,
        position: Some(pos),
    })
}

pub fn try_receive() -> ActionNode {
    ActionNode::new(SkillDef::TryReceive)
}
