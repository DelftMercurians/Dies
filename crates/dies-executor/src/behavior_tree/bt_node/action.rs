use dies_core::{debug_tree_node, Angle, PlayerId, Vector2};

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment,
};
use crate::{
    behavior_tree::{semaphore_node, sequence_node, Argument, BehaviorNode},
    control::{PlayerContext, PlayerControlInput, ShootTarget},
    skills::{
        Face, FetchBall, FetchBallWithPreshoot, GoToPosition, Kick, Shoot, Skill, SkillCtx,
        SkillProgress, SkillResult, TryReceive, Wait,
    },
};

#[derive(Clone)]
pub enum FaceTarget {
    Angle(Argument<Angle>),
    Position(Argument<Vector2>),
    OwnPlayer(Argument<u32>),
}

#[derive(Clone)]
pub enum HeadingTarget {
    Angle(Argument<Angle>),
    Position(Argument<Vector2>),
    OwnPlayer(Argument<u32>),
}

#[derive(Clone)]
pub enum SkillDefinition {
    GoToPosition {
        target_pos: Argument<Vector2>,
        target_heading: Option<Argument<Angle>>,
        target_velocity: Argument<Vector2>,
        pos_tolerance: Argument<f64>,
        velocity_tolerance: Argument<f64>,
        with_ball: Argument<bool>,
        avoid_ball: Argument<bool>,
    },
    Face {
        target: FaceTarget,
        with_ball: Argument<bool>,
    },
    Kick,
    Wait {
        duration_secs: Argument<f64>,
    },
    FetchBall,
    FetchBallWithPreshoot {
        distance_limit: f64,
        can_pass: bool,
    },
    Shoot {
        target: Argument<ShootTarget>,
    },
    TryReceive,
}

pub struct ActionNode {
    skill_def: SkillDefinition,
    active_skill: Option<Skill>,
    description_text: String,
    node_id_fragment: String,
}

impl ActionNode {
    pub fn new(skill_def: SkillDefinition, description: Option<String>) -> Self {
        let desc = description.unwrap_or_else(|| format!("Action"));
        Self {
            skill_def,
            active_skill: None,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let skill_info = match &self.skill_def {
            SkillDefinition::GoToPosition { .. } => "GoToPosition",
            SkillDefinition::Face { .. } => "Face",
            SkillDefinition::Kick => "Kick",
            SkillDefinition::Wait { .. } => "Wait",
            SkillDefinition::FetchBall => "FetchBall",
            SkillDefinition::FetchBallWithPreshoot { .. } => "FetchBallWithPreshoot",
            SkillDefinition::Shoot { .. } => "Shoot",
            SkillDefinition::TryReceive => "TryReceive",
        };
        let internal_state = self.active_skill.as_ref().map(|_| "Active".to_string());
        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&node_full_id),
            false, // We don't know if it's active without ticking
            "Action",
            internal_state,
            Some(skill_info.to_string()),
        );
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let skill_info = match &self.skill_def {
            SkillDefinition::GoToPosition { .. } => "GoToPosition",
            SkillDefinition::Face { .. } => "Face",
            SkillDefinition::Kick => "Kick",
            SkillDefinition::Wait { .. } => "Wait",
            SkillDefinition::FetchBall => "FetchBall",
            SkillDefinition::FetchBallWithPreshoot { .. } => "FetchBallWithPreshoot",
            SkillDefinition::Shoot { .. } => "Shoot",
            SkillDefinition::TryReceive => "TryReceive",
        };
        // Create a new skill if there is no active one
        if self.active_skill.is_none() {
            let new_skill = match &self.skill_def {
                SkillDefinition::GoToPosition {
                    target_pos,
                    target_heading,
                    target_velocity: _,
                    pos_tolerance: _,
                    velocity_tolerance: _,
                    with_ball,
                    avoid_ball,
                } => {
                    let target = target_pos.resolve(situation);
                    let heading = target_heading.as_ref().map(|h| h.resolve(situation));
                    let with_ball = with_ball.resolve(situation);
                    let avoid_ball = avoid_ball.resolve(situation);

                    let mut skill = GoToPosition::new(target);
                    if let Some(heading) = heading {
                        skill = skill.with_heading(heading);
                    }
                    if with_ball {
                        skill = skill.with_ball();
                    }
                    if avoid_ball {
                        skill = skill.avoid_ball();
                    }

                    Some(Skill::GoToPosition(skill))
                }
                SkillDefinition::Face { target, with_ball } => {
                    let with_ball = with_ball.resolve(situation);

                    let mut skill = match target {
                        FaceTarget::Angle(angle_rad) => Face::new(angle_rad.resolve(situation)),
                        FaceTarget::Position(pos) => Face::towards_position(pos.resolve(situation)),
                        FaceTarget::OwnPlayer(id) => {
                            Face::towards_own_player(PlayerId::new(id.resolve(situation)))
                        }
                    };

                    if with_ball {
                        skill = skill.with_ball();
                    }
                    Some(Skill::Face(skill))
                }
                SkillDefinition::Kick => Some(Skill::Kick(Kick::new())),
                SkillDefinition::Wait { duration_secs } => Some(Skill::Wait(Wait::new_secs_f64(
                    duration_secs.resolve(situation),
                ))),
                SkillDefinition::FetchBall => Some(Skill::FetchBall(FetchBall::new())),
                SkillDefinition::FetchBallWithPreshoot {
                    distance_limit,
                    can_pass,
                } => {
                    let skill = FetchBallWithPreshoot::new()
                        .with_distance_limit(*distance_limit)
                        .with_can_pass(*can_pass);
                    Some(Skill::FetchBallWithPreshoot(skill))
                }
                SkillDefinition::Shoot { target } => {
                    let target = target.resolve(situation);
                    Some(Skill::Shoot(Shoot::new(target)))
                }
                SkillDefinition::TryReceive => Some(Skill::TryReceive(TryReceive::new())),
            };
            self.active_skill = new_skill;
        }

        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let Some(skill) = self.active_skill.as_mut() else {
            return (BehaviorStatus::Failure, None);
        };

        let skill_ctx = SkillCtx {
            player: situation.player_data(),
            world: &situation.world,
            bt_context: situation.bt_context.clone(),
            viz_path_prefix: situation.debug_key(""),
            team_context: situation.team_context.clone(),
        };

        let (status, input) = match skill.update(skill_ctx) {
            SkillProgress::Continue(input) => (BehaviorStatus::Running, Some(input)),
            SkillProgress::Done(SkillResult::Success) => (BehaviorStatus::Success, None),
            SkillProgress::Done(SkillResult::Failure) => (BehaviorStatus::Failure, None),
        };

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        if matches!(status, BehaviorStatus::Failure) {
            dies_core::debug_string(
                format!("{}_p{}_skill", situation.team_color, situation.player_id),
                format!("{}: {}", skill_info, "failure"),
            );
        }
        dies_core::debug_string(
            format!(
                "team_{}.p{}.skill",
                situation.team_color, situation.player_id
            ),
            format!(
                "{}: {}{}",
                skill_info,
                if status == BehaviorStatus::Success {
                    "success"
                } else if status == BehaviorStatus::Failure {
                    "failure"
                } else {
                    "running"
                },
                if let Skill::Shoot(shoot) = skill {
                    format!(" ({})", shoot.state())
                } else if let Skill::FetchBallWithPreshoot(fetch_ball_with_preshoot) = skill {
                    format!(" ({})", fetch_ball_with_preshoot.state())
                } else {
                    "".to_string()
                },
            ),
        );

        // If the skill is done, remove it so it can be recreated on the next tick
        if status != BehaviorStatus::Running {
            self.active_skill = None;
        }

        let internal_state = if self.active_skill.is_some() {
            Some(format!("Running {}", skill_info))
        } else {
            Some("Idle".to_string())
        };
        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&node_full_id),
            is_active,
            "Action",
            internal_state,
            Some(skill_info.to_string()),
        );
        (status, input)
    }

    pub fn description(&self) -> String {
        self.description_text.clone()
    }

    pub fn get_node_id_fragment(&self) -> String {
        self.node_id_fragment.clone()
    }

    pub fn get_full_node_id(&self, current_path_prefix: &str) -> String {
        let fragment = self.get_node_id_fragment();
        if current_path_prefix.is_empty() {
            fragment
        } else {
            format!("{}.{}", current_path_prefix, fragment)
        }
    }

    pub fn get_child_node_ids(&self, _current_path_prefix: &str) -> Vec<String> {
        vec![]
    }
}

impl From<ActionNode> for BehaviorNode {
    fn from(node: ActionNode) -> Self {
        BehaviorNode::Action(node)
    }
}

pub struct GoToPositionBuilder {
    target_pos: Argument<Vector2>,
    target_heading: Option<Argument<Angle>>,
    target_velocity: Argument<Vector2>,
    pos_tolerance: Argument<f64>,
    velocity_tolerance: Argument<f64>,
    with_ball: Argument<bool>,
    avoid_ball: Argument<bool>,
    description: Option<String>,
}

impl GoToPositionBuilder {
    pub fn new(target_pos: impl Into<Argument<Vector2>>) -> Self {
        Self {
            target_pos: target_pos.into(),
            target_heading: None,
            target_velocity: Argument::Static(Vector2::zeros()),
            pos_tolerance: Argument::Static(0.1),
            velocity_tolerance: Argument::Static(0.1),
            with_ball: Argument::Static(false),
            avoid_ball: Argument::Static(false),
            description: None,
        }
    }

    pub fn with_heading(mut self, heading: impl Into<Argument<Angle>>) -> Self {
        self.target_heading = Some(heading.into());
        self
    }

    pub fn with_velocity(mut self, velocity: impl Into<Argument<Vector2>>) -> Self {
        self.target_velocity = velocity.into();
        self
    }

    pub fn with_pos_tolerance(mut self, tolerance: impl Into<Argument<f64>>) -> Self {
        self.pos_tolerance = tolerance.into();
        self
    }

    pub fn with_velocity_tolerance(mut self, tolerance: impl Into<Argument<f64>>) -> Self {
        self.velocity_tolerance = tolerance.into();
        self
    }

    pub fn with_ball(mut self) -> Self {
        self.with_ball = Argument::Static(true);
        self
    }

    pub fn avoid_ball(mut self) -> Self {
        self.avoid_ball = Argument::Static(true);
        self
    }

    pub fn description(mut self, desc: impl AsRef<str>) -> Self {
        self.description = Some(desc.as_ref().to_string());
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(
            SkillDefinition::GoToPosition {
                target_pos: self.target_pos,
                target_heading: self.target_heading,
                target_velocity: self.target_velocity,
                pos_tolerance: self.pos_tolerance,
                velocity_tolerance: self.velocity_tolerance,
                with_ball: self.with_ball,
                avoid_ball: self.avoid_ball,
            },
            self.description,
        )
    }
}

pub struct FaceBuilder {
    target: FaceTarget,
    with_ball: Argument<bool>,
    description: Option<String>,
}

impl FaceBuilder {
    pub fn angle(angle: impl Into<Argument<Angle>>) -> Self {
        Self {
            target: FaceTarget::Angle(angle.into()),
            with_ball: Argument::Static(false),
            description: None,
        }
    }

    pub fn position(pos: impl Into<Argument<Vector2>>) -> Self {
        Self {
            target: FaceTarget::Position(pos.into()),
            with_ball: Argument::Static(false),
            description: None,
        }
    }

    pub fn own_player(id: impl Into<Argument<u32>>) -> Self {
        Self {
            target: FaceTarget::OwnPlayer(id.into()),
            with_ball: Argument::Static(false),
            description: None,
        }
    }

    pub fn with_ball(mut self) -> Self {
        self.with_ball = Argument::Static(true);
        self
    }

    pub fn description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(
            SkillDefinition::Face {
                target: self.target,
                with_ball: self.with_ball,
            },
            self.description,
        )
    }
}

pub struct KickBuilder {
    description: Option<String>,
}

impl KickBuilder {
    pub fn new() -> Self {
        Self { description: None }
    }

    pub fn description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(SkillDefinition::Kick, self.description)
    }
}

pub struct WaitBuilder {
    duration_secs: Argument<f64>,
    description: Option<String>,
}

impl WaitBuilder {
    pub fn new(duration_secs: impl Into<Argument<f64>>) -> Self {
        Self {
            duration_secs: duration_secs.into(),
            description: None,
        }
    }

    pub fn description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(
            SkillDefinition::Wait {
                duration_secs: self.duration_secs,
            },
            self.description,
        )
    }
}

pub struct FetchBallBuilder {
    description: Option<String>,
}

impl FetchBallBuilder {
    pub fn new() -> Self {
        Self { description: None }
    }

    pub fn description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(SkillDefinition::FetchBall, self.description)
    }
}

pub struct FetchBallWithPreshootBuilder {
    distance_limit: f64,
    avoid_ball_care: f64,
    can_pass: bool,
    description: Option<String>,
}

impl FetchBallWithPreshootBuilder {
    pub fn new() -> Self {
        Self {
            distance_limit: 160.0,
            avoid_ball_care: 0.0,
            can_pass: true,
            description: None,
        }
    }

    pub fn description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    pub fn with_distance_limit(mut self, distance_limit: f64) -> Self {
        self.distance_limit = distance_limit;
        self
    }

    pub fn with_avoid_ball_care(mut self, avoid_ball_care: f64) -> Self {
        self.avoid_ball_care = avoid_ball_care;
        self
    }

    pub fn with_can_pass(mut self, can_pass: bool) -> Self {
        self.can_pass = can_pass;
        self
    }

    pub fn build(self) -> ActionNode {
        ActionNode::new(
            SkillDefinition::FetchBallWithPreshoot {
                distance_limit: self.distance_limit,
                can_pass: self.can_pass,
            },
            self.description,
        )
    }
}

// Convenience functions to create builders
pub fn go_to_position(target_pos: impl Into<Argument<Vector2>>) -> GoToPositionBuilder {
    GoToPositionBuilder::new(target_pos)
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

pub fn kick() -> KickBuilder {
    KickBuilder::new()
}

pub fn wait(duration_secs: impl Into<Argument<f64>>) -> WaitBuilder {
    WaitBuilder::new(duration_secs)
}

pub fn fetch_ball() -> FetchBallBuilder {
    FetchBallBuilder::new()
}

pub fn fetch_ball_with_preshoot() -> FetchBallWithPreshootBuilder {
    FetchBallWithPreshootBuilder::new()
}

pub fn shoot(target: impl Into<Argument<ShootTarget>>) -> ActionNode {
    ActionNode::new(
        SkillDefinition::Shoot {
            target: target.into(),
        },
        Some("Shoot".to_string()),
    )
}

pub fn at_goal(pos: impl Into<Argument<Vector2>>) -> Argument<ShootTarget> {
    pos.into().map(|p| ShootTarget::Goal(p))
}

pub fn to_player(id: impl Into<Argument<PlayerId>>) -> Argument<ShootTarget> {
    id.into()
        .map(|id| ShootTarget::Player { id, position: None })
}

pub fn to_player_at_pos(target: Argument<(PlayerId, Vector2)>) -> Argument<ShootTarget> {
    target.map(|(id, pos)| ShootTarget::Player {
        id,
        position: Some(pos),
    })
}

pub fn try_receive() -> ActionNode {
    return ActionNode::new(
        SkillDefinition::TryReceive,
        Some("Try receive?".to_string()),
    );
    // semaphore_node()
    //     .semaphore_id("try_receive".into())
    //     .max_entry(1)
    //     .do_then(
    // sequence_node()
    //     // .ignore_fail()
    //     .add(ActionNode::new(
    //         SkillDefinition::TryReceive,
    //         Some("Try receive?".to_string()),
    //     ))
    //     .add(fetch_ball_with_preshoot().build())
    //     .build()
    //     .into()
    // )
    // .build()
    // .into()
}
