use dies_core::{debug_tree_node, Angle, PlayerId, Vector2};
use rhai::Engine;

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment,
};
use crate::{
    behavior_tree::Argument,
    control::PlayerControlInput,
    roles::{
        skills::{
            ApproachBall, Face, FetchBall, FetchBallWithHeading, GoToPosition, InterceptBall, Kick,
            Wait,
        },
        Skill, SkillCtx, SkillProgress, SkillResult,
    },
};

#[derive(Clone)]
pub enum FaceTarget {
    Angle(Argument<f64>),
    Position(Argument<Vector2>),
    OwnPlayer(Argument<u32>),
}

#[derive(Clone)]
pub enum HeadingTarget {
    Angle(Argument<f64>),
    Position(Argument<Vector2>),
    OwnPlayer(Argument<u32>),
}

#[derive(Clone)]
pub enum SkillDefinition {
    GoToPosition {
        target_pos: Argument<Vector2>,
        target_heading: Option<Argument<f64>>,
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
    FetchBallWithHeading {
        target: HeadingTarget,
    },
    ApproachBall,
    InterceptBall,
}

#[derive(Clone)]
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

    pub fn debug_all_nodes(&self, situation: &RobotSituation, _engine: &Engine) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let skill_info = match &self.skill_def {
            SkillDefinition::GoToPosition { .. } => "GoToPosition",
            SkillDefinition::Face { .. } => "Face",
            SkillDefinition::Kick => "Kick",
            SkillDefinition::Wait { .. } => "Wait",
            SkillDefinition::FetchBall => "FetchBall",
            SkillDefinition::FetchBallWithHeading { .. } => "FetchBallWithHeading",
            SkillDefinition::ApproachBall => "ApproachBall",
            SkillDefinition::InterceptBall => "InterceptBall",
        };
        let internal_state = self.active_skill.as_ref().map(|_| "Active".to_string());
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
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
        engine: &Engine,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
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
                    let target = target_pos.resolve(situation, engine);
                    let heading = target_heading
                        .as_ref()
                        .map(|h| h.resolve(situation, engine));
                    let with_ball = with_ball.resolve(situation, engine);
                    let avoid_ball = avoid_ball.resolve(situation, engine);

                    if let (Ok(target), Ok(with_ball), Ok(avoid_ball)) =
                        (target, with_ball, avoid_ball)
                    {
                        let mut skill = GoToPosition::new(target);
                        if let Some(Ok(heading)) = heading {
                            skill = skill.with_heading(Angle::from_radians(heading));
                        }
                        if with_ball {
                            skill = skill.with_ball();
                        }
                        if avoid_ball {
                            skill = skill.avoid_ball();
                        }

                        Some(Skill::GoToPosition(skill))
                    } else {
                        log::error!(
                            "Failed to resolve arguments for GoToPosition: {:?}, {:?}",
                            target_pos,
                            target_heading,
                        );
                        None
                    }
                }
                SkillDefinition::Face { target, with_ball } => {
                    let with_ball = with_ball.resolve(situation, engine);
                    if let Ok(with_ball) = with_ball {
                        let skill = match target {
                            FaceTarget::Angle(angle_rad) => angle_rad
                                .resolve(situation, engine)
                                .ok()
                                .map(|rad| Face::new(Angle::from_radians(rad))),
                            FaceTarget::Position(pos) => pos
                                .resolve(situation, engine)
                                .ok()
                                .map(Face::towards_position),
                            FaceTarget::OwnPlayer(id) => id
                                .resolve(situation, engine)
                                .ok()
                                .map(|id| Face::towards_own_player(PlayerId::new(id))),
                        };

                        if let Some(mut skill) = skill {
                            if with_ball {
                                skill = skill.with_ball();
                            }
                            Some(Skill::Face(skill))
                        } else {
                            log::error!("Failed to resolve arguments for Face");
                            None
                        }
                    } else {
                        log::error!("Failed to resolve arguments for Face");
                        None
                    }
                }
                SkillDefinition::Kick => Some(Skill::Kick(Kick::new())),
                SkillDefinition::Wait { duration_secs } => duration_secs
                    .resolve(situation, engine)
                    .ok()
                    .map(|s| Skill::Wait(Wait::new_secs_f64(s))),
                SkillDefinition::FetchBall => Some(Skill::FetchBall(FetchBall::new())),
                SkillDefinition::FetchBallWithHeading { target } => {
                    let skill = match target {
                        HeadingTarget::Angle(angle_rad) => angle_rad
                            .resolve(situation, engine)
                            .ok()
                            .map(|rad| FetchBallWithHeading::new(Angle::from_radians(rad))),
                        HeadingTarget::Position(pos) => pos
                            .resolve(situation, engine)
                            .ok()
                            .map(FetchBallWithHeading::towards_position),
                        HeadingTarget::OwnPlayer(id) => id
                            .resolve(situation, engine)
                            .ok()
                            .map(|id| FetchBallWithHeading::towards_own_player(PlayerId::new(id))),
                    };
                    if skill.is_none() {
                        log::error!("Failed to resolve arguments for FetchBallWithHeading");
                    }
                    skill.map(Skill::FetchBallWithHeading)
                }
                SkillDefinition::ApproachBall => Some(Skill::ApproachBall(ApproachBall::new())),
                SkillDefinition::InterceptBall => Some(Skill::InterceptBall(InterceptBall::new())),
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
        };

        let (status, input) = match skill.update(skill_ctx) {
            SkillProgress::Continue(input) => (BehaviorStatus::Running, Some(input)),
            SkillProgress::Done(SkillResult::Success) => (BehaviorStatus::Success, None),
            SkillProgress::Done(SkillResult::Failure) => (BehaviorStatus::Failure, None),
        };

        // If the skill is done, remove it so it can be recreated on the next tick
        if status != BehaviorStatus::Running {
            self.active_skill = None;
        }

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        let skill_info = match &self.skill_def {
            SkillDefinition::GoToPosition { .. } => "GoToPosition",
            SkillDefinition::Face { .. } => "Face",
            SkillDefinition::Kick => "Kick",
            SkillDefinition::Wait { .. } => "Wait",
            SkillDefinition::FetchBall => "FetchBall",
            SkillDefinition::FetchBallWithHeading { .. } => "FetchBallWithHeading",
            SkillDefinition::ApproachBall => "ApproachBall",
            SkillDefinition::InterceptBall => "InterceptBall",
        };
        let internal_state = if self.active_skill.is_some() {
            Some(format!("Running {}", skill_info))
        } else {
            Some("Idle".to_string())
        };
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
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
            format!("{}/{}", current_path_prefix, fragment)
        }
    }

    pub fn get_child_node_ids(&self, _current_path_prefix: &str) -> Vec<String> {
        vec![]
    }
}
