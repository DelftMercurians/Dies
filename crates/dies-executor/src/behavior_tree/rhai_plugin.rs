use rhai::plugin::*;
use rhai::{
    Array, CustomType, Dynamic, Engine, EvalAltResult, FnPtr, Map, NativeCallContext, Position,
    Scope, TypeBuilder, AST,
};
use std::sync::Arc;

use crate::behavior_tree::{
    ActionNode, BehaviorNode as BehaviorNodeTypeEnum, BehaviorNode, GuardNode, ScoringSelectNode,
    SelectNode, SemaphoreNode, SequenceNode, Situation,
};
use crate::roles::skills::{
    ApproachBall as ApproachBallSkillInstance, Face as FaceSkillInstance,
    FetchBall as FetchBallSkillInstance, FetchBallWithHeading as FetchBallWithHeadingSkillInstance,
    GoToPosition as GoToPositionSkillInstance, InterceptBall as InterceptBallSkillInstance,
    Kick as KickSkillInstance, Wait as WaitSkillInstance,
};
use crate::roles::Skill as SkillEnum;
use dies_core::{Angle, PlayerId, Vector2 as CoreVector2};

#[derive(Clone)]
pub struct RhaiSkill(pub SkillEnum);

impl CustomType for RhaiSkill {
    fn build(mut builder: TypeBuilder<Self>) {
        builder.with_name("RhaiSkill");
    }
}

#[derive(Clone)]
pub struct RhaiBehaviorNode(pub BehaviorNode);

impl CustomType for RhaiBehaviorNode {
    fn build(mut builder: TypeBuilder<Self>) {
        builder.with_name("BehaviorNode");
    }
}

#[export_module]
pub mod bt_rhai_plugin {
    use crate::behavior_tree::BtCallback;

    use super::*; // Make imports from outer scope available

    // TYPE ALIASES
    // These tell Rhai to recognize RhaiBehaviorNode under the name "BehaviorNode" etc.
    pub type BehaviorNode = RhaiBehaviorNode;
    pub type SkillObject = RhaiSkill; // May not be directly used if skills return ActionNodes
    pub type Vec2 = CoreVector2;

    // VECTOR2 HELPERS
    #[rhai_fn(name = "vec2")]
    pub fn vec2_constructor(x: f64, y: f64) -> Vec2 {
        CoreVector2::new(x, y)
    }

    // BEHAVIOR NODE CONSTRUCTORS
    #[rhai_fn(name = "Select", return_raw)]
    pub fn select_node(
        children_dyn: Array,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let mut rust_children: Vec<BehaviorNodeTypeEnum> = Vec::new();
        for child_dyn in children_dyn {
            let type_name = child_dyn.type_name().to_string();
            let node_wrapper = child_dyn.try_cast::<RhaiBehaviorNode>().ok_or_else(|| {
                Box::new(EvalAltResult::ErrorMismatchDataType(
                    "Expected BehaviorNode".to_string(),
                    type_name.into(),
                    Position::NONE,
                ))
            })?;
            rust_children.push(node_wrapper.0);
        }
        Ok(RhaiBehaviorNode(BehaviorNodeTypeEnum::Select(
            SelectNode::new(rust_children, description),
        )))
    }

    #[rhai_fn(name = "Sequence", return_raw)]
    pub fn sequence_node(
        children_dyn: Array,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let mut rust_children: Vec<BehaviorNodeTypeEnum> = Vec::new();
        for child_dyn in children_dyn {
            let type_name = child_dyn.type_name().to_string();
            let node_wrapper = child_dyn.try_cast::<RhaiBehaviorNode>().ok_or_else(|| {
                Box::new(EvalAltResult::ErrorMismatchDataType(
                    "Expected BehaviorNode".to_string(),
                    type_name.into(),
                    Position::NONE,
                ))
            })?;
            rust_children.push(node_wrapper.0);
        }
        Ok(RhaiBehaviorNode(BehaviorNodeTypeEnum::Sequence(
            SequenceNode::new(rust_children, description),
        )))
    }

    #[rhai_fn(name = "Guard", return_raw)]
    pub fn guard_node(
        context: NativeCallContext,
        condition_fn_ptr: FnPtr,
        child_node: BehaviorNode,
        cond_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let callback_fn_name = condition_fn_ptr.fn_name().to_string();
        let description =
            cond_description.unwrap_or_else(|| format!("RhaiCond:{}", callback_fn_name));
        let situation = Situation::new_fn(
            BtCallback::new_rhai(&context, condition_fn_ptr),
            &description,
        );
        let guard_desc_override = Some(format!(
            "Guard_If_{}_Then_{}",
            description,
            child_node.0.description()
        ));
        Ok(RhaiBehaviorNode(BehaviorNodeTypeEnum::Guard(
            GuardNode::new(situation, child_node.0, guard_desc_override),
        )))
    }

    #[rhai_fn(name = "Semaphore", return_raw)]
    pub fn semaphore_node(
        child_node: BehaviorNode, // RhaiBehaviorNode
        id: String,
        max_count: i64,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        Ok(RhaiBehaviorNode(BehaviorNodeTypeEnum::Semaphore(
            SemaphoreNode::new(
                child_node.0, // .0 to access inner BehaviorNodeTypeEnum
                id,
                max_count as usize,
                description,
            ),
        )))
    }

    #[rhai_fn(name = "ScoringSelect", return_raw)]
    pub fn scoring_select_node(
        context: NativeCallContext,
        children_scorers_dyn: Array,
        hysteresis_margin: f64,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let mut rust_children_scorers: Vec<(BehaviorNodeTypeEnum, BtCallback<f64>)> = Vec::new();

        for item_dyn in children_scorers_dyn {
            let name = item_dyn.type_name().to_string();
            let item_map = item_dyn.try_cast::<Map>().ok_or_else(|| {
                Box::new(EvalAltResult::ErrorMismatchDataType(
                    "Expected map for child_scorer item".to_string(),
                    name.into(),
                    Position::NONE,
                ))
            })?;

            let node_dyn = item_map.get("node").ok_or_else(|| {
                Box::new(EvalAltResult::ErrorVariableNotFound(
                    "Missing 'node' in scorer item".to_string(),
                    Position::NONE,
                ))
            })?;
            let scorer_fn_ptr_dyn = item_map.get("scorer").ok_or_else(|| {
                Box::new(EvalAltResult::ErrorVariableNotFound(
                    "Missing 'scorer' in scorer item".to_string(),
                    Position::NONE,
                ))
            })?;

            let node_wrapper =
                node_dyn
                    .clone()
                    .try_cast::<RhaiBehaviorNode>()
                    .ok_or_else(|| {
                        Box::new(EvalAltResult::ErrorMismatchDataType(
                            "Expected BehaviorNode for 'node' field".to_string(),
                            node_dyn.type_name().into(),
                            Position::NONE,
                        ))
                    })?;

            let scorer_fn_ptr = scorer_fn_ptr_dyn.clone().cast::<FnPtr>();
            let scorer_callback = BtCallback::new_rhai(&context, scorer_fn_ptr);
            rust_children_scorers.push((node_wrapper.0, scorer_callback));
        }
        Ok(RhaiBehaviorNode(BehaviorNodeTypeEnum::ScoringSelect(
            ScoringSelectNode::new(rust_children_scorers, hysteresis_margin, description),
        )))
    }

    // Helper to create ActionNode from SkillEnum
    fn skill_to_action_node(
        skill_enum: SkillEnum,
        description: Option<String>,
        skill_name_for_auto_desc: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let final_description =
            description.unwrap_or_else(|| format!("Action_{}", skill_name_for_auto_desc));
        Ok(RhaiBehaviorNode(BehaviorNodeTypeEnum::Action(
            ActionNode::new(skill_enum, Some(final_description)),
        )))
    }

    // SKILL CONSTRUCTORS (returning Action Nodes)
    #[rhai_fn(name = "GoToPosition", return_raw)]
    pub fn goto_action(
        x: f64,
        y: f64,
        options: Option<Map>,
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let mut skill_instance = GoToPositionSkillInstance::new(CoreVector2::new(x, y));
        if let Some(opts) = options {
            if let Some(heading_dyn) = opts.get("heading") {
                if let Some(heading_f64) = heading_dyn.as_float().ok() {
                    skill_instance = skill_instance.with_heading(Angle::from_radians(heading_f64));
                } else {
                    return Err(Box::new(EvalAltResult::ErrorMismatchDataType(
                        "Expected float for heading".to_string(),
                        heading_dyn.type_name().into(),
                        Position::NONE,
                    )));
                }
            }
            if let Some(with_ball_dyn) = opts.get("with_ball") {
                if let Some(with_ball_bool) = with_ball_dyn.as_bool().ok() {
                    if with_ball_bool {
                        skill_instance = skill_instance.with_ball();
                    }
                } else {
                    return Err(Box::new(EvalAltResult::ErrorMismatchDataType(
                        "Expected bool for with_ball".to_string(),
                        with_ball_dyn.type_name().into(),
                        Position::NONE,
                    )));
                }
            }
            if let Some(avoid_ball_dyn) = opts.get("avoid_ball") {
                if let Some(avoid_ball_bool) = avoid_ball_dyn.as_bool().ok() {
                    if avoid_ball_bool {
                        skill_instance = skill_instance.avoid_ball();
                    }
                } else {
                    return Err(Box::new(EvalAltResult::ErrorMismatchDataType(
                        "Expected bool for avoid_ball".to_string(),
                        avoid_ball_dyn.type_name().into(),
                        Position::NONE,
                    )));
                }
            }
        }
        skill_to_action_node(
            SkillEnum::GoToPosition(skill_instance),
            action_description,
            "GoToPosition",
        )
    }

    #[rhai_fn(name = "FaceAngle", return_raw)]
    pub fn face_angle_action(
        angle_rad: f64,
        options: Option<Map>,
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let mut skill_instance = FaceSkillInstance::new(Angle::from_radians(angle_rad));
        if let Some(opts) = options {
            if let Some(with_ball_dyn) = opts.get("with_ball") {
                if let Some(with_ball_bool) = with_ball_dyn.as_bool().ok() {
                    if with_ball_bool {
                        skill_instance = skill_instance.with_ball();
                    }
                } // Error handling for type mismatch omitted for brevity, add if necessary
            }
        }
        skill_to_action_node(
            SkillEnum::Face(skill_instance),
            action_description,
            "FaceAngle",
        )
    }

    #[rhai_fn(name = "FaceTowardsPosition", return_raw)]
    pub fn face_towards_position_action(
        x: f64,
        y: f64,
        options: Option<Map>,
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let mut skill_instance = FaceSkillInstance::towards_position(CoreVector2::new(x, y));
        if let Some(opts) = options {
            if let Some(with_ball_dyn) = opts.get("with_ball") {
                if let Some(with_ball_bool) = with_ball_dyn.as_bool().ok() {
                    if with_ball_bool {
                        skill_instance = skill_instance.with_ball();
                    }
                }
            }
        }
        skill_to_action_node(
            SkillEnum::Face(skill_instance),
            action_description,
            "FaceTowardsPosition",
        )
    }

    #[rhai_fn(name = "FaceTowardsOwnPlayer", return_raw)]
    pub fn face_towards_own_player_action(
        player_id: i64, // Rhai typically uses i64 for integers
        options: Option<Map>,
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let id = PlayerId::new(player_id as u32);
        let mut skill_instance = FaceSkillInstance::towards_own_player(id);
        if let Some(opts) = options {
            if let Some(with_ball_dyn) = opts.get("with_ball") {
                if let Some(with_ball_bool) = with_ball_dyn.as_bool().ok() {
                    if with_ball_bool {
                        skill_instance = skill_instance.with_ball();
                    }
                }
            }
        }
        skill_to_action_node(
            SkillEnum::Face(skill_instance),
            action_description,
            "FaceTowardsOwnPlayer",
        )
    }

    #[rhai_fn(name = "Kick", return_raw)]
    pub fn kick_action(
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = KickSkillInstance::new();
        skill_to_action_node(SkillEnum::Kick(skill_instance), action_description, "Kick")
    }

    #[rhai_fn(name = "Wait", return_raw)]
    pub fn wait_action(
        duration_secs: f64,
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = WaitSkillInstance::new_secs_f64(duration_secs);
        skill_to_action_node(SkillEnum::Wait(skill_instance), action_description, "Wait")
    }

    #[rhai_fn(name = "FetchBall", return_raw)]
    pub fn fetch_ball_action(
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = FetchBallSkillInstance::new();
        skill_to_action_node(
            SkillEnum::FetchBall(skill_instance),
            action_description,
            "FetchBall",
        )
    }

    #[rhai_fn(name = "InterceptBall", return_raw)]
    pub fn intercept_ball_action(
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = InterceptBallSkillInstance::new();
        skill_to_action_node(
            SkillEnum::InterceptBall(skill_instance),
            action_description,
            "InterceptBall",
        )
    }

    #[rhai_fn(name = "ApproachBall", return_raw)]
    pub fn approach_ball_action(
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = ApproachBallSkillInstance::new();
        skill_to_action_node(
            SkillEnum::ApproachBall(skill_instance),
            action_description,
            "ApproachBall",
        )
    }

    #[rhai_fn(name = "FetchBallWithHeadingAngle", return_raw)]
    pub fn fetch_ball_with_heading_angle_action(
        angle_rad: f64,
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = FetchBallWithHeadingSkillInstance::new(Angle::from_radians(angle_rad));
        skill_to_action_node(
            SkillEnum::FetchBallWithHeading(skill_instance),
            action_description,
            "FetchBallWithHeadingAngle",
        )
    }

    #[rhai_fn(name = "FetchBallWithHeadingPosition", return_raw)]
    pub fn fetch_ball_with_heading_position_action(
        x: f64,
        y: f64,
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance =
            FetchBallWithHeadingSkillInstance::towards_position(CoreVector2::new(x, y));
        skill_to_action_node(
            SkillEnum::FetchBallWithHeading(skill_instance),
            action_description,
            "FetchBallWithHeadingPosition",
        )
    }

    #[rhai_fn(name = "FetchBallWithHeadingPlayer", return_raw)]
    pub fn fetch_ball_with_heading_player_action(
        player_id: i64,
        action_description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let id = PlayerId::new(player_id as u32);
        let skill_instance = FetchBallWithHeadingSkillInstance::towards_own_player(id);
        skill_to_action_node(
            SkillEnum::FetchBallWithHeading(skill_instance),
            action_description,
            "FetchBallWithHeadingPlayer",
        )
    }
}
