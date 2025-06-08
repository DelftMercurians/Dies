use rhai::plugin::*;
use rhai::{
    Array, CustomType, EvalAltResult, FnPtr, Map, NativeCallContext, Position, TypeBuilder,
};

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

#[allow(dead_code)]
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

    use super::*;

    pub type BehaviorNode = RhaiBehaviorNode;
    pub type Vec2 = CoreVector2;

    // PLAYER ID HELPERS
    #[rhai_fn(name = "to_string")]
    /// Returns the string representation of the player id.
    pub fn to_string(id: PlayerId) -> String {
        id.to_string()
    }

    #[rhai_fn(name = "hash_float")]
    /// Returns a float between 0 and 1 based on the player id.
    /// This is used to ensure that the same player id always produces the same hash.
    /// This can be used to induce different behavior for different players.
    pub fn hash(id: PlayerId) -> f64 {
        // Fast hash by multiplying by a large prime and taking fractional part
        (id.as_u32() as f64 * 0.6180339887498949) % 1.0
    }

    // VECTOR2 HELPERS
    #[rhai_fn(name = "vec2")]
    pub fn vec2_constructor(x: f64, y: f64) -> Vec2 {
        CoreVector2::new(x, y)
    }

    // BEHAVIOR NODE CONSTRUCTORS
    #[rhai_fn(name = "Select", return_raw)]
    pub fn select_node_with_description(
        children_dyn: Array,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        select_node_impl(children_dyn, Some(description))
    }

    #[rhai_fn(name = "Select", return_raw)]
    pub fn select_node_basic(children_dyn: Array) -> Result<BehaviorNode, Box<EvalAltResult>> {
        select_node_impl(children_dyn, None)
    }

    fn select_node_impl(
        children_dyn: Array,
        description: Option<&str>,
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
            SelectNode::new(rust_children, description.map(|s| s.to_string())),
        )))
    }

    #[rhai_fn(name = "Sequence", return_raw)]
    pub fn sequence_node_with_description(
        children_dyn: Array,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        sequence_node_impl(children_dyn, Some(description))
    }

    #[rhai_fn(name = "Sequence", return_raw)]
    pub fn sequence_node_basic(children_dyn: Array) -> Result<BehaviorNode, Box<EvalAltResult>> {
        sequence_node_impl(children_dyn, None)
    }

    fn sequence_node_impl(
        children_dyn: Array,
        description: Option<&str>,
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
            SequenceNode::new(rust_children, description.map(|s| s.to_string())),
        )))
    }

    #[rhai_fn(name = "Guard", return_raw)]
    pub fn guard_node(
        context: NativeCallContext,
        condition_fn_ptr: FnPtr,
        child_node: BehaviorNode,
        cond_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let situation = Situation::new_fn(
            BtCallback::new_rhai(&context, condition_fn_ptr),
            cond_description,
        );
        let guard_desc_override = Some(format!(
            "Guard_If_{}_Then_{}",
            cond_description,
            child_node.0.description()
        ));
        Ok(RhaiBehaviorNode(BehaviorNodeTypeEnum::Guard(
            GuardNode::new(situation, child_node.0, guard_desc_override),
        )))
    }

    #[rhai_fn(name = "Semaphore", return_raw)]
    pub fn semaphore_node_with_description(
        child_node: BehaviorNode,
        id: &str,
        max_count: i64,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        semaphore_node_impl(child_node, id, max_count, Some(description))
    }

    #[rhai_fn(name = "Semaphore", return_raw)]
    pub fn semaphore_node_basic(
        child_node: BehaviorNode,
        id: &str,
        max_count: i64,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        semaphore_node_impl(child_node, id, max_count, None)
    }

    fn semaphore_node_impl(
        child_node: BehaviorNode,
        id: &str,
        max_count: i64,
        description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        Ok(RhaiBehaviorNode(BehaviorNodeTypeEnum::Semaphore(
            SemaphoreNode::new(
                child_node.0,
                id.to_string(),
                max_count as usize,
                description.map(|s| s.to_string()),
            ),
        )))
    }

    #[rhai_fn(name = "ScoringSelect", return_raw)]
    pub fn scoring_select_node_with_description(
        context: NativeCallContext,
        children_scorers_dyn: Array,
        hysteresis_margin: f64,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        scoring_select_node_impl(
            context,
            children_scorers_dyn,
            hysteresis_margin,
            Some(description),
        )
    }

    #[rhai_fn(name = "ScoringSelect", return_raw)]
    pub fn scoring_select_node_basic(
        context: NativeCallContext,
        children_scorers_dyn: Array,
        hysteresis_margin: f64,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        scoring_select_node_impl(context, children_scorers_dyn, hysteresis_margin, None)
    }

    fn scoring_select_node_impl(
        context: NativeCallContext,
        children_scorers_dyn: Array,
        hysteresis_margin: f64,
        description: Option<&str>,
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
            ScoringSelectNode::new(
                rust_children_scorers,
                hysteresis_margin,
                description.map(|s| s.to_string()),
            ),
        )))
    }

    // Helper to create ActionNode from SkillEnum
    fn skill_to_action_node(
        skill_enum: SkillEnum,
        description: Option<&str>,
        skill_name_for_auto_desc: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let final_description = match description {
            Some(desc) => desc.to_string(),
            None => format!("Action_{}", skill_name_for_auto_desc),
        };
        Ok(RhaiBehaviorNode(BehaviorNodeTypeEnum::Action(
            ActionNode::new(skill_enum, Some(final_description)),
        )))
    }

    // SKILL CONSTRUCTORS (returning Action Nodes)
    // GoToPosition overloads for default parameter simulation
    #[rhai_fn(name = "GoToPosition", return_raw)]
    pub fn goto_action_full(
        x: f64,
        y: f64,
        options: Map,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        goto_action_with_options_impl(x, y, options, Some(action_description))
    }

    #[rhai_fn(name = "GoToPosition", return_raw)]
    pub fn goto_action_with_options(
        x: f64,
        y: f64,
        options: Map,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        goto_action_with_options_impl(x, y, options, None)
    }

    #[rhai_fn(name = "GoToPosition", return_raw)]
    pub fn goto_action_with_description(
        x: f64,
        y: f64,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        goto_action_basic_impl(x, y, Some(action_description))
    }

    #[rhai_fn(name = "GoToPosition", return_raw)]
    pub fn goto_action_basic(x: f64, y: f64) -> Result<BehaviorNode, Box<EvalAltResult>> {
        goto_action_basic_impl(x, y, None)
    }

    // Implementation functions for GoToPosition
    fn goto_action_with_options_impl(
        x: f64,
        y: f64,
        options: Map,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let mut skill_instance = GoToPositionSkillInstance::new(CoreVector2::new(x, y));
        if let Some(heading_dyn) = options.get("heading") {
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
        if let Some(with_ball_dyn) = options.get("with_ball") {
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
        if let Some(avoid_ball_dyn) = options.get("avoid_ball") {
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
        skill_to_action_node(
            SkillEnum::GoToPosition(skill_instance),
            action_description,
            "GoToPosition",
        )
    }

    fn goto_action_basic_impl(
        x: f64,
        y: f64,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = GoToPositionSkillInstance::new(CoreVector2::new(x, y));
        skill_to_action_node(
            SkillEnum::GoToPosition(skill_instance),
            action_description,
            "GoToPosition",
        )
    }

    // FaceAngle overloads for default parameter simulation
    #[rhai_fn(name = "FaceAngle", return_raw)]
    pub fn face_angle_action_full(
        angle_rad: f64,
        options: Map,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_angle_action_with_options_impl(angle_rad, options, Some(action_description))
    }

    #[rhai_fn(name = "FaceAngle", return_raw)]
    pub fn face_angle_action_with_options(
        angle_rad: f64,
        options: Map,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_angle_action_with_options_impl(angle_rad, options, None)
    }

    #[rhai_fn(name = "FaceAngle", return_raw)]
    pub fn face_angle_action_with_description(
        angle_rad: f64,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_angle_action_basic_impl(angle_rad, Some(action_description))
    }

    #[rhai_fn(name = "FaceAngle", return_raw)]
    pub fn face_angle_action_basic(angle_rad: f64) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_angle_action_basic_impl(angle_rad, None)
    }

    // Implementation functions for FaceAngle
    fn face_angle_action_with_options_impl(
        angle_rad: f64,
        options: Map,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let mut skill_instance = FaceSkillInstance::new(Angle::from_radians(angle_rad));
        if let Some(with_ball_dyn) = options.get("with_ball") {
            if let Some(with_ball_bool) = with_ball_dyn.as_bool().ok() {
                if with_ball_bool {
                    skill_instance = skill_instance.with_ball();
                }
            }
        }
        skill_to_action_node(
            SkillEnum::Face(skill_instance),
            action_description,
            "FaceAngle",
        )
    }

    fn face_angle_action_basic_impl(
        angle_rad: f64,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = FaceSkillInstance::new(Angle::from_radians(angle_rad));
        skill_to_action_node(
            SkillEnum::Face(skill_instance),
            action_description,
            "FaceAngle",
        )
    }

    // FaceTowardsPosition overloads for default parameter simulation
    #[rhai_fn(name = "FaceTowardsPosition", return_raw)]
    pub fn face_towards_position_action_full(
        x: f64,
        y: f64,
        options: Map,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_towards_position_action_with_options_impl(x, y, options, Some(action_description))
    }

    #[rhai_fn(name = "FaceTowardsPosition", return_raw)]
    pub fn face_towards_position_action_with_options(
        x: f64,
        y: f64,
        options: Map,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_towards_position_action_with_options_impl(x, y, options, None)
    }

    #[rhai_fn(name = "FaceTowardsPosition", return_raw)]
    pub fn face_towards_position_action_with_description(
        x: f64,
        y: f64,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_towards_position_action_basic_impl(x, y, Some(action_description))
    }

    #[rhai_fn(name = "FaceTowardsPosition", return_raw)]
    pub fn face_towards_position_action_basic(
        x: f64,
        y: f64,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_towards_position_action_basic_impl(x, y, None)
    }

    // Implementation functions for FaceTowardsPosition
    fn face_towards_position_action_with_options_impl(
        x: f64,
        y: f64,
        options: Map,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let mut skill_instance = FaceSkillInstance::towards_position(CoreVector2::new(x, y));
        if let Some(with_ball_dyn) = options.get("with_ball") {
            if let Some(with_ball_bool) = with_ball_dyn.as_bool().ok() {
                if with_ball_bool {
                    skill_instance = skill_instance.with_ball();
                }
            }
        }
        skill_to_action_node(
            SkillEnum::Face(skill_instance),
            action_description,
            "FaceTowardsPosition",
        )
    }

    fn face_towards_position_action_basic_impl(
        x: f64,
        y: f64,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = FaceSkillInstance::towards_position(CoreVector2::new(x, y));
        skill_to_action_node(
            SkillEnum::Face(skill_instance),
            action_description,
            "FaceTowardsPosition",
        )
    }

    // FaceTowardsOwnPlayer overloads for default parameter simulation
    #[rhai_fn(name = "FaceTowardsOwnPlayer", return_raw)]
    pub fn face_towards_own_player_action_full(
        player_id: i64,
        options: Map,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_towards_own_player_action_with_options_impl(
            player_id,
            options,
            Some(action_description),
        )
    }

    #[rhai_fn(name = "FaceTowardsOwnPlayer", return_raw)]
    pub fn face_towards_own_player_action_with_options(
        player_id: i64,
        options: Map,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_towards_own_player_action_with_options_impl(player_id, options, None)
    }

    #[rhai_fn(name = "FaceTowardsOwnPlayer", return_raw)]
    pub fn face_towards_own_player_action_with_description(
        player_id: i64,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_towards_own_player_action_basic_impl(player_id, Some(action_description))
    }

    #[rhai_fn(name = "FaceTowardsOwnPlayer", return_raw)]
    pub fn face_towards_own_player_action_basic(
        player_id: i64,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_towards_own_player_action_basic_impl(player_id, None)
    }

    // Implementation functions for FaceTowardsOwnPlayer
    fn face_towards_own_player_action_with_options_impl(
        player_id: i64,
        options: Map,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let id = PlayerId::new(player_id as u32);
        let mut skill_instance = FaceSkillInstance::towards_own_player(id);
        if let Some(with_ball_dyn) = options.get("with_ball") {
            if let Some(with_ball_bool) = with_ball_dyn.as_bool().ok() {
                if with_ball_bool {
                    skill_instance = skill_instance.with_ball();
                }
            }
        }
        skill_to_action_node(
            SkillEnum::Face(skill_instance),
            action_description,
            "FaceTowardsOwnPlayer",
        )
    }

    fn face_towards_own_player_action_basic_impl(
        player_id: i64,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let id = PlayerId::new(player_id as u32);
        let skill_instance = FaceSkillInstance::towards_own_player(id);
        skill_to_action_node(
            SkillEnum::Face(skill_instance),
            action_description,
            "FaceTowardsOwnPlayer",
        )
    }

    // Kick overloads for default parameter simulation
    #[rhai_fn(name = "Kick", return_raw)]
    pub fn kick_action_with_description(
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        kick_action_impl(Some(action_description))
    }

    #[rhai_fn(name = "Kick", return_raw)]
    pub fn kick_action_basic() -> Result<BehaviorNode, Box<EvalAltResult>> {
        kick_action_impl(None)
    }

    // Implementation function for Kick
    fn kick_action_impl(
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = KickSkillInstance::new();
        skill_to_action_node(SkillEnum::Kick(skill_instance), action_description, "Kick")
    }

    // Wait overloads for default parameter simulation
    #[rhai_fn(name = "Wait", return_raw)]
    pub fn wait_action_with_description(
        duration_secs: f64,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        wait_action_impl(duration_secs, Some(action_description))
    }

    #[rhai_fn(name = "Wait", return_raw)]
    pub fn wait_action_basic(duration_secs: f64) -> Result<BehaviorNode, Box<EvalAltResult>> {
        wait_action_impl(duration_secs, None)
    }

    // Implementation function for Wait
    fn wait_action_impl(
        duration_secs: f64,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = WaitSkillInstance::new_secs_f64(duration_secs);
        skill_to_action_node(SkillEnum::Wait(skill_instance), action_description, "Wait")
    }

    // FetchBall overloads for default parameter simulation
    #[rhai_fn(name = "FetchBall", return_raw)]
    pub fn fetch_ball_action_with_description(
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_action_impl(Some(action_description))
    }

    #[rhai_fn(name = "FetchBall", return_raw)]
    pub fn fetch_ball_action_basic() -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_action_impl(None)
    }

    // Implementation function for FetchBall
    fn fetch_ball_action_impl(
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = FetchBallSkillInstance::new();
        skill_to_action_node(
            SkillEnum::FetchBall(skill_instance),
            action_description,
            "FetchBall",
        )
    }

    // InterceptBall overloads for default parameter simulation
    #[rhai_fn(name = "InterceptBall", return_raw)]
    pub fn intercept_ball_action_with_description(
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        intercept_ball_action_impl(Some(action_description))
    }

    #[rhai_fn(name = "InterceptBall", return_raw)]
    pub fn intercept_ball_action_basic() -> Result<BehaviorNode, Box<EvalAltResult>> {
        intercept_ball_action_impl(None)
    }

    // Implementation function for InterceptBall
    fn intercept_ball_action_impl(
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = InterceptBallSkillInstance::new();
        skill_to_action_node(
            SkillEnum::InterceptBall(skill_instance),
            action_description,
            "InterceptBall",
        )
    }

    // ApproachBall overloads for default parameter simulation
    #[rhai_fn(name = "ApproachBall", return_raw)]
    pub fn approach_ball_action_with_description(
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        approach_ball_action_impl(Some(action_description))
    }

    #[rhai_fn(name = "ApproachBall", return_raw)]
    pub fn approach_ball_action_basic() -> Result<BehaviorNode, Box<EvalAltResult>> {
        approach_ball_action_impl(None)
    }

    // Implementation function for ApproachBall
    fn approach_ball_action_impl(
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = ApproachBallSkillInstance::new();
        skill_to_action_node(
            SkillEnum::ApproachBall(skill_instance),
            action_description,
            "ApproachBall",
        )
    }

    // FetchBallWithHeadingAngle overloads for default parameter simulation
    #[rhai_fn(name = "FetchBallWithHeadingAngle", return_raw)]
    pub fn fetch_ball_with_heading_angle_action_with_description(
        angle_rad: f64,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_angle_action_impl(angle_rad, Some(action_description))
    }

    #[rhai_fn(name = "FetchBallWithHeadingAngle", return_raw)]
    pub fn fetch_ball_with_heading_angle_action_basic(
        angle_rad: f64,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_angle_action_impl(angle_rad, None)
    }

    // Implementation function for FetchBallWithHeadingAngle
    fn fetch_ball_with_heading_angle_action_impl(
        angle_rad: f64,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance = FetchBallWithHeadingSkillInstance::new(Angle::from_radians(angle_rad));
        skill_to_action_node(
            SkillEnum::FetchBallWithHeading(skill_instance),
            action_description,
            "FetchBallWithHeadingAngle",
        )
    }

    // FetchBallWithHeadingPosition overloads for default parameter simulation
    #[rhai_fn(name = "FetchBallWithHeadingPosition", return_raw)]
    pub fn fetch_ball_with_heading_position_action_with_description(
        x: f64,
        y: f64,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_position_action_impl(x, y, Some(action_description))
    }

    #[rhai_fn(name = "FetchBallWithHeadingPosition", return_raw)]
    pub fn fetch_ball_with_heading_position_action_basic(
        x: f64,
        y: f64,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_position_action_impl(x, y, None)
    }

    // Implementation function for FetchBallWithHeadingPosition
    fn fetch_ball_with_heading_position_action_impl(
        x: f64,
        y: f64,
        action_description: Option<&str>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let skill_instance =
            FetchBallWithHeadingSkillInstance::towards_position(CoreVector2::new(x, y));
        skill_to_action_node(
            SkillEnum::FetchBallWithHeading(skill_instance),
            action_description,
            "FetchBallWithHeadingPosition",
        )
    }

    // FetchBallWithHeadingPlayer overloads for default parameter simulation
    #[rhai_fn(name = "FetchBallWithHeadingPlayer", return_raw)]
    pub fn fetch_ball_with_heading_player_action_with_description(
        player_id: i64,
        action_description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_player_action_impl(player_id, Some(action_description))
    }

    #[rhai_fn(name = "FetchBallWithHeadingPlayer", return_raw)]
    pub fn fetch_ball_with_heading_player_action_basic(
        player_id: i64,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_player_action_impl(player_id, None)
    }

    // Implementation function for FetchBallWithHeadingPlayer
    fn fetch_ball_with_heading_player_action_impl(
        player_id: i64,
        action_description: Option<&str>,
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
