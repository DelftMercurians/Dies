use rhai::plugin::*;
use rhai::{Array, Engine, EvalAltResult, FnPtr, Map, NativeCallContext, Position, INT};

use super::bt_callback::BtCallback;
use super::role_assignment::{Role, RoleAssignmentProblem, RoleBuilder};
use crate::behavior_tree::{
    ActionNode, BehaviorNode as BehaviorNodeTypeEnum, GuardNode, ScoringSelectNode, SelectNode,
    SemaphoreNode, SequenceNode, Situation,
};
use crate::behavior_tree::{BehaviorTree, RobotSituation};
use dies_core::{PlayerId, Vector2 as CoreVector2};

#[export_module]
pub mod bt_rhai_plugin {
    use crate::behavior_tree::rhai_types::RhaiBehaviorNode;
    use crate::behavior_tree::{Argument, FaceTarget, HeadingTarget, SkillDefinition};

    use super::*;

    pub type BehaviorNode = RhaiBehaviorNode;
    pub type Vec2 = CoreVector2;

    // PLAYER ID HELPERS
    #[rhai_fn(name = "to_string")]
    /// Returns the string representation of the player id.
    pub fn to_string(id: &mut PlayerId) -> String {
        id.to_string()
    }

    #[rhai_fn(name = "hash_float")]
    /// Returns a float between 0 and 1 based on the player id.
    /// This is used to ensure that the same player id always produces the same hash.
    /// This can be used to induce different behavior for different players.
    pub fn hash(id: &mut PlayerId) -> f64 {
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

    fn to_action_node(
        skill_def: SkillDefinition,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        Ok(RhaiBehaviorNode(BehaviorNodeTypeEnum::Action(
            ActionNode::new(skill_def, description),
        )))
    }

    // SKILL CONSTRUCTORS (returning Action Nodes)
    #[rhai_fn(name = "GoToPosition", return_raw)]
    pub fn goto_action(
        context: NativeCallContext,
        target: rhai::Dynamic,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        goto_action_impl(context, target, None, None)
    }

    #[rhai_fn(name = "GoToPosition", return_raw)]
    pub fn goto_action_with_options(
        context: NativeCallContext,
        target: rhai::Dynamic,
        options: Map,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        goto_action_impl(context, target, Some(options), None)
    }

    #[rhai_fn(name = "GoToPosition", return_raw)]
    pub fn goto_action_with_description(
        context: NativeCallContext,
        target: rhai::Dynamic,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        goto_action_impl(context, target, None, Some(description.to_string()))
    }

    #[rhai_fn(name = "GoToPosition", return_raw)]
    pub fn goto_action_with_options_and_description(
        context: NativeCallContext,
        target: rhai::Dynamic,
        options: Map,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        goto_action_impl(
            context,
            target,
            Some(options),
            Some(description.to_string()),
        )
    }

    fn goto_action_impl(
        context: NativeCallContext,
        target: rhai::Dynamic,
        options: Option<Map>,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let target_pos = Argument::from_rhai(target, &context)?;

        let mut heading_arg = None;
        let mut with_ball_arg = Argument::Static(false);
        let mut avoid_ball_arg = Argument::Static(false);

        if let Some(options_map) = options {
            if let Some(heading_dyn) = options_map.get("heading") {
                heading_arg = Some(Argument::from_rhai(heading_dyn.clone(), &context)?);
            }
            if let Some(with_ball_dyn) = options_map.get("with_ball") {
                with_ball_arg = Argument::from_rhai(with_ball_dyn.clone(), &context)?;
            }
            if let Some(avoid_ball_dyn) = options_map.get("avoid_ball") {
                avoid_ball_arg = Argument::from_rhai(avoid_ball_dyn.clone(), &context)?;
            }
        }

        let skill_def = SkillDefinition::GoToPosition {
            target_pos,
            target_heading: heading_arg,
            with_ball: with_ball_arg,
            avoid_ball: avoid_ball_arg,
            // Using reasonable defaults for unexposed parameters
            target_velocity: Argument::Static(CoreVector2::zeros()),
            pos_tolerance: Argument::Static(10.0),
            velocity_tolerance: Argument::Static(10.0),
        };

        to_action_node(skill_def, description)
    }

    #[rhai_fn(name = "FaceAngle", return_raw)]
    pub fn face_angle_action(
        context: NativeCallContext,
        angle_rad: rhai::Dynamic,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_angle_action_impl(context, angle_rad, None, None)
    }

    #[rhai_fn(name = "FaceAngle", return_raw)]
    pub fn face_angle_action_with_options(
        context: NativeCallContext,
        angle_rad: rhai::Dynamic,
        options: Map,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_angle_action_impl(context, angle_rad, Some(options), None)
    }

    #[rhai_fn(name = "FaceAngle", return_raw)]
    pub fn face_angle_action_with_description(
        context: NativeCallContext,
        angle_rad: rhai::Dynamic,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_angle_action_impl(context, angle_rad, None, Some(description.to_string()))
    }

    #[rhai_fn(name = "FaceAngle", return_raw)]
    pub fn face_angle_action_with_options_and_description(
        context: NativeCallContext,
        angle_rad: rhai::Dynamic,
        options: Map,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_angle_action_impl(
            context,
            angle_rad,
            Some(options),
            Some(description.to_string()),
        )
    }

    fn face_angle_action_impl(
        context: NativeCallContext,
        angle_rad: rhai::Dynamic,
        options: Option<Map>,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let angle_arg = Argument::from_rhai(angle_rad, &context)?;
        let mut with_ball_arg = Argument::Static(false);

        if let Some(options_map) = options {
            if let Some(with_ball_dyn) = options_map.get("with_ball") {
                with_ball_arg = Argument::from_rhai(with_ball_dyn.clone(), &context)?;
            }
        }

        let skill_def = SkillDefinition::Face {
            target: FaceTarget::Angle(angle_arg),
            with_ball: with_ball_arg,
        };
        to_action_node(skill_def, description)
    }

    #[rhai_fn(name = "FaceTowardsPosition", return_raw)]
    pub fn face_pos_action(
        context: NativeCallContext,
        target: rhai::Dynamic,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_pos_action_impl(context, target, None, None)
    }

    #[rhai_fn(name = "FaceTowardsPosition", return_raw)]
    pub fn face_pos_action_with_options(
        context: NativeCallContext,
        target: rhai::Dynamic,
        options: Map,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_pos_action_impl(context, target, Some(options), None)
    }

    #[rhai_fn(name = "FaceTowardsPosition", return_raw)]
    pub fn face_pos_action_with_description(
        context: NativeCallContext,
        target: rhai::Dynamic,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_pos_action_impl(context, target, None, Some(description.to_string()))
    }

    #[rhai_fn(name = "FaceTowardsPosition", return_raw)]
    pub fn face_pos_action_with_options_and_description(
        context: NativeCallContext,
        target: rhai::Dynamic,
        options: Map,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_pos_action_impl(
            context,
            target,
            Some(options),
            Some(description.to_string()),
        )
    }

    fn face_pos_action_impl(
        context: NativeCallContext,
        target: rhai::Dynamic,
        options: Option<Map>,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let target_arg = Argument::from_rhai(target, &context)?;
        let mut with_ball_arg = Argument::Static(false);

        if let Some(options_map) = options {
            if let Some(with_ball_dyn) = options_map.get("with_ball") {
                with_ball_arg = Argument::from_rhai(with_ball_dyn.clone(), &context)?;
            }
        }

        let skill_def = SkillDefinition::Face {
            target: FaceTarget::Position(target_arg),
            with_ball: with_ball_arg,
        };
        to_action_node(skill_def, description)
    }

    #[rhai_fn(name = "FaceTowardsOwnPlayer", return_raw)]
    pub fn face_player_action(
        context: NativeCallContext,
        player_id: rhai::Dynamic,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_player_action_impl(context, player_id, None, None)
    }

    #[rhai_fn(name = "FaceTowardsOwnPlayer", return_raw)]
    pub fn face_player_action_with_options(
        context: NativeCallContext,
        player_id: rhai::Dynamic,
        options: Map,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_player_action_impl(context, player_id, Some(options), None)
    }

    #[rhai_fn(name = "FaceTowardsOwnPlayer", return_raw)]
    pub fn face_player_action_with_description(
        context: NativeCallContext,
        player_id: rhai::Dynamic,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_player_action_impl(context, player_id, None, Some(description.to_string()))
    }

    #[rhai_fn(name = "FaceTowardsOwnPlayer", return_raw)]
    pub fn face_player_action_with_options_and_description(
        context: NativeCallContext,
        player_id: rhai::Dynamic,
        options: Map,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        face_player_action_impl(
            context,
            player_id,
            Some(options),
            Some(description.to_string()),
        )
    }

    fn face_player_action_impl(
        context: NativeCallContext,
        player_id: rhai::Dynamic,
        options: Option<Map>,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let player_id_arg = Argument::from_rhai(player_id, &context)?;
        let mut with_ball_arg = Argument::Static(false);

        if let Some(options_map) = options {
            if let Some(with_ball_dyn) = options_map.get("with_ball") {
                with_ball_arg = Argument::from_rhai(with_ball_dyn.clone(), &context)?;
            }
        }

        let skill_def = SkillDefinition::Face {
            target: FaceTarget::OwnPlayer(player_id_arg),
            with_ball: with_ball_arg,
        };
        to_action_node(skill_def, description)
    }

    #[rhai_fn(name = "Kick", return_raw)]
    pub fn kick_action(description: &str) -> Result<BehaviorNode, Box<EvalAltResult>> {
        to_action_node(SkillDefinition::Kick, Some(description.to_string()))
    }

    #[rhai_fn(name = "Kick", return_raw)]
    pub fn kick_action_basic() -> Result<BehaviorNode, Box<EvalAltResult>> {
        to_action_node(SkillDefinition::Kick, None)
    }

    #[rhai_fn(name = "Wait", return_raw)]
    pub fn wait_action(
        context: NativeCallContext,
        duration_secs: rhai::Dynamic,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        wait_action_impl(context, duration_secs, None)
    }

    #[rhai_fn(name = "Wait", return_raw)]
    pub fn wait_action_with_description(
        context: NativeCallContext,
        duration_secs: rhai::Dynamic,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        wait_action_impl(context, duration_secs, Some(description.to_string()))
    }

    fn wait_action_impl(
        context: NativeCallContext,
        duration_secs: rhai::Dynamic,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let duration_arg = Argument::from_rhai(duration_secs, &context)?;
        let skill_def = SkillDefinition::Wait {
            duration_secs: duration_arg,
        };
        to_action_node(skill_def, description)
    }

    #[rhai_fn(name = "FetchBall", return_raw)]
    pub fn fetch_ball_action(description: &str) -> Result<BehaviorNode, Box<EvalAltResult>> {
        to_action_node(SkillDefinition::FetchBall, Some(description.to_string()))
    }

    #[rhai_fn(name = "FetchBall", return_raw)]
    pub fn fetch_ball_action_basic() -> Result<BehaviorNode, Box<EvalAltResult>> {
        to_action_node(SkillDefinition::FetchBall, None)
    }

    #[rhai_fn(name = "InterceptBall", return_raw)]
    pub fn intercept_ball_action(description: &str) -> Result<BehaviorNode, Box<EvalAltResult>> {
        to_action_node(
            SkillDefinition::InterceptBall,
            Some(description.to_string()),
        )
    }

    #[rhai_fn(name = "InterceptBall", return_raw)]
    pub fn intercept_ball_action_basic() -> Result<BehaviorNode, Box<EvalAltResult>> {
        to_action_node(SkillDefinition::InterceptBall, None)
    }

    #[rhai_fn(name = "ApproachBall", return_raw)]
    pub fn approach_ball_action(description: &str) -> Result<BehaviorNode, Box<EvalAltResult>> {
        to_action_node(SkillDefinition::ApproachBall, Some(description.to_string()))
    }

    #[rhai_fn(name = "ApproachBall", return_raw)]
    pub fn approach_ball_action_basic() -> Result<BehaviorNode, Box<EvalAltResult>> {
        to_action_node(SkillDefinition::ApproachBall, None)
    }

    #[rhai_fn(name = "FetchBallWithHeadingAngle", return_raw)]
    pub fn fetch_ball_with_heading_angle_action(
        context: NativeCallContext,
        angle_rad: rhai::Dynamic,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_angle_impl(context, angle_rad, None)
    }

    #[rhai_fn(name = "FetchBallWithHeadingAngle", return_raw)]
    pub fn fetch_ball_with_heading_angle_action_with_description(
        context: NativeCallContext,
        angle_rad: rhai::Dynamic,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_angle_impl(context, angle_rad, Some(description.to_string()))
    }

    fn fetch_ball_with_heading_angle_impl(
        context: NativeCallContext,
        angle_rad: rhai::Dynamic,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let angle_arg = Argument::from_rhai(angle_rad, &context)?;
        let skill_def = SkillDefinition::FetchBallWithHeading {
            target: HeadingTarget::Angle(angle_arg),
        };
        to_action_node(skill_def, description)
    }

    #[rhai_fn(name = "FetchBallWithHeadingPosition", return_raw)]
    pub fn fetch_ball_with_heading_pos_action(
        context: NativeCallContext,
        target: rhai::Dynamic,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_pos_impl(context, target, None)
    }

    #[rhai_fn(name = "FetchBallWithHeadingPosition", return_raw)]
    pub fn fetch_ball_with_heading_pos_action_with_description(
        context: NativeCallContext,
        target: rhai::Dynamic,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_pos_impl(context, target, Some(description.to_string()))
    }

    fn fetch_ball_with_heading_pos_impl(
        context: NativeCallContext,
        target: rhai::Dynamic,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let target_arg = Argument::from_rhai(target, &context)?;
        let skill_def = SkillDefinition::FetchBallWithHeading {
            target: HeadingTarget::Position(target_arg),
        };
        to_action_node(skill_def, description)
    }

    #[rhai_fn(name = "FetchBallWithHeadingPlayer", return_raw)]
    pub fn fetch_ball_with_heading_player_action(
        context: NativeCallContext,
        player_id: rhai::Dynamic,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_player_impl(context, player_id, None)
    }

    #[rhai_fn(name = "FetchBallWithHeadingPlayer", return_raw)]
    pub fn fetch_ball_with_heading_player_action_with_description(
        context: NativeCallContext,
        player_id: rhai::Dynamic,
        description: &str,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        fetch_ball_with_heading_player_impl(context, player_id, Some(description.to_string()))
    }

    fn fetch_ball_with_heading_player_impl(
        context: NativeCallContext,
        player_id: rhai::Dynamic,
        description: Option<String>,
    ) -> Result<BehaviorNode, Box<EvalAltResult>> {
        let player_id_arg = Argument::from_rhai(player_id, &context)?;
        let skill_def = SkillDefinition::FetchBallWithHeading {
            target: HeadingTarget::OwnPlayer(player_id_arg),
        };
        to_action_node(skill_def, description)
    }

    // ROLE ASSIGNMENT API
    #[rhai_fn(name = "Role")]
    pub fn role_constructor(name: &str) -> RoleBuilder {
        RoleBuilder::new(name)
    }

    // RoleBuilder methods
    #[rhai_fn(name = "min", pure)]
    pub fn role_builder_min(builder: &mut RoleBuilder, count: INT) -> RoleBuilder {
        builder.clone().min(count as usize)
    }

    #[rhai_fn(name = "max", pure)]
    pub fn role_builder_max(builder: &mut RoleBuilder, count: INT) -> RoleBuilder {
        builder.clone().max(count as usize)
    }

    #[rhai_fn(name = "count", pure)]
    pub fn role_builder_count(builder: &mut RoleBuilder, count: INT) -> RoleBuilder {
        builder.clone().count(count as usize)
    }

    #[rhai_fn(name = "score", return_raw)]
    pub fn role_builder_score(
        context: NativeCallContext,
        builder: &mut RoleBuilder,
        scorer_fn: FnPtr,
    ) -> Result<RoleBuilder, Box<EvalAltResult>> {
        let callback = BtCallback::new_rhai(&context, scorer_fn);
        Ok(builder.clone().score(callback))
    }

    #[rhai_fn(name = "require", return_raw)]
    pub fn role_builder_require(
        context: NativeCallContext,
        builder: &mut RoleBuilder,
        filter_fn: FnPtr,
    ) -> Result<RoleBuilder, Box<EvalAltResult>> {
        let callback = BtCallback::new_rhai(&context, filter_fn);
        Ok(builder.clone().require(callback))
    }

    #[rhai_fn(name = "exclude", return_raw)]
    pub fn role_builder_exclude(
        context: NativeCallContext,
        builder: &mut RoleBuilder,
        filter_fn: FnPtr,
    ) -> Result<RoleBuilder, Box<EvalAltResult>> {
        let callback = BtCallback::new_rhai(&context, filter_fn);
        Ok(builder.clone().exclude(callback))
    }

    #[rhai_fn(name = "behavior", return_raw)]
    pub fn role_builder_behavior(
        context: NativeCallContext,
        builder: &mut RoleBuilder,
        builder_fn: FnPtr,
    ) -> Result<RoleBuilder, Box<EvalAltResult>> {
        let callback = BtCallback::new_rhai(&context, builder_fn);
        Ok(builder.clone().behavior(callback))
    }

    #[rhai_fn(name = "build", return_raw)]
    pub fn role_builder_build(builder: &mut RoleBuilder) -> Result<Role, Box<EvalAltResult>> {
        builder.clone().build().map_err(|e| {
            Box::new(EvalAltResult::ErrorRuntime(
                format!("Failed to build role: {}", e).into(),
                Position::NONE,
            ))
        })
    }

    // AssignRoles function
    #[rhai_fn(name = "AssignRoles", return_raw)]
    pub fn assign_roles(roles_array: Array) -> Result<RoleAssignmentProblem, Box<EvalAltResult>> {
        let mut roles = Vec::new();

        for role_dyn in roles_array {
            let type_name = role_dyn.type_name().to_string();
            let role = role_dyn.try_cast::<Role>().ok_or_else(|| {
                Box::new(EvalAltResult::ErrorMismatchDataType(
                    "Expected Role".to_string(),
                    type_name.into(),
                    Position::NONE,
                ))
            })?;
            roles.push(role);
        }

        Ok(RoleAssignmentProblem { roles })
    }
}
