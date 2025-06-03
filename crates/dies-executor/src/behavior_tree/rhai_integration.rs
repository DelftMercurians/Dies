use crate::behavior_tree::{
    ActionNode, BehaviorNode, GuardNode, RobotSituation, ScoringSelectNode, SelectNode,
    SemaphoreNode, SequenceNode, Situation,
};
use crate::roles::skills::GoToPosition;
use crate::roles::Skill;
use dies_core::Vector2;
use rhai::{
    Array, CustomType, Dynamic, Engine, EvalAltResult, FnPtr, Map, Position, Scope, TypeBuilder,
    AST,
};
use std::sync::Arc;

#[derive(Clone)]
pub struct RhaiCtx {
    engine: Arc<Engine>,
    ast: Arc<AST>,
}

#[derive(Clone)]
pub struct RhaiBehaviorNode(pub BehaviorNode);

impl CustomType for RhaiBehaviorNode {
    fn build(mut builder: TypeBuilder<Self>) {
        builder.with_name("BehaviorNode");
    }
}

pub fn create_rhai_situation_view(rs: &RobotSituation, _engine: &Engine) -> Dynamic {
    let mut map = Map::new();
    map.insert("player_id".into(), Dynamic::from(rs.player_id.to_string()));
    map.insert("has_ball".into(), Dynamic::from(rs.has_ball()));
    if let Some(ball) = rs.world.ball.as_ref() {
        let mut ball_map = Map::new();
        ball_map.insert("pos_x".into(), Dynamic::from(ball.position.x));
        ball_map.insert("pos_y".into(), Dynamic::from(ball.position.y));
        map.insert("ball".into(), Dynamic::from(ball_map));
    }
    let mut player_map = Map::new();
    player_map.insert("pos_x".into(), Dynamic::from(rs.player_data.position.x));
    player_map.insert("pos_y".into(), Dynamic::from(rs.player_data.position.y));
    map.insert("player".into(), Dynamic::from(player_map));
    Dynamic::from(map)
}

pub fn rhai_select_node(
    children_dyn: Array,
    description: Option<String>,
) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
    let mut rust_children: Vec<BehaviorNode> = Vec::new();
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
    Ok(RhaiBehaviorNode(BehaviorNode::Select(SelectNode::new(
        rust_children,
        description,
    ))))
}

pub fn rhai_sequence_node(
    children_dyn: Array,
    description: Option<String>,
) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
    let mut rust_children: Vec<BehaviorNode> = Vec::new();
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
    Ok(RhaiBehaviorNode(BehaviorNode::Sequence(SequenceNode::new(
        rust_children,
        description,
    ))))
}

#[derive(Clone)]
pub struct RhaiSkill(pub Skill);

impl CustomType for RhaiSkill {
    fn build(mut builder: TypeBuilder<Self>) {
        builder.with_name("RhaiSkill");
    }
}

pub fn rhai_goto_skill(
    x: f64,
    y: f64,
    options: Option<Map>,
) -> Result<RhaiSkill, Box<EvalAltResult>> {
    let mut skill = GoToPosition::new(Vector2::new(x, y));
    if let Some(opts) = options {
        if let Some(heading_dyn) = opts.get("heading") {
            if let Some(heading_f64) = heading_dyn.as_float().ok() {
                skill = skill.with_heading(dies_core::Angle::from_radians(heading_f64));
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
                    skill = skill.with_ball();
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
                    skill = skill.avoid_ball();
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
    Ok(RhaiSkill(Skill::GoToPosition(skill)))
}

pub fn rhai_action_node(
    skill_wrapper_dyn: Dynamic,
    description: Option<String>,
) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
    let name = skill_wrapper_dyn.type_name().to_string();
    let skill_wrapper = skill_wrapper_dyn.try_cast::<RhaiSkill>().ok_or_else(|| {
        Box::new(EvalAltResult::ErrorMismatchDataType(
            "Expected RhaiSkill".to_string(),
            name.into(),
            Position::NONE,
        ))
    })?;
    Ok(RhaiBehaviorNode(BehaviorNode::Action(ActionNode::new(
        skill_wrapper.0,
        description,
    ))))
}

pub fn rhai_guard_constructor(
    context: RhaiCtx,
    condition_fn_ptr: FnPtr,
    child_dyn: RhaiBehaviorNode,
    cond_description: Option<String>,
) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
    let callback_fn_name = condition_fn_ptr.fn_name().to_string();
    let description = cond_description.unwrap_or_else(|| format!("RhaiCond:{}", callback_fn_name));

    let actual_situation = Situation::new(
        move |rs: &RobotSituation| -> bool {
            let situation_view_dyn = create_rhai_situation_view(rs, &context.engine);
            match context.engine.call_fn::<bool>(
                &mut Scope::new(),
                &context.ast,
                &callback_fn_name,
                (situation_view_dyn,),
            ) {
                Ok(r) => r,
                Err(e) => {
                    log::error!(
                        "Error executing Rhai condition \'{}\': {:?}",
                        callback_fn_name,
                        e
                    );
                    false
                }
            }
        },
        &description,
    );

    let child_node = child_dyn.0;
    let guard_desc_override = Some(format!(
        "Guard_If_{}_Then_{}",
        description,
        child_node.description()
    ));

    Ok(RhaiBehaviorNode(BehaviorNode::Guard(GuardNode::new(
        actual_situation,
        child_node,
        guard_desc_override,
    ))))
}

pub fn rhai_scoring_select_node(
    context: RhaiCtx,
    children_scorers_dyn: Array,
    hysteresis_margin: f64,
    description: Option<String>,
) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
    let mut rust_children_scorers: Vec<(BehaviorNode, Arc<dyn Fn(&RobotSituation) -> f64>)> =
        Vec::new();

    for item_dyn in children_scorers_dyn {
        let name = item_dyn.type_name().to_string();
        let map = item_dyn.try_cast::<Map>().ok_or_else(|| {
            Box::new(EvalAltResult::ErrorMismatchDataType(
                "Expected map for child_scorer item".to_string(),
                name.into(),
                Position::NONE,
            ))
        })?;

        let node_dyn = map
            .get("node")
            .ok_or_else(|| {
                Box::new(EvalAltResult::ErrorVariableNotFound(
                    "Missing \'node\' in scorer item".to_string(),
                    Position::NONE,
                ))
            })?
            .clone();
        let scorer_fn_ptr = map
            .get("scorer")
            .ok_or_else(|| {
                Box::new(EvalAltResult::ErrorVariableNotFound(
                    "Missing \'scorer\' in scorer item".to_string(),
                    Position::NONE,
                ))
            })?
            .clone()
            .cast::<FnPtr>();

        let name = node_dyn.type_name().to_string();
        let node_wrapper = node_dyn.try_cast::<RhaiBehaviorNode>().ok_or_else(|| {
            Box::new(EvalAltResult::ErrorMismatchDataType(
                "Expected BehaviorNode for \'node\' field".to_string(),
                name.into(),
                Position::NONE,
            ))
        })?;

        let callback_fn_name = scorer_fn_ptr.fn_name().to_string();

        let engine = Arc::clone(&context.engine);
        let ast = Arc::clone(&context.ast);
        let scorer_closure = Arc::new(move |rs: &RobotSituation| -> f64 {
            let situation_view_dyn = create_rhai_situation_view(rs, &engine);
            match engine.call_fn::<f64>(
                &mut Scope::new(),
                &ast,
                &callback_fn_name,
                (situation_view_dyn,),
            ) {
                Ok(r) => r,
                Err(e) => {
                    log::error!(
                        "Error executing Rhai scorer \'{}\': {:?}",
                        callback_fn_name,
                        e
                    );
                    f64::NEG_INFINITY
                }
            }
        });
        rust_children_scorers.push((node_wrapper.0, scorer_closure));
    }
    Ok(RhaiBehaviorNode(BehaviorNode::ScoringSelect(
        ScoringSelectNode::new(rust_children_scorers, hysteresis_margin, description),
    )))
}

pub fn rhai_semaphore_node(
    child_dyn: RhaiBehaviorNode,
    id: String,
    max_count: i64,
    description: Option<String>,
) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
    let child_node = child_dyn.0;
    Ok(RhaiBehaviorNode(BehaviorNode::Semaphore(
        SemaphoreNode::new(child_node, id, max_count as usize, description),
    )))
}

pub fn rhai_face_angle_skill(
    angle_rad: f64,
    options: Option<Map>,
) -> Result<RhaiSkill, Box<EvalAltResult>> {
    let mut skill = crate::roles::skills::Face::new(dies_core::Angle::from_radians(angle_rad));
    if let Some(opts) = options {
        if let Some(with_ball_dyn) = opts.get("with_ball") {
            if let Some(with_ball_bool) = with_ball_dyn.as_bool().ok() {
                if with_ball_bool {
                    skill = skill.with_ball();
                }
            } else {
                return Err(Box::new(EvalAltResult::ErrorMismatchDataType(
                    "Expected bool for with_ball".to_string(),
                    with_ball_dyn.type_name().into(),
                    Position::NONE,
                )));
            }
        }
    }
    Ok(RhaiSkill(Skill::Face(skill)))
}

pub fn rhai_face_towards_position_skill(
    x: f64,
    y: f64,
    options: Option<Map>,
) -> Result<RhaiSkill, Box<EvalAltResult>> {
    let mut skill = crate::roles::skills::Face::towards_position(dies_core::Vector2::new(x, y));
    if let Some(opts) = options {
        if let Some(with_ball_dyn) = opts.get("with_ball") {
            if let Some(with_ball_bool) = with_ball_dyn.as_bool().ok() {
                if with_ball_bool {
                    skill = skill.with_ball();
                }
            } else {
                return Err(Box::new(EvalAltResult::ErrorMismatchDataType(
                    "Expected bool for with_ball".to_string(),
                    with_ball_dyn.type_name().into(),
                    Position::NONE,
                )));
            }
        }
    }
    Ok(RhaiSkill(Skill::Face(skill)))
}

pub fn rhai_face_towards_own_player_skill(
    player_id: i64,
    options: Option<Map>,
) -> Result<RhaiSkill, Box<EvalAltResult>> {
    let id = dies_core::PlayerId::new(player_id as u32);
    let mut skill = crate::roles::skills::Face::towards_own_player(id);
    if let Some(opts) = options {
        if let Some(with_ball_dyn) = opts.get("with_ball") {
            if let Some(with_ball_bool) = with_ball_dyn.as_bool().ok() {
                if with_ball_bool {
                    skill = skill.with_ball();
                }
            } else {
                return Err(Box::new(EvalAltResult::ErrorMismatchDataType(
                    "Expected bool for with_ball".to_string(),
                    with_ball_dyn.type_name().into(),
                    Position::NONE,
                )));
            }
        }
    }
    Ok(RhaiSkill(Skill::Face(skill)))
}

pub fn rhai_kick_skill() -> Result<RhaiSkill, Box<EvalAltResult>> {
    Ok(RhaiSkill(Skill::Kick(crate::roles::skills::Kick::new())))
}

pub fn rhai_wait_skill(duration_secs: f64) -> Result<RhaiSkill, Box<EvalAltResult>> {
    Ok(RhaiSkill(Skill::Wait(
        crate::roles::skills::Wait::new_secs_f64(duration_secs),
    )))
}

pub fn rhai_fetch_ball_skill() -> Result<RhaiSkill, Box<EvalAltResult>> {
    Ok(RhaiSkill(Skill::FetchBall(
        crate::roles::skills::FetchBall::new(),
    )))
}

pub fn rhai_intercept_ball_skill() -> Result<RhaiSkill, Box<EvalAltResult>> {
    Ok(RhaiSkill(Skill::InterceptBall(
        crate::roles::skills::InterceptBall::new(),
    )))
}

pub fn rhai_approach_ball_skill() -> Result<RhaiSkill, Box<EvalAltResult>> {
    Ok(RhaiSkill(Skill::ApproachBall(
        crate::roles::skills::ApproachBall::new(),
    )))
}

pub fn rhai_fetch_ball_with_heading_angle_skill(
    angle_rad: f64,
) -> Result<RhaiSkill, Box<EvalAltResult>> {
    Ok(RhaiSkill(Skill::FetchBallWithHeading(
        crate::roles::skills::FetchBallWithHeading::new(dies_core::Angle::from_radians(angle_rad)),
    )))
}

pub fn rhai_fetch_ball_with_heading_position_skill(
    x: f64,
    y: f64,
) -> Result<RhaiSkill, Box<EvalAltResult>> {
    Ok(RhaiSkill(Skill::FetchBallWithHeading(
        crate::roles::skills::FetchBallWithHeading::towards_position(dies_core::Vector2::new(x, y)),
    )))
}

pub fn rhai_fetch_ball_with_heading_player_skill(
    player_id: i64,
) -> Result<RhaiSkill, Box<EvalAltResult>> {
    let id = dies_core::PlayerId::new(player_id as u32);
    Ok(RhaiSkill(Skill::FetchBallWithHeading(
        crate::roles::skills::FetchBallWithHeading::towards_own_player(id),
    )))
}
