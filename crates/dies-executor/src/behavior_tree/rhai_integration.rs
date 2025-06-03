use crate::behavior_tree::{ActionNode, BehaviorNode, SelectNode, SequenceNode};
use crate::roles::skills::GoToPosition;
use crate::roles::Skill;
use dies_core::Vector2;
use rhai::{Array, CustomType, Dynamic, EvalAltResult, Position, TypeBuilder};

#[derive(Clone)]
pub struct RhaiBehaviorNode(pub BehaviorNode);

impl CustomType for RhaiBehaviorNode {
    fn build(mut builder: TypeBuilder<Self>) {
        builder.with_name("BehaviorNode");
    }
}

pub fn rhai_select_node(
    children_dyn: Array,
    description: Option<String>,
) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
    let mut rust_children: Vec<BehaviorNode> = Vec::new();
    for child_dyn in children_dyn {
        let node_wrapper = child_dyn.try_cast::<RhaiBehaviorNode>().ok_or_else(|| {
            Box::new(EvalAltResult::ErrorMismatchDataType(
                "Expected BehaviorNode".to_string(),
                child_dyn.type_name().to_string().into(),
                Position::NONE,
            ))
        })?;
        rust_children.push(node_wrapper.0);
    }
    Ok(RhaiBehaviorNode(Box::new(SelectNode::new(
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
        let node_wrapper = child_dyn.try_cast::<RhaiBehaviorNode>().ok_or_else(|| {
            Box::new(EvalAltResult::ErrorMismatchDataType(
                "Expected BehaviorNode".to_string(),
                child_dyn.type_name().to_string().into(),
                Position::NONE,
            ))
        })?;
        rust_children.push(node_wrapper.0);
    }
    Ok(RhaiBehaviorNode(Box::new(SequenceNode::new(
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

pub fn rhai_goto_skill(x: f64, y: f64) -> Result<RhaiSkill, Box<EvalAltResult>> {
    Ok(RhaiSkill(Box::new(GoToPosition::new(Vector2::new(x, y)))))
}

pub fn rhai_action_node(
    skill_wrapper_dyn: Dynamic,
    description: Option<String>,
) -> Result<RhaiBehaviorNode, Box<EvalAltResult>> {
    let skill_wrapper = skill_wrapper_dyn.try_cast::<RhaiSkill>().ok_or_else(|| {
        Box::new(EvalAltResult::ErrorMismatchDataType(
            "Expected RhaiSkill".to_string(),
            skill_wrapper_dyn.type_name().to_string().into(),
            Position::NONE,
        ))
    })?;
    Ok(RhaiBehaviorNode(Box::new(ActionNode::new(
        skill_wrapper.0,
        description,
    ))))
}
