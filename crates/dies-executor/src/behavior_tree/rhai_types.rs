use crate::behavior_tree::bt_node::BehaviorNode;
use crate::skills::Skill as SkillEnum;
use rhai::{CustomType, TypeBuilder};

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
