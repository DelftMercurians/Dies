use crate::behavior_tree::BehaviorNode;

#[derive(Clone)]
pub struct NoopNode;

impl NoopNode {
    pub fn new() -> Self {
        Self {}
    }
}

pub fn noop_node() -> NoopNode {
    NoopNode::new()
}

impl From<NoopNode> for BehaviorNode {
    fn from(node: NoopNode) -> Self {
        BehaviorNode::Noop(node)
    }
}
