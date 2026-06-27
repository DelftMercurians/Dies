use crate::{
    behavior_tree::{bt_node::sanitize_id_fragment, BehaviorNode, BehaviorStatus, RobotSituation},
    PlayerControlInput,
};

pub struct RepeatNode {
    child: Box<BehaviorNode>,
    node_id_fragment: String,
}

impl RepeatNode {
    pub fn new(description: impl AsRef<str>, child: BehaviorNode) -> Self {
        Self {
            child: Box::new(child),
            node_id_fragment: sanitize_id_fragment(&description.as_ref().to_string()),
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        // Repeat until the child returns Running (i.e., progress is Continue)
        loop {
            let (status, input) = self.child.tick(situation);
            if status == BehaviorStatus::Running {
                return (status, input);
            }
            // If the child returns Success or Failure, repeat (i.e., tick again)
            // This will keep repeating until Running is returned.
        }
    }

    pub fn description(&self) -> String {
        format!("Repeat({})", self.child.description())
    }

    pub fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        self.child.get_child_node_ids(current_path_prefix)
    }

    pub fn get_node_id_fragment(&self) -> String {
        self.node_id_fragment.clone()
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation) {
        self.child.debug_all_nodes(situation);
    }
}

pub struct RepeatNodeBuilder {
    child: BehaviorNode,
    description: Option<String>,
}

pub fn repeat_node(child: impl Into<BehaviorNode>) -> RepeatNodeBuilder {
    RepeatNodeBuilder {
        child: child.into(),
        description: None,
    }
}

impl RepeatNodeBuilder {
    pub fn description(mut self, description: impl AsRef<str>) -> Self {
        self.description = Some(description.as_ref().to_string());
        self
    }

    pub fn build(self) -> RepeatNode {
        RepeatNode::new(
            self.description.unwrap_or_else(|| "Repeat".to_string()),
            self.child,
        )
    }
}

impl From<RepeatNode> for BehaviorNode {
    fn from(node: RepeatNode) -> Self {
        BehaviorNode::Repeat(node)
    }
}

impl From<RepeatNodeBuilder> for BehaviorNode {
    fn from(builder: RepeatNodeBuilder) -> Self {
        builder.build().into()
    }
}
