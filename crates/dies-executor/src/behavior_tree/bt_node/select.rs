use dies_core::debug_tree_node;
use rhai::Engine;

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment, BehaviorNode,
};
use crate::{
    behavior_tree::{bt_callback::BtCallback, RhaiBehaviorNode},
    control::PlayerControlInput,
};

#[derive(Clone)]
pub enum ChildrenSource {
    Static(Vec<BehaviorNode>),
    Dynamic(BtCallback<Vec<RhaiBehaviorNode>>),
}

#[derive(Clone)]
pub struct SelectNode {
    children_source: ChildrenSource,
    description_text: String,
    node_id_fragment: String,
    last_children: Vec<BehaviorNode>,
    last_running_child_idx: Option<usize>,
}

impl SelectNode {
    pub fn new(children: Vec<BehaviorNode>, description: Option<String>) -> Self {
        let desc = description.unwrap_or_else(|| "Select".to_string());
        Self {
            children_source: ChildrenSource::Static(children),
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            last_children: Vec::new(),
            last_running_child_idx: None,
        }
    }

    pub fn new_dynamic(
        callback: BtCallback<Vec<RhaiBehaviorNode>>,
        description: Option<String>,
    ) -> Self {
        let desc = description.unwrap_or_else(|| "Select".to_string());
        Self {
            children_source: ChildrenSource::Dynamic(callback),
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            last_children: Vec::new(),
            last_running_child_idx: None,
        }
    }

    fn get_children(&mut self, situation: &RobotSituation, engine: &Engine) -> Vec<BehaviorNode> {
        match &self.children_source {
            ChildrenSource::Static(children) => children.clone(),
            ChildrenSource::Dynamic(callback) => match callback.call(situation, engine) {
                Ok(nodes) => nodes.into_iter().map(|n| n.0).collect(),
                Err(e) => {
                    log::error!("Failed to generate dynamic children for Select: {}", e);
                    vec![]
                }
            },
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation, engine: &Engine) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let children = match &self.children_source {
            ChildrenSource::Static(children) => children.clone(),
            ChildrenSource::Dynamic(_) => self.last_children.clone(),
        };

        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids_with_children(&situation.viz_path_prefix, &children),
            false,
            "Select",
            Some(format!("{} children", children.len())),
            Some("First success wins".to_string()),
        );

        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id;
        for child in &children {
            child.debug_all_nodes(&child_situation, engine);
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
        engine: &Engine,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);

        if let Some(running_idx) = self.last_running_child_idx {
            if let Some(child) = self.last_children.get_mut(running_idx) {
                let mut child_situation = situation.clone();
                child_situation.viz_path_prefix = node_full_id.clone();
                let (status, input) = child.tick(&mut child_situation, engine);
                if status == BehaviorStatus::Running {
                    return (BehaviorStatus::Running, input);
                }
            }
        }
        self.last_running_child_idx = None;

        let mut children = self.get_children(situation, engine);
        let mut final_status = BehaviorStatus::Failure;
        let mut final_input: Option<PlayerControlInput> = None;

        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();

        for (idx, child) in children.iter_mut().enumerate() {
            match child.tick(&mut child_situation, engine) {
                (BehaviorStatus::Success, input_opt) => {
                    final_status = BehaviorStatus::Success;
                    final_input = input_opt;
                    break;
                }
                (BehaviorStatus::Running, input_opt) => {
                    final_status = BehaviorStatus::Running;
                    final_input = input_opt;
                    self.last_running_child_idx = Some(idx);
                    break;
                }
                (BehaviorStatus::Failure, _) => continue,
            }
        }

        self.last_children = children;

        let is_active =
            final_status == BehaviorStatus::Running || final_status == BehaviorStatus::Success;
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids_with_children(&situation.viz_path_prefix, &self.last_children),
            is_active,
            "Select",
            Some(format!("{} children", self.last_children.len())),
            Some("First success wins".to_string()),
        );
        (final_status, final_input)
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

    pub fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        self.get_child_node_ids_with_children(current_path_prefix, &self.last_children)
    }

    fn get_child_node_ids_with_children(
        &self,
        current_path_prefix: &str,
        children: &[BehaviorNode],
    ) -> Vec<String> {
        let self_full_id = self.get_full_node_id(current_path_prefix);
        children
            .iter()
            .map(|c| c.get_full_node_id(&self_full_id))
            .collect()
    }
}
