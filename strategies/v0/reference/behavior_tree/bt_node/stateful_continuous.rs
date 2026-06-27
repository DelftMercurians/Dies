use std::any::Any;
use std::sync::Arc;

use dies_core::{debug_tree_node, Angle, Vector2};

use crate::{
    behavior_tree::{
        bt_node::sanitize_id_fragment, Argument, BehaviorNode, BehaviorStatus, RobotSituation,
    },
    PlayerControlInput,
};

pub type StatefulPositionCallback<S> =
    dyn Fn(&RobotSituation, Option<&S>) -> (Vector2, Option<S>) + Send + Sync;
pub type StatefulHeadingCallback<S> =
    dyn Fn(&RobotSituation, Option<&S>) -> (Angle, Option<S>) + Send + Sync;

pub struct StatefulContinuousNode {
    position_callback: Option<Arc<StatefulPositionCallback<Box<dyn Any + Send + Sync>>>>,
    heading_callback: Option<Arc<StatefulHeadingCallback<Box<dyn Any + Send + Sync>>>>,
    last_position: Option<Vector2>,
    last_heading: Option<Angle>,
    position_state: Option<Box<dyn Any + Send + Sync>>,
    heading_state: Option<Box<dyn Any + Send + Sync>>,
    description: String,
}

impl Clone for StatefulContinuousNode {
    fn clone(&self) -> Self {
        Self {
            position_callback: self.position_callback.clone(),
            heading_callback: self.heading_callback.clone(),
            last_position: self.last_position,
            last_heading: self.last_heading,
            position_state: None, // Reset state on clone
            heading_state: None,  // Reset state on clone
            description: self.description.clone(),
        }
    }
}

impl StatefulContinuousNode {
    pub fn new(
        position_callback: Option<Arc<StatefulPositionCallback<Box<dyn Any + Send + Sync>>>>,
        heading_callback: Option<Arc<StatefulHeadingCallback<Box<dyn Any + Send + Sync>>>>,
        description: String,
    ) -> Self {
        Self {
            position_callback,
            heading_callback,
            last_position: None,
            last_heading: None,
            position_state: None,
            heading_state: None,
            description,
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let mut control_input = PlayerControlInput::default();

        // Handle position callback
        if let Some(callback) = &self.position_callback {
            let (position, new_state) = callback(situation, self.position_state.as_ref());
            control_input.position = Some(position);
            self.last_position = Some(position);
            self.position_state = new_state;
        }

        // Handle heading callback
        if let Some(callback) = &self.heading_callback {
            let (heading, new_state) = callback(situation, self.heading_state.as_ref());
            control_input.yaw = Some(heading);
            self.last_heading = Some(heading);
            self.heading_state = new_state;
        }

        dies_core::debug_string(
            format!(
                "team_{}.p{}.skill",
                situation.team_color, situation.player_id
            ),
            format!("stateful_continuous: {}", self.description),
        );

        (BehaviorStatus::Running, Some(control_input))
    }

    pub fn description(&self) -> String {
        format!("StatefulContinuous: {}", self.description)
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let has_pos_state = self.position_state.is_some();
        let has_heading_state = self.heading_state.is_some();

        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&node_full_id),
            true,
            "StatefulContinuous",
            None,
            Some(format!(
                "position: {:?}, heading: {:?}, pos_state: {}, heading_state: {}",
                self.last_position, self.last_heading, has_pos_state, has_heading_state
            )),
        );
    }

    pub fn get_node_id_fragment(&self) -> String {
        sanitize_id_fragment(&self.description)
    }

    pub fn get_full_node_id(&self, current_path_prefix: &str) -> String {
        let fragment = self.get_node_id_fragment();
        if current_path_prefix.is_empty() {
            fragment
        } else {
            format!("{}.{}", current_path_prefix, fragment)
        }
    }

    pub fn get_child_node_ids(&self, _current_path_prefix: &str) -> Vec<String> {
        vec![]
    }
}

pub struct StatefulContinuousBuilder {
    position_callback: Option<Arc<StatefulPositionCallback<Box<dyn Any + Send + Sync>>>>,
    heading_callback: Option<Arc<StatefulHeadingCallback<Box<dyn Any + Send + Sync>>>>,
    description: String,
}

impl StatefulContinuousBuilder {
    pub fn new(description: impl AsRef<str>) -> Self {
        Self {
            position_callback: None,
            heading_callback: None,
            description: description.as_ref().to_string(),
        }
    }

    pub fn with_stateful_position<S: 'static + Send + Sync>(
        mut self,
        callback: impl Fn(&RobotSituation, Option<&S>) -> (Vector2, Option<S>) + Send + Sync + 'static,
    ) -> Self {
        let wrapped_callback = Arc::new(
            move |situation: &RobotSituation, state: Option<&Box<dyn Any + Send + Sync>>| {
                let typed_state = state.and_then(|s| s.downcast_ref::<S>());
                let (position, new_state) = callback(situation, typed_state);
                let boxed_state = new_state.map(|s| Box::new(s) as Box<dyn Any + Send + Sync>);
                (position, boxed_state)
            },
        );

        self.position_callback = Some(wrapped_callback);
        self
    }

    pub fn with_stateful_heading<S: 'static + Send + Sync>(
        mut self,
        callback: impl Fn(&RobotSituation, Option<&S>) -> (Angle, Option<S>) + Send + Sync + 'static,
    ) -> Self {
        let wrapped_callback = Arc::new(
            move |situation: &RobotSituation, state: Option<&Box<dyn Any + Send + Sync>>| {
                let typed_state = state.and_then(|s| s.downcast_ref::<S>());
                let (heading, new_state) = callback(situation, typed_state);
                let boxed_state = new_state.map(|s| Box::new(s) as Box<dyn Any + Send + Sync>);
                (heading, boxed_state)
            },
        );

        self.heading_callback = Some(wrapped_callback);
        self
    }

    pub fn build(self) -> StatefulContinuousNode {
        StatefulContinuousNode::new(
            self.position_callback,
            self.heading_callback,
            self.description,
        )
    }
}

pub fn stateful_continuous(description: impl AsRef<str>) -> StatefulContinuousBuilder {
    StatefulContinuousBuilder::new(description)
}

impl From<StatefulContinuousNode> for BehaviorNode {
    fn from(node: StatefulContinuousNode) -> Self {
        BehaviorNode::StatefulContinuous(node)
    }
}
