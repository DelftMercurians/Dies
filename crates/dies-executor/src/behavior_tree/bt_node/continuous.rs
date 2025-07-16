use dies_core::{debug_tree_node, Angle, TeamColor, Vector2};

use crate::{
    behavior_tree::{
        bt_node::sanitize_id_fragment, Argument, BehaviorNode, BehaviorStatus, RobotSituation,
    },
    PlayerControlInput,
};

#[derive(Clone)]
pub struct ContinuousNode {
    position: Option<Argument<Vector2>>,
    last_position: Option<Vector2>,
    heading: Option<Argument<Angle>>,
    last_heading: Option<Angle>,
    description: String,
}

impl ContinuousNode {
    pub fn new(
        position: Option<Argument<Vector2>>,
        heading: Option<Argument<Angle>>,
        description: String,
    ) -> Self {
        Self {
            position,
            last_position: None,
            heading,
            last_heading: None,
            description,
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let position = self.position.as_ref().map(|p| p.resolve(situation));
        let heading = self.heading.as_ref().map(|h| h.resolve(situation));

        let mut control_input = PlayerControlInput::default();
        control_input.position = position;
        control_input.yaw = heading;

        self.last_position = position;
        self.last_heading = heading;

        dies_core::debug_string(
            format!(
                "team_{}.p{}.skill",
                situation.team_color, situation.player_id
            ),
            format!("continuous: {}", self.description),
        );

        (BehaviorStatus::Running, Some(control_input))
    }

    pub fn description(&self) -> String {
        self.description.clone()
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&node_full_id),
            true,
            "Continuous",
            None,
            Some(format!(
                "position: {:?}, heading: {:?}",
                self.last_position, self.last_heading
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

pub struct ContinuousBuilder {
    position: Option<Argument<Vector2>>,
    heading: Option<Argument<Angle>>,
    description: String,
}

impl ContinuousBuilder {
    pub fn new(description: impl AsRef<str>) -> Self {
        Self {
            position: None,
            heading: None,
            description: description.as_ref().to_string(),
        }
    }

    pub fn position(mut self, position: impl Into<Argument<Vector2>>) -> Self {
        self.position = Some(position.into());
        self
    }

    pub fn heading(mut self, heading: impl Into<Argument<Angle>>) -> Self {
        self.heading = Some(heading.into());
        self
    }

    pub fn build(self) -> ContinuousNode {
        ContinuousNode::new(self.position, self.heading, self.description)
    }
}

pub fn continuous(description: impl AsRef<str>) -> ContinuousBuilder {
    ContinuousBuilder::new(description)
}

impl From<ContinuousNode> for BehaviorNode {
    fn from(node: ContinuousNode) -> Self {
        BehaviorNode::Continuous(node)
    }
}
