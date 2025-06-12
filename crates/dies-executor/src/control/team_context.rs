use dies_core::{debug_record, debug_remove};
use dies_core::{DebugColor, DebugShape, DebugValue, PlayerId, TeamColor, Vector2};

#[derive(Debug, Clone)]
pub struct TeamContext {
    team_color: TeamColor,
    debug_prefix: String,
}

impl TeamContext {
    pub fn new(team_color: TeamColor) -> Self {
        Self {
            team_color,
            debug_prefix: format!("team_{}", team_color),
        }
    }

    pub fn team_color(&self) -> TeamColor {
        self.team_color
    }

    pub fn debug_prefix(&self) -> &str {
        &self.debug_prefix
    }

    // Context-aware debug recording with automatic prefixing
    pub fn debug_record(&self, key: impl Into<String>, value: DebugValue) {
        let prefixed_key = format!("{}.{}", self.debug_prefix, key.into());
        debug_record(prefixed_key, value);
    }

    // Context-aware debug removal with automatic prefixing
    pub fn debug_remove(&self, key: impl Into<String>) {
        let prefixed_key = format!("{}.{}", self.debug_prefix, key.into());
        debug_remove(prefixed_key);
    }

    // Cross debug shape with default color
    pub fn debug_cross(&self, key: impl Into<String>, center: Vector2) {
        self.debug_cross_colored(key, center, DebugColor::default());
    }

    // Cross debug shape with specified color
    pub fn debug_cross_colored(&self, key: impl Into<String>, center: Vector2, color: DebugColor) {
        self.debug_record(key, DebugValue::Shape(DebugShape::Cross { center, color }));
    }

    // Filled circle debug shape with default color
    pub fn debug_circle_fill(&self, key: impl Into<String>, center: Vector2, radius: f64) {
        self.debug_circle_fill_colored(key, center, radius, DebugColor::default());
    }

    // Filled circle debug shape with specified color
    pub fn debug_circle_fill_colored(
        &self,
        key: impl Into<String>,
        center: Vector2,
        radius: f64,
        fill: DebugColor,
    ) {
        self.debug_record(
            key,
            DebugValue::Shape(DebugShape::Circle {
                center,
                radius,
                fill: Some(fill),
                stroke: None,
            }),
        );
    }

    // Hollow circle debug shape with default color
    pub fn debug_circle_stroke(&self, key: impl Into<String>, center: Vector2, radius: f64) {
        self.debug_circle_stroke_colored(key, center, radius, DebugColor::default());
    }

    // Hollow circle debug shape with specified color
    pub fn debug_circle_stroke_colored(
        &self,
        key: impl Into<String>,
        center: Vector2,
        radius: f64,
        stroke: DebugColor,
    ) {
        self.debug_record(
            key,
            DebugValue::Shape(DebugShape::Circle {
                center,
                radius,
                fill: None,
                stroke: Some(stroke),
            }),
        );
    }

    // Line debug shape with default color
    pub fn debug_line(&self, key: impl Into<String>, start: Vector2, end: Vector2) {
        self.debug_line_colored(key, start, end, DebugColor::default());
    }

    // Line debug shape with specified color
    pub fn debug_line_colored(
        &self,
        key: impl Into<String>,
        start: Vector2,
        end: Vector2,
        color: DebugColor,
    ) {
        self.debug_record(
            key,
            DebugValue::Shape(DebugShape::Line { start, end, color }),
        );
    }

    // Tree node debug shape
    pub fn debug_tree_node(
        &self,
        key: impl Into<String>,
        name: impl Into<String>,
        id: impl Into<String>,
        children_ids: impl Into<Vec<String>>,
        is_active: bool,
        node_type: impl Into<String>,
        internal_state: Option<String>,
        additional_info: Option<String>,
    ) {
        self.debug_record(
            key,
            DebugValue::Shape(DebugShape::TreeNode {
                name: name.into(),
                id: id.into(),
                children_ids: children_ids.into(),
                is_active,
                node_type: node_type.into(),
                internal_state,
                additional_info,
            }),
        );
    }

    // Numeric value debug
    pub fn debug_value(&self, key: impl Into<String>, value: f64) {
        self.debug_record(key, DebugValue::Number(value));
    }

    // String value debug
    pub fn debug_string(&self, key: impl Into<String>, value: impl Into<String>) {
        self.debug_record(key, DebugValue::String(value.into()));
    }

    // Create a player context for this team
    pub fn player_context(&self, player_id: PlayerId) -> PlayerContext {
        PlayerContext::new(self.clone(), player_id)
    }
}

#[derive(Debug, Clone)]
pub struct PlayerContext {
    team_context: TeamContext,
    player_id: PlayerId,
    debug_prefix: String,
}

impl PlayerContext {
    pub fn new(team_context: TeamContext, player_id: PlayerId) -> Self {
        let debug_prefix = format!("{}.p{}", team_context.debug_prefix, player_id);
        Self {
            team_context,
            player_id,
            debug_prefix,
        }
    }

    pub fn team_context(&self) -> &TeamContext {
        &self.team_context
    }

    pub fn player_id(&self) -> PlayerId {
        self.player_id
    }

    pub fn debug_prefix(&self) -> &str {
        &self.debug_prefix
    }

    // Context-aware debug recording with automatic prefixing (team.player)
    pub fn debug_record(&self, key: impl Into<String>, value: DebugValue) {
        let prefixed_key = format!("{}.{}", self.debug_prefix, key.into());
        debug_record(prefixed_key, value);
    }

    // Context-aware debug removal with automatic prefixing
    pub fn debug_remove(&self, key: impl Into<String>) {
        let prefixed_key = format!("{}.{}", self.debug_prefix, key.into());
        debug_remove(prefixed_key);
    }

    // Cross debug shape with default color
    pub fn debug_cross(&self, key: impl Into<String>, center: Vector2) {
        self.debug_cross_colored(key, center, DebugColor::default());
    }

    // Cross debug shape with specified color
    pub fn debug_cross_colored(&self, key: impl Into<String>, center: Vector2, color: DebugColor) {
        self.debug_record(key, DebugValue::Shape(DebugShape::Cross { center, color }));
    }

    // Filled circle debug shape with default color
    pub fn debug_circle_fill(&self, key: impl Into<String>, center: Vector2, radius: f64) {
        self.debug_circle_fill_colored(key, center, radius, DebugColor::default());
    }

    // Filled circle debug shape with specified color
    pub fn debug_circle_fill_colored(
        &self,
        key: impl Into<String>,
        center: Vector2,
        radius: f64,
        fill: DebugColor,
    ) {
        self.debug_record(
            key,
            DebugValue::Shape(DebugShape::Circle {
                center,
                radius,
                fill: Some(fill),
                stroke: None,
            }),
        );
    }

    // Hollow circle debug shape with default color
    pub fn debug_circle_stroke(&self, key: impl Into<String>, center: Vector2, radius: f64) {
        self.debug_circle_stroke_colored(key, center, radius, DebugColor::default());
    }

    // Hollow circle debug shape with specified color
    pub fn debug_circle_stroke_colored(
        &self,
        key: impl Into<String>,
        center: Vector2,
        radius: f64,
        stroke: DebugColor,
    ) {
        self.debug_record(
            key,
            DebugValue::Shape(DebugShape::Circle {
                center,
                radius,
                fill: None,
                stroke: Some(stroke),
            }),
        );
    }

    // Line debug shape with default color
    pub fn debug_line(&self, key: impl Into<String>, start: Vector2, end: Vector2) {
        self.debug_line_colored(key, start, end, DebugColor::default());
    }

    // Line debug shape with specified color
    pub fn debug_line_colored(
        &self,
        key: impl Into<String>,
        start: Vector2,
        end: Vector2,
        color: DebugColor,
    ) {
        self.debug_record(
            key,
            DebugValue::Shape(DebugShape::Line { start, end, color }),
        );
    }

    // Tree node debug shape
    pub fn debug_tree_node(
        &self,
        key: impl Into<String>,
        name: impl Into<String>,
        id: impl Into<String>,
        children_ids: impl Into<Vec<String>>,
        is_active: bool,
        node_type: impl Into<String>,
        internal_state: Option<String>,
        additional_info: Option<String>,
    ) {
        self.debug_record(
            key,
            DebugValue::Shape(DebugShape::TreeNode {
                name: name.into(),
                id: id.into(),
                children_ids: children_ids.into(),
                is_active,
                node_type: node_type.into(),
                internal_state,
                additional_info,
            }),
        );
    }

    // Numeric value debug
    pub fn debug_value(&self, key: impl Into<String>, value: f64) {
        self.debug_record(key, DebugValue::Number(value));
    }

    // String value debug
    pub fn debug_string(&self, key: impl Into<String>, value: impl Into<String>) {
        self.debug_record(key, DebugValue::String(value.into()));
    }
}
