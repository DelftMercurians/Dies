use dies_core::{
    Angle, ExecutorSettings, GameState, GameStateData, PlayerData, PlayerId, PlayerModel, Vector2,
    WorldData,
};

// Behavior Status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorStatus {
    Success,
    Failure,
    Running,
}

// Robot Situation
// This struct will hold the context for a single robot's decision-making.
pub struct RobotSituation<'a> {
    pub player_id: PlayerId,
    pub world: &'a WorldData,
    // Add more derived or pre-calculated data as needed for efficiency or convenience
}

impl<'a> RobotSituation<'a> {
    pub fn new(player_id: PlayerId, world: &'a WorldData) -> Self {
        Self { player_id, world }
    }

    // Helper to get this robot's data
    pub fn player_data(&self) -> Option<&'a PlayerData> {
        self.world.get_player(self.player_id)
    }

    // Example derived property: does the robot currently have the ball?
    pub fn has_ball(&self) -> bool {
        self.player_data()
            .map_or(false, |p| p.breakbeam_ball_detected)
    }

    // Placeholder for other derived properties mentioned in the design doc
    // pub fn shot_quality(&self) -> f64 { /* complex calculation */ 0.0 }
    // pub fn is_marked(&self) -> bool { /* complex calculation */ false }
    // pub fn teammate_has_ball(&self) -> bool { /* ... */ false }
    // pub fn defensive_emergency(&self) -> bool { /* ... */ false }
    // pub fn teammate_attacking(&self) -> bool { /* ... */ false }
}

// Situation (Conditional Logic)
pub struct Situation {
    condition: Box<dyn Fn(&RobotSituation) -> bool>,
    description: String,
    // pub visualization: Option<DebugVisualization>, // For future extension
}

impl Situation {
    pub fn new(condition: impl Fn(&RobotSituation) -> bool + 'static, description: &str) -> Self {
        Self {
            condition: Box::new(condition),
            description: description.to_string(),
            // visualization: None,
        }
    }

    pub fn check(&self, situation: &RobotSituation) -> bool {
        (self.condition)(situation)
    }

    pub fn and(self, other: Situation) -> Self {
        let desc = format!("({}) AND ({})", self.description, other.description);
        Situation::new(move |s| self.check(s) && other.check(s), &desc)
    }

    pub fn or(self, other: Situation) -> Self {
        let desc = format!("({}) OR ({})", self.description, other.description);
        Situation::new(move |s| self.check(s) || other.check(s), &desc)
    }

    pub fn not(self) -> Self {
        let desc = format!("NOT ({})", self.description);
        Situation::new(move |s| !self.check(s), &desc)
    }
}

// Behavior Node Trait
pub trait BehaviorNode {
    fn tick(&mut self, situation: &RobotSituation) -> BehaviorStatus;
    fn description(&self) -> String;
    // fn reset(&mut self); // Optional: For nodes that need explicit reset
}

// Select Node (Selector / Fallback)
// Tries children in priority order until one succeeds or is running.
pub struct SelectNode {
    children: Vec<Box<dyn BehaviorNode>>,
    description: String,
    // No need for current_child_index for a typical stateless selector that always re-evaluates from the start
}

impl SelectNode {
    pub fn new(children: Vec<Box<dyn BehaviorNode>>, description: Option<String>) -> Self {
        Self {
            children,
            description: description.unwrap_or_else(|| "Select".to_string()),
        }
    }
}

impl BehaviorNode for SelectNode {
    fn tick(&mut self, situation: &RobotSituation) -> BehaviorStatus {
        for child in self.children.iter_mut() {
            match child.tick(situation) {
                BehaviorStatus::Success => return BehaviorStatus::Success,
                BehaviorStatus::Running => return BehaviorStatus::Running,
                BehaviorStatus::Failure => continue, // Try next child
            }
        }
        BehaviorStatus::Failure // All children failed
    }

    fn description(&self) -> String {
        let child_descs = self
            .children
            .iter()
            .map(|c| c.description())
            .collect::<Vec<_>>()
            .join(" OR ");
        format!("{}: ({})", self.description, child_descs)
    }
}

// Sequence Node
// Executes children in order. All must succeed for the sequence to succeed.
pub struct SequenceNode {
    children: Vec<Box<dyn BehaviorNode>>,
    description: String,
    current_child_index: usize, // To remember which child to tick next if one is Running
}

impl SequenceNode {
    pub fn new(children: Vec<Box<dyn BehaviorNode>>, description: Option<String>) -> Self {
        Self {
            children,
            description: description.unwrap_or_else(|| "Sequence".to_string()),
            current_child_index: 0,
        }
    }
}

impl BehaviorNode for SequenceNode {
    fn tick(&mut self, situation: &RobotSituation) -> BehaviorStatus {
        while self.current_child_index < self.children.len() {
            match self.children[self.current_child_index].tick(situation) {
                BehaviorStatus::Success => {
                    self.current_child_index += 1;
                }
                BehaviorStatus::Running => {
                    return BehaviorStatus::Running;
                }
                BehaviorStatus::Failure => {
                    self.current_child_index = 0; // Reset for next time
                    return BehaviorStatus::Failure;
                }
            }
        }
        self.current_child_index = 0; // Reset for next time
        BehaviorStatus::Success
    }

    fn description(&self) -> String {
        let child_descs = self
            .children
            .iter()
            .map(|c| c.description())
            .collect::<Vec<_>>()
            .join(" THEN ");
        format!("{}: ({})", self.description, child_descs)
    }
}

// Guard Node (Conditional Node)
// Executes a child behavior only if a condition (Situation) is met.
pub struct GuardNode {
    condition: Situation,
    child: Box<dyn BehaviorNode>,
    description_override: Option<String>, // Allows custom description for the guard itself
}

impl GuardNode {
    pub fn new(
        condition: Situation,
        child: Box<dyn BehaviorNode>,
        description_override: Option<String>,
    ) -> Self {
        Self {
            condition,
            child,
            description_override,
        }
    }
}

impl BehaviorNode for GuardNode {
    fn tick(&mut self, situation: &RobotSituation) -> BehaviorStatus {
        if self.condition.check(situation) {
            self.child.tick(situation)
        } else {
            BehaviorStatus::Failure // Condition not met
        }
    }

    fn description(&self) -> String {
        if let Some(ref d) = self.description_override {
            return d.clone();
        }
        format!(
            "IF ({}) THEN ({})",
            self.condition.description,
            self.child.description()
        )
    }
}
