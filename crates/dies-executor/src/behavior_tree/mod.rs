use crate::control::PlayerControlInput;
use crate::roles::{Skill, SkillCtx, SkillProgress, SkillResult};
use dies_core::{
    Angle, ExecutorSettings, GameState, GameStateData, PlayerData, PlayerId, PlayerModel, Vector2,
    WorldData,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

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
    pub player_data: &'a PlayerData,
    pub team_context: &'a TeamContext, // Added for Phase 2
                                       // Add more derived or pre-calculated data as needed for efficiency or convenience
}

impl<'a> RobotSituation<'a> {
    pub fn new(
        player_id: PlayerId,
        world: &'a WorldData,
        team_context: &'a TeamContext, // Added for Phase 2
    ) -> Self {
        let player_data = world.get_player(player_id);
        Self {
            player_id,
            world,
            player_data,
            team_context, // Added for Phase 2
        }
    }

    pub fn has_ball(&self) -> bool {
        self.player_data.breakbeam_ball_detected
    }
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
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>);
    fn description(&self) -> String;
    // fn reset(&mut self); // Optional: For nodes that need explicit reset
    // fn get_unique_id(&self) -> String; // For Phase 4: Visualization
    // fn get_children_unique_ids(&self) -> Vec<String>; // For Phase 4: Visualization
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
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        for child in self.children.iter_mut() {
            match child.tick(situation) {
                (BehaviorStatus::Success, input_opt) => {
                    return (BehaviorStatus::Success, input_opt)
                }
                (BehaviorStatus::Running, input_opt) => {
                    return (BehaviorStatus::Running, input_opt)
                }
                (BehaviorStatus::Failure, _) => continue, // Try next child
            }
        }
        (BehaviorStatus::Failure, None) // All children failed
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
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let mut last_input_on_success: Option<PlayerControlInput> = None;
        while self.current_child_index < self.children.len() {
            match self.children[self.current_child_index].tick(situation) {
                (BehaviorStatus::Success, input_opt) => {
                    self.current_child_index += 1;
                    last_input_on_success = input_opt;
                }
                (BehaviorStatus::Running, input_opt) => {
                    return (BehaviorStatus::Running, input_opt);
                }
                (BehaviorStatus::Failure, _input_opt) => {
                    // Assuming None for Failure as per ActionNode example
                    self.current_child_index = 0; // Reset for next time
                    return (BehaviorStatus::Failure, None);
                }
            }
        }
        self.current_child_index = 0; // Reset for next time
        (BehaviorStatus::Success, last_input_on_success)
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
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        if self.condition.check(situation) {
            self.child.tick(situation)
        } else {
            (BehaviorStatus::Failure, None) // Condition not met
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

// Action Node
// Executes a skill and translates its progress to behavior status and player control input.
pub struct ActionNode {
    skill: Box<dyn Skill>,
    description: String,
    // unique_id: String, // For Phase 4: Visualization
}

impl ActionNode {
    pub fn new(
        skill: Box<dyn Skill>,
        description: Option<String>, /*, unique_id: String*/
    ) -> Self {
        Self {
            skill,
            description: description.unwrap_or_else(|| "Action: <Unnamed Skill>".to_string()),
            // unique_id,
        }
    }
}

impl BehaviorNode for ActionNode {
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        // Create SkillCtx from RobotSituation
        let skill_ctx = SkillCtx {
            player: situation.player_data,
            world: situation.world,
        };

        // Call the skill's update method and translate SkillProgress
        match self.skill.update(skill_ctx) {
            SkillProgress::Continue(input) => (BehaviorStatus::Running, Some(input)),
            SkillProgress::Done(SkillResult::Success) => (BehaviorStatus::Success, None),
            SkillProgress::Done(SkillResult::Failure) => (BehaviorStatus::Failure, None),
        }
    }

    fn description(&self) -> String {
        self.description.clone()
    }
}

// TeamContext (New for Phase 2)
// Shared context for team-level coordination, like semaphores.
pub struct TeamContext {
    semaphores: Arc<RwLock<HashMap<String, (usize, HashSet<PlayerId>)>>>,
    // usize: current count of players holding the semaphore
    // HashSet<PlayerId>: set of players currently holding the semaphore
}

impl TeamContext {
    pub fn new() -> Self {
        Self {
            semaphores: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Attempts to acquire a semaphore for the given player.
    /// Returns true if the semaphore is acquired or if the player already holds it
    /// and the max_count is not exceeded by others.
    pub fn try_acquire_semaphore(&self, id: &str, max_count: usize, player_id: PlayerId) -> bool {
        let mut semaphores = self.semaphores.write().unwrap();
        let entry = semaphores
            .entry(id.to_string())
            .or_insert((0, HashSet::new()));

        if entry.1.contains(&player_id) {
            // Player already holds the semaphore
            return true;
        }

        if entry.0 < max_count {
            entry.0 += 1;
            entry.1.insert(player_id);
            true
        } else {
            false
        }
    }

    /// Releases a semaphore held by the given player.
    pub fn release_semaphore(&self, id: &str, player_id: PlayerId) {
        let mut semaphores = self.semaphores.write().unwrap();
        if let Some(entry) = semaphores.get_mut(id) {
            if entry.1.remove(&player_id) {
                entry.0 = entry.0.saturating_sub(1);
            }
            // Clean up if no one is holding and count is zero (optional, good practice)
            // if entry.0 == 0 && entry.1.is_empty() {
            //     semaphores.remove(id);
            // }
        }
    }

    /// Clears all semaphores. Useful for resetting state each tick.
    pub fn clear_semaphores(&self) {
        let mut semaphores = self.semaphores.write().unwrap();
        semaphores.clear();
    }
}

impl Default for TeamContext {
    fn default() -> Self {
        Self::new()
    }
}

// Semaphore Node (New for Phase 2)
// Limits concurrent execution of its child based on a team-wide semaphore.
pub struct SemaphoreNode {
    child: Box<dyn BehaviorNode>,
    semaphore_id: String,
    max_count: usize,
    description: String,
    // Tracks if this node instance successfully acquired the semaphore for the current player
    // during a 'Running' state of its child.
    player_id_holding_via_this_node: Option<PlayerId>,
    // unique_id: String, // For Phase 4: Visualization
}

impl SemaphoreNode {
    pub fn new(
        child: Box<dyn BehaviorNode>,
        semaphore_id: String,
        max_count: usize,
        description: Option<String>,
        // unique_id: String, // For Phase 4
    ) -> Self {
        Self {
            child,
            semaphore_id: semaphore_id.clone(),
            max_count,
            description: description.unwrap_or_else(|| format!("Semaphore({})", semaphore_id)),
            player_id_holding_via_this_node: None,
            // unique_id,
        }
    }
}

impl BehaviorNode for SemaphoreNode {
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let player_id = situation.player_id;

        // Check if this node instance was already holding the semaphore for this player
        if self.player_id_holding_via_this_node == Some(player_id) {
            let (child_status, child_input) = self.child.tick(situation);
            match child_status {
                BehaviorStatus::Success | BehaviorStatus::Failure => {
                    situation
                        .team_context
                        .release_semaphore(&self.semaphore_id, player_id);
                    self.player_id_holding_via_this_node = None;
                }
                BehaviorStatus::Running => {
                    // Still running, keep holding
                }
            }
            return (child_status, child_input);
        }

        // If not already holding, try to acquire
        if situation.team_context.try_acquire_semaphore(
            &self.semaphore_id,
            self.max_count,
            player_id,
        ) {
            self.player_id_holding_via_this_node = Some(player_id); // Acquired
            let (child_status, child_input) = self.child.tick(situation);

            match child_status {
                BehaviorStatus::Success | BehaviorStatus::Failure => {
                    // Child finished, release semaphore
                    situation
                        .team_context
                        .release_semaphore(&self.semaphore_id, player_id);
                    self.player_id_holding_via_this_node = None;
                }
                BehaviorStatus::Running => {
                    // Child is running, semaphore remains held by this node for this player
                }
            }
            (child_status, child_input)
        } else {
            // Failed to acquire semaphore
            (BehaviorStatus::Failure, None)
        }
    }

    fn description(&self) -> String {
        format!("{} [Child: {}]", self.description, self.child.description())
    }

    // get_unique_id and get_children_unique_ids for Phase 4
    // fn get_unique_id(&self) -> String { self.unique_id.clone() }
    // fn get_children_unique_ids(&self) -> Vec<String> { vec![self.child.get_unique_id()] }
}

// Note: Drop trait for SemaphoreNode to auto-release is not implemented here.
// The SOW suggests relying on tick logic and TeamController's context management
// (e.g., clearing semaphores per tick) as a safer pattern.
// If a SemaphoreNode is removed from the tree while its child was Running and holding a semaphore,
// that semaphore slot might remain acquired until the TeamContext is cleared/reset.

// Scoring Select Node
// Selects a child based on dynamically calculated scores, incorporating hysteresis.
pub struct ScoringSelectNode {
    children_with_scorers: Vec<(Box<dyn BehaviorNode>, Box<dyn Fn(&RobotSituation) -> f64>)>,
    hysteresis_margin: f64,
    description: String,
    current_best_child_index: Option<usize>,
    current_best_child_score: f64,
    // unique_id: String, // For Phase 4: Visualization
}

impl ScoringSelectNode {
    pub fn new(
        children_with_scorers: Vec<(Box<dyn BehaviorNode>, Box<dyn Fn(&RobotSituation) -> f64>)>,
        hysteresis_margin: f64,
        description: Option<String>,
        // unique_id: String, // For Phase 4
    ) -> Self {
        Self {
            children_with_scorers,
            hysteresis_margin,
            description: description.unwrap_or_else(|| "ScoringSelect".to_string()),
            current_best_child_index: None,
            current_best_child_score: f64::NEG_INFINITY,
            // unique_id,
        }
    }
}

impl BehaviorNode for ScoringSelectNode {
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        if self.children_with_scorers.is_empty() {
            return (BehaviorStatus::Failure, None);
        }

        let mut new_highest_score = f64::NEG_INFINITY;
        let mut new_best_child_candidate_index = 0;

        let mut scores = Vec::with_capacity(self.children_with_scorers.len());
        for (i, (_, scorer)) in self.children_with_scorers.iter().enumerate() {
            let score = scorer(situation);
            scores.push(score);
            if score > new_highest_score {
                new_highest_score = score;
                new_best_child_candidate_index = i;
            }
        }

        let final_child_to_tick_idx: usize;

        if let Some(active_child_idx) = self.current_best_child_index {
            // A child was previously selected and presumably running.
            let score_of_active_child = scores[active_child_idx]; // Its current score

            if score_of_active_child >= new_highest_score - self.hysteresis_margin {
                // Stick with the current child
                final_child_to_tick_idx = active_child_idx;
            } else {
                // Switch to the new best candidate
                final_child_to_tick_idx = new_best_child_candidate_index;
                self.current_best_child_index = Some(new_best_child_candidate_index);
                self.current_best_child_score = new_highest_score;
            }
        } else {
            // No child was previously selected (first tick, or previous child completed)
            final_child_to_tick_idx = new_best_child_candidate_index;
            self.current_best_child_index = Some(new_best_child_candidate_index);
            self.current_best_child_score = new_highest_score;
        }

        let (child_node, _) = &mut self.children_with_scorers[final_child_to_tick_idx];
        let (status, input) = child_node.tick(situation);

        match status {
            BehaviorStatus::Success | BehaviorStatus::Failure => {
                // If the child succeeded or failed, it's no longer the active running child.
                // Reset to force re-evaluation without hysteresis bias from this child.
                self.current_best_child_index = None;
                // Resetting score as well, though it's mainly the index that drives logic.
                self.current_best_child_score = f64::NEG_INFINITY;
            }
            BehaviorStatus::Running => {
                // Child is still running, keep current_best_child_index pointing to it.
                // The score self.current_best_child_score is the score that *selected* it.
                // It's not updated while it's running, only when a new selection is made.
            }
        }
        (status, input)
    }

    fn description(&self) -> String {
        // You might want to enhance this description, e.g., by including child count or active child.
        // For now, just using the provided or default description.
        format!(
            "{}{}",
            self.description,
            if let Some(idx) = self.current_best_child_index {
                format!(" (Active: Index {})", idx)
            } else {
                "".to_string()
            }
        )
    }
}
