use crate::control::PlayerControlInput;
use crate::roles::{Skill, SkillCtx, SkillProgress, SkillResult};
use dies_core::{debug_tree_node, PlayerData, PlayerId, WorldData};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

// Behavior Status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorStatus {
    Success,
    Failure,
    Running,
}

// Helper for creating safe ID fragments
fn sanitize_id_fragment(text: &str) -> String {
    text.chars()
        .filter_map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' => Some(c.to_ascii_lowercase()),
            ' ' | '_' | '-' => Some('_'),
            _ => None, // Skip other characters
        })
        .collect::<String>()
}

// Robot Situation
// This struct will hold the context for a single robot's decision-making.
pub struct RobotSituation<'a> {
    pub player_id: PlayerId,
    pub world: &'a WorldData,
    pub player_data: &'a PlayerData,
    pub team_context: &'a TeamContext, // Added for Phase 2
    pub viz_path_prefix: String,       // Added for Phase 4: Visualization
}

impl<'a> RobotSituation<'a> {
    pub fn new(
        player_id: PlayerId,
        world: &'a WorldData,
        team_context: &'a TeamContext, // Added for Phase 2
        viz_path_prefix: String,       // Added for Phase 4: Visualization
    ) -> Self {
        let player_data = world.get_player(player_id);
        Self {
            player_id,
            world,
            player_data,
            team_context,    // Added for Phase 2
            viz_path_prefix, // Added for Phase 4: Visualization
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
    fn get_node_id_fragment(&self) -> String;
    fn get_full_node_id(&self, current_path_prefix: &str) -> String {
        let fragment = self.get_node_id_fragment();
        if current_path_prefix.is_empty() {
            fragment
        } else {
            format!("{}/{}", current_path_prefix, fragment)
        }
    }
    fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String>;
    // fn reset(&mut self); // Optional: For nodes that need explicit reset
}

// Select Node (Selector / Fallback)
// Tries children in priority order until one succeeds or is running.
pub struct SelectNode {
    children: Vec<Box<dyn BehaviorNode>>,
    description_text: String,
    node_id_fragment: String,
}

impl SelectNode {
    pub fn new(children: Vec<Box<dyn BehaviorNode>>, description: Option<String>) -> Self {
        let desc = description.unwrap_or_else(|| "Select".to_string());
        Self {
            children,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
        }
    }
}

impl BehaviorNode for SelectNode {
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let mut final_status = BehaviorStatus::Failure;
        let mut final_input: Option<PlayerControlInput> = None;

        for child in self.children.iter_mut() {
            match child.tick(situation) {
                (BehaviorStatus::Success, input_opt) => {
                    final_status = BehaviorStatus::Success;
                    final_input = input_opt;
                    break;
                }
                (BehaviorStatus::Running, input_opt) => {
                    final_status = BehaviorStatus::Running;
                    final_input = input_opt;
                    break;
                }
                (BehaviorStatus::Failure, _) => continue,
            }
        }

        let is_active =
            final_status == BehaviorStatus::Running || final_status == BehaviorStatus::Success;
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            is_active,
        );
        (final_status, final_input)
    }

    fn description(&self) -> String {
        self.description_text.clone()
    }

    fn get_node_id_fragment(&self) -> String {
        self.node_id_fragment.clone()
    }

    fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        let self_full_id = self.get_full_node_id(current_path_prefix);
        self.children
            .iter()
            .map(|c| c.get_full_node_id(&self_full_id))
            .collect()
    }
}

// Sequence Node
// Executes children in order. All must succeed for the sequence to succeed.
pub struct SequenceNode {
    children: Vec<Box<dyn BehaviorNode>>,
    description_text: String,
    node_id_fragment: String,
    current_child_index: usize, // To remember which child to tick next if one is Running
}

impl SequenceNode {
    pub fn new(children: Vec<Box<dyn BehaviorNode>>, description: Option<String>) -> Self {
        let desc = description.unwrap_or_else(|| "Sequence".to_string());
        Self {
            children,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            current_child_index: 0,
        }
    }
}

impl BehaviorNode for SequenceNode {
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let mut last_input_on_success: Option<PlayerControlInput> = None;

        while self.current_child_index < self.children.len() {
            match self.children[self.current_child_index].tick(situation) {
                (BehaviorStatus::Success, input_opt) => {
                    self.current_child_index += 1;
                    last_input_on_success = input_opt; // Potentially store this from the last succeeding child
                }
                (BehaviorStatus::Running, input_opt) => {
                    debug_tree_node(
                        format!("bt.p{}.{}", situation.player_id, node_full_id),
                        self.description(),
                        node_full_id.clone(),
                        self.get_child_node_ids(&situation.viz_path_prefix),
                        true, // Active if running
                    );
                    return (BehaviorStatus::Running, input_opt);
                }
                (BehaviorStatus::Failure, _input_opt) => {
                    self.current_child_index = 0; // Reset for next time
                    debug_tree_node(
                        format!("bt.p{}.{}", situation.player_id, node_full_id),
                        self.description(),
                        node_full_id.clone(),
                        self.get_child_node_ids(&situation.viz_path_prefix),
                        false, // Not active if failed
                    );
                    return (BehaviorStatus::Failure, None);
                }
            }
        }

        self.current_child_index = 0; // Reset for next full run
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            true,
        );
        (BehaviorStatus::Success, last_input_on_success)
    }

    fn description(&self) -> String {
        self.description_text.clone()
    }

    fn get_node_id_fragment(&self) -> String {
        self.node_id_fragment.clone()
    }

    fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        let self_full_id = self.get_full_node_id(current_path_prefix);
        self.children
            .iter()
            .map(|c| c.get_full_node_id(&self_full_id))
            .collect()
    }
}

// Guard Node (Conditional Node)
// Executes a child behavior only if a condition (Situation) is met.
pub struct GuardNode {
    condition: Situation,
    child: Box<dyn BehaviorNode>,
    description_text: String,
    node_id_fragment: String,
}

impl GuardNode {
    pub fn new(
        condition: Situation,
        child: Box<dyn BehaviorNode>,
        description_override: Option<String>,
    ) -> Self {
        let desc = description_override.unwrap_or_else(|| {
            format!(
                "Guard_If_{}_Then_{}",
                sanitize_id_fragment(&condition.description),
                child.get_node_id_fragment()
            )
        });
        Self {
            condition,
            child,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
        }
    }
}

impl BehaviorNode for GuardNode {
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let (status, input) = if self.condition.check(situation) {
            self.child.tick(situation)
        } else {
            (BehaviorStatus::Failure, None)
        };

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            is_active && self.condition.check(situation), // Active if condition met and child is active
        );
        (status, input)
    }

    fn description(&self) -> String {
        // Use stored description_text which might be an override or a generated one.
        format!(
            "IF ({}) THEN ({})",
            self.condition.description,
            self.child.description()
        )
    }

    fn get_node_id_fragment(&self) -> String {
        self.node_id_fragment.clone()
    }

    fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        let self_full_id = self.get_full_node_id(current_path_prefix);
        vec![self.child.get_full_node_id(&self_full_id)]
    }
}

// Action Node
// Executes a skill and translates its progress to behavior status and player control input.
pub struct ActionNode {
    skill: Box<dyn Skill>,
    description_text: String,
    node_id_fragment: String,
}

impl ActionNode {
    pub fn new(skill: Box<dyn Skill>, description: Option<String>) -> Self {
        let desc = description.unwrap_or_else(|| format!("Action")); // TODO: Add skill name
        Self {
            skill,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
        }
    }
}

impl BehaviorNode for ActionNode {
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let skill_ctx = SkillCtx {
            player: situation.player_data,
            world: situation.world,
        };

        let (status, input) = match self.skill.update(skill_ctx) {
            SkillProgress::Continue(input) => (BehaviorStatus::Running, Some(input)),
            SkillProgress::Done(SkillResult::Success) => (BehaviorStatus::Success, None),
            SkillProgress::Done(SkillResult::Failure) => (BehaviorStatus::Failure, None),
        };

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix), // empty vec
            is_active,
        );
        (status, input)
    }

    fn description(&self) -> String {
        self.description_text.clone()
    }

    fn get_node_id_fragment(&self) -> String {
        self.node_id_fragment.clone()
    }

    fn get_child_node_ids(&self, _current_path_prefix: &str) -> Vec<String> {
        vec![] // Action nodes are leaves
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
    semaphore_id_str: String, // Renamed to avoid conflict with id method
    max_count: usize,
    description_text: String,
    node_id_fragment: String,
    player_id_holding_via_this_node: Option<PlayerId>,
}

impl SemaphoreNode {
    pub fn new(
        child: Box<dyn BehaviorNode>,
        semaphore_id: String,
        max_count: usize,
        description: Option<String>,
    ) -> Self {
        let desc = description.unwrap_or_else(|| format!("Semaphore_{}", semaphore_id));
        Self {
            child,
            semaphore_id_str: semaphore_id,
            max_count,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            player_id_holding_via_this_node: None,
        }
    }
}

impl BehaviorNode for SemaphoreNode {
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let player_id = situation.player_id;
        let mut result_status = BehaviorStatus::Failure;
        let mut result_input: Option<PlayerControlInput> = None;

        let acquired_semaphore = if self.player_id_holding_via_this_node == Some(player_id) {
            true // Already holding
        } else {
            // Try to acquire
            if situation.team_context.try_acquire_semaphore(
                &self.semaphore_id_str,
                self.max_count,
                player_id,
            ) {
                self.player_id_holding_via_this_node = Some(player_id);
                true
            } else {
                false // Failed to acquire
            }
        };

        if acquired_semaphore {
            let (child_status, child_input) = self.child.tick(situation);
            result_status = child_status;
            result_input = child_input;

            match child_status {
                BehaviorStatus::Success | BehaviorStatus::Failure => {
                    situation
                        .team_context
                        .release_semaphore(&self.semaphore_id_str, player_id);
                    self.player_id_holding_via_this_node = None;
                }
                BehaviorStatus::Running => {
                    // Keep holding
                }
            }
        } else {
            // Failed to acquire semaphore
            result_status = BehaviorStatus::Failure;
            result_input = None;
        }

        let is_active = result_status == BehaviorStatus::Running
            || (result_status == BehaviorStatus::Success && acquired_semaphore);
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            is_active,
        );
        (result_status, result_input)
    }

    fn description(&self) -> String {
        self.description_text.clone()
    }

    fn get_node_id_fragment(&self) -> String {
        self.node_id_fragment.clone()
    }

    fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        let self_full_id = self.get_full_node_id(current_path_prefix);
        vec![self.child.get_full_node_id(&self_full_id)]
    }
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
    description_text: String,
    node_id_fragment: String,
    current_best_child_index: Option<usize>,
    current_best_child_score: f64,
}

impl ScoringSelectNode {
    pub fn new(
        children_with_scorers: Vec<(Box<dyn BehaviorNode>, Box<dyn Fn(&RobotSituation) -> f64>)>,
        hysteresis_margin: f64,
        description: Option<String>,
    ) -> Self {
        let desc = description.unwrap_or_else(|| "ScoringSelect".to_string());
        Self {
            children_with_scorers,
            hysteresis_margin,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            current_best_child_index: None,
            current_best_child_score: f64::NEG_INFINITY,
        }
    }
}

impl BehaviorNode for ScoringSelectNode {
    fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);

        if self.children_with_scorers.is_empty() {
            debug_tree_node(
                format!("bt.p{}.{}", situation.player_id, node_full_id),
                self.description(),
                node_full_id.clone(),
                vec![],
                false, // Not active if no children
            );
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
            let score_of_active_child = scores[active_child_idx];
            if score_of_active_child >= new_highest_score - self.hysteresis_margin {
                final_child_to_tick_idx = active_child_idx;
            } else {
                final_child_to_tick_idx = new_best_child_candidate_index;
            }
        } else {
            final_child_to_tick_idx = new_best_child_candidate_index;
        }

        // Update state *before* ticking child, if selection changes or is made first time
        if self.current_best_child_index != Some(final_child_to_tick_idx) {
            self.current_best_child_index = Some(final_child_to_tick_idx);
            self.current_best_child_score = scores[final_child_to_tick_idx]; // Store the score that led to its selection
        }

        let (child_node, _) = &mut self.children_with_scorers[final_child_to_tick_idx];
        let (status, input) = child_node.tick(situation);

        match status {
            BehaviorStatus::Success | BehaviorStatus::Failure => {
                self.current_best_child_index = None;
                self.current_best_child_score = f64::NEG_INFINITY;
            }
            BehaviorStatus::Running => {
                // current_best_child_index and current_best_child_score remain to reflect the running child
            }
        }

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(), // Or a more detailed one including active child
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            is_active,
        );
        (status, input)
    }

    fn description(&self) -> String {
        // Consider adding active child info to description if useful for debugging.
        self.description_text.clone()
    }

    fn get_node_id_fragment(&self) -> String {
        self.node_id_fragment.clone()
    }

    fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        let self_full_id = self.get_full_node_id(current_path_prefix);
        self.children_with_scorers
            .iter()
            .map(|(c, _)| c.get_full_node_id(&self_full_id))
            .collect()
    }
}
