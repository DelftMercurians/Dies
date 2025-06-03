use crate::control::PlayerControlInput;
use crate::roles::{Skill, SkillCtx, SkillProgress, SkillResult};
use dies_core::{debug_tree_node, PlayerData, PlayerId, WorldData};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

pub mod rhai_integration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BehaviorStatus {
    Success,
    Failure,
    Running,
}

#[derive(Clone)]
pub enum BehaviorNode {
    Select(SelectNode),
    Sequence(SequenceNode),
    Guard(GuardNode),
    Action(ActionNode),
    Semaphore(SemaphoreNode),
    ScoringSelect(ScoringSelectNode),
}

impl BehaviorNode {
    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        match self {
            BehaviorNode::Select(node) => node.tick(situation),
            BehaviorNode::Sequence(node) => node.tick(situation),
            BehaviorNode::Guard(node) => node.tick(situation),
            BehaviorNode::Action(node) => node.tick(situation),
            BehaviorNode::Semaphore(node) => node.tick(situation),
            BehaviorNode::ScoringSelect(node) => node.tick(situation),
        }
    }

    pub fn description(&self) -> String {
        match self {
            BehaviorNode::Select(node) => node.description(),
            BehaviorNode::Sequence(node) => node.description(),
            BehaviorNode::Guard(node) => node.description(),
            BehaviorNode::Action(node) => node.description(),
            BehaviorNode::Semaphore(node) => node.description(),
            BehaviorNode::ScoringSelect(node) => node.description(),
        }
    }

    pub fn get_node_id_fragment(&self) -> String {
        match self {
            BehaviorNode::Select(node) => node.get_node_id_fragment(),
            BehaviorNode::Sequence(node) => node.get_node_id_fragment(),
            BehaviorNode::Guard(node) => node.get_node_id_fragment(),
            BehaviorNode::Action(node) => node.get_node_id_fragment(),
            BehaviorNode::Semaphore(node) => node.get_node_id_fragment(),
            BehaviorNode::ScoringSelect(node) => node.get_node_id_fragment(),
        }
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
        match self {
            BehaviorNode::Select(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::Sequence(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::Guard(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::Action(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::Semaphore(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::ScoringSelect(node) => node.get_child_node_ids(current_path_prefix),
        }
    }
}

fn sanitize_id_fragment(text: &str) -> String {
    text.chars()
        .filter_map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' => Some(c.to_ascii_lowercase()),
            ' ' | '_' | '-' => Some('_'),
            _ => None,
        })
        .collect::<String>()
}

#[derive(Clone)]
pub struct RobotSituation<'a> {
    pub player_id: PlayerId,
    pub world: &'a WorldData,
    pub player_data: &'a PlayerData,
    pub team_context: &'a TeamContext,
    pub viz_path_prefix: String,
}

impl<'a> RobotSituation<'a> {
    pub fn new(
        player_id: PlayerId,
        world: &'a WorldData,
        team_context: &'a TeamContext,
        viz_path_prefix: String,
    ) -> Self {
        let player_data = world.get_player(player_id);
        Self {
            player_id,
            world,
            player_data,
            team_context,
            viz_path_prefix,
        }
    }

    pub fn has_ball(&self) -> bool {
        self.player_data.breakbeam_ball_detected
    }
}

#[derive(Clone)]
pub struct Situation {
    condition: Arc<dyn Fn(&RobotSituation) -> bool>,
    description: String,
}

impl Situation {
    pub fn new(condition: impl Fn(&RobotSituation) -> bool + 'static, description: &str) -> Self {
        Self {
            condition: Arc::new(condition),
            description: description.to_string(),
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

#[derive(Clone)]
pub struct SelectNode {
    children: Vec<BehaviorNode>,
    description_text: String,
    node_id_fragment: String,
}

impl SelectNode {
    pub fn new(children: Vec<BehaviorNode>, description: Option<String>) -> Self {
        let desc = description.unwrap_or_else(|| "Select".to_string());
        Self {
            children,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
        }
    }

    pub fn tick(
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
            self.get_child_node_ids(&node_full_id),
            is_active,
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
        let self_full_id = self.get_full_node_id(current_path_prefix);
        self.children
            .iter()
            .map(|c| c.get_full_node_id(&self_full_id))
            .collect()
    }
}

#[derive(Clone)]
pub struct SequenceNode {
    children: Vec<BehaviorNode>,
    description_text: String,
    node_id_fragment: String,
    current_child_index: usize,
}

impl SequenceNode {
    pub fn new(children: Vec<BehaviorNode>, description: Option<String>) -> Self {
        let desc = description.unwrap_or_else(|| "Sequence".to_string());
        Self {
            children,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            current_child_index: 0,
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let mut last_input_on_success: Option<PlayerControlInput> = None;

        while self.current_child_index < self.children.len() {
            match self.children[self.current_child_index].tick(situation) {
                (BehaviorStatus::Success, input_opt) => {
                    self.current_child_index += 1;
                    last_input_on_success = input_opt;
                }
                (BehaviorStatus::Running, input_opt) => {
                    debug_tree_node(
                        format!("bt.p{}.{}", situation.player_id, node_full_id),
                        self.description(),
                        node_full_id.clone(),
                        self.get_child_node_ids(&node_full_id),
                        true,
                    );
                    return (BehaviorStatus::Running, input_opt);
                }
                (BehaviorStatus::Failure, _input_opt) => {
                    self.current_child_index = 0;
                    debug_tree_node(
                        format!("bt.p{}.{}", situation.player_id, node_full_id),
                        self.description(),
                        node_full_id.clone(),
                        self.get_child_node_ids(&node_full_id),
                        false,
                    );
                    return (BehaviorStatus::Failure, None);
                }
            }
        }

        self.current_child_index = 0;
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&node_full_id),
            true,
        );
        (BehaviorStatus::Success, last_input_on_success)
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
        let self_full_id = self.get_full_node_id(current_path_prefix);
        self.children
            .iter()
            .map(|c| c.get_full_node_id(&self_full_id))
            .collect()
    }
}

#[derive(Clone)]
pub struct GuardNode {
    condition: Situation,
    child: Box<BehaviorNode>,
    description_text: String,
    node_id_fragment: String,
}

impl GuardNode {
    pub fn new(
        condition: Situation,
        child: BehaviorNode,
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
            child: Box::new(child),
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
        }
    }

    pub fn tick(
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
            self.get_child_node_ids(&node_full_id),
            is_active && self.condition.check(situation),
        );
        (status, input)
    }

    pub fn description(&self) -> String {
        format!(
            "IF ({}) THEN ({})",
            self.condition.description,
            self.child.description()
        )
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
        let self_full_id = self.get_full_node_id(current_path_prefix);
        vec![self.child.get_full_node_id(&self_full_id)]
    }
}

#[derive(Clone)]
pub struct ActionNode {
    skill: Skill,
    description_text: String,
    node_id_fragment: String,
}

impl ActionNode {
    pub fn new(skill: Skill, description: Option<String>) -> Self {
        let desc = description.unwrap_or_else(|| format!("Action"));
        Self {
            skill,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
        }
    }

    pub fn tick(
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
            self.get_child_node_ids(&node_full_id),
            is_active,
        );
        (status, input)
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

    pub fn get_child_node_ids(&self, _current_path_prefix: &str) -> Vec<String> {
        vec![]
    }
}

#[derive(Clone)]
pub struct TeamContext {
    semaphores: Arc<RwLock<HashMap<String, (usize, HashSet<PlayerId>)>>>,
}

impl TeamContext {
    pub fn new() -> Self {
        Self {
            semaphores: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn try_acquire_semaphore(&self, id: &str, max_count: usize, player_id: PlayerId) -> bool {
        let mut semaphores = self.semaphores.write().unwrap();
        let entry = semaphores
            .entry(id.to_string())
            .or_insert((0, HashSet::new()));

        if entry.1.contains(&player_id) {
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

    pub fn release_semaphore(&self, id: &str, player_id: PlayerId) {
        let mut semaphores = self.semaphores.write().unwrap();
        if let Some(entry) = semaphores.get_mut(id) {
            if entry.1.remove(&player_id) {
                entry.0 = entry.0.saturating_sub(1);
            }
        }
    }

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

#[derive(Clone)]
pub struct SemaphoreNode {
    child: Box<BehaviorNode>,
    semaphore_id_str: String,
    max_count: usize,
    description_text: String,
    node_id_fragment: String,
    player_id_holding_via_this_node: Option<PlayerId>,
}

impl SemaphoreNode {
    pub fn new(
        child: BehaviorNode,
        semaphore_id: String,
        max_count: usize,
        description: Option<String>,
    ) -> Self {
        let desc = description.unwrap_or_else(|| format!("Semaphore_{}", semaphore_id));
        Self {
            child: Box::new(child),
            semaphore_id_str: semaphore_id,
            max_count,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            player_id_holding_via_this_node: None,
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let player_id = situation.player_id;
        let mut result_status = BehaviorStatus::Failure;
        let mut result_input: Option<PlayerControlInput> = None;

        let acquired_semaphore = if self.player_id_holding_via_this_node == Some(player_id) {
            true
        } else {
            if situation.team_context.try_acquire_semaphore(
                &self.semaphore_id_str,
                self.max_count,
                player_id,
            ) {
                self.player_id_holding_via_this_node = Some(player_id);
                true
            } else {
                false
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
                BehaviorStatus::Running => {}
            }
        } else {
            result_status = BehaviorStatus::Failure;
            result_input = None;
        }

        let is_active = result_status == BehaviorStatus::Running
            || (result_status == BehaviorStatus::Success && acquired_semaphore);
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&node_full_id),
            is_active,
        );
        (result_status, result_input)
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
        let self_full_id = self.get_full_node_id(current_path_prefix);
        vec![self.child.get_full_node_id(&self_full_id)]
    }
}

#[derive(Clone)]
pub struct ScoringSelectNode {
    children_with_scorers: Vec<(BehaviorNode, Arc<dyn Fn(&RobotSituation) -> f64>)>,
    hysteresis_margin: f64,
    description_text: String,
    node_id_fragment: String,
    current_best_child_index: Option<usize>,
    current_best_child_score: f64,
}

impl ScoringSelectNode {
    pub fn new(
        children_with_scorers_boxed: Vec<(BehaviorNode, Box<dyn Fn(&RobotSituation) -> f64>)>,
        hysteresis_margin: f64,
        description: Option<String>,
    ) -> Self {
        let desc = description.unwrap_or_else(|| "ScoringSelect".to_string());
        let children_with_scorers = children_with_scorers_boxed
            .into_iter()
            .map(|(node, scorer_box)| (node, Arc::from(scorer_box)))
            .collect();

        Self {
            children_with_scorers,
            hysteresis_margin,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            current_best_child_index: None,
            current_best_child_score: f64::NEG_INFINITY,
        }
    }

    pub fn tick(
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
                false,
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

        if self.current_best_child_index != Some(final_child_to_tick_idx) {
            self.current_best_child_index = Some(final_child_to_tick_idx);
            self.current_best_child_score = scores[final_child_to_tick_idx];
        }

        let (child_node, _) = &mut self.children_with_scorers[final_child_to_tick_idx];
        let (status, input) = child_node.tick(situation);

        match status {
            BehaviorStatus::Success | BehaviorStatus::Failure => {
                self.current_best_child_index = None;
                self.current_best_child_score = f64::NEG_INFINITY;
            }
            BehaviorStatus::Running => {}
        }

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&node_full_id),
            is_active,
        );
        (status, input)
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
        let self_full_id = self.get_full_node_id(current_path_prefix);
        self.children_with_scorers
            .iter()
            .map(|(c, _)| c.get_full_node_id(&self_full_id))
            .collect()
    }
}
