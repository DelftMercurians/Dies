//! Composite / decorator nodes: the control-flow of the tree.

use std::collections::HashMap;
use std::sync::Arc;

use dies_strategy_protocol::PlayerId;

use super::super::situation::{BehaviorStatus, RobotSituation};
use super::super::BtCallback;
use super::{BehaviorNode, TickResult};

// ─────────────────────────────────────────────────────────────────────────────
// Select — first child that succeeds or is running wins.
// ─────────────────────────────────────────────────────────────────────────────

pub struct SelectNode {
    children: Vec<BehaviorNode>,
}

impl SelectNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        for child in self.children.iter_mut() {
            match child.tick(situation) {
                (BehaviorStatus::Success, input) => return (BehaviorStatus::Success, input),
                (BehaviorStatus::Running, input) => return (BehaviorStatus::Running, input),
                (BehaviorStatus::Failure, _) => continue,
            }
        }
        (BehaviorStatus::Failure, None)
    }
}

impl From<SelectNode> for BehaviorNode {
    fn from(n: SelectNode) -> Self {
        BehaviorNode::Select(n)
    }
}

#[derive(Default)]
pub struct SelectNodeBuilder {
    children: Vec<BehaviorNode>,
}

impl SelectNodeBuilder {
    pub fn add(mut self, child: impl Into<BehaviorNode>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn description(self, _description: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> SelectNode {
        SelectNode {
            children: self.children,
        }
    }
}

pub fn select_node() -> SelectNodeBuilder {
    SelectNodeBuilder::default()
}

// ─────────────────────────────────────────────────────────────────────────────
// Sequence — run children in order; restart on failure.
// ─────────────────────────────────────────────────────────────────────────────

pub struct SequenceNode {
    children: Vec<BehaviorNode>,
    current_child_index: usize,
}

impl SequenceNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        let mut last_input = None;
        while self.current_child_index < self.children.len() {
            match self.children[self.current_child_index].tick(situation) {
                (BehaviorStatus::Success, input) => {
                    self.current_child_index += 1;
                    last_input = input;
                }
                (BehaviorStatus::Running, input) => return (BehaviorStatus::Running, input),
                (BehaviorStatus::Failure, _) => {
                    self.current_child_index = 0;
                    return (BehaviorStatus::Failure, None);
                }
            }
        }
        self.current_child_index = 0;
        (BehaviorStatus::Success, last_input)
    }
}

impl From<SequenceNode> for BehaviorNode {
    fn from(n: SequenceNode) -> Self {
        BehaviorNode::Sequence(n)
    }
}

#[derive(Default)]
pub struct SequenceNodeBuilder {
    children: Vec<BehaviorNode>,
}

impl SequenceNodeBuilder {
    pub fn add(mut self, child: impl Into<BehaviorNode>) -> Self {
        self.children.push(child.into());
        self
    }

    pub fn description(self, _description: impl AsRef<str>) -> Self {
        self
    }

    pub fn ignore_fail(self) -> Self {
        self
    }

    pub fn build(self) -> SequenceNode {
        SequenceNode {
            children: self.children,
            current_child_index: 0,
        }
    }
}

pub fn sequence_node() -> SequenceNodeBuilder {
    SequenceNodeBuilder::default()
}

// ─────────────────────────────────────────────────────────────────────────────
// Guard — run child only while condition holds.
// ─────────────────────────────────────────────────────────────────────────────

pub struct GuardNode {
    condition: Arc<dyn BtCallback<bool>>,
    child: Box<BehaviorNode>,
}

impl GuardNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        if (self.condition)(situation) {
            self.child.tick(situation)
        } else {
            (BehaviorStatus::Failure, None)
        }
    }
}

impl From<GuardNode> for BehaviorNode {
    fn from(n: GuardNode) -> Self {
        BehaviorNode::Guard(n)
    }
}

#[derive(Default)]
pub struct GuardNodeBuilder {
    condition: Option<Arc<dyn BtCallback<bool>>>,
    child: Option<BehaviorNode>,
}

impl GuardNodeBuilder {
    pub fn condition(mut self, condition: impl BtCallback<bool>) -> Self {
        self.condition = Some(Arc::new(condition));
        self
    }

    pub fn then(mut self, child: impl Into<BehaviorNode>) -> Self {
        self.child = Some(child.into());
        self
    }

    pub fn description(self, _description: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> GuardNode {
        GuardNode {
            condition: self.condition.expect("Guard condition is required"),
            child: Box::new(self.child.expect("Guard child is required")),
        }
    }
}

pub fn guard_node() -> GuardNodeBuilder {
    GuardNodeBuilder::default()
}

// ─────────────────────────────────────────────────────────────────────────────
// GuardWithHysteresis — like Guard, but the condition must hold/clear for a few
// ticks before switching (debounce).
// ─────────────────────────────────────────────────────────────────────────────

pub struct GuardWithHysteresisNode {
    condition: Arc<dyn BtCallback<bool>>,
    child: Box<BehaviorNode>,
    hysteresis_margin: f64,
    last_condition_result: Option<bool>,
    activation_count: u32,
    deactivation_count: u32,
}

impl GuardWithHysteresisNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        let raw = (self.condition)(situation);
        let threshold = (self.hysteresis_margin * 10.0) as u32;
        let effective = match self.last_condition_result {
            None => {
                self.last_condition_result = Some(raw);
                raw
            }
            Some(last) => {
                if last && !raw {
                    self.deactivation_count += 1;
                    if self.deactivation_count >= threshold {
                        self.last_condition_result = Some(false);
                        self.deactivation_count = 0;
                        self.activation_count = 0;
                        false
                    } else {
                        true
                    }
                } else if !last && raw {
                    self.activation_count += 1;
                    if self.activation_count >= threshold {
                        self.last_condition_result = Some(true);
                        self.activation_count = 0;
                        self.deactivation_count = 0;
                        true
                    } else {
                        false
                    }
                } else {
                    self.activation_count = 0;
                    self.deactivation_count = 0;
                    last
                }
            }
        };

        if effective {
            self.child.tick(situation)
        } else {
            (BehaviorStatus::Failure, None)
        }
    }
}

impl From<GuardWithHysteresisNode> for BehaviorNode {
    fn from(n: GuardWithHysteresisNode) -> Self {
        BehaviorNode::GuardWithHysteresis(n)
    }
}

pub struct GuardWithHysteresisNodeBuilder {
    condition: Option<Arc<dyn BtCallback<bool>>>,
    child: Option<BehaviorNode>,
    hysteresis_margin: f64,
}

impl Default for GuardWithHysteresisNodeBuilder {
    fn default() -> Self {
        Self {
            condition: None,
            child: None,
            hysteresis_margin: 0.1,
        }
    }
}

impl GuardWithHysteresisNodeBuilder {
    pub fn condition(mut self, condition: impl BtCallback<bool>) -> Self {
        self.condition = Some(Arc::new(condition));
        self
    }

    pub fn then(mut self, child: impl Into<BehaviorNode>) -> Self {
        self.child = Some(child.into());
        self
    }

    pub fn hysteresis_margin(mut self, margin: f64) -> Self {
        self.hysteresis_margin = margin;
        self
    }

    pub fn description(self, _description: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> GuardWithHysteresisNode {
        GuardWithHysteresisNode {
            condition: self.condition.expect("Guard condition is required"),
            child: Box::new(self.child.expect("Guard child is required")),
            hysteresis_margin: self.hysteresis_margin,
            last_condition_result: None,
            activation_count: 0,
            deactivation_count: 0,
        }
    }
}

pub fn guard_with_hysteresis_node() -> GuardWithHysteresisNodeBuilder {
    GuardWithHysteresisNodeBuilder::default()
}

// ─────────────────────────────────────────────────────────────────────────────
// CommittingGuard — once the entry condition fires, stay committed to the child
// until it finishes or the cancel condition fires.
// ─────────────────────────────────────────────────────────────────────────────

pub struct CommittingGuardNode {
    entry_condition: Arc<dyn BtCallback<bool>>,
    cancel_condition: Arc<dyn BtCallback<bool>>,
    child: Box<BehaviorNode>,
    is_committed: bool,
}

impl CommittingGuardNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        if !self.is_committed {
            if (self.entry_condition)(situation) {
                self.is_committed = true;
            } else {
                return (BehaviorStatus::Failure, None);
            }
        }

        if (self.cancel_condition)(situation) {
            self.is_committed = false;
            return (BehaviorStatus::Failure, None);
        }

        let (status, input) = self.child.tick(situation);
        if matches!(status, BehaviorStatus::Success | BehaviorStatus::Failure) {
            self.is_committed = false;
        }
        (status, input)
    }
}

impl From<CommittingGuardNode> for BehaviorNode {
    fn from(n: CommittingGuardNode) -> Self {
        BehaviorNode::CommittingGuard(n)
    }
}

#[derive(Default)]
pub struct CommittingGuardNodeBuilder {
    entry_condition: Option<Arc<dyn BtCallback<bool>>>,
    cancel_condition: Option<Arc<dyn BtCallback<bool>>>,
    child: Option<BehaviorNode>,
}

impl CommittingGuardNodeBuilder {
    pub fn when(mut self, condition: impl BtCallback<bool>) -> Self {
        self.entry_condition = Some(Arc::new(condition));
        self
    }

    pub fn commit_to(mut self, child: impl Into<BehaviorNode>) -> Self {
        self.child = Some(child.into());
        self
    }

    pub fn until(mut self, condition: impl BtCallback<bool>) -> Self {
        self.cancel_condition = Some(Arc::new(condition));
        self
    }

    pub fn description(self, _description: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> CommittingGuardNode {
        CommittingGuardNode {
            entry_condition: self.entry_condition.expect("entry condition is required"),
            cancel_condition: self.cancel_condition.expect("cancel condition is required"),
            child: Box::new(self.child.expect("child is required")),
            is_committed: false,
        }
    }
}

pub fn committing_guard_node() -> CommittingGuardNodeBuilder {
    CommittingGuardNodeBuilder::default()
}

// ─────────────────────────────────────────────────────────────────────────────
// ScoringSelect — pick the highest-scoring child (with hysteresis on switching).
// ─────────────────────────────────────────────────────────────────────────────

struct Scorer {
    node: BehaviorNode,
    callback: Arc<dyn BtCallback<f64>>,
    key: String,
}

pub struct ScoringSelectNode {
    scorers: Vec<Scorer>,
    hysteresis_margin: f64,
    current_best_child_key: Option<String>,
}

impl ScoringSelectNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        if self.scorers.is_empty() {
            return (BehaviorStatus::Failure, None);
        }

        let mut scores = HashMap::new();
        for scorer in self.scorers.iter() {
            scores.insert(scorer.key.clone(), (scorer.callback)(situation));
        }

        let (best_key, &highest) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let final_key = if let Some(active) = self.current_best_child_key.as_ref() {
            match scores.get(active) {
                Some(&active_score) if active_score >= highest - self.hysteresis_margin => {
                    active.clone()
                }
                _ => best_key.clone(),
            }
        } else {
            best_key.clone()
        };
        self.current_best_child_key = Some(final_key.clone());

        let idx = self
            .scorers
            .iter()
            .position(|s| s.key == final_key)
            .unwrap();
        let (status, input) = self.scorers[idx].node.tick(situation);
        if matches!(status, BehaviorStatus::Success | BehaviorStatus::Failure) {
            self.current_best_child_key = None;
        }
        (status, input)
    }
}

impl From<ScoringSelectNode> for BehaviorNode {
    fn from(n: ScoringSelectNode) -> Self {
        BehaviorNode::ScoringSelect(n)
    }
}

pub struct ScoringSelectNodeBuilder {
    children: Vec<(BehaviorNode, Arc<dyn BtCallback<f64>>)>,
    hysteresis_margin: f64,
}

impl Default for ScoringSelectNodeBuilder {
    fn default() -> Self {
        Self {
            children: Vec::new(),
            hysteresis_margin: 0.1,
        }
    }
}

impl ScoringSelectNodeBuilder {
    pub fn add_child(
        mut self,
        child: impl Into<BehaviorNode>,
        scorer: impl BtCallback<f64>,
    ) -> Self {
        self.children.push((child.into(), Arc::new(scorer)));
        self
    }

    pub fn hysteresis_margin(mut self, margin: f64) -> Self {
        self.hysteresis_margin = margin;
        self
    }

    pub fn description(self, _description: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> ScoringSelectNode {
        let scorers = self
            .children
            .into_iter()
            .enumerate()
            .map(|(i, (node, callback))| Scorer {
                node,
                callback,
                key: i.to_string(),
            })
            .collect();
        ScoringSelectNode {
            scorers,
            hysteresis_margin: self.hysteresis_margin,
            current_best_child_key: None,
        }
    }
}

pub fn scoring_select_node() -> ScoringSelectNodeBuilder {
    ScoringSelectNodeBuilder::default()
}

// ─────────────────────────────────────────────────────────────────────────────
// Semaphore — cooperatively cap how many robots run the child subtree at once.
// ─────────────────────────────────────────────────────────────────────────────

pub struct SemaphoreNode {
    child: Box<BehaviorNode>,
    semaphore_id: String,
    max_count: usize,
    holding: Option<PlayerId>,
}

impl SemaphoreNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        let player_id = situation.player_id;
        let acquired = situation.bt_context.try_acquire_semaphore(
            &self.semaphore_id,
            self.max_count,
            player_id,
        );

        if !acquired {
            self.holding = None;
            return (BehaviorStatus::Failure, None);
        }
        self.holding = Some(player_id);

        let (status, input) = self.child.tick(situation);
        if matches!(status, BehaviorStatus::Success | BehaviorStatus::Failure) {
            situation
                .bt_context
                .release_semaphore(&self.semaphore_id, player_id);
            self.holding = None;
        }
        (status, input)
    }
}

impl From<SemaphoreNode> for BehaviorNode {
    fn from(n: SemaphoreNode) -> Self {
        BehaviorNode::Semaphore(n)
    }
}

#[derive(Default)]
pub struct SemaphoreNodeBuilder {
    child: Option<BehaviorNode>,
    semaphore_id: Option<String>,
    max_count: Option<usize>,
}

impl SemaphoreNodeBuilder {
    pub fn do_then(mut self, child: impl Into<BehaviorNode>) -> Self {
        self.child = Some(child.into());
        self
    }

    pub fn semaphore_id(mut self, semaphore_id: impl Into<String>) -> Self {
        self.semaphore_id = Some(semaphore_id.into());
        self
    }

    pub fn max_entry(mut self, max_count: usize) -> Self {
        self.max_count = Some(max_count);
        self
    }

    pub fn description(self, _description: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> SemaphoreNode {
        SemaphoreNode {
            child: Box::new(self.child.expect("semaphore child is required")),
            semaphore_id: self.semaphore_id.expect("semaphore id is required"),
            max_count: self.max_count.expect("semaphore max count is required"),
            holding: None,
        }
    }
}

pub fn semaphore_node() -> SemaphoreNodeBuilder {
    SemaphoreNodeBuilder::default()
}

// ─────────────────────────────────────────────────────────────────────────────
// Repeat — re-tick the child until it reports Running.
// ─────────────────────────────────────────────────────────────────────────────

pub struct RepeatNode {
    child: Box<BehaviorNode>,
}

impl RepeatNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        // Bounded to avoid an infinite loop if the child resolves instantly every
        // time; in practice a child either runs a skill (Running) or flips a
        // cheap condition.
        for _ in 0..64 {
            let (status, input) = self.child.tick(situation);
            if status == BehaviorStatus::Running {
                return (status, input);
            }
        }
        (BehaviorStatus::Running, None)
    }
}

impl From<RepeatNode> for BehaviorNode {
    fn from(n: RepeatNode) -> Self {
        BehaviorNode::Repeat(n)
    }
}

pub struct RepeatNodeBuilder {
    child: BehaviorNode,
}

impl RepeatNodeBuilder {
    pub fn description(self, _description: impl AsRef<str>) -> Self {
        self
    }

    pub fn build(self) -> RepeatNode {
        RepeatNode {
            child: Box::new(self.child),
        }
    }
}

pub fn repeat_node(child: impl Into<BehaviorNode>) -> RepeatNodeBuilder {
    RepeatNodeBuilder {
        child: child.into(),
    }
}

impl From<RepeatNodeBuilder> for BehaviorNode {
    fn from(builder: RepeatNodeBuilder) -> Self {
        builder.build().into()
    }
}
