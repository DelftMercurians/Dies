use dies_core::{PlayerData, PlayerId, TeamData};
use rhai::Engine;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::PlayerControlInput;

use super::{BehaviorNode, NoopNode};

#[derive(Clone)]
pub struct BehaviorTree {
    root_node: BehaviorNode,
}

impl BehaviorTree {
    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
        engine: &Engine,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        // First, generate debug info for the entire tree structure
        self.root_node.debug_all_nodes(situation, engine);

        // Then tick the tree normally
        self.root_node.tick(situation, engine)
    }
}

impl BehaviorTree {
    pub fn new(root_node: BehaviorNode) -> Self {
        Self { root_node }
    }
}

impl Default for BehaviorTree {
    fn default() -> Self {
        Self::new(BehaviorNode::Noop(NoopNode::new()))
    }
}

#[derive(Clone)]
pub struct BtContext {
    semaphores: Arc<RwLock<HashMap<String, (usize, HashSet<PlayerId>)>>>,
}

impl BtContext {
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

impl Default for BtContext {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BehaviorStatus {
    Success,
    Failure,
    #[default]
    Running,
}

#[derive(Clone)]
pub struct RobotSituation {
    pub player_id: PlayerId,
    pub world: Arc<TeamData>,
    pub team_context: BtContext,
    pub viz_path_prefix: String,
}

impl RobotSituation {
    pub fn new(
        player_id: PlayerId,
        world: Arc<TeamData>,
        team_context: BtContext,
        viz_path_prefix: String,
    ) -> Self {
        Self {
            player_id,
            world,
            team_context,
            viz_path_prefix,
        }
    }

    pub fn player_data(&self) -> &PlayerData {
        self.world.get_player(self.player_id)
    }

    pub fn has_ball(&self) -> bool {
        self.player_data().breakbeam_ball_detected
    }
}
