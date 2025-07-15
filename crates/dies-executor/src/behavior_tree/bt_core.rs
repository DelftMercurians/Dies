use dies_core::{Angle, GameState, PlayerData, PlayerId, TeamColor, TeamData, Vector2};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::behavior_tree::NoopNode;
use crate::PlayerControlInput;

use super::BehaviorNode;

pub struct BehaviorTree {
    pub name: String,
    root_node: BehaviorNode,
}

impl BehaviorTree {
    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        // First, generate debug info for the entire tree structure
        self.root_node.debug_all_nodes(situation);

        // Then tick the tree normally
        self.root_node.tick(situation)
    }
}

impl BehaviorTree {
    pub fn new(name: String, root_node: BehaviorNode) -> Self {
        Self { name, root_node }
    }
}

impl Default for BehaviorTree {
    fn default() -> Self {
        Self::new("noop".to_string(), BehaviorNode::Noop(NoopNode::new()))
    }
}

#[derive(Clone)]
pub struct PassingTarget {
    pub id: PlayerId,
    pub position: Vector2,
}

#[derive(Clone)]
pub struct BtContext {
    semaphores: Arc<RwLock<HashMap<String, HashSet<PlayerId>>>>,
    passing_target: Arc<RwLock<Option<PassingTarget>>>,
}

impl BtContext {
    pub fn new() -> Self {
        Self {
            semaphores: Arc::new(RwLock::new(HashMap::new())),
            passing_target: Arc::new(RwLock::new(None)),
        }
    }

    pub fn try_acquire_semaphore(&self, id: &str, max_count: usize, player_id: PlayerId) -> bool {
        let mut semaphores = self.semaphores.write().unwrap();
        let entry = semaphores.entry(id.to_string()).or_insert(HashSet::new());

        if entry.contains(&player_id) {
            return true;
        }

        if entry.len() < max_count {
            entry.insert(player_id);
            true
        } else {
            false
        }
    }

    pub fn release_semaphore(&self, id: &str, player_id: PlayerId) {
        let mut semaphores = self.semaphores.write().unwrap();
        if let Some(entry) = semaphores.get_mut(id) {
            entry.remove(&player_id);
        }
    }

    pub fn clear_semaphores(&self) {
        let mut semaphores = self.semaphores.write().unwrap();
        semaphores.clear();
    }

    pub fn clear_semaphores_for_player(&self, player_id: PlayerId) {
        let mut semaphores = self.semaphores.write().unwrap();
        for (_, player_set) in semaphores.iter_mut() {
            player_set.remove(&player_id);
        }
    }

    pub fn cleanup_empty_semaphores(&self) {
        let mut semaphores = self.semaphores.write().unwrap();
        semaphores.retain(|_, player_set| !player_set.is_empty());
    }

    pub fn set_passing_target(&self, target: PassingTarget) {
        *self.passing_target.write().unwrap() = Some(target);
    }

    pub fn is_passing_target(&self, player_id: PlayerId) -> bool {
        self.passing_target
            .read()
            .unwrap()
            .as_ref()
            .map(|p| p.id == player_id)
            .unwrap_or(false)
    }

    pub fn take_passing_target(&self, player_id: PlayerId) -> Option<PassingTarget> {
        let mut passing_target = self.passing_target.write().unwrap();
        if let Some(target) = passing_target.as_ref() {
            if target.id == player_id {
                passing_target.take()
            } else {
                None
            }
        } else {
            None
        }
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
    pub bt_context: BtContext,
    pub viz_path_prefix: String,
    pub role_assignments: Arc<HashMap<PlayerId, String>>,
    pub team_color: TeamColor,
}

impl RobotSituation {
    pub fn new(
        player_id: PlayerId,
        world: Arc<TeamData>,
        team_context: BtContext,
        viz_path_prefix: String,
        role_assignments: Arc<HashMap<PlayerId, String>>,
        team_color: TeamColor,
    ) -> Self {
        Self {
            player_id,
            world,
            bt_context: team_context,
            viz_path_prefix,
            role_assignments,
            team_color,
        }
    }

    pub fn debug_key(&self, key: impl Into<String>) -> String {
        format!("{}.{}", self.viz_path_prefix, key.into())
    }

    pub fn player_id(&self) -> PlayerId {
        self.player_id
    }

    pub fn ball_position(&self) -> Vector2 {
        self.world
            .ball
            .as_ref()
            .map(|b| b.position.xy())
            .unwrap_or_default()
    }

    pub fn ball_velocity(&self) -> Vector2 {
        self.world
            .ball
            .as_ref()
            .map(|b| b.velocity.xy())
            .unwrap_or_default()
    }

    pub fn ball_speed(&self) -> f64 {
        self.ball_velocity().norm()
    }

    pub fn player_id_hash(&self) -> f64 {
        (self.player_id.as_u32() as f64 * 0.6180339887498949) % 1.0
    }

    pub fn game_state(&self) -> GameState {
        self.world.current_game_state.game_state
    }

    pub fn game_state_is(&self, state: GameState) -> bool {
        self.game_state() == state
    }

    pub fn game_state_is_not(&self, state: GameState) -> bool {
        self.game_state() != state
    }

    pub fn game_state_is_one_of(&self, states: &[GameState]) -> bool {
        states.contains(&self.game_state())
    }

    pub fn game_state_is_none_of(&self, states: &[GameState]) -> bool {
        !states.contains(&self.game_state())
    }

    pub fn player_data(&self) -> &PlayerData {
        self.world.get_player(self.player_id)
    }

    pub fn position(&self) -> Vector2 {
        self.player_data().position
    }

    pub fn heading(&self) -> Angle {
        self.player_data().yaw
    }

    pub fn velocity(&self) -> Vector2 {
        self.player_data().velocity
    }

    pub fn has_ball(&self) -> bool {
        self.player_data().breakbeam_ball_detected
    }

    pub fn ball_in_our_half(&self) -> bool {
        self.world
            .ball
            .as_ref()
            .map(|b| b.position.x < 0.0)
            .unwrap_or(false)
    }

    pub fn ball_in_opponent_half(&self) -> bool {
        self.world
            .ball
            .as_ref()
            .map(|b| b.position.x > 0.0)
            .unwrap_or(false)
    }

    // Player role checks
    pub fn is_goalkeeper(&self) -> bool {
        self.player_id == PlayerId::new(0)
    }

    // Game state checks
    pub fn we_are_attacking(&self) -> bool {
        self.world.current_game_state.us_operating
    }

    pub fn is_kickoff_state(&self) -> bool {
        matches!(
            self.world.current_game_state.game_state,
            GameState::Kickoff | GameState::PrepareKickoff
        )
    }

    pub fn is_penalty_state(&self) -> bool {
        matches!(
            self.world.current_game_state.game_state,
            GameState::Penalty | GameState::PreparePenalty | GameState::PenaltyRun
        )
    }

    pub fn constrain_to_field(&self, pos: Vector2) -> Vector2 {
        if let Some(field) = &self.world.field_geom {
            let half_length = field.field_length / 2.0;
            let half_width = field.field_width / 2.0;

            let x = pos.x.max(-half_length + 100.0).min(half_length - 100.0);
            let y = pos.y.max(-half_width + 100.0).min(half_width - 100.0);

            Vector2::new(x, y)
        } else {
            pos
        }
    }

    pub fn is_free_kick_state(&self) -> bool {
        self.world.current_game_state.game_state == GameState::FreeKick
    }

    pub fn is_normal_play_state(&self) -> bool {
        self.world.current_game_state.game_state == GameState::Run
    }

    pub fn is_stopped_state(&self) -> bool {
        matches!(
            self.world.current_game_state.game_state,
            GameState::Stop | GameState::Halt
        )
    }

    pub fn close_to_ball(&self) -> bool {
        self.distance_to_ball() < 500.0
    }

    pub fn very_close_to_ball(&self) -> bool {
        self.distance_to_ball() < 300.0
    }

    pub fn far_from_ball(&self) -> bool {
        self.distance_to_ball() > 2000.0
    }

    pub fn close_to_own_goal(&self) -> bool {
        self.distance_to_position(self.get_own_goal_position()) < 4000.0
    }

    pub fn close_to_opp_goal(&self) -> bool {
        self.distance_to_position(self.get_opp_goal_position()) < 4000.0
    }

    pub fn get_own_goal_position(&self) -> Vector2 {
        self.world
            .field_geom
            .as_ref()
            .map(|f| Vector2::new(-f.field_length / 2.0, 0.0))
            .unwrap_or_else(|| Vector2::new(-4500.0, 0.0))
    }

    pub fn get_opp_goal_position(&self) -> Vector2 {
        self.world
            .field_geom
            .as_ref()
            .map(|f| Vector2::new(f.field_length / 2.0, 0.0))
            .unwrap_or_else(|| Vector2::new(4500.0, 0.0))
    }

    pub fn get_field_center(&self) -> Vector2 {
        Vector2::zeros()
    }

    pub fn is_position_in_center_circle(&self, pos: Vector2) -> bool {
        self.world
            .field_geom
            .as_ref()
            .map(|f| pos.norm() <= f.center_circle_radius)
            .unwrap_or(false)
    }

    pub fn distance_to_ball(&self) -> f64 {
        self.world
            .ball
            .as_ref()
            .map(|b| (self.player_data().position - b.position.xy()).norm())
            .unwrap_or(f64::INFINITY)
    }

    pub fn distance_to_position(&self, pos: Vector2) -> f64 {
        (self.player_data().position - pos).norm()
    }

    pub fn distance_to_player(&self, player_id: PlayerId) -> f64 {
        let my_pos = self.player_data().position;

        // Check own players first
        if let Some(player) = self.world.own_players.iter().find(|p| p.id == player_id) {
            return (my_pos - player.position).norm();
        }

        // Check opponent players
        if let Some(player) = self.world.opp_players.iter().find(|p| p.id == player_id) {
            return (my_pos - player.position).norm();
        }

        f64::INFINITY
    }

    pub fn am_closest_to_ball(&self) -> bool {
        let my_dist = self.distance_to_ball();
        self.world
            .own_players
            .iter()
            .filter(|p| p.id != self.player_id)
            .all(|p| {
                self.world
                    .ball
                    .as_ref()
                    .map(|b| (p.position - b.position.xy()).norm() > my_dist)
                    .unwrap_or(true)
            })
    }

    pub fn find_own_player_min_by(&self, key: impl Fn(&PlayerData) -> f64) -> Option<PlayerData> {
        self.world
            .own_players
            .iter()
            .filter(|p| p.id != self.player_id)
            .min_by(|p, q| key(p).partial_cmp(&key(q)).unwrap())
            .map(|p| p.clone())
    }

    pub fn find_opp_player_min_by(&self, key: impl Fn(&PlayerData) -> f64) -> Option<PlayerData> {
        self.world
            .opp_players
            .iter()
            .min_by(|p, q| key(p).partial_cmp(&key(q)).unwrap())
            .map(|p| p.clone())
    }

    pub fn get_closest_own_player_to_ball(&self) -> Option<PlayerData> {
        self.world.ball.as_ref().and_then(|ball| {
            let ball_pos = ball.position.xy();
            self.world
                .own_players
                .iter()
                .filter(|p| p.id != self.player_id)
                .min_by_key(|p| ((p.position - ball_pos).norm() * 1000.0) as i64)
                .map(|p| p.clone())
        })
    }

    pub fn distance_of_closest_own_player_to_ball(&self) -> f64 {
        self.get_closest_own_player_to_ball()
            .map(|p| (p.position - self.ball_position()).norm())
            .unwrap_or(f64::INFINITY)
    }

    pub fn get_closest_opp_player_to_ball(&self) -> Option<PlayerData> {
        self.world.ball.as_ref().and_then(|ball| {
            let ball_pos = ball.position.xy();
            self.world
                .opp_players
                .iter()
                .min_by_key(|p| ((p.position - ball_pos).norm() * 1000.0) as i64)
                .map(|p| p.clone())
        })
    }

    pub fn distance_of_closest_opp_player_to_ball(&self) -> f64 {
        self.get_closest_opp_player_to_ball()
            .map(|p| (p.position - self.ball_position()).norm())
            .unwrap_or(f64::INFINITY)
    }

    pub fn get_all_players_within_radius(&self, center: Vector2, radius: f64) -> Vec<PlayerData> {
        let mut players = Vec::new();

        for p in &self.world.own_players {
            if p.id == self.player_id {
                continue;
            }
            if (p.position - center).norm() < radius {
                players.push(p.clone());
            }
        }

        for p in &self.world.opp_players {
            if (p.position - center).norm() < radius {
                players.push(p.clone());
            }
        }

        players
    }

    pub fn get_all_players_within_radius_of_me(&self, radius: f64) -> Vec<PlayerData> {
        self.get_all_players_within_radius(self.player_data().position, radius)
    }

    pub fn get_own_players_within_radius(&self, center: Vector2, radius: f64) -> Vec<PlayerData> {
        self.world
            .own_players
            .iter()
            .filter(|p| p.id != self.player_id)
            .filter(|p| (p.position - center).norm() < radius)
            .map(|p| p.clone())
            .collect()
    }

    pub fn get_own_players_within_radius_of_me(&self, radius: f64) -> Vec<PlayerData> {
        self.get_own_players_within_radius(self.player_data().position, radius)
    }

    pub fn get_opp_players_within_radius(&self, center: Vector2, radius: f64) -> Vec<PlayerData> {
        self.world
            .opp_players
            .iter()
            .filter(|p| (p.position - center).norm() < radius)
            .map(|p| p.clone())
            .collect()
    }

    pub fn get_opp_players_within_radius_of_me(&self, radius: f64) -> Vec<PlayerData> {
        self.get_opp_players_within_radius(self.player_data().position, radius)
    }

    pub fn ball_in_penalty_area(&self) -> bool {
        self.ball_in_own_penalty_area() || self.ball_in_opp_penalty_area()
    }

    pub fn get_own_penalty_mark(&self) -> Vector2 {
        if let Some(field) = &self.world.field_geom {
            Vector2::new(
                -field.field_length / 2.0 + field.goal_line_to_penalty_mark,
                0.0,
            )
        } else {
            Vector2::new(-3500.0, 0.0) // Default
        }
    }

    pub fn get_opp_penalty_mark(&self) -> Vector2 {
        if let Some(field) = &self.world.field_geom {
            Vector2::new(
                field.field_length / 2.0 - field.goal_line_to_penalty_mark,
                0.0,
            )
        } else {
            Vector2::new(3500.0, 0.0) // Default
        }
    }

    pub fn ball_in_own_penalty_area(&self) -> bool {
        self.world
            .ball
            .as_ref()
            .zip(self.world.field_geom.as_ref())
            .map_or(false, |(ball, field)| {
                let ball_pos = ball.position.xy();
                let half_length = field.field_length / 2.0;
                let half_penalty_width = field.penalty_area_width / 2.0;

                ball_pos.x <= -half_length + field.penalty_area_depth
                    && ball_pos.y >= -half_penalty_width
                    && ball_pos.y <= half_penalty_width
            })
    }

    pub fn ball_in_opp_penalty_area(&self) -> bool {
        self.world
            .ball
            .as_ref()
            .zip(self.world.field_geom.as_ref())
            .map_or(false, |(ball, field)| {
                let ball_pos = ball.position.xy();
                let half_length = field.field_length / 2.0;
                let half_penalty_width = field.penalty_area_width / 2.0;

                ball_pos.x >= half_length - field.penalty_area_depth
                    && ball_pos.y >= -half_penalty_width
                    && ball_pos.y <= half_penalty_width
            })
    }

    pub fn is_passing_target(&self) -> bool {
        self.bt_context
            .passing_target
            .read()
            .unwrap()
            .as_ref()
            .map(|p| p.id == self.player_id)
            .unwrap_or(false)
    }

    pub fn accept_passing_target(&mut self) -> bool {
        if self.is_passing_target() {
            self.bt_context.take_passing_target(self.player_id);
            true
        } else {
            false
        }
    }

    pub fn set_passing_target(&mut self, player_id: PlayerId) {
        self.bt_context.set_passing_target(PassingTarget {
            id: player_id,
            position: self.world.get_player(player_id).position,
        });
    }
}
