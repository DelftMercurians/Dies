//! `RobotSituation` — the per-robot read-only query layer the behavior tree ticks
//! against, re-implemented over the IPC [`WorldSnapshot`] instead of the old
//! in-executor `TeamData`. All the geometric/query helpers the v0 roles rely on
//! live here.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use dies_core::{Angle, FieldGeometry, Vector2};
use dies_strategy_protocol::{
    BallState, GameState, Handicap, PlayerId, PlayerState, SkillStatus, WorldSnapshot,
};

/// Tri-state result of ticking a behavior node, matching the classic BT trio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BehaviorStatus {
    Success,
    Failure,
    #[default]
    Running,
}

/// Where a fetched ball should be sent.
#[derive(Clone, Debug)]
pub enum ShootTarget {
    /// Aim at a fixed position (e.g. a point in the opponent goal mouth, or a
    /// clearance target).
    Goal(Vector2),
    /// Aim at a teammate; `position` overrides their live position when set.
    Player {
        id: PlayerId,
        position: Option<Vector2>,
    },
}

/// Shared, cross-robot blackboard for the behavior trees: cooperative semaphores
/// (so only N robots run a guarded subtree) and a single passing-target slot.
/// Cloned cheaply (Arc-backed) into every `RobotSituation`.
#[derive(Clone, Default)]
pub struct BtContext {
    semaphores: Arc<RwLock<HashMap<String, HashSet<PlayerId>>>>,
    pub(crate) passing_target: Arc<RwLock<Option<PassingTarget>>>,
}

#[derive(Clone, Debug)]
pub struct PassingTarget {
    pub shooter_id: PlayerId,
    pub id: PlayerId,
    pub position: Vector2,
}

impl BtContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn try_acquire_semaphore(&self, id: &str, max_count: usize, player_id: PlayerId) -> bool {
        let mut semaphores = self.semaphores.write().unwrap();
        let entry = semaphores.entry(id.to_string()).or_default();
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

    /// Drop all semaphore claims. Called once per frame by the runtime so claims
    /// are rebuilt fresh as the trees re-tick (cross-frame stability comes from
    /// the committing-guard nodes, not from sticky semaphore membership).
    pub fn begin_frame(&self) {
        self.semaphores.write().unwrap().clear();
    }

    pub fn set_passing_target(&self, target: PassingTarget) {
        *self.passing_target.write().unwrap() = Some(target);
    }

    pub fn take_passing_target(&self, player_id: PlayerId) -> Option<PassingTarget> {
        let mut slot = self.passing_target.write().unwrap();
        if slot.as_ref().map(|p| p.id == player_id).unwrap_or(false) {
            slot.take()
        } else {
            None
        }
    }
}

/// Read-only view of the world from one robot's perspective, plus the shared BT
/// blackboard and this robot's live skill status (used by action nodes to detect
/// skill completion). All coordinates are team-relative (+x toward the opponent
/// goal), exactly as the IPC `World` provides them.
#[derive(Clone)]
pub struct RobotSituation {
    pub player_id: PlayerId,
    pub world: Arc<WorldSnapshot>,
    pub bt_context: BtContext,
    pub role_assignments: Arc<HashMap<PlayerId, String>>,
    /// Status of the skill this robot is currently running, as last reported by
    /// the executor. `Idle` during role assignment (no tree ticked yet).
    pub skill_status: SkillStatus,
}

impl RobotSituation {
    pub fn new(
        player_id: PlayerId,
        world: Arc<WorldSnapshot>,
        bt_context: BtContext,
        role_assignments: Arc<HashMap<PlayerId, String>>,
        skill_status: SkillStatus,
    ) -> Self {
        Self {
            player_id,
            world,
            bt_context,
            role_assignments,
            skill_status,
        }
    }

    pub fn player_id(&self) -> PlayerId {
        self.player_id
    }

    // ── Field geometry ──────────────────────────────────────────────────

    pub fn field(&self) -> FieldGeometry {
        self.world.field_geom.clone().unwrap_or_default()
    }

    pub fn field_length(&self) -> f64 {
        self.field().field_length
    }

    pub fn field_width(&self) -> f64 {
        self.field().field_width
    }

    pub fn half_field_length(&self) -> f64 {
        self.field_length() / 2.0
    }

    pub fn half_field_width(&self) -> f64 {
        self.field_width() / 2.0
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

    pub fn get_own_penalty_mark(&self) -> Vector2 {
        if let Some(field) = &self.world.field_geom {
            Vector2::new(-field.field_length / 2.0 + field.goal_line_to_penalty_mark, 0.0)
        } else {
            Vector2::new(-3500.0, 0.0)
        }
    }

    pub fn get_opp_penalty_mark(&self) -> Vector2 {
        if let Some(field) = &self.world.field_geom {
            Vector2::new(field.field_length / 2.0 - field.goal_line_to_penalty_mark, 0.0)
        } else {
            Vector2::new(3500.0, 0.0)
        }
    }

    pub fn constrain_to_field(&self, pos: Vector2) -> Vector2 {
        if let Some(field) = &self.world.field_geom {
            let half_length = field.boundary_width + field.field_length / 2.0;
            let half_width = field.boundary_width + field.field_width / 2.0;
            let x = pos.x.clamp(-half_length + 100.0, half_length - 100.0);
            let y = pos.y.clamp(-half_width + 100.0, half_width - 100.0);
            Vector2::new(x, y)
        } else {
            pos
        }
    }

    pub fn is_position_in_center_circle(&self, pos: Vector2) -> bool {
        self.world
            .field_geom
            .as_ref()
            .map(|f| pos.norm() <= f.center_circle_radius)
            .unwrap_or(false)
    }

    // ── Ball ────────────────────────────────────────────────────────────

    pub fn ball(&self) -> Option<&BallState> {
        self.world.ball.as_ref()
    }

    pub fn ball_position(&self) -> Vector2 {
        self.world.ball.as_ref().map(|b| b.position).unwrap_or_default()
    }

    pub fn ball_velocity(&self) -> Vector2 {
        self.world.ball.as_ref().map(|b| b.velocity).unwrap_or_default()
    }

    pub fn ball_speed(&self) -> f64 {
        self.ball_velocity().norm()
    }

    pub fn ball_in_our_half(&self) -> bool {
        self.world.ball.as_ref().map(|b| b.position.x < 0.0).unwrap_or(false)
    }

    pub fn ball_in_opponent_half(&self) -> bool {
        self.world.ball.as_ref().map(|b| b.position.x > 0.0).unwrap_or(false)
    }

    pub fn ball_in_own_penalty_area(&self) -> bool {
        self.world
            .ball
            .as_ref()
            .zip(self.world.field_geom.as_ref())
            .map_or(false, |(ball, field)| {
                let p = ball.position;
                let half_length = field.field_length / 2.0;
                let half_pw = field.penalty_area_width / 2.0;
                p.x <= -half_length + field.penalty_area_depth && p.y.abs() <= half_pw
            })
    }

    pub fn ball_in_opp_penalty_area(&self) -> bool {
        self.world
            .ball
            .as_ref()
            .zip(self.world.field_geom.as_ref())
            .map_or(false, |(ball, field)| {
                let p = ball.position;
                let half_length = field.field_length / 2.0;
                let half_pw = field.penalty_area_width / 2.0;
                p.x >= half_length - field.penalty_area_depth && p.y.abs() <= half_pw
            })
    }

    pub fn ball_in_penalty_area(&self) -> bool {
        self.ball_in_own_penalty_area() || self.ball_in_opp_penalty_area()
    }

    // ── Game state ──────────────────────────────────────────────────────

    pub fn game_state(&self) -> GameState {
        self.world.game_state
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

    pub fn we_are_attacking(&self) -> bool {
        self.world.us_operating
    }

    pub fn is_kickoff_state(&self) -> bool {
        matches!(self.game_state(), GameState::Kickoff | GameState::PrepareKickoff)
    }

    pub fn is_free_kick_state(&self) -> bool {
        self.game_state() == GameState::FreeKick
    }

    pub fn is_normal_play_state(&self) -> bool {
        self.game_state() == GameState::Run
    }

    pub fn is_stopped_state(&self) -> bool {
        matches!(self.game_state(), GameState::Stop | GameState::Halt)
    }

    /// Whether this robot is allowed to touch the ball (false only when it is the
    /// designated free-kick/kickoff kicker that already touched it, to avoid a
    /// double-touch).
    pub fn can_touch_ball(&self) -> bool {
        !matches!(self.world.freekick_kicker, Some(k) if k == self.player_id)
    }

    // ── This robot ──────────────────────────────────────────────────────

    pub fn player_data(&self) -> PlayerState {
        self.world
            .own_players
            .iter()
            .find(|p| p.id == self.player_id)
            .cloned()
            .unwrap_or_else(|| {
                PlayerState::new(self.player_id, Vector2::zeros(), Vector2::zeros(), Angle::from_radians(0.0))
            })
    }

    pub fn position(&self) -> Vector2 {
        self.player_data().position
    }

    pub fn heading(&self) -> Angle {
        self.player_data().heading
    }

    pub fn velocity(&self) -> Vector2 {
        self.player_data().velocity
    }

    pub fn has_ball(&self) -> bool {
        self.player_data().has_ball
    }

    pub fn has_handicap(&self, handicap: Handicap) -> bool {
        self.player_data().handicaps.contains(&handicap)
    }

    pub fn does_not_have_handicap(&self, handicap: Handicap) -> bool {
        !self.has_handicap(handicap)
    }

    pub fn has_any_handicap(&self, handicaps: &[Handicap]) -> bool {
        self.player_data().handicaps.iter().any(|h| handicaps.contains(h))
    }

    pub fn has_none_of_handicaps(&self, handicaps: &[Handicap]) -> bool {
        self.player_data().handicaps.iter().all(|h| !handicaps.contains(h))
    }

    pub fn player_id_hash(&self) -> f64 {
        (self.player_id.as_u32() as f64 * 0.618_033_988_749_894_9) % 1.0
    }

    // ── Roles ───────────────────────────────────────────────────────────

    pub fn current_role(&self) -> String {
        self.role_assignments.get(&self.player_id).cloned().unwrap_or_default()
    }

    pub fn current_role_is(&self, role: &str) -> bool {
        self.current_role().contains(role)
    }

    pub fn get_players_with_role(&self, role: &str) -> Vec<PlayerState> {
        self.world
            .own_players
            .iter()
            .filter(|p| self.role_assignments.get(&p.id).map(|r| r.contains(role)).unwrap_or(false))
            .cloned()
            .collect()
    }

    // ── Distances / nearest ─────────────────────────────────────────────

    pub fn distance_to_ball(&self) -> f64 {
        self.world
            .ball
            .as_ref()
            .map(|b| (self.position() - b.position).norm())
            .unwrap_or(f64::INFINITY)
    }

    pub fn distance_to_position(&self, pos: Vector2) -> f64 {
        (self.position() - pos).norm()
    }

    pub fn distance_to_player(&self, player_id: PlayerId) -> f64 {
        let my_pos = self.position();
        if let Some(p) = self.world.own_players.iter().find(|p| p.id == player_id) {
            return (my_pos - p.position).norm();
        }
        if let Some(p) = self.world.opp_players.iter().find(|p| p.id == player_id) {
            return (my_pos - p.position).norm();
        }
        f64::INFINITY
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

    pub fn am_closest_to_ball(&self) -> bool {
        let my_dist = self.distance_to_ball();
        let ball = self.ball_position();
        self.world
            .own_players
            .iter()
            .filter(|p| p.id != self.player_id)
            .all(|p| (p.position - ball).norm() > my_dist)
    }

    pub fn find_own_player_min_by(&self, key: impl Fn(&PlayerState) -> f64) -> Option<PlayerState> {
        self.world
            .own_players
            .iter()
            .filter(|p| p.id != self.player_id)
            .min_by(|p, q| key(p).partial_cmp(&key(q)).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
    }

    pub fn find_opp_player_min_by(&self, key: impl Fn(&PlayerState) -> f64) -> Option<PlayerState> {
        self.world
            .opp_players
            .iter()
            .min_by(|p, q| key(p).partial_cmp(&key(q)).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
    }

    pub fn get_closest_own_player_to_ball(&self) -> Option<PlayerState> {
        let ball = self.ball_position();
        self.world
            .own_players
            .iter()
            .filter(|p| p.id != self.player_id)
            .min_by(|a, b| {
                (a.position - ball)
                    .norm()
                    .partial_cmp(&(b.position - ball).norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    pub fn distance_of_closest_own_player_to_ball(&self) -> f64 {
        self.get_closest_own_player_to_ball()
            .map(|p| (p.position - self.ball_position()).norm())
            .unwrap_or(f64::INFINITY)
    }

    pub fn get_closest_opp_player_to_ball(&self) -> Option<PlayerState> {
        let ball = self.ball_position();
        self.world
            .opp_players
            .iter()
            .min_by(|a, b| {
                (a.position - ball)
                    .norm()
                    .partial_cmp(&(b.position - ball).norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    pub fn distance_of_closest_opp_player_to_ball(&self) -> f64 {
        self.get_closest_opp_player_to_ball()
            .map(|p| (p.position - self.ball_position()).norm())
            .unwrap_or(f64::INFINITY)
    }

    pub fn get_all_players_within_radius(&self, center: Vector2, radius: f64) -> Vec<PlayerState> {
        let mut out = Vec::new();
        for p in &self.world.own_players {
            if p.id != self.player_id && (p.position - center).norm() < radius {
                out.push(p.clone());
            }
        }
        for p in &self.world.opp_players {
            if (p.position - center).norm() < radius {
                out.push(p.clone());
            }
        }
        out
    }

    pub fn get_all_players_within_radius_of_me(&self, radius: f64) -> Vec<PlayerState> {
        self.get_all_players_within_radius(self.position(), radius)
    }

    pub fn get_own_players_within_radius(&self, center: Vector2, radius: f64) -> Vec<PlayerState> {
        self.world
            .own_players
            .iter()
            .filter(|p| p.id != self.player_id && (p.position - center).norm() < radius)
            .cloned()
            .collect()
    }

    pub fn get_own_players_within_radius_of_me(&self, radius: f64) -> Vec<PlayerState> {
        self.get_own_players_within_radius(self.position(), radius)
    }

    pub fn get_opp_players_within_radius(&self, center: Vector2, radius: f64) -> Vec<PlayerState> {
        self.world
            .opp_players
            .iter()
            .filter(|p| (p.position - center).norm() < radius)
            .cloned()
            .collect()
    }

    pub fn get_opp_players_within_radius_of_me(&self, radius: f64) -> Vec<PlayerState> {
        self.get_opp_players_within_radius(self.position(), radius)
    }

    // ── Passing target slot ─────────────────────────────────────────────

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
        if let Some(p) = self.world.own_players.iter().find(|p| p.id == player_id) {
            self.bt_context.set_passing_target(PassingTarget {
                shooter_id: self.player_id,
                id: player_id,
                position: p.position,
            });
        }
    }
}
