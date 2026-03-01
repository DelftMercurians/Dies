//! World state types for strategy communication.
//!
//! These types represent the world state as seen by a strategy. All coordinates
//! are in the team-relative reference frame where +x points toward the opponent's goal.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::{Angle, FieldGeometry, PlayerId, Vector2, Vector3};

/// Complete world state snapshot sent to strategies each frame.
///
/// All coordinates are in the normalized team-relative frame:
/// - **+x**: Toward opponent's goal (attacking direction)
/// - **-x**: Toward our own goal (defending direction)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldSnapshot {
    /// Unix timestamp of the frame, in seconds.
    pub timestamp: f64,

    /// Time since the last frame was received, in seconds.
    pub dt: f64,

    /// Field geometry (dimensions, penalty areas, etc.).
    pub field_geom: Option<FieldGeometry>,

    /// Ball state, if the ball is detected.
    pub ball: Option<BallState>,

    /// Our team's players.
    pub own_players: Vec<PlayerState>,

    /// Opponent team's players.
    pub opp_players: Vec<PlayerState>,

    /// Current game state.
    pub game_state: GameState,

    /// Whether it's our team's turn to act (e.g., we have a free kick).
    pub us_operating: bool,

    /// Our goalkeeper's player ID, if designated.
    pub our_keeper_id: Option<PlayerId>,

    /// The player who performed a free kick or kickoff (for double-touch tracking).
    /// Only `Some` until another player touches the ball.
    pub freekick_kicker: Option<PlayerId>,
}

/// Ball state in the world.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BallState {
    /// Ball position in mm (x, y only; z is rarely used).
    pub position: Vector2,

    /// Ball velocity in mm/s.
    pub velocity: Vector2,

    /// Whether the ball is currently being detected by the vision system.
    pub detected: bool,
}

impl BallState {
    /// Create a new ball state.
    pub fn new(position: Vector2, velocity: Vector2, detected: bool) -> Self {
        Self {
            position,
            velocity,
            detected,
        }
    }

    /// Create a ball state from a 3D position and velocity.
    pub fn from_3d(position: Vector3, velocity: Vector3, detected: bool) -> Self {
        Self {
            position: Vector2::new(position.x, position.y),
            velocity: Vector2::new(velocity.x, velocity.y),
            detected,
        }
    }
}

/// A handicap that limits a robot's capabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Handicap {
    /// Robot's kicker is not working.
    NoKicker,
    /// Robot's dribbler is not working.
    NoDribbler,
    /// Robot's breakbeam sensor is not working.
    NoBreakbeam,
}

impl From<dies_core::Handicap> for Handicap {
    fn from(h: dies_core::Handicap) -> Self {
        match h {
            dies_core::Handicap::NoKicker => Handicap::NoKicker,
            dies_core::Handicap::NoDribbler => Handicap::NoDribbler,
            dies_core::Handicap::NoBreakbeam => Handicap::NoBreakbeam,
        }
    }
}

/// Player (robot) state in the world.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlayerState {
    /// Player's unique ID.
    pub id: PlayerId,

    /// Position in mm.
    pub position: Vector2,

    /// Velocity in mm/s.
    pub velocity: Vector2,

    /// Heading (yaw) angle.
    pub heading: Angle,

    /// Angular velocity in rad/s.
    pub angular_velocity: f64,

    /// Whether the breakbeam sensor detects a ball (robot has the ball).
    ///
    /// Only available for own players; always `false` for opponent players.
    pub has_ball: bool,

    /// Set of handicaps affecting this robot.
    pub handicaps: HashSet<Handicap>,
}

impl PlayerState {
    /// Create a new player state with minimal data.
    pub fn new(id: PlayerId, position: Vector2, velocity: Vector2, heading: Angle) -> Self {
        Self {
            id,
            position,
            velocity,
            heading,
            angular_velocity: 0.0,
            has_ball: false,
            handicaps: HashSet::new(),
        }
    }

    /// Builder method to set angular velocity.
    pub fn with_angular_velocity(mut self, angular_velocity: f64) -> Self {
        self.angular_velocity = angular_velocity;
        self
    }

    /// Builder method to set ball possession.
    pub fn with_has_ball(mut self, has_ball: bool) -> Self {
        self.has_ball = has_ball;
        self
    }

    /// Builder method to add a handicap.
    pub fn with_handicap(mut self, handicap: Handicap) -> Self {
        self.handicaps.insert(handicap);
        self
    }

    /// Builder method to set all handicaps.
    pub fn with_handicaps(mut self, handicaps: HashSet<Handicap>) -> Self {
        self.handicaps = handicaps;
        self
    }
}

/// Game state as reported by the game controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GameState {
    /// Unknown or uninitialized state.
    #[default]
    Unknown,
    /// All robots must stop immediately.
    Halt,
    /// Timeout.
    Timeout,
    /// Stop - robots must stay 500mm from ball.
    Stop,
    /// Preparing for kickoff.
    PrepareKickoff,
    /// Ball placement in progress.
    BallPlacement,
    /// Preparing for penalty kick.
    PreparePenalty,
    /// Kickoff is happening (ball in play but must be touched by kicker first).
    Kickoff,
    /// Free kick in progress.
    FreeKick,
    /// Penalty kick setup (before the kick).
    Penalty,
    /// Penalty kick in progress.
    PenaltyRun,
    /// Normal play - ball is in play.
    Run,
}

impl GameState {
    /// Returns `true` if the ball is in play (can be contested).
    pub fn is_ball_in_play(&self) -> bool {
        matches!(
            self,
            GameState::Kickoff | GameState::PenaltyRun | GameState::Run | GameState::FreeKick
        )
    }
}

impl From<dies_core::GameState> for GameState {
    fn from(gs: dies_core::GameState) -> Self {
        match gs {
            dies_core::GameState::Unknown => GameState::Unknown,
            dies_core::GameState::Halt => GameState::Halt,
            dies_core::GameState::Timeout => GameState::Timeout,
            dies_core::GameState::Stop => GameState::Stop,
            dies_core::GameState::PrepareKickoff => GameState::PrepareKickoff,
            dies_core::GameState::BallReplacement(_) => GameState::BallPlacement,
            dies_core::GameState::PreparePenalty => GameState::PreparePenalty,
            dies_core::GameState::Kickoff => GameState::Kickoff,
            dies_core::GameState::FreeKick => GameState::FreeKick,
            dies_core::GameState::Penalty => GameState::Penalty,
            dies_core::GameState::PenaltyRun => GameState::PenaltyRun,
            dies_core::GameState::Run => GameState::Run,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_snapshot_serialization() {
        let snapshot = WorldSnapshot {
            timestamp: 1234567890.123,
            dt: 0.016,
            field_geom: None,
            ball: Some(BallState::new(
                Vector2::new(100.0, 200.0),
                Vector2::new(50.0, -30.0),
                true,
            )),
            own_players: vec![PlayerState::new(
                PlayerId::new(1),
                Vector2::new(1000.0, 500.0),
                Vector2::new(0.0, 0.0),
                Angle::from_radians(0.0),
            )
            .with_has_ball(true)],
            opp_players: vec![PlayerState::new(
                PlayerId::new(2),
                Vector2::new(-1000.0, -500.0),
                Vector2::new(100.0, 0.0),
                Angle::from_radians(std::f64::consts::PI),
            )],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: Some(PlayerId::new(0)),
            freekick_kicker: None,
        };

        let encoded = bincode::serialize(&snapshot).unwrap();
        let decoded: WorldSnapshot = bincode::deserialize(&encoded).unwrap();

        assert!((decoded.timestamp - 1234567890.123).abs() < 1e-6);
        assert!(decoded.ball.is_some());
        assert_eq!(decoded.own_players.len(), 1);
        assert_eq!(decoded.opp_players.len(), 1);
        assert_eq!(decoded.game_state, GameState::Run);
        assert!(decoded.us_operating);
        assert_eq!(decoded.our_keeper_id, Some(PlayerId::new(0)));
    }

    #[test]
    fn test_ball_state_serialization() {
        let ball = BallState::new(
            Vector2::new(100.0, -200.0),
            Vector2::new(500.0, 300.0),
            true,
        );

        let encoded = bincode::serialize(&ball).unwrap();
        let decoded: BallState = bincode::deserialize(&encoded).unwrap();

        assert!((decoded.position.x - 100.0).abs() < 1e-6);
        assert!((decoded.position.y - (-200.0)).abs() < 1e-6);
        assert!((decoded.velocity.x - 500.0).abs() < 1e-6);
        assert!((decoded.velocity.y - 300.0).abs() < 1e-6);
        assert!(decoded.detected);
    }

    #[test]
    fn test_player_state_with_handicaps() {
        let mut handicaps = HashSet::new();
        handicaps.insert(Handicap::NoKicker);
        handicaps.insert(Handicap::NoDribbler);

        let player = PlayerState::new(
            PlayerId::new(5),
            Vector2::new(0.0, 0.0),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        )
        .with_handicaps(handicaps);

        let encoded = bincode::serialize(&player).unwrap();
        let decoded: PlayerState = bincode::deserialize(&encoded).unwrap();

        assert_eq!(decoded.id, PlayerId::new(5));
        assert!(decoded.handicaps.contains(&Handicap::NoKicker));
        assert!(decoded.handicaps.contains(&Handicap::NoDribbler));
        assert!(!decoded.handicaps.contains(&Handicap::NoBreakbeam));
    }

    #[test]
    fn test_game_state_is_ball_in_play() {
        assert!(GameState::Run.is_ball_in_play());
        assert!(GameState::Kickoff.is_ball_in_play());
        assert!(GameState::FreeKick.is_ball_in_play());
        assert!(GameState::PenaltyRun.is_ball_in_play());

        assert!(!GameState::Halt.is_ball_in_play());
        assert!(!GameState::Stop.is_ball_in_play());
        assert!(!GameState::PrepareKickoff.is_ball_in_play());
        assert!(!GameState::PreparePenalty.is_ball_in_play());
    }
}
