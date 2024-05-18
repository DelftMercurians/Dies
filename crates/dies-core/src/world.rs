use serde::Serialize;

use crate::{FieldGeometry, PlayerId, Vector2, Vector3};

/// The game state, as reported by the referee.
#[derive(Serialize, Clone, Debug, PartialEq, Copy, Default)]
pub enum GameState {
    #[default]
    Unknown,
    Halt,
    Timeout,
    Stop,
    PrepareKickoff,
    BallReplacement(Vector2),
    PreparePenalty,
    Kickoff,
    FreeKick,
    Penalty,
    PenaltyRun,
    Run,
}

/// A struct to store the ball state from a single frame.
#[derive(Serialize, Clone, Debug)]
pub struct BallData {
    /// Unix timestamp of the recorded frame from which this data was extracted (in
    /// seconds). This is the time that ssl-vision received the frame.
    pub timestamp: f64,
    /// Position of the ball filtered by us, in mm, in dies coordinates
    pub position: Vector3,
    /// Velocity of the ball in mm/s, in dies coordinates
    pub velocity: Vector3,
}

#[derive(Serialize, Clone, Debug, Default)]
pub struct GameStateData {
    /// The state of current game
    pub game_state: GameState,
    /// If we are the main party currently performing tasks in the state.
    /// true for symmetric states(halt stop run timout)
    pub us_operating: bool,
}

/// A struct to store the player state from a single frame.
#[derive(Serialize, Clone, Debug)]
pub struct PlayerData {
    /// Unix timestamp of the recorded frame from which this data was extracted (in
    /// seconds). This is the time that ssl-vision received the frame.
    pub timestamp: f64,
    /// The player's unique id
    pub id: PlayerId,
    /// Unfiltered position as reported by vision
    pub raw_position: Vector2,
    /// Position of the player filtered by us in mm, in dies coordinates
    pub position: Vector2,
    /// Velocity of the player in mm/s, in dies coordinates
    pub velocity: Vector2,
    /// Orientation of the player, in radians, (`-pi`, `pi`), where `0` is the positive
    /// x direction, and `pi/2` is the positive y direction.
    pub orientation: f64,
    /// Angular speed of the player (in rad/s)
    pub angular_speed: f64,
}

/// A struct to store the world state from a single frame.
#[derive(Serialize, Clone, Debug, Default)]
pub struct WorldData {
    pub own_players: Vec<PlayerData>,
    pub opp_players: Vec<PlayerData>,
    pub ball: Option<BallData>,
    pub field_geom: Option<FieldGeometry>,
    pub current_game_state: GameStateData,
}
