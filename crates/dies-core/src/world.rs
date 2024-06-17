use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::{player::PlayerId, Angle, FieldGeometry, Vector2, Vector3};

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub struct WorldUpdate {
    pub world_data: WorldData,
}

/// The game state, as reported by the referee.
#[derive(Serialize, Clone, Debug, PartialEq, Copy, Default)]
#[typeshare]
#[serde(tag = "type", content = "data")]
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
#[typeshare]
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
#[typeshare]
pub struct GameStateData {
    /// The state of current game
    pub game_state: GameState,
    /// If we are the main party currently performing tasks in the state.
    /// true for symmetric states(halt stop run timout)
    pub us_operating: bool,
}

/// A struct to store the player state from a single frame.
#[derive(Serialize, Clone, Debug)]
#[typeshare]
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
    /// Yaw of the player, in radians, (`-pi`, `pi`), where `0` is the positive
    /// x direction, and `pi/2` is the positive y direction.
    pub yaw: Angle,
    /// Angular speed of the player (in rad/s)
    pub angular_speed: f64,
}

impl PlayerData {
    pub fn new(id: PlayerId) -> Self {
        Self {
            timestamp: 0.0,
            id,
            raw_position: Vector2::zeros(),
            position: Vector2::zeros(),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            angular_speed: 0.0,
        }
    }
}

/// A struct to store the world state from a single frame.
#[derive(Serialize, Clone, Debug, Default)]
#[typeshare]
pub struct WorldData {
    pub own_players: Vec<PlayerData>,
    pub opp_players: Vec<PlayerData>,
    pub ball: Option<BallData>,
    pub field_geom: Option<FieldGeometry>,
    pub current_game_state: GameStateData,
    // duration between the last two data generations
    pub duration: f64,
}
