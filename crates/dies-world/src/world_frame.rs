use std::{fmt::Display, hash::Hash, time::Instant};

use dies_core::{Angle, PlayerId, SysStatus, Vector2, Vector3};
use serde::{Deserialize, Serialize};

use crate::FieldGeometry;

/// A struct to store the world state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorldFrame {
    /// Timestamp of the frame, in seconds relative to tracking start
    pub t_received: f64,
    /// Recording timestamp of the frame from vision, in seconds
    pub t_capture: f64,
    /// The time since the last frame was received, in seconds
    pub dt: f64,
    pub blue_team: Vec<PlayerFrame>,
    pub yellow_team: Vec<PlayerFrame>,
    pub ball: Option<BallFrame>,
    pub field_geom: Option<FieldGeometry>,
    pub current_game_state: GameState,
    pub side_assignment: SideAssignment,
}

impl WorldFrame {
    pub fn get_team(&self, team: Team) -> &Vec<PlayerFrame> {
        match team {
            Team::Blue => &self.blue_team,
            Team::Yellow => &self.yellow_team,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Team {
    Blue,
    Yellow,
}

impl Team {
    pub fn opponent(&self) -> Team {
        match self {
            Team::Blue => Team::Yellow,
            Team::Yellow => Team::Blue,
        }
    }
}

/// The side the two teams are assigned to.
///
/// `BluePositive` means that the blue team's goal is the positive side of the field.
/// `YellowPositive` means that the yellow team's goal is the positive side of the field.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideAssignment {
    BluePositive,
    YellowPositive,
}

impl SideAssignment {
    /// Get the sign of the x-coordinate for the blue team's goal's side.
    pub fn blue_goal_side(&self) -> f64 {
        match self {
            SideAssignment::BluePositive => 1.0,
            SideAssignment::YellowPositive => -1.0,
        }
    }

    /// Get the sign of the x-coordinate for the yellow team's goal's side.
    pub fn yellow_goal_side(&self) -> f64 {
        match self {
            SideAssignment::BluePositive => -1.0,
            SideAssignment::YellowPositive => 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum WorldInstant {
    Real(Instant),
    Simulated(f64),
}

impl WorldInstant {
    pub fn now_real() -> Self {
        WorldInstant::Real(Instant::now())
    }

    pub fn simulated(t: f64) -> Self {
        WorldInstant::Simulated(t)
    }

    pub fn duration_since(&self, earlier: &Self) -> f64 {
        match (earlier, self) {
            (WorldInstant::Real(earlier), WorldInstant::Real(later)) => {
                later.duration_since(*earlier).as_secs_f64()
            }
            (WorldInstant::Simulated(earlier), WorldInstant::Simulated(later)) => later - earlier,
            _ => panic!("Cannot compare real and simulated instants"),
        }
    }
}

/// The game state, as reported by the referee.
#[derive(Serialize, Deserialize, Clone, Debug, Copy, Default)]
#[serde(tag = "type", content = "data")]
pub enum GameStateType {
    #[default]
    Unknown,
    Halt,
    Timeout,
    Stop,
    PrepareKickoff,
    BallPlacement(Vector2),
    PreparePenalty,
    Kickoff,
    FreeKick,
    Penalty,
    PenaltyRun,
    Run,
}

impl Display for GameStateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            GameStateType::Unknown => "Unknown".to_string(),
            GameStateType::Halt => "Halt".to_string(),
            GameStateType::Timeout => "Timeout".to_string(),
            GameStateType::Stop => "Stop".to_string(),
            GameStateType::PrepareKickoff => "PrepareKickoff".to_string(),
            GameStateType::BallPlacement(_) => "BallPlacement".to_string(),
            GameStateType::PreparePenalty => "PreparePenalty".to_string(),
            GameStateType::Kickoff => "Kickoff".to_string(),
            GameStateType::FreeKick => "FreeKick".to_string(),
            GameStateType::Penalty => "Penalty".to_string(),
            GameStateType::PenaltyRun => "PenaltyRun".to_string(),
            GameStateType::Run => "Run".to_string(),
        };
        write!(f, "{}", str)
    }
}
impl Hash for GameStateType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_string().hash(state);
    }
}

impl PartialEq for GameStateType {
    fn eq(&self, other: &Self) -> bool {
        self.to_string() == other.to_string()
    }
}

impl Eq for GameStateType {}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct GameState {
    /// The state of current game
    pub game_state: GameStateType,
    /// The team (if any) currently operating in asymmetric states
    pub operating_team: Option<Team>,
}

/// A struct to store the player state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PlayerFrame {
    /// The player's unique id
    pub id: PlayerId,
    /// Position of the player filtered by us in mm
    pub position: Vector2,
    /// Velocity of the player in mm/s
    pub velocity: Vector2,
    /// Yaw of the player, in radians, (-pi, pi)
    pub yaw: Angle,
    /// Angular speed of the player (in rad/s)
    pub angular_speed: f64,
    /// Feedback from the player, if it is controlled
    pub feedback: PlayerFeedback,
}

impl PlayerFrame {
    pub fn new(id: PlayerId) -> Self {
        Self {
            id,
            position: Vector2::zeros(),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            angular_speed: 0.0,
            feedback: PlayerFeedback::NotControlled,
        }
    }
}

/// The feedback that the player receives from the world, if it is controlled.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PlayerFeedback {
    NotControlled,
    NotReceived,
    Controlled {
        /// The overall status of the robot.
        primary_status: Option<SysStatus>,
        /// The voltage of the kicker capacitor (in V).
        kicker_cap_voltage: Option<f32>,
        /// The temperature of the kicker.
        kicker_temp: Option<f32>,
        /// The voltages of the battery packs.
        pack_voltages: Option<[f32; 2]>,
        /// Whether the breakbeam sensor detected a ball.
        breakbeam_ball_detected: bool,
        /// The status of the IMU.
        imu_status: Option<SysStatus>,
        /// The status of the kicker.
        kicker_status: Option<SysStatus>,
    },
}

/// A struct to store the ball state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BallFrame {
    /// Position of the ball filtered by us, in mm, in dies coordinates
    pub position: Vector3,
    /// Velocity of the ball in mm/s, in dies coordinates
    pub velocity: Vector3,
    /// Whether the ball is being detected
    pub detected: bool,
}

pub fn mock_world_frame() -> WorldFrame {
    WorldFrame {
        blue_team: vec![PlayerFrame {
            id: PlayerId::new(0),
            position: Vector2::new(1000.0, 1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            angular_speed: 0.0,
            feedback: PlayerFeedback::NotControlled,
        }],
        yellow_team: vec![PlayerFrame {
            id: PlayerId::new(1),
            position: Vector2::new(-1000.0, -1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            angular_speed: 0.0,
            feedback: PlayerFeedback::NotControlled,
        }],
        field_geom: Some(FieldGeometry {
            field_length: 9000.0,
            field_width: 6000.0,
            ..Default::default()
        }),
        t_received: 0.0,
        t_capture: 0.0,
        dt: 1.0,
        ball: None,
        current_game_state: GameState {
            game_state: GameStateType::Run,
            operating_team: None,
        },
        side_assignment: SideAssignment::BluePositive,
    }
}
