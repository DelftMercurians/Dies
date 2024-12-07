use std::{collections::HashSet, fmt::Display, hash::Hash, time::Instant};

use dies_core::{Angle, PlayerId, RoleType, SysStatus, Vector2, Vector3};
use serde::{Deserialize, Serialize};

use crate::FieldGeometry;

const STOP_BALL_AVOIDANCE_RADIUS: f64 = 800.0;
const PLAYER_RADIUS: f64 = 90.0;
const DRIBBLER_ANGLE_DEG: f64 = 56.0;

// Enum to represent different obstacle types
#[derive(Debug, Clone, Serialize)]
pub enum Obstacle {
    Circle { center: Vector2, radius: f64 },
    Rectangle { min: Vector2, max: Vector2 },
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

impl Display for GameState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            GameState::Unknown => "Unknown".to_string(),
            GameState::Halt => "Halt".to_string(),
            GameState::Timeout => "Timeout".to_string(),
            GameState::Stop => "Stop".to_string(),
            GameState::PrepareKickoff => "PrepareKickoff".to_string(),
            GameState::BallReplacement(_) => "BallReplacement".to_string(),
            GameState::PreparePenalty => "PreparePenalty".to_string(),
            GameState::Kickoff => "Kickoff".to_string(),
            GameState::FreeKick => "FreeKick".to_string(),
            GameState::Penalty => "Penalty".to_string(),
            GameState::PenaltyRun => "PenaltyRun".to_string(),
            GameState::Run => "Run".to_string(),
        };
        write!(f, "{}", str)
    }
}
impl Hash for GameState {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_string().hash(state);
    }
}

impl PartialEq for GameState {
    fn eq(&self, other: &Self) -> bool {
        self.to_string() == other.to_string()
    }
}

impl Eq for GameState {}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct GameStateData {
    /// The state of current game
    pub game_state: GameState,
    /// The team (if any) currently operating in asymmetric states
    pub operating_team: Option<Team>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Team {
    Blue,
    Yellow,
}

/// A struct to store the player state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PlayerData {
    /// Unix timestamp of the recorded frame from which this data was extracted
    pub timestamp: f64,
    /// The player's unique id
    pub id: PlayerId,
    /// Unfiltered position as reported by vision
    pub raw_position: Vector2,
    /// Position of the player filtered by us in mm
    pub position: Vector2,
    /// Velocity of the player in mm/s
    pub velocity: Vector2,
    /// Yaw of the player, in radians, (-pi, pi)
    pub yaw: Angle,
    /// Unfiltered yaw as reported by vision
    pub raw_yaw: Angle,
    /// Angular speed of the player (in rad/s)
    pub angular_speed: f64,

    /// Whether this player is controlled (receives feedback)
    pub is_controlled: bool,
    /// The overall status of the robot. Only available for controlled players.
    pub primary_status: Option<SysStatus>,
    /// The voltage of the kicker capacitor (in V). Only available for controlled players.
    pub kicker_cap_voltage: Option<f32>,
    /// The temperature of the kicker. Only available for controlled players.
    pub kicker_temp: Option<f32>,
    /// The voltages of the battery packs. Only available for controlled players.
    pub pack_voltages: Option<[f32; 2]>,
    /// Whether the breakbeam sensor detected a ball. Only available for controlled players.
    pub breakbeam_ball_detected: bool,
    pub imu_status: Option<SysStatus>,
    pub kicker_status: Option<SysStatus>,
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
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            is_controlled: false,
            primary_status: None,
            kicker_cap_voltage: None,
            kicker_temp: None,
            pack_voltages: None,
            imu_status: None,
            breakbeam_ball_detected: false,
            kicker_status: None,
        }
    }
}
/// A struct to store the ball state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BallData {
    /// Unix timestamp of the recorded frame from which this data was extracted (in
    /// seconds). This is the time that ssl-vision received the frame.
    pub timestamp: f64,
    /// Position of the ball filtered by us, in mm, in dies coordinates
    pub position: Vector3,
    /// Raw position as reported by vision
    pub raw_position: Vec<Vector3>,
    /// Velocity of the ball in mm/s, in dies coordinates
    pub velocity: Vector3,
    /// Whether the ball is being detected
    pub detected: bool,
}

pub enum BallPrediction {
    Linear(Vector2),
    Collision(Vector2),
}

/// A struct to store the world state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorldFrame {
    /// Timestamp of the frame, in seconds relative to tracking start
    pub t_received: f64,
    /// Recording timestamp of the frame from vision, in seconds
    pub t_capture: f64,
    /// The time since the last frame was received, in seconds
    pub dt: f64,
    pub blue_team: Vec<PlayerData>,
    pub yellow_team: Vec<PlayerData>,
    pub ball: Option<BallData>,
    pub field_geom: Option<FieldGeometry>,
    pub current_game_state: GameStateData,
}

impl WorldFrame {
    pub fn get_obstacles_for_player(&self, role: RoleType) -> Vec<Obstacle> {
        if let Some(field_geom) = &self.field_geom {
            let field_boundary = {
                let hl = field_geom.field_length / 2.0;
                let hw = field_geom.field_width / 2.0;
                Obstacle::Rectangle {
                    min: Vector2::new(
                        -hl - field_geom.boundary_width,
                        -hw - field_geom.boundary_width,
                    ),
                    max: Vector2::new(
                        hl + field_geom.boundary_width,
                        hw + field_geom.boundary_width,
                    ),
                }
            };
            let mut obstacles = vec![field_boundary];

            // Add own defence area for non-keeper robots
            if role != RoleType::Goalkeeper {
                let lower = Vector2::new(-10_000.0, -field_geom.penalty_area_width / 2.0);
                let upper = Vector2::new(
                    -field_geom.field_length / 2.0 + field_geom.penalty_area_depth + 50.0,
                    field_geom.penalty_area_width / 2.0,
                );

                let defence_area = Obstacle::Rectangle {
                    min: lower,
                    max: upper,
                };
                obstacles.push(defence_area);
            }

            // Add opponent defence area for all robots
            let lower = Vector2::new(
                field_geom.field_length / 2.0 - field_geom.penalty_area_depth - 50.0,
                -field_geom.penalty_area_width / 2.0,
            );
            let upper = Vector2::new(10_0000.0, field_geom.penalty_area_width / 2.0);
            let defence_area = Obstacle::Rectangle {
                min: lower,
                max: upper,
            };
            obstacles.push(defence_area);

            match self.current_game_state.game_state {
                GameState::Stop => {
                    // Add obstacle to prevent getting close to the ball
                    if let Some(ball) = &self.ball {
                        obstacles.push(Obstacle::Circle {
                            center: ball.position.xy(),
                            radius: STOP_BALL_AVOIDANCE_RADIUS,
                        });
                    }
                }
                GameState::Kickoff | GameState::PrepareKickoff => match role {
                    RoleType::KickoffKicker => {}
                    _ => {
                        // Add center circle for non kicker robots
                        obstacles.push(Obstacle::Circle {
                            center: Vector2::zeros(),
                            radius: field_geom.center_circle_radius,
                        });
                    }
                },
                GameState::BallReplacement(_) => {}
                GameState::PreparePenalty => {}
                GameState::FreeKick => {}
                GameState::Penalty => {}
                GameState::PenaltyRun => {}
                GameState::Run | GameState::Halt | GameState::Timeout | GameState::Unknown => {
                    // Nothing to do
                }
            };

            obstacles
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone)]
pub enum Avoid {
    Line { start: Vector2, end: Vector2 },
    Circle { center: Vector2 },
}

impl Avoid {
    fn distance_to(&self, pos: Vector2) -> f64 {
        match self {
            Avoid::Line { start, end } => distance_to_line(*start, *end, pos),
            Avoid::Circle { center } => (center - pos).norm(),
        }
    }
}

fn distance_to_line(start: Vector2, end: Vector2, pos: Vector2) -> f64 {
    let line = end - start;
    let line_norm = line.norm();
    let line_dir = line / line_norm;
    let pos_dir = pos - start;
    let proj = pos_dir.dot(&line_dir);
    if proj < 0.0 {
        (pos - start).norm()
    } else if proj > line_norm {
        (pos - end).norm()
    } else {
        let proj_vec = line_dir * proj;
        (pos_dir - proj_vec).norm()
    }
}

pub fn nearest_safe_pos(
    avoding_point: Avoid,
    min_distance: f64,
    initial_pos: Vector2,
    target_pos: Vector2,
    max_radius: i32,
    field: &FieldGeometry,
) -> Vector2 {
    let mut best_pos = Vector2::new(f64::INFINITY, f64::INFINITY);
    let mut found_better = false;
    let min_theta = 0;
    let max_theta = 360;
    let mut i = 0;
    for theta in (min_theta..max_theta).step_by(10) {
        let theta = Angle::from_degrees(theta as f64);
        for radius in (0..max_radius).step_by(50) {
            let position = initial_pos + theta.to_vector() * (radius as f64);
            if is_pos_in_field(position, field)
                && avoding_point.distance_to(position) > min_distance
            {
                if (position - target_pos).norm() < (best_pos - target_pos).norm() {
                    // crate::debug_cross(format!("{i}"), position, crate::DebugColor::Green);
                    best_pos = position;
                    found_better = true;
                }
            } else {
                // crate::debug_cross(format!("{i}"), position, crate::DebugColor::Red);
            }
            i += 1;
        }
    }
    if !found_better {
        log::warn!("Could not find a safe position from {initial_pos}, avoiding {avoding_point:?}");
    }

    best_pos
}

pub fn is_pos_in_field(pos: Vector2, field: &FieldGeometry) -> bool {
    const MARGIN: f64 = 100.0;
    // check if pos outside field
    if pos.x.abs() > field.field_length / 2.0 - MARGIN
        || pos.y.abs() > field.field_width / 2.0 - MARGIN
    {
        return false;
    }

    true
}

pub fn mock_world_data() -> WorldFrame {
    WorldFrame {
        blue_team: vec![PlayerData {
            id: PlayerId::new(0),
            position: Vector2::new(1000.0, 1000.0),
            timestamp: 0.0,
            raw_position: Vector2::new(1000.0, 1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            is_controlled: true,
            primary_status: Some(SysStatus::Ready),
            kicker_cap_voltage: Some(0.0),
            kicker_temp: Some(0.0),
            pack_voltages: Some([0.0, 0.0]),
            breakbeam_ball_detected: false,
            imu_status: Some(SysStatus::Ready),
            kicker_status: Some(SysStatus::Standby),
        }],
        yellow_team: vec![PlayerData {
            id: PlayerId::new(1),
            position: Vector2::new(-1000.0, -1000.0),
            timestamp: 0.0,
            raw_position: Vector2::new(-1000.0, -1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            is_controlled: false,
            primary_status: None,
            kicker_cap_voltage: None,
            kicker_temp: None,
            pack_voltages: None,
            breakbeam_ball_detected: false,
            imu_status: None,
            kicker_status: None,
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
        current_game_state: GameStateData {
            game_state: GameState::Run,
            operating_team: None,
        },
    }
}
