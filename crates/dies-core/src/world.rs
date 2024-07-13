use std::time::Instant;

use dodgy_2d::Obstacle;
use serde::Serialize;
use typeshare::typeshare;

use crate::{player::PlayerId, Angle, FieldGeometry, RoleType, SysStatus, Vector2, Vector3};

const STOP_BALL_AVOIDANCE_RADIUS: f64 = 500.0;

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

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub struct WorldUpdate {
    pub world_data: WorldData,
}

/// The game state, as reported by the referee.
#[derive(Serialize, Clone, Debug, PartialEq, Copy, Default)]
#[serde(tag = "type", content = "data")]
#[typeshare]
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
    /// Raw position as reported by vision
    pub raw_position: Vec<Vector3>,
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
    /// Unfiltered yaw as reported by vision
    pub raw_yaw: Angle,
    /// Angular speed of the player (in rad/s)
    pub angular_speed: f64,

    /// The overall status of the robot. Only available for own players.
    pub primary_status: Option<SysStatus>,
    /// The voltage of the kicker capacitor (in V). Only available for own players.
    pub kicker_cap_voltage: Option<f32>,
    /// The temperature of the kicker. Only available for own players.
    pub kicker_temp: Option<f32>,
    /// The voltages of the battery packs. Only available for own players.
    pub pack_voltages: Option<[f32; 2]>,
    /// Whether the breakbeam sensor detected a ball. Only available for own players.
    pub breakbeam_ball_detected: bool,
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
            primary_status: None,
            kicker_cap_voltage: None,
            kicker_temp: None,
            pack_voltages: None,
            breakbeam_ball_detected: false,
        }
    }
}

/// A struct to store the world state from a single frame.
#[derive(Serialize, Clone, Debug, Default)]
#[typeshare]
pub struct WorldData {
    /// Timestamp of the frame, in seconds. This timestamp is relative to the time the
    /// world tracking was started.
    pub t_received: f64,
    /// Recording timestamp of the frame, in seconds, as reported by vision. This
    /// timestamp is relative to the time the first image was captured.
    pub t_capture: f64,
    /// The time since the last frame was received, in seconds
    pub dt: f64,
    pub own_players: Vec<PlayerData>,
    pub opp_players: Vec<PlayerData>,
    pub ball: Option<BallData>,
    pub field_geom: Option<FieldGeometry>,
    pub current_game_state: GameStateData,
}

impl WorldData {
    pub fn get_player(&self, id: PlayerId) -> Option<&PlayerData> {
        self.own_players.iter().find(|p| p.id == id)
    }

    pub fn players_within_radius(&self, pos: Vector2, radius: f64) -> Vec<&PlayerData> {
        self.own_players
            .iter()
            .chain(self.opp_players.iter())
            .filter(|p| (p.position.xy() - pos).norm() < radius)
            .collect()
    }

    pub fn get_obstacles_for_player(&self, role: RoleType) -> Vec<Obstacle> {
        if let Some(field_geom) = &self.field_geom {
            let field_boundary = {
                let hl = field_geom.field_length as f32 / 2.0;
                let hw = field_geom.field_width as f32 / 2.0;
                Obstacle::Closed {
                    vertices: vec![
                        // Clockwise -> prevent leaving the field
                        dodgy_2d::Vec2::new(-hl, -hw),
                        dodgy_2d::Vec2::new(-hl, hw),
                        dodgy_2d::Vec2::new(hl, hw),
                        dodgy_2d::Vec2::new(hl, -hw),
                    ],
                }
            };
            let mut obstacles = vec![field_boundary];

            // // Add own defence area for non-keeper robots
            // if role != RoleType::Goalkeeper {
            //     let defence_area = create_bbox_from_rect(
            //         Vector2::new(
            //             -field_geom.field_length + field_geom.penalty_area_depth / 2.0,
            //             0.0,
            //         ),
            //         field_geom.penalty_area_depth,
            //         field_geom.penalty_area_width,
            //     );
            //     obstacles.push(defence_area);
            // }

            // // Add opponent defence area for all robots
            // let defence_area = create_bbox_from_rect(
            //     Vector2::new(
            //         field_geom.field_length - field_geom.penalty_area_depth / 2.0,
            //         0.0,
            //     ),
            //     field_geom.penalty_area_depth,
            //     field_geom.penalty_area_width,
            // );
            // obstacles.push(defence_area);

            match self.current_game_state.game_state {
                GameState::Stop => {
                    // Add obstacle to prevent getting close to the ball
                    if let Some(ball) = &self.ball {
                        obstacles.push(create_bbox_from_circle(
                            ball.position.xy(),
                            STOP_BALL_AVOIDANCE_RADIUS,
                        ));
                    }
                }
                GameState::Kickoff | GameState::PrepareKickoff => match role {
                    RoleType::KickoffKicker => {}
                    _ => {
                        // Add center circle for non kicker robots
                        obstacles.push(create_bbox_from_circle(
                            Vector2::zeros(),
                            field_geom.center_circle_radius,
                        ));
                    }
                },
                GameState::BallReplacement(_) => todo!(),
                GameState::PreparePenalty => todo!(),
                GameState::FreeKick => todo!(),
                GameState::Penalty => todo!(),
                GameState::PenaltyRun => todo!(),
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

fn create_bbox_from_circle(center: Vector2, radius: f64) -> Obstacle {
    let hw = radius as f32 / 2.0;
    let x = center.x as f32;
    let y = center.y as f32;
    Obstacle::Closed {
        vertices: vec![
            // Counter-clockwise -> prevent getting into the loop
            dodgy_2d::Vec2::new(x - hw, y - hw),
            dodgy_2d::Vec2::new(x + hw, y - hw),
            dodgy_2d::Vec2::new(x + hw, y + hw),
            dodgy_2d::Vec2::new(x - hw, y + hw),
        ],
    }
}

fn create_bbox_from_rect(center: Vector2, width: f64, height: f64) -> Obstacle {
    let hw = width as f32 / 2.0;
    let hh = height as f32 / 2.0;
    let x = center.x as f32;
    let y = center.y as f32;
    Obstacle::Closed {
        vertices: vec![
            dodgy_2d::Vec2::new(x - hw, y - hh),
            dodgy_2d::Vec2::new(x + hw, y - hh),
            dodgy_2d::Vec2::new(x + hw, y + hh),
            dodgy_2d::Vec2::new(x - hw, y + hh),
        ],
    }
}
