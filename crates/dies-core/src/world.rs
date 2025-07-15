use std::{collections::HashSet, fmt::Display, hash::Hash, time::Instant};

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::{
    distance_to_line, player::PlayerId, Angle, FieldGeometry, Handicap, RoleType, SideAssignment,
    SysStatus, TeamColor, TeamPlayerId, Vector2, Vector3,
};

const STOP_BALL_AVOIDANCE_RADIUS: f64 = 800.0;
pub const PLAYER_RADIUS: f64 = 90.0;
pub const BALL_RADIUS: f64 = 21.5;
const MAX_SPEED: f64 = 10000.0;
const MAX_ACCELERATION: f64 = 125000.0;

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

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub struct WorldUpdate {
    pub world_data: WorldData,
}

/// The game state, as reported by the referee.
#[derive(Serialize, Deserialize, Clone, Debug, Copy, Default)]
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

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum StrategyGameStateMacther {
    #[default]
    Any,
    Specific(GameState),
    AnyOf(HashSet<GameState>),
}

impl StrategyGameStateMacther {
    pub fn matches(&self, state: &GameState) -> bool {
        match self {
            StrategyGameStateMacther::Any => true,
            StrategyGameStateMacther::Specific(s) => s == state,
            StrategyGameStateMacther::AnyOf(states) => states.contains(state),
        }
    }

    pub fn any() -> Self {
        StrategyGameStateMacther::Any
    }

    pub fn specific(state: GameState) -> Self {
        StrategyGameStateMacther::Specific(state)
    }

    pub fn any_of(states: &[GameState]) -> Self {
        StrategyGameStateMacther::AnyOf(states.iter().cloned().collect())
    }
}

/// A struct to store the ball state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
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
    /// Whether the ball is being detected
    pub detected: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[typeshare]
pub struct GameStateData {
    /// The state of current game
    pub game_state: GameState,
    /// If we are the main party currently performing tasks in the state.
    /// true for symmetric states(halt stop run timout)
    pub us_operating: bool,
    #[typeshare(serialized_as = "u32")]
    pub yellow_cards: usize,
    /// The player who performed the freekick, if it was us (for double touch tracking)
    /// Only Some until another player touches the ball
    pub freekick_kicker: Option<PlayerId>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct RawGameStateData {
    /// The state of current game
    pub game_state: GameState,
    /// The team that is currently performing tasks in the state.
    pub operating_team: TeamColor,
    /// The player who performed the freekick (for double touch tracking)
    pub freekick_kicker: Option<TeamPlayerId>,
    #[typeshare(serialized_as = "u32")]
    pub blue_team_yellow_cards: usize,
    #[typeshare(serialized_as = "u32")]
    pub yellow_team_yellow_cards: usize,
}

/// A struct to store the player state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
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
    pub imu_status: Option<SysStatus>,
    pub kicker_status: Option<SysStatus>,

    pub handicaps: HashSet<Handicap>,
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
            imu_status: None,
            breakbeam_ball_detected: false,
            kicker_status: None,
            handicaps: HashSet::new(),
        }
    }
}

pub enum BallPrediction {
    Linear(Vector2),
    Collision(Vector2),
}

#[derive(Serialize, Deserialize, Clone, Debug)]
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
    pub blue_team: Vec<PlayerData>,
    pub yellow_team: Vec<PlayerData>,
    pub ball: Option<BallData>,
    pub field_geom: Option<FieldGeometry>,
    pub game_state: RawGameStateData,
    pub side_assignment: SideAssignment,
}

impl WorldData {
    pub fn get_team_players(&self, color: TeamColor) -> &Vec<PlayerData> {
        match color {
            TeamColor::Blue => &self.blue_team,
            TeamColor::Yellow => &self.yellow_team,
        }
    }

    pub fn get_opp_team_players(&self, color: TeamColor) -> &Vec<PlayerData> {
        match color {
            TeamColor::Blue => &self.yellow_team,
            TeamColor::Yellow => &self.blue_team,
        }
    }

    pub fn get_team_data(&self, color: TeamColor) -> TeamData {
        self.side_assignment.transform_to_team_coords(color, self)
    }
}

/// A struct to store the world state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
#[typeshare]
pub struct TeamData {
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

impl TeamData {
    pub fn get_player(&self, id: PlayerId) -> &PlayerData {
        self.own_players.iter().find(|p| p.id == id).unwrap()
    }

    pub fn players_within_radius(&self, pos: Vector2, radius: f64) -> Vec<&PlayerData> {
        self.own_players
            .iter()
            .chain(self.opp_players.iter())
            .filter(|p| (p.position.xy() - pos).norm() < radius)
            .collect()
    }

    /// Compute the approximate time it will take for the player to reach the target point
    pub fn time_to_reach_point(&self, player: &PlayerData, target: Vector2) -> f64 {
        let max_speed = MAX_SPEED;
        let max_acceleration = MAX_ACCELERATION;
        let current_speed = player.velocity.norm();
        let dist = (target - player.position.xy()).norm();

        // Acceleration phase
        let t1 = (max_speed - current_speed) / max_acceleration;
        let d1 = current_speed * t1 + 0.5 * max_acceleration * t1.powi(2);

        // Deceleration phase
        let t3 = max_speed / max_acceleration;
        let d3 = 0.5 * max_speed * t3;

        // Constant speed phase
        let d2 = dist - d1 - d3;

        if d2 > 0.0 {
            // The robot reaches max speed
            let t2 = d2 / max_speed;
            t1 + t2 + t3
        } else {
            // The robot doesn't reach max speed
            // Calculate time considering both acceleration and deceleration
            let v_peak = (max_acceleration * dist).sqrt();
            let t_acc = (v_peak - current_speed) / max_acceleration;
            let t_dec = v_peak / max_acceleration;
            t_acc + t_dec
        }
    }

    /// Cast a ray from the start point in the given direction and return the intersection point with the first
    /// player or wall that it.
    pub fn cast_ray(&self, start: Vector2, direction: Vector2) -> Option<Vector2> {
        let normalized_direction = direction.normalize();
        let mut closest_intersection: Option<(f64, Vector2)> = None;

        let update_closest = |closest: &mut Option<(f64, Vector2)>, t: f64, point: Vector2| {
            if t > 0.0 && (closest.is_none() || t < closest.unwrap().0) {
                *closest = Some((t, point));
            }
        };

        // Check intersections with players
        for player in self.own_players.iter().chain(self.opp_players.iter()) {
            let oc = start - player.position.xy();
            let a = normalized_direction.dot(&normalized_direction);
            let b = 2.0 * oc.dot(&normalized_direction);
            let c = oc.dot(&oc) - PLAYER_RADIUS * PLAYER_RADIUS;
            let discriminant = b * b - 4.0 * a * c;

            if discriminant >= 0.0 {
                let t = (-b - discriminant.sqrt()) / (2.0 * a);
                if t > 0.0 {
                    let intersection_point = start + normalized_direction * t;
                    update_closest(&mut closest_intersection, t, intersection_point);
                }
            }
        }

        // Check intersections with field boundaries
        // if let Some(field_geom) = &self.field_geom {
        //     let half_length = field_geom.field_length / 2.0;
        //     let half_width = field_geom.field_width / 2.0;

        //     let check_boundary = |p1: Vector2, p2: Vector2| {
        //         let v1 = p2 - p1;
        //         if let Some(intersection) = find_intersection(p1, v1, start, direction) {
        //             let t = (intersection - start).dot(&normalized_direction);
        //             update_closest(&mut closest_intersection, t, intersection);
        //         }
        //     };

        // Check all four boundaries
        /*
        check_boundary(
            Vector2::new(-half_length, half_width),
            Vector2::new(half_length, half_width),
        );
        check_boundary(
            Vector2::new(-half_length, -half_width),
            Vector2::new(half_length, -half_width),
        );
        check_boundary(
            Vector2::new(-half_length, -half_width),
            Vector2::new(-half_length, half_width),
        );
        check_boundary(
            Vector2::new(half_length, -half_width),
            Vector2::new(half_length, half_width),
        );
        */
        // }

        closest_intersection.map(|(_, point)| point)
    }

    /// Predict the position of the ball at time `t` seconds in the future assuming it
    /// moves in a straight line.
    pub fn predict_ball_position(&self, t: f64) -> Option<BallPrediction> {
        if let Some(ball) = &self.ball {
            let current_pos = ball.position.xy();
            let dp = ball.velocity.xy() * t;
            let distance = dp.norm();
            let pred_pos = ball.position.xy() + dp;

            if let Some(intersection) = self.cast_ray(current_pos, dp) {
                if (intersection - current_pos).norm() < distance {
                    Some(BallPrediction::Collision(intersection))
                } else {
                    Some(BallPrediction::Linear(pred_pos))
                }
            } else {
                Some(BallPrediction::Linear(pred_pos))
            }
        } else {
            None
        }
    }

    pub fn get_best_kick_direction(&self, player_id: PlayerId) -> Angle {
        let player = self.get_player(player_id);
        let mut score = 0.0;
        let mut best_angle: Angle = Angle::default();
        for theta in (-90..90).step_by(20) {
            let angle = Angle::from_degrees(theta as f64);
            let direction = player.position + angle.to_vector() * 1000.0;

            let mut min_dist: f64 = 0.0;
            for p in self.opp_players.iter() {
                let dist = distance_to_line(player.position, direction, p.position);
                if dist < min_dist {
                    min_dist = dist;
                }
            }

            if min_dist > score {
                score = min_dist;
                best_angle = angle;
            }
        }
        best_angle
    }

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
                        // TODO: fix
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

    fn intersects_line(&self, start: Vector2, end: Vector2) -> bool {
        match self {
            Avoid::Line {
                start: line_start,
                end: line_end,
            } => {
                let line_seg = *line_end - *line_start;
                let test_seg = end - start;

                let denom = line_seg.x * test_seg.y - line_seg.y * test_seg.x;
                if denom.abs() < 1e-10 {
                    return false; // Lines are parallel
                }

                let t = ((start.x - line_start.x) * test_seg.y
                    - (start.y - line_start.y) * test_seg.x)
                    / denom;
                let u = ((start.x - line_start.x) * line_seg.y
                    - (start.y - line_start.y) * line_seg.x)
                    / denom;

                t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0
            }
            Avoid::Circle { center } => {
                // Calculate minimum distance from circle center to line segment
                let line_vec = end - start;
                let line_len_sq = line_vec.norm_squared();

                if line_len_sq == 0.0 {
                    return false;
                }

                let t = ((center - start).dot(&line_vec) / line_len_sq).clamp(0.0, 1.0);
                let closest_point = start + line_vec * t;
                let dist_to_line = (center - closest_point).norm();

                dist_to_line < PLAYER_RADIUS
            }
        }
    }
}

pub fn nearest_safe_pos(
    entity_to_avoid: Avoid,
    min_distance: f64,
    initial_pos: Vector2,
    target_pos: Vector2,
    max_radius: i32,
    field: &FieldGeometry,
) -> Vector2 {
    let mut best_pos = target_pos;
    let mut best_loss = f64::INFINITY;
    let min_theta = 0;
    let max_theta = 360;

    for theta in (min_theta..max_theta).step_by(10) {
        let theta = Angle::from_degrees(theta as f64);
        for radius in (0..max_radius).step_by(50) {
            let position = initial_pos + theta.to_vector() * (radius as f64);
            if is_pos_in_field(position, field)
                && entity_to_avoid.distance_to(position) > min_distance
            {
                let distnorm = (initial_pos - target_pos).norm() + 1e-6;
                let dist_score = (position - target_pos).norm() / distnorm;
                let closeness_score = (position - initial_pos).norm() / distnorm;
                let not_gay_score =
                    entity_to_avoid.intersects_line(initial_pos, position) as i32 as f64;
                let loss = dist_score + closeness_score * 0.5 + not_gay_score;

                if loss < best_loss {
                    best_loss = loss;
                    best_pos = position;
                }
            }
        }
    }
    if best_loss == f64::INFINITY {
        log::warn!("Could not find a safe position from {initial_pos}, avoiding {entity_to_avoid:?}, {best_loss}");
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

pub fn mock_world_data() -> WorldData {
    WorldData {
        t_received: 0.0,
        t_capture: 0.0,
        dt: 1.0,
        blue_team: vec![PlayerData {
            id: PlayerId::new(0),
            position: Vector2::new(1000.0, 1000.0),
            timestamp: 0.0,
            raw_position: Vector2::new(1000.0, 1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            primary_status: Some(SysStatus::Ready),
            kicker_cap_voltage: Some(0.0),
            kicker_temp: Some(0.0),
            pack_voltages: Some([0.0, 0.0]),
            breakbeam_ball_detected: false,
            imu_status: Some(SysStatus::Ready),
            kicker_status: Some(SysStatus::Standby),
            handicaps: HashSet::new(),
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
            primary_status: Some(SysStatus::Ready),
            kicker_cap_voltage: Some(0.0),
            kicker_temp: Some(0.0),
            pack_voltages: Some([0.0, 0.0]),
            breakbeam_ball_detected: false,
            imu_status: Some(SysStatus::Ready),
            kicker_status: Some(SysStatus::Standby),
            handicaps: HashSet::new(),
        }],
        field_geom: Default::default(),
        ball: None,
        game_state: RawGameStateData {
            game_state: GameState::Run,
            operating_team: TeamColor::Blue,
            freekick_kicker: None,
            blue_team_yellow_cards: 0,
            yellow_team_yellow_cards: 0,
        },
        side_assignment: SideAssignment::YellowOnPositive,
    }
}

pub fn mock_team_data() -> TeamData {
    TeamData {
        own_players: vec![PlayerData {
            id: PlayerId::new(0),
            position: Vector2::new(1000.0, 1000.0),
            timestamp: 0.0,
            raw_position: Vector2::new(1000.0, 1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            primary_status: Some(SysStatus::Ready),
            kicker_cap_voltage: Some(0.0),
            kicker_temp: Some(0.0),
            pack_voltages: Some([0.0, 0.0]),
            breakbeam_ball_detected: false,
            imu_status: Some(SysStatus::Ready),
            kicker_status: Some(SysStatus::Standby),
            handicaps: HashSet::new(),
        }],
        opp_players: vec![PlayerData {
            id: PlayerId::new(1),
            position: Vector2::new(-1000.0, -1000.0),
            timestamp: 0.0,
            raw_position: Vector2::new(-1000.0, -1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            raw_yaw: Angle::default(),
            angular_speed: 0.0,
            primary_status: Some(SysStatus::Ready),
            kicker_cap_voltage: Some(0.0),
            kicker_temp: Some(0.0),
            pack_voltages: Some([0.0, 0.0]),
            breakbeam_ball_detected: false,
            imu_status: Some(SysStatus::Ready),
            kicker_status: Some(SysStatus::Standby),
            handicaps: HashSet::new(),
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
            us_operating: true,
            yellow_cards: 0,
            freekick_kicker: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_ray_player_intersection() {
        let world = mock_team_data();
        let start = Vector2::new(0.0, 0.0);
        let direction = Vector2::new(1.0, 1.0);

        let result = world.cast_ray(start, direction);
        assert!(result.is_some());
        let intersection = result.unwrap();
        assert_relative_eq!(intersection.x, 955.0, epsilon = 1.0);
        assert_relative_eq!(intersection.y, 955.0, epsilon = 1.0);
    }

    #[test]
    fn test_ray_wall_intersection() {
        let world = mock_team_data();
        let start = Vector2::new(0.0, 0.0);
        let direction = Vector2::new(1.0, 0.0);

        let result = world.cast_ray(start, direction);
        assert!(result.is_some());
        let intersection = result.unwrap();
        assert_relative_eq!(intersection.x, 4500.0, epsilon = 1.0);
        assert_relative_eq!(intersection.y, 0.0, epsilon = 1.0);
    }

    #[test]
    fn test_no_intersection() {
        let world = mock_team_data();
        let start = Vector2::new(0.0, 0.0);
        let direction = Vector2::new(0.0, 1.0);

        let result = world.cast_ray(start, direction);
        assert!(result.is_some());
        let intersection = result.unwrap();
        assert_relative_eq!(intersection.x, 0.0, epsilon = 1.0);
        assert_relative_eq!(intersection.y, 3000.0, epsilon = 1.0);
    }

    #[test]
    fn test_ray_origin_inside_player() {
        let world = mock_team_data();
        let start = Vector2::new(1000.0, 1000.0); // Inside the player
        let direction = Vector2::new(1.0, 0.0);

        let result = world.cast_ray(start, direction);
        assert!(result.is_some());
        let intersection = result.unwrap();
        assert_relative_eq!(intersection.x, 1090.0, epsilon = 1.0);
        assert_relative_eq!(intersection.y, 1000.0, epsilon = 1.0);
    }

    #[test]
    fn test_ray_parallel_to_wall() {
        let world = mock_team_data();
        let start = Vector2::new(0.0, 3000.0);
        let direction = Vector2::new(1.0, 0.0);

        let result = world.cast_ray(start, direction);
        assert!(result.is_some());
        let intersection = result.unwrap();
        assert_relative_eq!(intersection.x, 4500.0, epsilon = 1.0);
        assert_relative_eq!(intersection.y, 3000.0, epsilon = 1.0);
    }

    #[test]
    fn test_ray_away_from_everything() {
        let world = mock_team_data();
        let start = Vector2::new(0.0, 0.0);
        let direction = Vector2::new(-1.0, -1.0);

        let result = world.cast_ray(start, direction);
        assert!(result.is_some());
        let intersection = result.unwrap();
        assert_relative_eq!(intersection.x, -4500.0, epsilon = 1.0);
        assert_relative_eq!(intersection.y, -4500.0, epsilon = 1.0);
    }

    #[test]
    fn test_ball_placement_equals() {
        assert!(
            GameState::BallReplacement(Vector2::zeros())
                == GameState::BallReplacement(Vector2::x())
        )
    }

    #[test]
    fn test_ball_placement_matches() {
        assert!(StrategyGameStateMacther::any_of(
            vec![GameState::BallReplacement(Vector2::zeros())].as_slice()
        )
        .matches(&GameState::BallReplacement(Vector2::x())))
    }
}
