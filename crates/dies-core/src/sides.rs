use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::{
    Angle, AutorefKickedBallTeam, BallData, GameStateData, PlayerData, PlayerGlobalMoveCmd,
    PlayerId, PlayerMoveCmd, RobotCmd, RotationDirection, TeamData, Vector2, Vector3, WorldData,
};

/// # Team-Specific Coordinate System
///
/// This module provides coordinate system transformations to create a consistent
/// team-specific reference frame where the +x axis always points towards the
/// enemy goal, regardless of which side of the field the team is defending.
///
/// ## Coordinate System Philosophy
///
/// In RoboCup SSL, teams can be assigned to defend either the positive or negative
/// x side of the field. To simplify strategy code, we transform all coordinates
/// into a team-specific coordinate system where:
///
/// - **+x axis**: Always points towards the enemy goal (attacking direction)
/// - **-x axis**: Always points towards our own goal (defending direction)
/// - **y axis**: Remains unchanged (left/right from team perspective)
///
/// ## How It Works
///
/// The transformation is based on two key pieces of information:
/// 1. **SideAssignment**: Which team (Blue/Yellow) defends the positive x side
/// 2. **TeamColor**: The color of our team (Blue/Yellow)
///
/// ### Transformation Rules
///
/// - If our team attacks in the same direction as world +x: coordinates remain unchanged
/// - If our team attacks in the opposite direction: x coordinates are negated, angles are mirrored
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[typeshare]
pub enum TeamColor {
    Blue,
    Yellow,
}

impl TeamColor {
    /// Returns the opposite team color.
    pub fn opposite(&self) -> Self {
        match self {
            TeamColor::Blue => TeamColor::Yellow,
            TeamColor::Yellow => TeamColor::Blue,
        }
    }
}

impl std::fmt::Display for TeamColor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeamColor::Blue => write!(f, "Blue"),
            TeamColor::Yellow => write!(f, "Yellow"),
        }
    }
}

/// Represents which team defends the positive x side of the field.
///
/// In RoboCup SSL, the field coordinate system is fixed, but teams can be
/// assigned to defend either side. This enum tracks that assignment.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[typeshare]
pub enum SideAssignment {
    /// Blue team defends the positive x side (+x goal)
    BlueOnPositive,
    /// Yellow team defends the positive x side (+x goal)
    YellowOnPositive,
}

impl SideAssignment {
    /// Transforms world data to team-specific coordinates.
    pub fn transform_to_team_coords(&self, color: TeamColor, world_data: &WorldData) -> TeamData {
        TeamData {
            t_received: world_data.t_received,
            t_capture: world_data.t_capture,
            dt: world_data.dt,
            own_players: world_data
                .get_team_players(color)
                .iter()
                .map(|p| self.transform_to_team_coords_player(color, p))
                .collect(),
            opp_players: world_data
                .get_team_players(color.opposite())
                .iter()
                .map(|p| self.transform_to_team_coords_player(color, p))
                .collect(),
            ball: world_data
                .ball
                .as_ref()
                .map(|b| self.transform_to_team_coords_ball(color, &b)),
            field_geom: world_data.field_geom.clone(),
            current_game_state: GameStateData {
                game_state: world_data.game_state.game_state,
                us_operating: world_data.game_state.operating_team == color,
                yellow_cards: match color {
                    TeamColor::Blue => world_data.game_state.blue_team_yellow_cards,
                    TeamColor::Yellow => world_data.game_state.yellow_team_yellow_cards,
                },
                freekick_kicker: if let Some(kicker) = world_data.game_state.freekick_kicker {
                    if kicker.team_color == color {
                        Some(kicker.player_id)
                    } else {
                        None
                    }
                } else {
                    None
                },
                max_allowed_bots: match color {
                    TeamColor::Blue => world_data.game_state.blue_team_max_allowed_bots,
                    TeamColor::Yellow => world_data.game_state.yellow_team_max_allowed_bots,
                },
            },
            ball_on_our_side: match color {
                TeamColor::Blue => world_data.ball_on_blue_side,
                TeamColor::Yellow => world_data.ball_on_yellow_side,
            },
            ball_on_opp_side: match color {
                TeamColor::Blue => world_data.ball_on_yellow_side,
                TeamColor::Yellow => world_data.ball_on_blue_side,
            },
            kicked_ball: None,
            // world_data
            //     .autoref_info
            //     .as_ref()
            //     .map(|info| AutorefKickedBallTeam {
            //         pos: info.kicked_ball.as_ref().map(|k| k.pos),
            //         vel: info.kicked_ball.as_ref().map(|k| k.vel),
            //         start_timestamp: info.kicked_ball.as_ref().map(|k| k.start_timestamp),
            //         stop_timestamp: info.kicked_ball.as_ref().map(|k| k.stop_timestamp),
            //         stop_pos: info.kicked_ball.as_ref().map(|k| k.stop_pos),
            //         robot_id: info.kicked_ball.as_ref().map(|k| k.robot_id),
            //         we_kicked: info.kicked_ball.as_ref().map(|k| k.we_kicked),
            //     }),
        }
    }

    /// Returns the direction multiplier for transforming to team coordinates.
    ///
    /// This returns the sign that indicates which direction the specified team
    /// should attack (towards the enemy goal) in the world coordinate system.
    ///
    /// # Returns
    /// - `1.0`: Team attacks towards +x (coordinates unchanged)
    /// - `-1.0`: Team attacks towards -x (coordinates need to be flipped)
    pub fn attacking_direction_sign(&self, color: TeamColor) -> f64 {
        match (self, color) {
            (SideAssignment::BlueOnPositive, TeamColor::Blue) => -1.0,
            (SideAssignment::BlueOnPositive, TeamColor::Yellow) => 1.0,
            (SideAssignment::YellowOnPositive, TeamColor::Blue) => 1.0,
            (SideAssignment::YellowOnPositive, TeamColor::Yellow) => -1.0,
        }
    }

    pub fn is_on_own_side(&self, color: TeamColor, position: &Vector2) -> bool {
        match (self, color) {
            (SideAssignment::BlueOnPositive, TeamColor::Blue) => position.x > 0.0,
            (SideAssignment::BlueOnPositive, TeamColor::Yellow) => position.x < 0.0,
            (SideAssignment::YellowOnPositive, TeamColor::Blue) => position.x < 0.0,
            (SideAssignment::YellowOnPositive, TeamColor::Yellow) => position.x > 0.0,
        }
    }

    pub fn is_on_own_side_vec3(&self, color: TeamColor, position: &Vector3) -> bool {
        self.is_on_own_side(color, &Vector2::new(position.x, position.y))
    }

    pub fn is_on_opp_side(&self, color: TeamColor, position: &Vector2) -> bool {
        !self.is_on_own_side(color, position)
    }

    pub fn is_on_opp_side_vec3(&self, color: TeamColor, position: &Vector3) -> bool {
        !self.is_on_own_side_vec3(color, position)
    }

    pub fn transform_vec2(&self, color: TeamColor, vec: &Vector2) -> Vector2 {
        Vector2::new(vec.x * self.attacking_direction_sign(color), vec.y)
    }

    pub fn transform_vec3(&self, color: TeamColor, vec: &Vector3) -> Vector3 {
        Vector3::new(vec.x * self.attacking_direction_sign(color), vec.y, vec.z)
    }

    fn transform_angle(&self, color: TeamColor, angle: Angle) -> Angle {
        let sign = self.attacking_direction_sign(color);
        if sign > 0.0 {
            angle
        } else {
            // Mirror around y-axis: angle -> π - angle (or -π - angle for negative angles)
            if angle.radians() >= 0.0 {
                Angle::from_radians(std::f64::consts::PI - angle.radians())
            } else {
                Angle::from_radians(-std::f64::consts::PI - angle.radians())
            }
        }
    }

    fn transform_to_team_coords_player(&self, color: TeamColor, player: &PlayerData) -> PlayerData {
        PlayerData {
            position: self.transform_vec2(color, &player.position),
            velocity: self.transform_vec2(color, &player.velocity),
            yaw: self.transform_angle(color, player.yaw),
            angular_speed: player.angular_speed * self.attacking_direction_sign(color),
            raw_position: self.transform_vec2(color, &player.raw_position),
            raw_yaw: self.transform_angle(color, player.raw_yaw),
            primary_status: player.primary_status,
            kicker_cap_voltage: player.kicker_cap_voltage,
            kicker_temp: player.kicker_temp,
            pack_voltages: player.pack_voltages,
            breakbeam_ball_detected: player.breakbeam_ball_detected,
            imu_status: player.imu_status,
            kicker_status: player.kicker_status,
            id: player.id,
            timestamp: player.timestamp,
            handicaps: player.handicaps.clone(),
        }
    }

    fn transform_to_team_coords_ball(&self, color: TeamColor, ball: &BallData) -> BallData {
        BallData {
            position: self.transform_vec3(color, &ball.position),
            velocity: self.transform_vec3(color, &ball.velocity),
            detected: ball.detected,
            raw_position: ball
                .raw_position
                .iter()
                .map(|p| self.transform_vec3(color, p))
                .collect(),
            timestamp: ball.timestamp,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlayerCmdUntransformer {
    side_assignment: SideAssignment,
    team_color: TeamColor,
    target_velocity: Option<Vector2>,
    target_yaw: Option<Angle>,
    w: Option<f64>,
    dribble_speed: Option<f64>,
    fan_speed: Option<f64>,
    kick_speed: Option<f64>,
    robot_cmd: Option<RobotCmd>,
    kick_counter: Option<u8>,
    max_yaw_rate: Option<f64>,
    preferred_rotation_direction: Option<RotationDirection>,
}

impl PlayerCmdUntransformer {
    pub fn new(side_assignment: SideAssignment, team_color: TeamColor) -> Self {
        Self {
            side_assignment,
            team_color,
            target_velocity: None,
            target_yaw: None,
            w: None,
            dribble_speed: None,
            fan_speed: None,
            kick_speed: None,
            robot_cmd: None,
            kick_counter: None,
            max_yaw_rate: None,
            preferred_rotation_direction: None,
        }
    }

    pub fn set_target_velocity(&mut self, target_velocity: Vector2) -> &mut Self {
        self.target_velocity = Some(target_velocity);
        self
    }

    pub fn set_target_yaw(&mut self, target_yaw: Angle) -> &mut Self {
        self.target_yaw = Some(target_yaw);
        self
    }

    pub fn set_w(&mut self, w: f64) -> &mut Self {
        self.w = Some(w);
        self
    }

    pub fn set_dribble_speed(&mut self, dribble_speed: f64) -> &mut Self {
        self.dribble_speed = Some(dribble_speed);
        self
    }

    pub fn set_fan_speed(&mut self, fan_speed: f64) -> &mut Self {
        self.fan_speed = Some(fan_speed);
        self
    }

    pub fn set_kick_speed(&mut self, kick_speed: f64) -> &mut Self {
        self.kick_speed = Some(kick_speed);
        self
    }

    pub fn set_robot_cmd(&mut self, robot_cmd: RobotCmd) -> &mut Self {
        self.robot_cmd = Some(robot_cmd);
        self
    }

    pub fn set_kick_counter(&mut self, kick_counter: u8) -> &mut Self {
        self.kick_counter = Some(kick_counter);
        self
    }

    pub fn set_max_yaw_rate(&mut self, max_yaw_rate: f64) -> &mut Self {
        self.max_yaw_rate = Some(max_yaw_rate);
        self
    }

    pub fn set_preferred_rotation_direction(
        &mut self,
        preferred_rotation_direction: RotationDirection,
    ) -> &mut Self {
        self.preferred_rotation_direction = Some(preferred_rotation_direction);
        self
    }

    pub fn untransform_move_cmd(&self, id: PlayerId, yaw: Angle) -> PlayerMoveCmd {
        let target_velocity_local = if let Some(target_velocity) = self.target_velocity {
            self.side_assignment
                .transform_angle(self.team_color, yaw)
                .inv()
                .rotate_vector(
                    &(self
                        .side_assignment
                        .transform_vec2(self.team_color, &target_velocity)),
                )
        } else {
            Vector2::zeros()
        };

        PlayerMoveCmd {
            id,
            sx: target_velocity_local.x / 1000.0, // Convert to m/s
            sy: -target_velocity_local.y / 1000.0, // Convert to m/s
            w: -self
                .side_assignment
                .attacking_direction_sign(self.team_color)
                * self.w.unwrap_or(0.0),
            dribble_speed: self.dribble_speed.unwrap_or(0.0),
            fan_speed: self.fan_speed.unwrap_or(0.0),
            kick_speed: self.kick_speed.unwrap_or(0.0),
            robot_cmd: self.robot_cmd.unwrap_or(RobotCmd::None),
        }
    }

    pub fn untransform_global_move_cmd(&self, id: PlayerId, yaw: Angle) -> PlayerGlobalMoveCmd {
        let target_velocity = self.side_assignment.transform_vec2(
            self.team_color,
            &self.target_velocity.unwrap_or(Vector2::zeros()),
        );
        let target_yaw = self
            .target_yaw
            .map(|yaw| {
                self.side_assignment
                    .transform_angle(self.team_color, yaw)
                    .radians()
            })
            .unwrap_or(f64::NAN);
        let last_yaw = self
            .side_assignment
            .transform_angle(self.team_color, yaw)
            .radians();
        PlayerGlobalMoveCmd {
            id,
            global_x: target_velocity.x / 1000.0,
            global_y: target_velocity.y / 1000.0,
            heading_setpoint: target_yaw,
            last_heading: last_yaw,
            dribble_speed: self.dribble_speed.unwrap_or(0.0),
            kick_counter: self.kick_counter.unwrap_or(0),
            robot_cmd: self.robot_cmd.unwrap_or(RobotCmd::None),
            max_yaw_rate: self.max_yaw_rate.unwrap_or(10_000.0),
            preferred_rotation_direction: self
                .preferred_rotation_direction
                .unwrap_or(RotationDirection::NoPreference),
        }
    }

    pub fn untrasform_set_heading(&self, heading: Angle) -> f64 {
        self.side_assignment
            .transform_angle(self.team_color, heading)
            .degrees()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Angle, Vector2};
    use approx::assert_relative_eq;

    const BLUE_ON_POSITIVE: SideAssignment = SideAssignment::BlueOnPositive;
    const YELLOW_ON_POSITIVE: SideAssignment = SideAssignment::YellowOnPositive;

    #[test]
    fn test_attacking_direction_sign() {
        // Blue team
        assert_eq!(
            BLUE_ON_POSITIVE.attacking_direction_sign(TeamColor::Blue),
            -1.0
        );
        assert_eq!(
            YELLOW_ON_POSITIVE.attacking_direction_sign(TeamColor::Blue),
            1.0
        );

        // Yellow team
        assert_eq!(
            BLUE_ON_POSITIVE.attacking_direction_sign(TeamColor::Yellow),
            1.0
        );
        assert_eq!(
            YELLOW_ON_POSITIVE.attacking_direction_sign(TeamColor::Yellow),
            -1.0
        );
    }

    #[test]
    fn test_transform_to_team_coords_vec2() {
        let vec = Vector2::new(1.0, 2.0);

        // Attacks -x
        let transformed = BLUE_ON_POSITIVE.transform_vec2(TeamColor::Blue, &vec);
        assert_eq!(transformed, Vector2::new(-1.0, 2.0));

        // Attacks +x
        let transformed = YELLOW_ON_POSITIVE.transform_vec2(TeamColor::Blue, &vec);
        assert_eq!(transformed, Vector2::new(1.0, 2.0));
    }

    #[test]
    fn test_transform_to_team_coords_angle() {
        let angle_pos = Angle::from_degrees(45.0);
        let angle_neg = Angle::from_degrees(-45.0);

        // Attacks +x (no change)
        let transformed_pos = YELLOW_ON_POSITIVE.transform_angle(TeamColor::Blue, angle_pos);
        assert_relative_eq!(transformed_pos.degrees(), 45.0);
        let transformed_neg = YELLOW_ON_POSITIVE.transform_angle(TeamColor::Blue, angle_neg);
        assert_relative_eq!(transformed_neg.degrees(), -45.0);

        // Attacks -x (flip)
        let transformed_pos = BLUE_ON_POSITIVE.transform_angle(TeamColor::Blue, angle_pos);
        assert_relative_eq!(transformed_pos.degrees(), 135.0);
        let transformed_neg = BLUE_ON_POSITIVE.transform_angle(TeamColor::Blue, angle_neg);
        assert_relative_eq!(transformed_neg.degrees(), -135.0);
    }
}
