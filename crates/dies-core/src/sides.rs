use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::{
    Angle, BallData, GameStateData, PlayerCmd, PlayerData, PlayerMoveCmd, TeamData, Vector2,
    Vector3, WorldData,
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
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
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
            },
        }
    }

    pub fn untransform_player_cmd(&self, color: TeamColor, cmd: &PlayerCmd) -> PlayerCmd {
        match cmd {
            PlayerCmd::Move(cmd) => PlayerCmd::Move(self.untransform_player_move_cmd(color, cmd)),
            PlayerCmd::SetHeading { id, heading } => PlayerCmd::SetHeading {
                id: *id,
                heading: self
                    .transform_to_team_coords_angle(color, Angle::from_degrees(*heading))
                    .degrees(),
            },
        }
    }

    fn untransform_player_move_cmd(&self, color: TeamColor, cmd: &PlayerMoveCmd) -> PlayerMoveCmd {
        PlayerMoveCmd {
            id: cmd.id,
            sx: cmd.sx * self.attacking_direction_sign(color),
            sy: cmd.sy,
            w: cmd.w * self.attacking_direction_sign(color),
            dribble_speed: cmd.dribble_speed,
            robot_cmd: cmd.robot_cmd,
            fan_speed: cmd.fan_speed,
            kick_speed: cmd.kick_speed,
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
    fn attacking_direction_sign(&self, color: TeamColor) -> f64 {
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

    fn transform_to_team_coords_vec2(&self, color: TeamColor, vec: &Vector2) -> Vector2 {
        Vector2::new(vec.x * self.attacking_direction_sign(color), vec.y)
    }

    fn transform_to_team_coords_vec3(&self, color: TeamColor, vec: &Vector3) -> Vector3 {
        Vector3::new(vec.x * self.attacking_direction_sign(color), vec.y, vec.z)
    }

    fn transform_to_team_coords_angle(&self, color: TeamColor, angle: Angle) -> Angle {
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
            position: self.transform_to_team_coords_vec2(color, &player.position),
            velocity: self.transform_to_team_coords_vec2(color, &player.velocity),
            yaw: self.transform_to_team_coords_angle(color, player.yaw),
            angular_speed: player.angular_speed * self.attacking_direction_sign(color),
            raw_position: self.transform_to_team_coords_vec2(color, &player.raw_position),
            raw_yaw: self.transform_to_team_coords_angle(color, player.raw_yaw),
            primary_status: player.primary_status,
            kicker_cap_voltage: player.kicker_cap_voltage,
            kicker_temp: player.kicker_temp,
            pack_voltages: player.pack_voltages,
            breakbeam_ball_detected: player.breakbeam_ball_detected,
            imu_status: player.imu_status,
            kicker_status: player.kicker_status,
            id: player.id,
            timestamp: player.timestamp,
        }
    }

    fn transform_to_team_coords_ball(&self, color: TeamColor, ball: &BallData) -> BallData {
        BallData {
            position: self.transform_to_team_coords_vec3(color, &ball.position),
            velocity: self.transform_to_team_coords_vec3(color, &ball.velocity),
            detected: ball.detected,
            raw_position: ball
                .raw_position
                .iter()
                .map(|p| self.transform_to_team_coords_vec3(color, p))
                .collect(),
            timestamp: ball.timestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Angle, PlayerId, Vector2};
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
        let transformed = BLUE_ON_POSITIVE.transform_to_team_coords_vec2(TeamColor::Blue, &vec);
        assert_eq!(transformed, Vector2::new(-1.0, 2.0));

        // Attacks +x
        let transformed = YELLOW_ON_POSITIVE.transform_to_team_coords_vec2(TeamColor::Blue, &vec);
        assert_eq!(transformed, Vector2::new(1.0, 2.0));
    }

    #[test]
    fn test_transform_to_team_coords_angle() {
        let angle_pos = Angle::from_degrees(45.0);
        let angle_neg = Angle::from_degrees(-45.0);

        // Attacks +x (no change)
        let transformed_pos =
            YELLOW_ON_POSITIVE.transform_to_team_coords_angle(TeamColor::Blue, angle_pos);
        assert_relative_eq!(transformed_pos.degrees(), 45.0);
        let transformed_neg =
            YELLOW_ON_POSITIVE.transform_to_team_coords_angle(TeamColor::Blue, angle_neg);
        assert_relative_eq!(transformed_neg.degrees(), -45.0);

        // Attacks -x (flip)
        let transformed_pos =
            BLUE_ON_POSITIVE.transform_to_team_coords_angle(TeamColor::Blue, angle_pos);
        assert_relative_eq!(transformed_pos.degrees(), 135.0);
        let transformed_neg =
            BLUE_ON_POSITIVE.transform_to_team_coords_angle(TeamColor::Blue, angle_neg);
        assert_relative_eq!(transformed_neg.degrees(), -135.0);
    }

    #[test]
    fn test_untransform_player_move_cmd() {
        let cmd = PlayerMoveCmd {
            id: PlayerId::new(0),
            sx: 1.0,
            sy: 2.0,
            w: 0.5,
            dribble_speed: 1.0,
            robot_cmd: crate::player::RobotCmd::None,
            fan_speed: 0.0,
            kick_speed: 0.0,
        };

        // Attacks -x
        let untransformed = BLUE_ON_POSITIVE.untransform_player_move_cmd(TeamColor::Blue, &cmd);
        assert_eq!(untransformed.id, cmd.id);
        assert_relative_eq!(untransformed.sx, -1.0);
        assert_relative_eq!(untransformed.sy, 2.0);
        assert_relative_eq!(untransformed.w, -0.5);

        // Attacks +x
        let untransformed = YELLOW_ON_POSITIVE.untransform_player_move_cmd(TeamColor::Blue, &cmd);
        assert_eq!(untransformed.id, cmd.id);
        assert_relative_eq!(untransformed.sx, 1.0);
        assert_relative_eq!(untransformed.sy, 2.0);
        assert_relative_eq!(untransformed.w, 0.5);
    }

    #[test]
    fn test_untransform_player_set_heading_cmd() {
        let cmd = PlayerCmd::SetHeading {
            id: PlayerId::new(0),
            heading: 45.0,
        };

        // Attacks -x
        let untransformed = BLUE_ON_POSITIVE.untransform_player_cmd(TeamColor::Blue, &cmd);
        if let PlayerCmd::SetHeading { heading, .. } = untransformed {
            assert_relative_eq!(heading, 135.0, epsilon = 1e-5);
        } else {
            panic!("Expected SetHeading command");
        }

        // Attacks +x
        let untransformed = YELLOW_ON_POSITIVE.untransform_player_cmd(TeamColor::Blue, &cmd);
        if let PlayerCmd::SetHeading { heading, .. } = untransformed {
            assert_relative_eq!(heading, 45.0, epsilon = 1e-5);
        } else {
            panic!("Expected SetHeading command");
        }
    }
}
