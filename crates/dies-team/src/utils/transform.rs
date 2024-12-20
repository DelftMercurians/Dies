/// Describes the side of the field a team is assigned to.
#[derive(Debug, Clone, Copy)]
pub struct SideTransform {
    team: Team,
    side_assignment: SideAssignment,
}

impl SideTransform {
    pub fn new(team: Team, side_assignment: SideAssignment) -> Self {
        Self {
            team,
            side_assignment,
        }
    }

    /// Returns the sign of the x-coordinate for the given team's **opponent's** goal's side.
    ///
    /// All x coordinates need to be mutiplied by this value to convert from the team's side to the
    pub fn x_sign(&self) -> f64 {
        match self.team {
            Team::Blue => self.side_assignment.yellow_goal_side(),
            Team::Yellow => self.side_assignment.blue_goal_side(),
        }
    }

    pub fn reverse_x_sign(&self) -> f64 {
        -self.x_sign()
    }

    pub fn transform_vector2(&self, v: Vector2) -> Vector2 {
        Vector2::new(v.x * self.x_sign(), v.y)
    }

    pub fn reverse_vector2(&self, v: Vector2) -> Vector2 {
        Vector2::new(v.x * self.reverse_x_sign(), v.y)
    }

    pub fn transform_vector3(&self, v: Vector3) -> Vector3 {
        Vector3::new(v.x * self.x_sign(), v.y, v.z)
    }

    pub fn reverse_vector3(&self, v: Vector3) -> Vector3 {
        Vector3::new(v.x * self.reverse_x_sign(), v.y, v.z)
    }

    pub fn transform_angle(&self, angle: Angle) -> Angle {
        let sign = self.x_sign();
        if sign > 0.0 {
            angle
        } else {
            // Invert around y-axis
            if angle.radians() >= 0.0 {
                Angle::from_radians(std::f64::consts::PI - angle.radians())
            } else {
                Angle::from_radians(-std::f64::consts::PI - angle.radians())
            }
        }
    }

    pub fn reverse_angle(&self, angle: Angle) -> Angle {
        let sign = self.reverse_x_sign();
        if sign > 0.0 {
            angle
        } else {
            // Invert around y-axis
            if angle.radians() >= 0.0 {
                Angle::from_radians(std::f64::consts::PI - angle.radians())
            } else {
                Angle::from_radians(-std::f64::consts::PI - angle.radians())
            }
        }
    }

    pub fn transform_player(&self, p: &PlayerFrame) -> PlayerFrame {
        PlayerFrame {
            timestamp: p.timestamp,
            id: p.id,
            position: self.transform_vector2(p.position),
            velocity: self.transform_vector2(p.velocity),
            yaw: self.transform_angle(p.yaw),
            angular_speed: self.x_sign() * p.angular_speed,
            feedback: p.feedback.clone(),
        }
    }

    pub fn transform_ball(&self, b: &BallFrame) -> BallFrame {
        BallFrame {
            timestamp: b.timestamp,
            position: self.transform_vector3(b.position),
            velocity: self.transform_vector3(b.velocity),
            detected: b.detected,
        }
    }

    pub fn transform_game_state(&self, state: &GameState) -> TeamGameState {
        let transformed_state = match state.game_state {
            GameStateType::BallReplacement(v) => {
                GameStateType::BallReplacement(self.transform_vector2(v))
            }
            GameStateType::Unknown
            | GameStateType::Halt
            | GameStateType::Timeout
            | GameStateType::Stop
            | GameStateType::PrepareKickoff
            | GameStateType::PreparePenalty
            | GameStateType::Kickoff
            | GameStateType::FreeKick
            | GameStateType::Penalty
            | GameStateType::PenaltyRun
            | GameStateType::Run => state.game_state,
        };
        TeamGameState {
            game_state: transformed_state,
            us_operating: state
                .operating_team
                .as_ref()
                .map(|t| *t == self.team)
                .unwrap_or(false),
        }
    }
}
