use std::f64::consts::PI;

use dies_core::{BallData, PlayerData, Vector2, WorldData};

use super::control_output::ControlOutputItem;

pub struct OwnPlayerState {
    pub position: Vector2,
    pub orientation: f64,
}

impl OwnPlayerState {
    pub fn new(data: &PlayerData) -> Self {
        Self {
            position: data.position,
            orientation: data.orientation,
        }
    }

    pub fn step(&mut self, dt: f64, control: &ControlOutputItem) {
        self.position += control.velocity * dt;

        let new_orientation = self.orientation + control.angular_velocity * dt;
        // Wrap orientation to [-pi, pi]
        self.orientation = (new_orientation + PI).rem_euclid(2.0 * PI) - PI;
    }
}

pub struct OppPlayerState {
    pub position: Vector2,
    pub velocity: Vector2,
}

impl OppPlayerState {
    pub fn new(data: &PlayerData) -> Self {
        Self {
            position: data.position,
            velocity: data.velocity,
        }
    }

    pub fn step(&mut self, dt: f64) {
        self.position += self.velocity * dt;
    }
}

pub struct BallState {
    pub position: Vector2,
    pub velocity: Vector2,
}

impl BallState {
    pub fn new(data: &BallData) -> Self {
        Self {
            position: data.position.xy(),
            velocity: data.velocity.xy(),
        }
    }

    pub fn step(&mut self, dt: f64) {
        self.position += self.velocity * dt;
    }
}

pub struct State {
    pub own_players: Vec<OwnPlayerState>,
    pub opp_players: Vec<OppPlayerState>,
    pub ball: Option<BallState>,
}

impl State {
    pub fn new(world: &WorldData) -> Self {
        Self {
            own_players: world.own_players.iter().map(OwnPlayerState::new).collect(),
            opp_players: world.opp_players.iter().map(OppPlayerState::new).collect(),
            ball: world.ball.as_ref().map(BallState::new),
        }
    }

    pub fn ball_position(&self) -> Option<Vector2> {
        self.ball.as_ref().map(|ball| ball.position)
    }

    pub fn step(&mut self, dt: f64, u: &Vec<ControlOutputItem>) {
        for (player, control) in self.own_players.iter_mut().zip(u.iter()) {
            player.step(dt, control);
        }

        for player in self.opp_players.iter_mut() {
            player.step(dt);
        }

        if let Some(ball) = self.ball.as_mut() {
            ball.step(dt);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use dies_core::{PlayerId, Vector3};

    #[test]
    fn test_state() {
        let player_data = vec![
            PlayerData {
                id: PlayerId::new(0),
                timestamp: 0.0,
                position: Vector2::new(0.0, 100.0),
                raw_position: Vector2::new(0.0, 100.0),
                velocity: Vector2::new(0.0, 0.0),
                orientation: 0.0,
                angular_speed: 0.0,
            },
            PlayerData {
                id: PlayerId::new(1),
                timestamp: 0.0,
                position: Vector2::new(100.0, 0.0),
                raw_position: Vector2::new(100.0, 0.0),
                velocity: Vector2::new(0.0, 0.0),
                orientation: 0.0,
                angular_speed: 0.0,
            },
        ];
        let ball_data = BallData {
            position: Vector3::new(0.0, 0.0, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            timestamp: 0.0,
        };
        let world = WorldData {
            own_players: player_data,
            opp_players: vec![],
            ball: Some(ball_data),
            ..Default::default()
        };

        let mut state = State::new(&world);
        assert_eq!(state.own_players.len(), 2);

        let control = vec![
            ControlOutputItem {
                velocity: Vector2::new(10.0, 0.0),
                angular_velocity: 0.0,
            },
            ControlOutputItem {
                velocity: Vector2::new(0.0, 10.0),
                angular_velocity: 0.0,
            },
        ];

        state.step(0.1, &control);

        assert_eq!(state.own_players[0].position, Vector2::new(1.0, 100.0));
        assert_eq!(state.own_players[1].position, Vector2::new(100.0, 1.0));

        assert_eq!(state.ball_position(), Some(Vector2::new(0.0, 0.0)));
    }
}
