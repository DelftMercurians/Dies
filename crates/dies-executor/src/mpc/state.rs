use dies_core::{BallData, PlayerData, Vector2, Vector3, WorldData};
use nalgebra::{Dyn, Matrix1xX, Matrix2xX, MatrixView1xX, MatrixView2xX, U0, U2};

use super::cost::ControlOutput;

pub struct OwnPlayersState {
    pub positions: Matrix2xX<f64>,
    pub orientations: Matrix1xX<f64>,
}

impl OwnPlayersState {
    pub fn new(data: &WorldData) -> Self {
        let mut positions = Matrix2xX::zeros(data.own_players.len());
        let mut orientations = Matrix1xX::zeros(data.own_players.len());
        for (i, player) in data.own_players.iter().enumerate() {
            positions[(0, i)] = player.position.x;
            positions[(1, i)] = player.position.y;
            orientations[(0, i)] = player.orientation;
        }

        Self {
            positions,
            orientations,
        }
    }

    pub fn step(
        &mut self,
        dt: f64,
        velocites: MatrixView2xX<f64>,
        angular_velocities: MatrixView1xX<f64>,
    ) {
        self.positions += velocites * dt;
        self.orientations += angular_velocities * dt;
    }
}

pub struct OppPlayersState {
    pub positions: Matrix2xX<f64>,
    pub velocities: Matrix2xX<f64>,
}

impl OppPlayersState {
    pub fn new(data: &WorldData) -> Self {
        let mut positions = Matrix2xX::zeros(data.opp_players.len());
        let mut velocities = Matrix2xX::zeros(data.opp_players.len());
        for (i, player) in data.opp_players.iter().enumerate() {
            positions[(0, i)] = player.position.x;
            positions[(1, i)] = player.position.y;
            velocities[(0, i)] = player.velocity.x;
            velocities[(1, i)] = player.velocity.y;
        }

        Self {
            positions,
            velocities,
        }
    }

    pub fn step(&mut self, dt: f64) {
        self.positions += self.velocities.scale(dt);
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
    pub own_players: OwnPlayersState,
    pub opp_players: OppPlayersState,
    pub ball: Option<BallState>,
}

impl State {
    pub fn new(world: &WorldData) -> Self {
        Self {
            own_players: OwnPlayersState::new(&world),
            opp_players: OppPlayersState::new(&world),
            ball: world.ball.as_ref().map(BallState::new),
        }
    }

    pub fn position(&self, player_idx: usize) -> Vector2 {
        self.own_players.positions.column(player_idx).into()
    }

    pub fn heading(&self, player_idx: usize) -> f64 {
        self.own_players.orientations[player_idx]
    }

    pub fn ball_position(&self) -> Option<Vector2> {
        self.ball.as_ref().map(|b| b.position)
    }

    pub fn step(
        &mut self,
        dt: f64,
        velocities: MatrixView2xX<f64>,
        angular_velocities: MatrixView1xX<f64>,
    ) {
        self.own_players.step(dt, velocities, angular_velocities);
        self.opp_players.step(dt);
        self.ball.as_mut().map(|ball| ball.step(dt));
    }
}
