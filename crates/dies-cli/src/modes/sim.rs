use std::time::Duration;

use anyhow::Result;
use dies_core::{PlayerData, WorldData};
use dies_executor::{
    strategy::{AdHocStrategy, Role, Strategy},
    Executor, PlayerControlInput,
};
use dies_simulator::{SimulationBuilder, SimulationConfig};
use dies_world::WorldConfig;
use nalgebra::{Vector2, Vector3};
use tokio::sync::broadcast;

struct Passer;
struct Receiver;

impl Receiver {
    fn normalized_perpendicular(&self, _player_data: &PlayerData, _world: &WorldData) -> Vector2<f32>{
        // Normalized perpendicular vector to the line between the reciever and the passer
        let passer_pos = _world.own_players.iter().find(|p| p.id == 15).unwrap().position;
        let dx = _player_data.position.x - passer_pos.x;
        let dy = _player_data.position.y - passer_pos.y;
        let normalized_perpendicular = Vector2::new(-dy, dx).normalize();
        return normalized_perpendicular;
    }

    fn find_intersection(&self, _player_data: &PlayerData, _world: &WorldData) -> Vector2<f32> {
        // Find the intersection point of the line between the ball and the passer and the line perpendicular to the player
        let normalized_perpendicular = self.normalized_perpendicular(_player_data, _world);
        let x1 = _world.ball.as_ref().unwrap().position.x;
        let y1 = _world.ball.as_ref().unwrap().position.y;
        let x2 = _world.ball.as_ref().unwrap().position.x - _world.ball.as_ref().unwrap().velocity.x;
        let y2 = _world.ball.as_ref().unwrap().position.y - _world.ball.as_ref().unwrap().velocity.y;
        let x3 = _player_data.position.x;
        let y3 = _player_data.position.y;
        let x4 = _player_data.position.x + normalized_perpendicular.x;
        let y4 = _player_data.position.y + normalized_perpendicular.y;
        let denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

        if denominator == 0.0 {
            return Vector2::new(0.0, 0.0);
        }

        let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator;
        let px = x1 + t * (x2 - x1);
        let py = y1 + t * (y2 - y1);

        return Vector2::new(px, py);
    }

    fn angle_to_passer(&self, _player_data: &PlayerData, _world: &WorldData) -> f32 {
        // Angle needed to face the passer (using arc tangent)
        let dx = _world.ball.as_ref().unwrap().position.x - _player_data.position.x;
        let dy = _world.ball.as_ref().unwrap().position.y - _player_data.position.y;
        return dy.atan2(dx);
    }
    
}

impl Passer {
    fn angle_to_receiver(&self, _player_data: &PlayerData, _world: &WorldData) -> f32 {
        let receiver_pos = _world.own_players.iter().find(|p| p.id == 16).unwrap().position;
        let dx = receiver_pos.x - _player_data.position.x;
        let dy = receiver_pos.y - _player_data.position.y;
        return dy.atan2(dx);
    }

}

impl Role for Receiver {
    // passer id = 15 using world data to get position of passer

    fn update(&mut self, _player_data: &PlayerData, _world: &WorldData) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();
        let target_pos = self.find_intersection(_player_data, _world);
        let target_angle = self.angle_to_passer(_player_data, _world);
        input.with_position(target_pos);
        input.with_orientation(target_angle);
        return input;
    }
}

impl Role for Passer {
    
    fn update(&mut self, _player_data: &PlayerData, _world: &WorldData) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();
        let target_angle = self.angle_to_receiver(_player_data, _world);
        input.with_orientation(target_angle);
        return input;
    }
}

pub async fn run(_args: crate::Args, stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let simulator = SimulationBuilder::new(SimulationConfig::default())
        .add_own_player(Vector2::new(0.0, -1000.0), 0.0)
        .add_own_player_with_id(14, Vector2::new(0.0, 1000.0), 0.0)
        .add_ball(Vector3::zeros())
        .build();
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Receiver));
    strategy.add_role(Box::new(Passer));

    let mut builder = Executor::builder();
    builder.with_world_config(WorldConfig {
        is_blue: true,
        initial_opp_goal_x: 1.0,
    });
    builder.with_strategy(Box::new(strategy) as Box<dyn Strategy>);
    builder.with_simulator(simulator, Duration::from_millis(10));

    let executor = builder.build()?;

    // Spawn webui
    let executor_rx = executor.subscribe();
    let (ui_command_tx, _) = broadcast::channel(16);
    let webui_shutdown_rx = stop_rx.resubscribe();
    tokio::spawn(
        async move { dies_webui::start(executor_rx, ui_command_tx, webui_shutdown_rx).await },
    );

    executor.run_real_time(stop_rx).await
}
