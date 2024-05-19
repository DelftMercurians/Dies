use std::time::Duration;

use anyhow::Result;
use dies_core::{PlayerData, WorldData};
use dies_executor::{
    strategy::{AdHocStrategy, Role, Strategy},
    Executor, PlayerControlInput,
};
use dies_simulator::{SimulationBuilder, SimulationConfig};
use dies_webui::UiSettings;
use dies_world::WorldConfig;
use nalgebra::{Vector2, Vector3};
use tokio::sync::broadcast;
use dies_core::PlayerId;

struct Passer;
struct Receiver;
static PASSER_ID: PlayerId = PlayerId::new(0);
static RECEIVER_ID: PlayerId = PlayerId::new(14);


impl Receiver {
    fn normalized_perpendicular(&self, _player_data: &PlayerData, _world: &WorldData) -> Vector2<f64>{
        // Normalized perpendicular vector to the line between the reciever and the passer
        let passer_pos = _world.own_players.iter().find(|p| p.id == PASSER_ID).unwrap().position;
        let receiver_pos = _world.own_players.iter().find(|p| p.id == RECEIVER_ID).unwrap().position;
        let dx = receiver_pos.x - passer_pos.x;
        let dy = receiver_pos.y - passer_pos.y;
        // if dx == 0.0 && dy == 0.0 {
        //     return Vector2::new(-dy, dx);
        // }
        let normalized_perpendicular = Vector2::new(-dy, dx).normalize();
        return normalized_perpendicular;
    }

    fn find_intersection(&self, _player_data: &PlayerData, _world: &WorldData) -> Vector2<f64> {
        // Find the intersection point of the line between the ball and the passer and the line perpendicular to the player
        let receiver_pos = _world.own_players.iter().find(|p| p.id == RECEIVER_ID).unwrap().position;
        let normalized_perpendicular = self.normalized_perpendicular(_player_data, _world);
        let ball_pos = _world.ball.as_ref().unwrap().position;
        let ball_vel = _world.ball.as_ref().unwrap().velocity;
        let second_point = ball_pos - ball_vel;
        let fourth_point = receiver_pos + normalized_perpendicular;
        let x1 = ball_pos.x;
        let y1 = ball_pos.y;
        let x2 = second_point.x;
        let y2 = second_point.y;

        let x3 = receiver_pos.x;
        let y3 = receiver_pos.y;

        let x4 = fourth_point.x;
        let y4 = fourth_point.y;
        let denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

        if denominator == 0.0 {
            // println!("denominator is 0");
            return receiver_pos;
        }

        if ball_vel.norm() == 0.0 {
            // println!("ball velocity is 0");
            return Vector2::new(x1, x2);
        }

        let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator;
        let px = x1 + t * (x2 - x1);
        let py = y1 + t * (y2 - y1);

        return Vector2::new(px, py);
    }

    fn angle_to_ball(&self, _player_data: &PlayerData, _world: &WorldData) -> f64 {
        // Angle needed to face the passer (using arc tangent)
        let receiver_pos = _world.own_players.iter().find(|p| p.id == RECEIVER_ID).unwrap().position;
        let ball_pos = _world.ball.as_ref().unwrap().position;
        let dx = ball_pos.x - receiver_pos.x;
        let dy = ball_pos.y - receiver_pos.y;
        let angle = dy.atan2(dx);
        // println!("angle: {}", angle);

        dy.atan2(dx)
    }
    
}

impl Passer {
    fn angle_to_receiver(&self, _player_data: &PlayerData, _world: &WorldData) -> f64 {
        let receiver_pos = _world.own_players.iter().find(|p| p.id == RECEIVER_ID).unwrap().position;
        let passer_pos = _world.own_players.iter().find(|p| p.id == PASSER_ID).unwrap().position;
        let dx = receiver_pos.x - passer_pos.x;
        let dy = receiver_pos.y - passer_pos.y;
        // println!("dx: {}", dx);
        // println!("dy: {}", dy);
        if dx == 0.0 && dy == 0.0 {
            return 0.0;
        }

        dy.atan2(dx)
    }

}

impl Role for Receiver {
    // passer id = 15 using world data to get position of passer

    fn update(&mut self, _player_data: &PlayerData, _world: &WorldData) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();
        let target_pos: nalgebra::Matrix<f64, nalgebra::Const<2>, nalgebra::Const<1>, nalgebra::ArrayStorage<f64, 2, 1>> = self.find_intersection(_player_data, _world);
        let target_angle = self.angle_to_ball(_player_data, _world);
        // print!("target_pos: {}", target_pos);
        input.with_position(target_pos);
        input.with_orientation(target_angle);

        input
    }
}

impl Role for Passer {
    
    fn update(&mut self, _player_data: &PlayerData, _world: &WorldData) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();
        let target_angle = self.angle_to_receiver(_player_data, _world);
        input.with_orientation(target_angle);
        input
    }
}

pub async fn run(_args: crate::Args, stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let simulator = SimulationBuilder::new(SimulationConfig::default())
        .add_own_player_with_id(14, Vector2::new(2600.0, -1000.0), 0.0)
        .add_own_player_with_id(0, Vector2::new(-2500.0, 0.0), 0.0)
        .add_ball(Vector3::new(3000.0, 0.0, 0.0))
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
    let settings = UiSettings { can_control: true };
    tokio::spawn(async move {
        println!("Webui running at http://localhost:5555");
        dies_webui::start(settings, executor_rx, ui_command_tx, webui_shutdown_rx).await
    });

    // tokio::time::sleep(Duration::from_secs(30)).await;
    executor.run_real_time(stop_rx).await
}
