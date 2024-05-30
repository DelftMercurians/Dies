use std::{sync::{atomic::AtomicBool, Arc}, time::Duration};

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
use tokio::{sync::broadcast, time::Instant};
use dies_core::PlayerId;

struct Passer{
    timestamp: Instant,
    is_armed: bool,
    has_kicked: Arc<AtomicBool>,
}
struct Receiver{
    has_passer_kicked: Arc<AtomicBool>,
}
static PASSER_ID: PlayerId = PlayerId::new(0);
static RECEIVER_ID: PlayerId = PlayerId::new(5);
static GOALKEEPER_ID: PlayerId = PlayerId::new(1);

struct Goalkeeper{
    has_passer_kicked: Arc<AtomicBool>,
}


impl Goalkeeper {

    fn find_intersection(&self, _player_data: &PlayerData, _world: &WorldData) -> Vector2<f64> {
        // Find the goalkeeper's position
        let goalkeeper_pos = _world.own_players.iter().find(|p| p.id == GOALKEEPER_ID).unwrap().position;
    
        // Get the ball's current position and velocity
        let ball_pos = _world.ball.as_ref().unwrap().position;
        let ball_vel = _world.ball.as_ref().unwrap().velocity;
    
        // Calculate the second point based on the ball's trajectory
        let second_point = ball_pos - ball_vel;
    
        // Coordinates of the ball's trajectory points
        let x1 = ball_pos.x;
        let y1 = ball_pos.y;
        let x2 = second_point.x;
        let y2 = second_point.y;
    
        // Vertical line's x-coordinate is the goalkeeper's x-coordinate
        // Maybe it would be better to get a static value so that it doesn't deviate over time...
        let x_vertical = goalkeeper_pos.x;
    
        // Handle the special case where the ball's trajectory is vertical
        if x1 == x2 {
            return Vector2::new(x_vertical, ball_pos.y);
        }
    
        // Calculate the parameter t to find the y-coordinate at the intersection with the vertical line
        let t = (x_vertical - x1) / (x2 - x1);
        let py = y1 + t * (y2 - y1);
    
        // Get the goal width from the world data
        let goal_width = _world.field_geom.as_ref().unwrap().goal_width as f64;
        // let goal_width = 1000.0;
        // print!("goal width: {}", goal_width);
    
        // Calculate the min and max y-coordinates the goalkeeper can move to, constrained by the goal width
        let y_min = goalkeeper_pos.y - goal_width / 2.0;
        let y_max = goalkeeper_pos.y + goal_width / 2.0;
    
        // Clamp the y-coordinate to ensure it stays within the goal width range
        let py_clamped = py.max(y_min).min(y_max);
    
        // Return the clamped intersection point
        return Vector2::new(x_vertical, py_clamped);
    }

    fn angle_to_ball(&self, _player_data: &PlayerData, _world: &WorldData) -> f64 {
        // Angle needed to face the passer (using arc tangent)
        let receiver_pos = _world.own_players.iter().find(|p| p.id == GOALKEEPER_ID).unwrap().position;
        let ball_pos = _world.ball.as_ref().unwrap().position;
        let dx = ball_pos.x - receiver_pos.x;
        let dy = ball_pos.y - receiver_pos.y;
        let angle = dy.atan2(dx);
        // println!("angle: {}", angle);

        dy.atan2(dx)
    }
    
}


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
            println!("denominator is 0");
            return receiver_pos;
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

// impl Role for Receiver {

//     fn update(&mut self, _player_data: &PlayerData, _world: &WorldData) -> PlayerControlInput {
//         let mut input = PlayerControlInput::new();

//         // don't move until the passer kicks the ball
//         if !self.has_passer_kicked.load(std::sync::atomic::Ordering::Relaxed) {
//             println!("[RECEIVER]: Waiting for passer to kick the ball");
//             return input;
//         }
        
//         let target_pos: nalgebra::Matrix<f64, nalgebra::Const<2>, nalgebra::Const<1>, nalgebra::ArrayStorage<f64, 2, 1>>;

//         let ball_pos = _world.ball.as_ref().unwrap().position;
//         let ball_vel = _world.ball.as_ref().unwrap().velocity;
        
//         //println!("ball velocity: {}", Vector2::new(ball_vel.x, ball_vel.y).norm());
//         let ball_vel_norm = Vector2::new(ball_vel.x, ball_vel.y).norm();
//         let ball_vel_threshold = 90.0;
//         if ball_vel_norm < ball_vel_threshold {
//             println!("[RECEIVER]: ball velocity is below {}", ball_vel_threshold);
//             target_pos = Vector2::new(ball_pos.x, ball_pos.y);
//         } else {
//             target_pos = self.find_intersection(_player_data, _world);
//         }

//         let target_pos = Vector2::new(0.0,0.0);
//         // let target_pos: nalgebra::Matrix<f64, nalgebra::Const<2>, nalgebra::Const<1>, nalgebra::ArrayStorage<f64, 2, 1>> = self.find_intersection(_player_data, _world);
//         let target_angle = self.angle_to_ball(_player_data, _world);
//         print!("target_pos: {}", target_pos);

//         input.with_position(target_pos);
//         input.with_orientation(target_angle);

//         input
//     }
// }

impl Role for Passer {

    // assume for now that we stand close to the ball
    // and we can kick it after a few seconds

    fn update(&mut self, _player_data: &PlayerData, _world: &WorldData) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();
        // let target_angle = self.angle_to_receiver(_player_data, _world);
        // input.with_orientation(target_angle);
        
        if (self.timestamp.elapsed().as_secs() > 3) && !self.is_armed {
            self.is_armed = true;
            self.timestamp = Instant::now();
            let kicker = dies_executor::KickerControlInput::Arm;
            
            println!("[PASSER]:Armed");
            input.with_kicker(kicker);

            return input;
        } else if  self.timestamp.elapsed().as_secs() > 1 &&  self.is_armed && !self.has_kicked.load(std::sync::atomic::Ordering::Relaxed) {
            
            input.with_dribbling(0.0);

            self.has_kicked.store(true, std::sync::atomic::Ordering::Relaxed);
            self.timestamp = Instant::now();
            
            let kicker = dies_executor::KickerControlInput::Kick;
            input.with_kicker(kicker);
            
            println!("[PASSER]: Kicked");

            return input
        } else if self.timestamp.elapsed().as_secs_f64() < 1.1 && self.has_kicked.load(std::sync::atomic::Ordering::Relaxed) {
            
            // kick for 0.1 seconds
            
            input.with_dribbling(0.0);
            let kicker = dies_executor::KickerControlInput::Kick;
            input.with_kicker(kicker);
            
            println!("[PASSER]: Kicked2");

            return input
            
        }
        input

    }
}

impl Role for Goalkeeper {

    fn update(&mut self, _player_data: &PlayerData, _world: &WorldData) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();

        // don't move until the passer kicks the ball
        if !self.has_passer_kicked.load(std::sync::atomic::Ordering::Relaxed) {
            println!("[RECEIVER]: Waiting for passer to kick the ball");
            return input;
        }
        
        let target_pos: nalgebra::Matrix<f64, nalgebra::Const<2>, nalgebra::Const<1>, nalgebra::ArrayStorage<f64, 2, 1>>;
        target_pos = self.find_intersection(_player_data, _world);

        // let target_pos: nalgebra::Matrix<f64, nalgebra::Const<2>, nalgebra::Const<1>, nalgebra::ArrayStorage<f64, 2, 1>> = self.find_intersection(_player_data, _world);
        let target_angle = self.angle_to_ball(_player_data, _world);

        input.with_position(target_pos);
        input.with_orientation(target_angle);

        input
    }
}

pub async fn run(_args: crate::Args, stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let simulator = SimulationBuilder::new(SimulationConfig::default())
        .add_own_player_with_id(GOALKEEPER_ID.as_u32(), Vector2::new(2600.0, -1000.0), 0.0)
        .add_own_player_with_id(PASSER_ID.as_u32(), Vector2::new(-510.0, 0.0), 0.0) // Position was -1245.0, 0.0
        .add_ball(Vector3::new(-265.0, 0.0, 0.0)) // Position before was -1000.0 0.0
        .build();
    let mut strategy = AdHocStrategy::new();

    // use Arc to share the state between the passer and the receiver
    let has_kicked_communication = Arc::new(AtomicBool::new(false));
    strategy.add_role_with_id(GOALKEEPER_ID, Box::new(Goalkeeper{has_passer_kicked: has_kicked_communication.clone()}));
    let timestamp_instant = tokio::time::Instant::now();
    strategy.add_role_with_id(PASSER_ID, Box::new(Passer{timestamp: timestamp_instant, is_armed: false, has_kicked: has_kicked_communication.clone()}));

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
