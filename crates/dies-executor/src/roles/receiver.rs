use std::sync::{atomic::AtomicBool, Arc};

use dies_core::{PlayerData, WorldData};
use nalgebra::Vector2;

use super::passer::{PASSER_ID, RECEIVER_ID};

pub struct Receiver {
    has_passer_kicked: Arc<AtomicBool>,
}

impl Receiver {
    fn normalized_perpendicular(
        &self,
        _player_data: &PlayerData,
        _world: &WorldData,
    ) -> Vector2<f64> {
        // Normalized perpendicular vector to the line between the reciever and the passer
        let passer_pos = _world
            .own_players
            .iter()
            .find(|p| p.id == PASSER_ID)
            .unwrap()
            .position;
        let receiver_pos = _world
            .own_players
            .iter()
            .find(|p| p.id == RECEIVER_ID)
            .unwrap()
            .position;
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
        let receiver_pos = _world
            .own_players
            .iter()
            .find(|p| p.id == RECEIVER_ID)
            .unwrap()
            .position;
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
        let receiver_pos = _world
            .own_players
            .iter()
            .find(|p| p.id == RECEIVER_ID)
            .unwrap()
            .position;
        let ball_pos = _world.ball.as_ref().unwrap().position;
        let dx = ball_pos.x - receiver_pos.x;
        let dy = ball_pos.y - receiver_pos.y;
        let angle = dy.atan2(dx);
        // println!("angle: {}", angle);

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
