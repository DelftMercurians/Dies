use std::sync::{atomic::AtomicBool, Arc};

use dies_core::PlayerId;
use dies_core::{PlayerData, WorldData};
use nalgebra::Vector2;

use crate::strategy::Role;
use crate::PlayerControlInput;

static GOALKEEPER_ID: PlayerId = PlayerId::new(1);

struct Goalkeeper {
    has_passer_kicked: Arc<AtomicBool>,
}

impl Goalkeeper {
    fn find_intersection(&self, _player_data: &PlayerData, _world: &WorldData) -> Vector2<f64> {
        // Find the goalkeeper's position
        let goalkeeper_pos = _world
            .own_players
            .iter()
            .find(|p| p.id == GOALKEEPER_ID)
            .unwrap()
            .position;

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
        let receiver_pos = _world
            .own_players
            .iter()
            .find(|p| p.id == GOALKEEPER_ID)
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

impl Role for Goalkeeper {
    fn update(&mut self, _player_data: &PlayerData, _world: &WorldData) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();

        // don't move until the passer kicks the ball
        if !self
            .has_passer_kicked
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            println!("[RECEIVER]: Waiting for passer to kick the ball");
            return input;
        }

        let target_pos: nalgebra::Matrix<
            f64,
            nalgebra::Const<2>,
            nalgebra::Const<1>,
            nalgebra::ArrayStorage<f64, 2, 1>,
        >;
        target_pos = self.find_intersection(_player_data, _world);

        // let target_pos: nalgebra::Matrix<f64, nalgebra::Const<2>, nalgebra::Const<1>, nalgebra::ArrayStorage<f64, 2, 1>> = self.find_intersection(_player_data, _world);
        let target_angle = self.angle_to_ball(_player_data, _world);

        input.with_position(target_pos);
        input.with_orientation(target_angle);

        input
    }
}
