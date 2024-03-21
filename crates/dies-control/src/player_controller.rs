use dies_core::{PlayerCmd, PlayerData};
use nalgebra::Vector2;

use crate::pid::PID;
pub struct PlayerController {
    id: u32,
    position_pid: PID<Vector2<f32>>,
    heading_pid: PID<f32>,
}

impl PlayerController {
    /// Create a new player controller with the given ID.
    pub fn new(id: u32, pos: Vector2<f32>) -> Self {
        let mut heading_pid = PID::new(2.0, 0.002, 0.0);
        // Fix the heading for now
        heading_pid.set_setpoint(0.0);
        Self {
            id,
            position_pid: PID::new(0.7, 0.0, 0.0),
            heading_pid,
        }
    }

    /// Set the target position for the player.
    pub fn set_target_pos(&mut self, setpoint: Vector2<f32>) {
        self.position_pid.set_setpoint(setpoint);
    }

    /// check if the player is in the target position
    pub fn is_in_target_pos(&self, state: &PlayerData) -> bool {
        let eps: f32 = 10.0; // 1cm
        let setpoint = self.position_pid.get_setpoint();
        let error = setpoint - state.position;
        error.norm() < eps
    }

    /// Set the target heading for the player.
    pub fn set_target_heading(&mut self, setpoint: f32) {
        self.heading_pid.set_setpoint(setpoint);
    }

    /// check if the player is in the target heading
    pub fn is_in_target_heading(&self, state: &PlayerData) -> bool {
        let eps: f32 = 0.1; // 0.1 rad
        let setpoint = self.heading_pid.get_setpoint();
        let error = setpoint - state.orientation;
        error.abs() < eps
    }

    /// Update the controller with the current state of the player.
    pub fn update(&mut self, state: &PlayerData) -> PlayerCmd {
        let pos_u = self.position_pid.update(state.position);
        let head_u = self.heading_pid.update(state.orientation);
        let local_u = rotate_vector(Vector2::new(pos_u.x, pos_u.y), -state.orientation);
            PlayerCmd {
            id: self.id,
            sx: local_u.x,
            sy: local_u.y,
            w: head_u,
            ..Default::default()
        }
    }

    // stop the player
    pub fn stop(&mut self) -> PlayerCmd {
        PlayerCmd::zero(self.id)
    }


}

fn rotate_vector(v: Vector2<f32>, angle: f32) -> Vector2<f32> {
    let rot = nalgebra::Rotation2::new(angle);
    rot * v
}
