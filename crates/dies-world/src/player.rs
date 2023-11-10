use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PlayerData {
    // Whether the player is friendly -- on our team
    pub is_ours: bool,
    // Position of the player
    pub position: Vector2<f32>,
    // Velocity of the player
    pub velocity: Vector2<f32>,
    // Orientation of the player [-pi, pi]
    pub orientation: f32,
    // Angular speed of the player
    pub angular_speed: f32,
}

// Create tracker types for players and the ball.
#[derive(Serialize, Clone, Debug)]
pub struct PlayerTracker {
    position: Vector2<f32>,
    // Velocity of the player
    velocity: Vector2<f32>,
    // Orientation of the player [-pi, pi]
    orientation: f32,
    // Angular speed of the player
    angular_speed: f32,
}

impl PlayerTracker {
    pub fn new() -> PlayerTracker {
        PlayerTracker {
            position: Vector2::zeros(),
            velocity: Vector2::zeros(),
            orientation: 0.0,
            angular_speed: 0.0,
        }
    }

    pub fn update(&mut self, player_data: &PlayerData) {
        // Update the player tracker with new data.
        // Update fields in the PlayerTracker based on player_data.
        self.position = player_data.position.clone();
        self.velocity = player_data.velocity.clone();
        self.orientation = player_data.orientation;
        self.angular_speed = player_data.angular_speed;
    }
}
