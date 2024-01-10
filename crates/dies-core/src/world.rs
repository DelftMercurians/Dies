use nalgebra::{Vector2, Vector3};
use serde::Serialize;

use crate::FieldGeometry;

/// A struct to store the ball state from a single frame.
#[derive(Serialize, Clone, Debug)]
pub struct BallData {
    /// Unix timestamp of the recorded frame from which this data was extracted (in
    /// seconds). This is the time that ssl-vision received the frame.
    pub timestamp: f64,
    /// Position of the ball in mm, in dies coordinates
    pub position: Vector3<f32>,
    /// Velocity of the ball in mm/s, in dies coordinates
    pub velocity: Vector3<f32>,
}

/// A struct to store the player state from a single frame.
#[derive(Serialize, Clone, Debug)]
pub struct PlayerData {
    /// Unix timestamp of the recorded frame from which this data was extracted (in
    /// seconds). This is the time that ssl-vision received the frame.
    pub timestamp: f64,
    /// The player's unique id
    pub id: u32,
    /// Position of the player in mm, in dies coordinates
    pub position: Vector2<f32>,
    /// Velocity of the player in mm/s, in dies coordinates
    pub velocity: Vector2<f32>,
    /// Orientation of the player, in radians, (`-pi`, `pi`), where `0` is the positive
    /// y direction, and `pi/2` is the positive x direction (regardless of the goal
    /// sign).
    pub orientation: f32,
    /// Angular speed of the player (in rad/s)
    pub angular_speed: f32,
}

/// A struct to store the world state from a single frame.
#[derive(Serialize, Clone, Debug)]
pub struct WorldData {
    pub own_players: Vec<PlayerData>,
    pub opp_players: Vec<PlayerData>,
    pub ball: Option<BallData>,
    pub field_geom: FieldGeometry,
}
