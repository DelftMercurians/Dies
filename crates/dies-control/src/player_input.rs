use std::collections::HashMap;

use nalgebra::Vector2;

/// A collection of player inputs.
pub struct PlayerInputs {
    inputs: HashMap<u32, PlayerControlInput>,
}

impl PlayerInputs {
    /// Create a new instance of `PlayerInputs`.
    pub fn new() -> Self {
        Self {
            inputs: HashMap::with_capacity(6),
        }
    }

    /// Get an iterator over the player inputs.
    pub fn iter(&self) -> impl Iterator<Item = (&u32, &PlayerControlInput)> {
        self.inputs.iter()
    }

    /// Get the mutable input for a player, creating a new one if it doesn't exist.
    pub fn player_mut(&mut self, id: u32) -> &mut PlayerControlInput {
        self.inputs.entry(id).or_insert(PlayerControlInput::new())
    }

    /// Get the input for a player, or an empty one if it doesn't exist.
    pub fn player(&self, id: u32) -> PlayerControlInput {
        self.inputs
            .get(&id)
            .cloned()
            .unwrap_or(PlayerControlInput::new())
    }
}

impl IntoIterator for PlayerInputs {
    type Item = (u32, PlayerControlInput);
    type IntoIter = std::collections::hash_map::IntoIter<u32, PlayerControlInput>;

    fn into_iter(self) -> Self::IntoIter {
        self.inputs.into_iter()
    }
}

impl FromIterator<(u32, PlayerControlInput)> for PlayerInputs {
    fn from_iter<T: IntoIterator<Item = (u32, PlayerControlInput)>>(iter: T) -> Self {
        Self {
            inputs: iter.into_iter().collect(),
        }
    }
}

/// Input to the player controller.
#[derive(Debug, Clone, Default)]
pub struct PlayerControlInput {
    /// Target position. If `None`, the player will just follow the given velocity
    pub position: Option<Vector2<f32>>,
    /// Target velocity (in global frame). This is added to the output of the position
    /// controller.
    pub velocity: Vector2<f32>,
    /// Target orientation. If `None` the player will just follow the given angula
    /// velocity
    pub orientation: Option<f32>,
    /// Target angular velocity. This is added to the output of the controller.
    pub angular_velocity: f32,
    /// Dribbler speed normalised to [0, 1]
    pub dribbling_speed: f32,
    /// Kicker control input
    pub kicker: KickerControlInput,
}

impl PlayerControlInput {
    /// Create a new instance of `PlayerControlInput`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target position of the player.
    pub fn with_position(&mut self, pos: Vector2<f32>) -> &mut Self {
        self.position = Some(pos);
        self
    }

    /// Set the target heading of the player.
    pub fn with_orientation(&mut self, orientation: f32) -> &mut Self {
        self.orientation = Some(orientation);
        self
    }

    /// Set the dribbling speed of the player.
    pub fn with_dribbling(&mut self, speed: f32) -> &mut Self {
        self.dribbling_speed = speed;
        self
    }

    /// Set the kicker control input.
    pub fn with_kicker(&mut self, kicker: KickerControlInput) -> &mut Self {
        self.kicker = kicker;
        self
    }
}

/// Kicker state in the current update.
#[derive(Debug, Clone, Default)]
pub enum KickerControlInput {
    /// Kicker is not used
    #[default]
    Idle,
    /// Charge the kicker capacitor
    Arm,
    /// Engage the kicker. Should be sent after ~10s of charging and only once.
    Kick,
    /// Discharge the kicker capacitor without kicking
    Disarm,
}