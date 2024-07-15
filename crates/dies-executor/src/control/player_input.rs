use std::collections::HashMap;

use dies_core::{Angle, PlayerId, Vector2};

/// A collection of player inputs.
pub struct PlayerInputs {
    inputs: HashMap<PlayerId, PlayerControlInput>,
}

impl Default for PlayerInputs {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayerInputs {
    /// Create a new instance of `PlayerInputs`.
    pub fn new() -> Self {
        Self {
            inputs: HashMap::with_capacity(6),
        }
    }

    /// Get an iterator over the player inputs.
    pub fn iter(&self) -> impl Iterator<Item = (&PlayerId, &PlayerControlInput)> {
        self.inputs.iter()
    }

    /// Get the mutable input for a player, creating a new one if it doesn't exist.
    pub fn player_mut(&mut self, id: PlayerId) -> &mut PlayerControlInput {
        self.inputs.entry(id).or_default()
    }

    /// Get the input for a player, or an empty one if it doesn't exist.
    pub fn player(&self, id: PlayerId) -> PlayerControlInput {
        self.inputs
            .get(&id)
            .cloned()
            .unwrap_or(PlayerControlInput::new())
    }

    /// Set the input for a player
    pub fn insert(&mut self, id: PlayerId, input: PlayerControlInput) {
        self.inputs.insert(id, input);
    }
}

impl IntoIterator for PlayerInputs {
    type Item = (PlayerId, PlayerControlInput);
    type IntoIter = std::collections::hash_map::IntoIter<PlayerId, PlayerControlInput>;

    fn into_iter(self) -> Self::IntoIter {
        self.inputs.into_iter()
    }
}

impl FromIterator<(PlayerId, PlayerControlInput)> for PlayerInputs {
    fn from_iter<T: IntoIterator<Item = (PlayerId, PlayerControlInput)>>(iter: T) -> Self {
        Self {
            inputs: iter.into_iter().collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Velocity {
    Global(Vector2),
    Local(Vector2),
}

impl Velocity {
    pub fn local(v: Vector2) -> Self {
        Self::Local(v)
    }

    pub fn global(v: Vector2) -> Self {
        Self::Global(v)
    }

    pub fn to_local(&self, yaw: Angle) -> Vector2 {
        match self {
            Self::Global(v) => yaw.inv().rotate_vector(v),
            Self::Local(v) => *v,
        }
    }

    pub fn map(&self, f: impl FnOnce(&Vector2) -> Vector2) -> Self {
        match self {
            Velocity::Global(v) => Velocity::Global(f(v)),
            Velocity::Local(v) => Velocity::Local(f(v)),
        }
    }

    pub fn cap_magnitude(&self, max: f64) -> Self {
        self.map(|v| v.cap_magnitude(max))
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Velocity::Global(v) => v.norm() < 1e-6,
            Velocity::Local(v) => v.norm() < 1e-6,
        }
    }
}

impl Default for Velocity {
    fn default() -> Self {
        Self::Local(Vector2::zeros())
    }
}

/// Input to the player controller.
#[derive(Debug, Clone, Default)]
pub struct PlayerControlInput {
    /// Target position. If `None`, the player will just follow the given velocity
    pub position: Option<Vector2>,
    /// Target velocity. This is added to the output of the position
    /// controller.
    pub velocity: Velocity,
    /// Target yaw. If `None` the player will just follow the given angula
    /// velocity
    pub yaw: Option<Angle>,
    /// Target angular velocity. This is added to the output of the controller.
    pub angular_velocity: f64,
    /// Dribbler speed normalised to \[0, 1\]
    pub dribbling_speed: f64,
    /// Kicker control input
    pub kicker: KickerControlInput,
}

impl PlayerControlInput {
    /// Create a new instance of `PlayerControlInput`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target position of the player.
    pub fn with_position(&mut self, pos: Vector2) -> &mut Self {
        self.position = Some(pos);
        self
    }

    /// Set the target yaw of the player.
    pub fn with_yaw(&mut self, yaw: Angle) -> &mut Self {
        self.yaw = Some(yaw);
        self
    }

    /// Set the dribbling speed of the player.
    pub fn with_dribbling(&mut self, speed: f64) -> &mut Self {
        self.dribbling_speed = speed;
        self
    }

    /// Set the kicker control input.
    pub fn with_kicker(&mut self, kicker: KickerControlInput) -> &mut Self {
        self.kicker = kicker;
        self
    }

    /// Set the target velocity of the player.
    pub fn add_global_velocity(&mut self, vel: Vector2) -> &mut Self {
        self.velocity = match self.velocity {
            Velocity::Global(v) => Velocity::Global(v + vel),
            Velocity::Local(_) => Velocity::Global(vel),
        };

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
