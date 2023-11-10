pub mod workspace_utils;

pub type VisionMsg = dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
pub type GcRefereeMsg = dies_protos::ssl_gc_referee_message::Referee;

#[derive(Debug, Clone)]
pub enum EnvEvent {
    VisionMsg(VisionMsg),
    GcRefereeMsg(GcRefereeMsg),
}

/// A command to one of our players (placeholder)
///
/// All values are relative to the robots local frame and are in meters.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerCmd {
    /// The robot's ID
    pub id: u32,
    /// The player's x (left-right, with `+` left) velocity \[m/s]
    pub sx: f32,
    /// The player's y (forward-backward, with `+` forward) velocity \[m/s]
    pub sy: f32,
    /// The player's angular velocity (with `+` counter-clockwise, `-` clockwise) \[rad/s]
    pub w: f32,
}

/// A trait for all environments
pub trait Env {
    /// Receive zero or more events from the environment.
    ///
    /// This will block until at least one event is received.
    fn recv(&mut self) -> Vec<EnvEvent>;

    /// Send a command to a player
    fn send_player(&self, cmd: PlayerCmd) -> anyhow::Result<()>;
}
