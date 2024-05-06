use super::{
    pid::PID,
    player_input::{KickerControlInput, PlayerControlInput},
};
use dies_core::{PlayerCmd, PlayerData};
use nalgebra::Vector2;

const MISSING_FRAMES_THRESHOLD: u32 = 50;
const MAX_DRIBBLE_SPEED: f32 = 100.0;

enum KickerState {
    Disarming,
    Arming,
    Kicking,
}

pub struct PlayerController {
    id: u32,
    position_pid: PID<Vector2<f32>>,
    heading_pid: PID<f32>,
    last_pos: Vector2<f32>,
    last_orientation: f32,
    frame_missings: u32,

    /// Output velocity \[mm/s\]
    target_velocity: Vector2<f32>,
    /// Output angular velocity \[rad/s\]
    target_angular_velocity: f32,
    /// Kicker control
    kicker: KickerState,
    /// Dribble speed normalized to \[0, 1\]
    dribble_speed: f32,
}

impl PlayerController {
    /// Create a new player controller with the given ID.
    pub fn new(id: u32) -> Self {
        let heading_pid = PID::new(2.0, 0.002, 0.0);
        Self {
            id,
            position_pid: PID::new(0.7, 0.0, 0.0),
            heading_pid,
            last_pos: Vector2::new(0.0, 0.0),
            last_orientation: 0.0,
            frame_missings: 0,
            target_velocity: Vector2::new(0.0, 0.0),
            target_angular_velocity: 0.0,
            kicker: KickerState::Disarming,
            dribble_speed: 0.0,
        }
    }

    /// Get the ID of the player.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Get the current command for the player.
    pub fn command(&mut self) -> PlayerCmd {
        let mut cmd = PlayerCmd {
            id: self.id,
            // In the robot's local frame +sx means left, +sy means forward, so we
            // need to swap the x and y components here.
            sy: self.target_velocity.x / 1000.0, // Convert to m/s
            sx: self.target_velocity.y / 1000.0, // Convert to m/s
            w: self.target_angular_velocity,
            dribble_speed: self.dribble_speed * MAX_DRIBBLE_SPEED,
            ..Default::default()
        };

        match self.kicker {
            KickerState::Arming => {
                cmd.arm = true;
            }
            KickerState::Kicking => {
                cmd.kick = true;
                self.kicker = KickerState::Disarming;
            }
            _ => {}
        }

        cmd
    }

    /// Increment the missing frame count, stops the robot if it is too high.
    pub fn increment_frames_missings(&mut self) {
        self.frame_missings += 1;
        if self.frame_missings > MISSING_FRAMES_THRESHOLD {
            tracing::warn!("Player {} has missing frames, stopping", self.id);
            self.target_velocity = Vector2::new(0.0, 0.0);
            self.target_angular_velocity = 0.0;
        }
    }

    /// Update the controller with the current state of the player.
    pub fn update(&mut self, state: &PlayerData, input: PlayerControlInput) {
        // Calculate velocity using the PID controller
        self.last_orientation = state.orientation;
        self.last_pos = state.position;

        if let Some(pos_target) = input.position {
            self.position_pid.set_setpoint(pos_target);
            let pos_u = self.position_pid.update(self.last_pos);
            let local_u = rotate_vector(pos_u, -self.last_orientation);
            self.target_velocity = local_u;
        }
        let local_vel = rotate_vector(input.velocity, -self.last_orientation);
        self.target_velocity += local_vel;

        if let Some(orientation) = input.orientation {
            self.heading_pid.set_setpoint(orientation);
            let head_u = self.heading_pid.update(self.last_orientation);
            self.target_angular_velocity = head_u;
        }
        self.target_angular_velocity += input.angular_velocity;

        // Set dribbling speed
        self.dribble_speed = input.dribbling_speed;

        // Set kicker control
        match input.kicker {
            KickerControlInput::Arm => {
                self.kicker = KickerState::Arming;
            }
            KickerControlInput::Kick => {
                self.kicker = KickerState::Kicking;
            }
            _ => {
                self.kicker = KickerState::Disarming;
            }
        }
    }
}

fn rotate_vector(v: Vector2<f32>, angle: f32) -> Vector2<f32> {
    let rot = nalgebra::Rotation2::new(angle);
    rot * v
}
