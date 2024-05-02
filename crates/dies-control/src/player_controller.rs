use crate::pid::PID;
use dies_core::{PlayerCmd, PlayerData};
use nalgebra::Vector2;
use std::time::{Duration, Instant};

const MISSING_FRAMES_THRESHOLD: u32 = 50;

/// Input to the player controller.
#[derive(Debug, Clone, Default)]
pub struct PlayerControlInput {
    /// Player ID
    pub id: u32,
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
    /// Create a new instance of `PlayerControlInput` with the given ID
    pub fn new(id: u32) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    /// Set the target position of the player.
    pub fn with_position(mut self, pos: Vector2<f32>) -> Self {
        self.position = Some(pos);
        self
    }

    /// Set the target heading of the player.
    pub fn with_orientation(mut self, orientation: f32) -> Self {
        self.orientation = Some(orientation);
        self
    }

    /// Set the dribbling speed of the player.
    pub fn with_dribbling(mut self, speed: f32) -> Self {
        self.dribbling_speed = speed;
        self
    }

    /// Set the kicker control input.
    pub fn with_kicker(mut self, kicker: KickerControlInput) -> Self {
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

pub struct PlayerController {
    id: u32,
    position_pid: PID<Vector2<f32>>,
    heading_pid: PID<f32>,
    kick_timer: Option<Instant>,
    last_pos: Vector2<f32>,
    last_orientation: f32,
    frame_missings: u32,
    current_command: PlayerCmd,
}

impl PlayerController {
    /// Create a new player controller with the given ID.
    pub fn new(id: u32) -> Self {
        let heading_pid = PID::new(2.0, 0.002, 0.0);
        Self {
            id,
            position_pid: PID::new(0.7, 0.0, 0.0),
            heading_pid,
            kick_timer: None,
            last_pos: Vector2::new(0.0, 0.0),
            last_orientation: 0.0,
            frame_missings: 0,
            current_command: PlayerCmd::default(),
        }
    }

    /// Get the ID of the player.
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Get the current command for the player.
    pub fn command(&self) -> PlayerCmd {
        self.current_command.clone()
    }

    /// Increment the missing frame count, stops the robot if it is too high.
    pub fn increment_frames_missings(&mut self) {
        self.frame_missings += 1;
        if self.frame_missings > MISSING_FRAMES_THRESHOLD {
            self.current_command.sx = 0.0;
            self.current_command.sy = 0.0;
            self.current_command.w = 0.0;
        }
    }

    /// Update the controller with the current state of the player.
    pub fn update(&mut self, state: &PlayerData, input: PlayerControlInput) {
        self.current_command = PlayerCmd {
            id: self.id,
            ..Default::default()
        };

        // Calculate velocity using the PID controller
        self.last_orientation = state.orientation;
        self.last_pos = state.position;

        if let Some(pos_target) = input.position {
            self.position_pid.set_setpoint(pos_target);
            let pos_u = self.position_pid.update(self.last_pos);
            let local_u = rotate_vector(Vector2::new(pos_u.x, pos_u.y), -self.last_orientation);
            self.current_command.sx = local_u.x;
            self.current_command.sy = local_u.y;
        }
        let local_vel = rotate_vector(input.velocity, -self.last_orientation);
        self.current_command.sx += local_vel.x;
        self.current_command.sy += local_vel.y;

        if let Some(orientation) = input.orientation {
            self.heading_pid.set_setpoint(orientation);
            let head_u = self.heading_pid.update(self.last_orientation);
            self.current_command.w = head_u;
        }
        self.current_command.w += input.angular_velocity;

        // Set dribbling speed
        self.current_command.dribble_speed = input.dribbling_speed * 60.0;

        // Set kicker control
        match input.kicker {
            KickerControlInput::Arm => {
                if let Some(timer) = self.kick_timer {
                    // Safety: if it has been armed for more than 30s, disarm
                    if timer.elapsed() >= Duration::from_secs(30) {
                        self.current_command.disarm = true;
                        self.kick_timer = None;
                    } else {
                        self.current_command.arm = true;
                    }
                } else {
                    // Start arming
                    self.kick_timer = Some(Instant::now());
                    self.current_command.arm = true;
                }
            }
            KickerControlInput::Kick => {
                if self.kick_timer.is_none() {
                    self.kick_timer = Some(Instant::now());
                } else {
                    let elapsed = self.kick_timer.unwrap().elapsed();
                    if elapsed >= Duration::from_millis(1000) {
                        self.current_command.kick = true;
                        self.kick_timer = None;
                    }
                }
            }
            KickerControlInput::Disarm => {
                self.current_command.disarm = true;
            }
            _ => {}
        }
    }

    /// speed limit check
    pub fn speed_limit_check(&self, cmd: &mut PlayerCmd, velocity: Vector2<f32>) {
        // for example 1500 mm/s
        let max_speed = 1500.0;
        let speed = (velocity.x.powi(2) + velocity.y.powi(2)).sqrt();
        if speed > max_speed {
            let scale = max_speed / speed;
            cmd.sx *= scale;
            cmd.sy *= scale;
        }
    }

    /// out of bound detection
    pub fn out_of_bound_detection(&self) {
        // TODO:
        // what is the excepted behavior of this?? I think pid will drag it back when
        // it is out of bound, and since pid is the only way to control the robot, it may be unnecessary
        // to do this check.
    }
}

fn rotate_vector(v: Vector2<f32>, angle: f32) -> Vector2<f32> {
    let rot = nalgebra::Rotation2::new(angle);
    rot * v
}
