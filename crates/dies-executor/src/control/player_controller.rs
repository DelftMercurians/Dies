use super::{
    pid::PID,
    player_input::{KickerControlInput, PlayerControlInput},
};
use dies_core::{Angle, PlayerCmd, PlayerData, PlayerId, Vector2};

const MISSING_FRAMES_THRESHOLD: usize = 50;
const MAX_DRIBBLE_SPEED: f64 = 100.0;

// maximum acceleration unit: mm/s2
const MAX_ACC: f64 = 500.0;

// maximum acceleration unit: radius/s2
const MAX_ACC_RADIUS: f64 = 1.5;

enum KickerState {
    Disarming,
    Arming,
    Kicking,
}

pub struct PlayerController {
    id: PlayerId,
    position_pid: PID<Vector2>,
    yaw_pid: PID<f64>,
    last_pos: Vector2,
    last_yaw: Angle,
    frame_missings: usize,

    /// Output velocity \[mm/s\]
    target_velocity: Vector2,
    /// Output angular velocity \[rad/s\]
    target_angular_velocity: f64,
    /// Kicker control
    kicker: KickerState,
    /// Dribble speed normalized to \[0, 1\]
    dribble_speed: f64,
}

impl PlayerController {
    /// Create a new player controller with the given ID.
    pub fn new(id: PlayerId) -> Self {
        let yaw_pid = PID::new(2.0, 0.002, 0.0);
        Self {
            id,
            position_pid: PID::new(0.7, 0.0, 0.0),
            yaw_pid,
            last_pos: Vector2::new(0.0, 0.0),
            last_yaw: Angle::from_radians(0.0),
            frame_missings: 0,
            target_velocity: Vector2::new(0.0, 0.0),
            target_angular_velocity: 0.0,
            kicker: KickerState::Disarming,
            dribble_speed: 0.0,
        }
    }

    /// Get the ID of the player.
    pub fn id(&self) -> PlayerId {
        self.id
    }

    /// Get the current command for the player.
    pub fn command(&mut self) -> PlayerCmd {
        let mut cmd = PlayerCmd {
            id: self.id,
            // In the robot's local frame +sx means forward and +sy means right
            sx: self.target_velocity.x / 1000.0, // Convert to m/s
            sy: self.target_velocity.y / 1000.0, // Convert to m/s
            w: self.target_angular_velocity,
            dribble_speed: self.dribble_speed * MAX_DRIBBLE_SPEED,
            arm: false,
            disarm: false,
            kick: false,
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
            log::warn!("Player {} has missing frames, stopping", self.id);
            self.target_velocity = Vector2::new(0.0, 0.0);
            self.target_angular_velocity = 0.0;
        }
    }

    /// Update the controller with the current state of the player.
    pub fn update(&mut self, state: &PlayerData, input: &PlayerControlInput, duration: f64) {
        // Calculate velocity using the PID controller
        self.last_yaw = state.yaw;
        self.last_pos = state.position;
        let last_vel: Vector2 = state.velocity;
        if let Some(pos_target) = input.position {
            self.position_pid.set_setpoint(pos_target);
            let pos_u = self.position_pid.update(self.last_pos);
            let local_u = self.last_yaw.inv().rotate_vector(&pos_u);
            self.target_velocity = local_u;
        }
        let local_vel = input.velocity.to_local(self.last_yaw);
        self.target_velocity += local_vel;

        // Cap the velocity
        let mut v_diff = self.target_velocity - last_vel;
        v_diff = v_diff.cap_magnitude(MAX_ACC * duration);
        self.target_velocity = last_vel + v_diff;

        let last_ang_vel = state.angular_speed;
        if let Some(yaw) = input.yaw {
            // TODO: Use Angle directly
            self.yaw_pid.set_setpoint(yaw.radians());
            let head_u = self.yaw_pid.update(self.last_yaw.radians());
            self.target_angular_velocity = head_u;
        }
        self.target_angular_velocity += input.angular_velocity;

        // Cap the angular velocity
        let ang_diff = self.target_angular_velocity - last_ang_vel;
        self.target_angular_velocity = last_ang_vel
            + ang_diff
                .max(-MAX_ACC_RADIUS * duration)
                .min(MAX_ACC_RADIUS * duration);

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
