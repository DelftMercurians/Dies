use std::time::Duration;

use super::{
    mtp::MTP,
    player_input::{KickerControlInput, PlayerControlInput},
};
use dies_core::{Angle, ControllerSettings, KickerCmd, PlayerCmd, PlayerData, PlayerId, Vector2};

const MISSING_FRAMES_THRESHOLD: usize = 50;
const MAX_DRIBBLE_SPEED: f64 = 100.0;

enum KickerState {
    Disarming,
    Arming,
    Kicking,
}

pub struct PlayerController {
    id: PlayerId,
    position_mtp: MTP<Vector2>,
    last_pos: Vector2,
    /// Output velocity \[mm/s\]
    target_velocity: Vector2,

    yaw_mtp: MTP<f64>,
    last_yaw: Angle,
    /// Output angular velocity \[rad/s\]
    target_angular_velocity: f64,

    frame_misses: usize,

    /// Kicker control
    kicker: KickerState,
    /// Dribble speed normalized to \[0, 1\]
    dribble_speed: f64,
}

impl PlayerController {
    /// Create a new player controller with the given ID.
    pub fn new(id: PlayerId, settings: &ControllerSettings) -> Self {
        let mut instance = Self {
            id,

            position_mtp: MTP::new(
                settings.max_acceleration,
                settings.max_velocity,
                settings.max_deceleration,
            ),
            last_pos: Vector2::new(0.0, 0.0),
            target_velocity: Vector2::new(0.0, 0.0),

            yaw_mtp: MTP::new(
                settings.max_angular_acceleration,
                settings.max_angular_velocity,
                settings.max_angular_deceleration,
            ),
            last_yaw: Angle::from_radians(0.0),
            target_angular_velocity: 0.0,

            frame_misses: 0,
            kicker: KickerState::Disarming,
            dribble_speed: 0.0,
        };
        instance.update_settings(settings);
        instance
    }

    /// Update the controller settings.
    pub fn update_settings(&mut self, settings: &ControllerSettings) {
        self.position_mtp.update_settings(
            settings.max_acceleration,
            settings.max_velocity,
            settings.max_deceleration,
            settings.position_kp,
            Duration::from_secs_f64(settings.position_proportional_time_window),
            settings.position_cutoff_distance,
        );
        self.yaw_mtp.update_settings(
            settings.max_angular_acceleration,
            settings.max_angular_velocity,
            settings.max_angular_deceleration,
            settings.angle_kp,
            Duration::from_secs_f64(settings.angle_proportional_time_window),
            settings.angle_cutoff_distance,
        );
    }

    /// Get the ID of the player.
    pub fn id(&self) -> PlayerId {
        self.id
    }

    /// Get the current target velocity of the player.
    pub(super) fn target_velocity(&self) -> Vector2 {
        self.target_velocity
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
            kicker_cmd: KickerCmd::None,
        };

        match self.kicker {
            KickerState::Arming => {
                cmd.kicker_cmd = KickerCmd::Arm;
            }
            KickerState::Kicking => {
                cmd.kicker_cmd = KickerCmd::Kick;
                self.kicker = KickerState::Disarming;
            }
            _ => {}
        }

        dies_core::debug_value(format!("p{}.sx", self.id), cmd.sx);
        dies_core::debug_value(format!("p{}.sy", self.id), cmd.sy);
        dies_core::debug_value(format!("p{}.w", self.id), cmd.w);
        dies_core::debug_value(format!("p{}.dribble_speed", self.id), cmd.dribble_speed);
        dies_core::debug_string(
            format!("p{}.kicker_cmd", self.id),
            format!("{:?}", cmd.kicker_cmd),
        );

        cmd
    }

    /// Increment the missing frame count, stops the robot if it is too high.
    pub fn increment_frames_misses(&mut self) {
        self.frame_misses += 1;
        if self.frame_misses > MISSING_FRAMES_THRESHOLD {
            log::warn!("Player {} has missing frames, stopping", self.id);
            self.target_velocity = Vector2::new(0.0, 0.0);
            self.target_angular_velocity = 0.0;
        }
    }

    /// Update the controller with the current state of the player.
    pub fn update(&mut self, state: &PlayerData, input: &PlayerControlInput, dt: f64) {
        // Calculate velocity using the MTP controller
        self.last_yaw = state.raw_yaw;
        self.last_pos = state.position;
        self.target_velocity = Vector2::zeros();
        if let Some(pos_target) = input.position {
            self.position_mtp.set_setpoint(pos_target);
            dies_core::debug_cross(
                format!("p{}.position_target", self.id),
                pos_target,
                dies_core::DebugColor::Purple,
            );
            let pos_u = self.position_mtp.update(self.last_pos, state.velocity, dt);
            let local_u = self.last_yaw.inv().rotate_vector(&pos_u);
            self.target_velocity = local_u;
        } else {
            dies_core::debug_remove(format!("p{}.position_target", self.id));
        }
        let local_vel = input.velocity.to_local(self.last_yaw);
        self.target_velocity += local_vel;

        // Draw the velocity
        dies_core::debug_line(
            format!("p{}.target_velocity", self.id),
            self.last_pos,
            self.last_pos + self.last_yaw.rotate_vector(&self.target_velocity),
            dies_core::DebugColor::Red,
        );

        self.target_angular_velocity = 0.0;
        dies_core::debug_value(
            format!("p{}.angular_speed", self.id),
            state.angular_speed.to_degrees(),
        );
        if let Some(yaw) = input.yaw {
            // draw target yaw
            dies_core::debug_value(format!("p{}.target_yaw", self.id), yaw.degrees());
            dies_core::debug_line(
                format!("p{}.target_yaw_line", self.id),
                self.last_pos,
                self.last_pos + yaw.rotate_vector(&Vector2::new(200.0, 0.0)),
                dies_core::DebugColor::Green,
            );

            // TODO: Use Angle directly
            self.yaw_mtp.set_setpoint(yaw.radians());
            let head_u = self
                .yaw_mtp
                .update(self.last_yaw.radians(), state.angular_speed, dt);
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

    pub fn update_target_velocity_with_avoidance(&mut self, target_velocity: Vector2) {
        self.target_velocity = target_velocity;
        dies_core::debug_line(
            format!("p{}.target_velocity", self.id),
            self.last_pos,
            self.last_pos + self.last_yaw.rotate_vector(&self.target_velocity),
            dies_core::DebugColor::Red,
        );
    }
}
