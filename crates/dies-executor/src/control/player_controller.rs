use std::{
    f64::{consts::PI, EPSILON},
    time::{Duration, Instant},
};

use dies_core::{
    to_dies_coords2, to_dies_yaw, Angle, ControllerSettings, ExecutorSettings, Obstacle, PlayerCmd,
    PlayerData, PlayerId, PlayerMoveCmd, RobotCmd, SysStatus, Vector2, WorldData,
};

use super::{
    mtp::MTP,
    player_input::{KickerControlInput, PlayerControlInput},
    rvo::velocity_obstacle_update,
    yaw_control::YawController,
    Velocity,
};

const MISSING_FRAMES_THRESHOLD: usize = 50;
const MAX_DRIBBLE_SPEED: f64 = 1_000.0;

#[derive(Debug, Clone, Copy, PartialEq)]
enum KickerState {
    Disarming,
    Arming,
    Kicking,
}

pub struct PlayerController {
    id: PlayerId,
    position_mtp: MTP,
    last_pos: Vector2,
    if_gate_keeper: bool,
    /// Output velocity \[mm/s\]
    target_velocity: Vector2,

    yaw_control: YawController,
    last_yaw: Angle,
    /// Output angular velocity or heading
    target_z: f64,

    frame_misses: usize,

    /// Kicker control
    kicker: KickerState,
    /// Dribble speed normalized to \[0, 1\]
    dribble_speed: f64,

    max_accel: f64,
    max_speed: f64,
    max_decel: f64,
    max_angular_velocity: f64,
    max_angular_acceleration: f64,

    have_imu: bool,
    has_target_heading: bool,
    /// If Some(true) we need to switch to imu heading, if Some(false) we need to switch
    /// to yaw control, and if None we don't need to do anything.
    switch_heading: Option<bool>,
    heading_interval: IntervalTrigger,

    opp_goal_sign: f64,
}

impl PlayerController {
    pub fn set_opp_goal_sign(&mut self, opp_goal_sign: f64) {
        self.opp_goal_sign = opp_goal_sign;
    }

    /// Create a new player controller with the given ID.
    pub fn new(id: PlayerId, settings: &ExecutorSettings) -> Self {
        let mut instance = Self {
            id,

            position_mtp: MTP::new(),
            last_pos: Vector2::new(0.0, 0.0),
            target_velocity: Vector2::new(0.0, 0.0),

            yaw_control: YawController::new(
                settings.controller_settings.angle_kp,
                settings.controller_settings.angle_cutoff_distance,
            ),
            last_yaw: Angle::from_radians(0.0),
            target_z: 0.0,

            frame_misses: 0,
            kicker: KickerState::Disarming,
            dribble_speed: 0.0,
            if_gate_keeper: false,

            max_accel: settings.controller_settings.max_acceleration,
            max_speed: settings.controller_settings.max_velocity,
            max_decel: settings.controller_settings.max_deceleration,
            max_angular_velocity: settings.controller_settings.max_angular_velocity,
            max_angular_acceleration: settings.controller_settings.max_angular_acceleration,

            have_imu: false,
            has_target_heading: false,
            switch_heading: None,
            heading_interval: IntervalTrigger::new(Duration::from_secs_f64(0.5)),
            opp_goal_sign: settings.tracker_settings.initial_opp_goal_x,
        };
        instance.update_settings(&settings.controller_settings);
        instance
    }

    /// Update the controller settings.
    pub fn update_settings(&mut self, settings: &ControllerSettings) {
        self.position_mtp.update_settings(
            settings.position_kp,
            Duration::from_secs_f64(settings.position_proportional_time_window),
            settings.position_cutoff_distance,
        );
        self.yaw_control
            .update_settings(settings.angle_kp, settings.angle_cutoff_distance);
    }

    /// Get the ID of the player.
    pub fn id(&self) -> PlayerId {
        self.id
    }

    /// set the player as the gate keeper
    pub fn set_gate_keeper(&mut self) {
        self.if_gate_keeper = true;
    }

    pub fn target_velocity(&self) -> Vector2 {
        self.target_velocity
    }

    /// Get the current command for the player.
    pub fn command(&mut self) -> PlayerCmd {
        if self.heading_interval.trigger() {
            return PlayerCmd::SetHeading {
                id: self.id,
                heading: -to_dies_yaw(self.last_yaw, self.opp_goal_sign).radians(),
            };
        }

        // Priority list: 1. Kick, 2. Switch heading, 3. Anything else
        let robot_cmd = match (self.kicker, self.switch_heading) {
            (_, Some(heading)) if self.have_imu => {
                self.switch_heading = None;
                if heading {
                    RobotCmd::HeadingControl
                } else {
                    RobotCmd::YawRateControl
                }
            }
            (KickerState::Kicking, _) => {
                self.kicker = KickerState::Disarming;
                RobotCmd::Kick
            }
            (KickerState::Arming, None) => RobotCmd::Arm,
            _ => RobotCmd::Arm,
        };

        let target_velocity = to_dies_yaw(self.last_yaw, self.opp_goal_sign)
            .inv()
            .rotate_vector(&to_dies_coords2(self.target_velocity, self.opp_goal_sign));

        let cmd = PlayerMoveCmd {
            id: self.id,
            // In the robot's local frame +sx means forward and +sy means right
            sx: target_velocity.x / 1000.0,  // Convert to m/s
            sy: -target_velocity.y / 1000.0, // Convert to m/s
            w: if self.have_imu && self.has_target_heading {
                -to_dies_yaw(Angle::from_radians(self.target_z), self.opp_goal_sign).radians()
            } else {
                -self.target_z * self.opp_goal_sign
            },
            dribble_speed: self.dribble_speed * MAX_DRIBBLE_SPEED,
            robot_cmd,
        };

        dies_core::debug_value(format!("p{}.sx", self.id), cmd.sx);
        dies_core::debug_value(format!("p{}.sy", self.id), cmd.sy);
        dies_core::debug_value(format!("p{}.w", self.id), cmd.w);
        dies_core::debug_value(format!("p{}.dribble_speed", self.id), cmd.dribble_speed);
        dies_core::debug_string(
            format!("p{}.robot_cmd", self.id),
            format!("{:?}", cmd.robot_cmd),
        );

        PlayerCmd::Move(cmd)
    }

    /// Increment the missing frame count, stops the robot if it is too high.
    pub fn increment_frames_misses(&mut self) {
        self.frame_misses += 1;
        if self.frame_misses > MISSING_FRAMES_THRESHOLD {
            log::warn!("Player {} has missing frames, stopping", self.id);
        }
    }

    /// Update the controller with the current state of the player.
    pub fn update(
        &mut self,
        state: &PlayerData,
        world: &WorldData,
        input: &PlayerControlInput,
        dt: f64,
        is_manual_override: bool,
        obstacles: Vec<Obstacle>,
        all_players: &[&PlayerData],
    ) {
        self.have_imu = false; // matches!(
                               //     state.imu_status,
                               //     Some(SysStatus::Ok) | Some(SysStatus::Ready)
                               // );

        if is_about_to_collide(state, world, 3.0 * dt) {
            dies_core::debug_string(format!("p{}.collision", self.id), "true");
        }

        // Calculate velocity using the MTP controller
        self.last_yaw = state.raw_yaw;
        self.last_pos = state.position;
        self.target_velocity = Vector2::zeros();
        if let Some(pos_target) = input.position {
            self.position_mtp.set_setpoint(pos_target);
            dies_core::debug_cross(
                format!("p{}.control.target", self.id),
                pos_target,
                dies_core::DebugColor::Red,
            );

            let pos_u = self.position_mtp.update(
                self.last_pos,
                state.velocity,
                dt,
                input.acceleration_limit.unwrap_or(self.max_accel),
                input.speed_limit.unwrap_or(self.max_speed),
                input.acceleration_limit.unwrap_or(self.max_decel),
                input.care,
            );
            // let local_u = self.last_yaw.inv().rotate_vector(&pos_u);
            self.target_velocity = pos_u;
        } else {
            dies_core::debug_remove(format!("p{}.control.target", self.id));
        }
        let add_vel = match input.velocity {
            Velocity::Global(v) => v,
            Velocity::Local(v) => self.last_yaw.rotate_vector(&v),
        };
        self.target_velocity += add_vel;

        if !is_manual_override {
            let obstacles = if input.avoid_ball {
                if let Some(ball) = world.ball.as_ref() {
                    let mut obstacles = obstacles;
                    obstacles.push(Obstacle::Circle {
                        center: ball.position.xy(),
                        radius: 50.0,
                    });
                    obstacles
                } else {
                    obstacles
                }
            } else {
                obstacles
            };

            self.target_velocity = velocity_obstacle_update(
                state,
                &self.target_velocity,
                all_players,
                obstacles.as_slice(),
                &world.player_model,
                super::rvo::VelocityObstacleType::VO,
            );
        }

        // Draw the velocity
        dies_core::debug_line(
            format!("p{}.velocity", self.id),
            self.last_pos,
            self.last_pos + state.velocity,
            dies_core::DebugColor::Red,
        );

        if let Some(yaw) = input.yaw {
            if !self.has_target_heading {
                self.has_target_heading = true;
                self.switch_heading = Some(true);
            }

            dies_core::debug_line(
                format!("p{}.target_yaw_line", self.id),
                self.last_pos,
                self.last_pos + yaw.rotate_vector(&Vector2::new(30.0, 0.0)),
                dies_core::DebugColor::Purple,
            );

            if self.have_imu {
                dies_core::debug_string(format!("p{}.yaw_control", self.id), "heading");
                self.target_z = yaw.radians();
            } else {
                dies_core::debug_string(format!("p{}.yaw_control", self.id), "yaw rate");
                self.yaw_control.set_setpoint(yaw);
                let head_u = self.yaw_control.update(
                    self.last_yaw,
                    state.angular_speed,
                    dt,
                    input
                        .angular_speed_limit
                        .unwrap_or(self.max_angular_velocity),
                    input
                        .angular_acceleration_limit
                        .unwrap_or(self.max_angular_acceleration),
                    // input.care,
                );
                self.target_z = head_u;
            }
        } else {
            if self.has_target_heading {
                self.has_target_heading = false;
                self.switch_heading = Some(false);
            }
            self.target_z = 0.0;
        }

        // Set dribbling speed
        self.dribble_speed = input.dribbling_speed;

        // Set kicker control
        match input.kicker {
            KickerControlInput::Arm => {
                self.kicker = KickerState::Arming;
            }
            KickerControlInput::Kick => match self.kicker {
                KickerState::Disarming | KickerState::Arming => {
                    self.kicker = KickerState::Kicking;
                }
                KickerState::Kicking => {}
            },
            KickerControlInput::Disarm | KickerControlInput::Idle => {
                // Make sure kick goes through
                if self.kicker != KickerState::Kicking {
                    self.kicker = KickerState::Disarming;
                }
            }
        }
    }

    pub fn update_target_velocity_with_avoidance(&mut self, target_velocity: Vector2) {
        self.target_velocity = target_velocity;
        dies_core::debug_line(
            format!("p{}.target_velocity", self.id),
            self.last_pos,
            self.last_pos + self.target_velocity,
            dies_core::DebugColor::Green,
        );
    }
}

fn is_about_to_collide(player: &PlayerData, world: &WorldData, time_horizon: f64) -> bool {
    // Check if the player is about to collide with any other player
    for other in world.own_players.iter().chain(world.opp_players.iter()) {
        if player.position == other.position {
            continue;
        }

        let dist = (player.position - other.position).norm();
        let relative_velocity = player.velocity - other.velocity;
        let time_to_collision = dist / relative_velocity.norm();
        if time_to_collision < time_horizon && dist < 140.0 {
            return true;
        }
    }

    // Check if the player is about to leave the field
    if let Some(geom) = world.field_geom.as_ref() {
        let hw = geom.field_width / 2.0;
        let hl = geom.field_length / 2.0;
        let pos = player.position;
        let vel = player.velocity;
        let new_pos = pos + vel * time_horizon;
        if new_pos.x.abs() > hw || new_pos.y.abs() > hl {
            return true;
        }
    }

    false
}

/// A struct that triggers an event periodically at a given interval.
struct IntervalTrigger {
    interval: Duration,
    next_trigger: Instant,
}

impl IntervalTrigger {
    /// Creates a new `Interval` with the given interval.
    fn new(interval: Duration) -> Self {
        Self {
            interval,
            next_trigger: Instant::now() + interval,
        }
    }

    /// Returns true if the event should be triggered at the given time.
    fn trigger(&mut self) -> bool {
        if Instant::now() >= self.next_trigger {
            self.next_trigger += self.interval;
            true
        } else {
            false
        }
    }
}
