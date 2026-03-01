use std::time::Duration;

use dies_core::{
    Angle, ControllerSettings, ExecutorSettings, Obstacle, PlayerCmd, PlayerCmdUntransformer,
    PlayerData, PlayerGlobalMoveCmd, PlayerId, RobotCmd, TeamData, Vector2,
};

use super::{
    mtp::MTP,
    player_input::{KickerControlInput, PlayerControlInput},
    team_context::PlayerContext,
    two_step_mtp::TwoStepMTP,
    yaw_control::YawController,
    Velocity,
};

const MISSING_FRAMES_THRESHOLD: usize = 50;
const MAX_DRIBBLE_SPEED: f64 = 1000.0;

#[derive(Debug, Clone, Copy, PartialEq)]
enum KickerState {
    Disarming,
    Arming,
    Kicking,
}

pub struct PlayerController {
    id: PlayerId,
    position_mtp: MTP,
    two_step_mtp: TwoStepMTP,
    use_mpc: bool,
    use_two_step_mtp: bool,
    last_pos: Vector2,

    /// Output velocity \[mm/s\]
    target_velocity_global: Vector2,

    yaw_control: YawController,
    last_yaw: Angle,

    /// Target heading
    target_z: f64,
    /// Local angular velocity
    w: f64,
    /// Kick counter
    kick_counter: u8,

    frame_misses: usize,

    /// Kicker control
    kicker: KickerState,
    /// Dribble speed normalized to \[0, 1\]
    dribble_speed: f64,

    max_accel: f64,
    max_speed: f64,
    max_decel: f64,
    max_angular_acceleration: f64,
    last_yaw_rate_limit: Option<f64>,

    fan_speed: f64,
    kick_speed: f64,
}

impl PlayerController {
    #[cfg(feature = "mpc")]
    pub fn get_max_speed(&self) -> f64 {
        self.max_speed
    }

    /// Create a new player controller with the given ID.
    pub fn new(id: PlayerId, settings: &ExecutorSettings) -> Self {
        let mut instance = Self {
            id,

            position_mtp: MTP::new(),
            two_step_mtp: TwoStepMTP::new(),
            use_mpc: cfg!(feature = "mpc"), // Default to using MPC
            use_two_step_mtp: true,         // Default to double step MTP
            last_pos: Vector2::new(0.0, 0.0),
            target_velocity_global: Vector2::new(0.0, 0.0),
            w: 0.0,
            yaw_control: YawController::new(
                settings.controller_settings.angle_kp,
                settings.controller_settings.angle_cutoff_distance,
            ),
            last_yaw: Angle::from_radians(0.0),
            target_z: 0.0,
            kick_counter: 0,

            frame_misses: 0,
            kicker: KickerState::Disarming,
            dribble_speed: 0.0,

            max_accel: settings.controller_settings.max_acceleration,
            max_speed: settings.controller_settings.max_velocity,
            max_decel: settings.controller_settings.max_deceleration,
            max_angular_acceleration: settings.controller_settings.max_angular_acceleration,

            fan_speed: 0.0,
            kick_speed: 0.0,

            last_yaw_rate_limit: None,
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
        self.two_step_mtp.update_settings(
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

    /// Set the target velocity (used when MPC is computed externally)
    pub fn set_target_velocity(&mut self, velocity: Vector2) {
        self.target_velocity_global = velocity;
    }

    /// Check if using MPC
    pub fn use_mpc(&self) -> bool {
        self.use_mpc
    }

    /// Get the current command for the player.
    pub fn command(
        &mut self,
        player_context: &PlayerContext,
        mut untransformer: PlayerCmdUntransformer,
    ) -> PlayerCmd {
        if self.frame_misses > MISSING_FRAMES_THRESHOLD {
            return PlayerCmd::GlobalMove(PlayerGlobalMoveCmd::zero(self.id));
        }

        // Priority list: 1. Kick, 2. Anything else
        let _robot_cmd = match self.kicker {
            KickerState::Kicking => {
                self.kicker = KickerState::Disarming;
                self.kick_counter = self.kick_counter.wrapping_add(1);
                RobotCmd::Kick
            }
            _ => RobotCmd::Arm,
        };

        let global_cmd = untransformer
            .set_target_velocity(self.target_velocity_global)
            .set_target_yaw(Angle::from_radians(self.target_z))
            .set_w(self.w)
            .set_dribble_speed(self.dribble_speed * MAX_DRIBBLE_SPEED)
            .set_kick_speed(self.kick_speed)
            .set_kick_counter(self.kick_counter)
            .set_robot_cmd(RobotCmd::Arm)
            .set_max_yaw_rate(self.last_yaw_rate_limit.unwrap_or(100.0))
            .untransform_global_move_cmd(self.id, self.last_yaw);

        player_context.debug_string(
            "target_vel",
            format!(
                "{} {}",
                self.target_velocity_global.x, self.target_velocity_global.y
            ),
        );
        player_context.debug_value("heading_setpoint", global_cmd.heading_setpoint);
        player_context.debug_value("dribble_speed", global_cmd.dribble_speed);
        player_context.debug_value("kick_counter", global_cmd.kick_counter as f64);

        PlayerCmd::GlobalMove(global_cmd)
    }

    /// Increment the missing frame count, stops the robot if it is too high.
    pub fn increment_frames_misses(&mut self) {
        self.frame_misses += 1;
        // if self.frame_misses > MISSING_FRAMES_THRESHOLD {
        //     log::warn!("Player {} has missing frames, stopping", self.id);
        // }
    }

    /// Update the controller with the current state of the player.
    pub fn update(
        &mut self,
        state: &PlayerData,
        world: &TeamData,
        input: &PlayerControlInput,
        dt: f64,
        is_manual_override: bool,
        obstacles: Vec<Obstacle>,
        _all_players: &[&PlayerData],
        player_context: &PlayerContext,
        avoid_goal_area: bool,
        avoid_goal_area_margin: f64,
        avoid_opp_robots: bool,
    ) {
        self.frame_misses = 0;

        if let Some(fan_speed) = input.fan_speed {
            self.fan_speed = fan_speed;
        }
        if let Some(kick_speed) = input.kick_speed {
            self.kick_speed = kick_speed;
        }

        if state.handicaps.len() > 0 {
            player_context.debug_string(
                "handicaps",
                format!(
                    "{:?}",
                    state
                        .handicaps
                        .iter()
                        .map(|h| format!("{}", h).replace("No ", ""))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
        } else {
            player_context.debug_remove("handicaps");
        }

        player_context.debug_string(
            "can_touch_ball",
            format!(
                "{}",
                !matches!(
                    world.current_game_state.freekick_kicker,
                    Some(kicker) if kicker == self.id
                )
            ),
        );

        // Calculate velocity using MTP controller (MPC is handled at team level)
        self.last_yaw = state.raw_yaw;
        self.last_pos = state.position;
        if let Some(pos_target) = input.position {
            self.position_mtp.set_setpoint(pos_target);
            player_context.debug_cross_colored(
                "control.target",
                pos_target,
                dies_core::DebugColor::Red,
            );
        }

        // Always compute MTP first as fallback
        self.target_velocity_global = Vector2::zeros();
        if let Some(pos_target) = input.position {
            // Choose between regular MTP and two-step MTP
            let pos_u = if self.use_two_step_mtp && !is_manual_override {
                self.two_step_mtp.set_setpoint(pos_target);
                self.two_step_mtp.update(
                    self.last_pos,
                    state.velocity,
                    dt,
                    input.acceleration_limit.unwrap_or(self.max_accel),
                    input.speed_limit.unwrap_or(self.max_speed),
                    input.acceleration_limit.unwrap_or(self.max_decel),
                    input.care,
                    input.aggressiveness,
                    player_context,
                    world,
                    state,
                    avoid_goal_area,
                    avoid_goal_area_margin,
                    avoid_opp_robots,
                    obstacles,
                    input.control_paramer_override.clone(),
                )
            } else {
                self.position_mtp.set_setpoint(pos_target);
                self.position_mtp.update(
                    self.last_pos,
                    state.velocity,
                    dt,
                    input.acceleration_limit.unwrap_or(self.max_accel),
                    input.speed_limit.unwrap_or(self.max_speed),
                    input.acceleration_limit.unwrap_or(self.max_decel),
                    input.care,
                    player_context,
                )
            };
            self.target_velocity_global = pos_u;

            // Debug string will be updated by team controller if MPC overrides
            let controller_name = if self.use_two_step_mtp && !is_manual_override {
                "TwoStepMTP"
            } else {
                "MTP"
            };
            player_context.debug_string("controller", controller_name);
        } else {
            dies_core::debug_remove(format!("p{}.control.target", self.id));
        }
        let add_vel = match input.velocity {
            Velocity::Global(v) => v,
            Velocity::Local(v) => self.last_yaw.rotate_vector(&v),
        };

        let Some(geom) = world.field_geom.as_ref() else {
            return;
        };
        let maxdist = 400.0;
        let add_vel_factor =
            if self.is_in_prohibited_zone(state.position, geom, avoid_goal_area, maxdist) {
                let mut actual_dist = 0.0;
                for margin in (1..=(maxdist as i32)).step_by(10) {
                    if self.is_in_prohibited_zone(
                        state.position,
                        geom,
                        avoid_goal_area,
                        margin as f64,
                    ) {
                        actual_dist = margin as f64;
                        break;
                    }
                }
                ((actual_dist - 80.0) / (maxdist - 80.0)).max(0.0)
            } else {
                1.0
            };

        if !is_manual_override {
            self.target_velocity_global += add_vel * add_vel_factor;
            self.target_velocity_global = self
                .target_velocity_global
                .cap_magnitude(input.speed_limit.unwrap_or(self.max_speed));
        } else {
            self.target_velocity_global += add_vel;
        }

        // Draw the velocity
        player_context.debug_line_colored(
            "velocity",
            self.last_pos,
            self.last_pos + state.velocity,
            dies_core::DebugColor::Red,
        );

        if let Some(yaw) = input.yaw {
            player_context.debug_line_colored(
                "target_yaw_line",
                self.last_pos,
                self.last_pos + yaw.rotate_vector(&Vector2::new(30.0, 0.0)),
                dies_core::DebugColor::Purple,
            );

            self.target_z = yaw.radians();

            self.yaw_control.set_setpoint(yaw);
            let head_u = self.yaw_control.update(
                self.last_yaw,
                state.angular_speed,
                dt,
                input
                    .angular_speed_limit
                    .unwrap_or(self.last_yaw_rate_limit.unwrap_or(100.0)),
                input
                    .angular_acceleration_limit
                    .unwrap_or(self.max_angular_acceleration),
                input.care,
            );
            self.w = head_u;
        } else {
            self.target_z = f64::NAN;
            self.w = 0.0;
        }

        // Ramp up yaw rate with proprtional control
        if let Some(target_yaw_rate) = input.angular_speed_limit {
            let last_yaw_rate = self.last_yaw_rate_limit.unwrap_or(0.0);
            let yaw_rate_error = target_yaw_rate - last_yaw_rate;
            let yaw_rate_output = yaw_rate_error * 0.2; // kp
            self.last_yaw_rate_limit = Some(last_yaw_rate + yaw_rate_output);
        } else {
            self.last_yaw_rate_limit = None;
        }

        // If there is angular speed, add it
        if let Some(angular_velocity) = input.angular_velocity {
            self.w = angular_velocity;
        }

        // Set dribbling speed
        self.dribble_speed = input.dribbling_speed;

        // Set kicker control
        match input.kicker {
            KickerControlInput::Arm => {
                self.kicker = KickerState::Arming;
            }
            KickerControlInput::Kick { .. } => match self.kicker {
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

    fn is_in_prohibited_zone(
        &self,
        position: Vector2,
        geometry: &dies_core::FieldGeometry,
        avoid_goal_area: bool,
        margin: f64,
    ) -> bool {
        let field_half_width = geometry.boundary_width + geometry.field_width / 2.0;
        let field_half_length = geometry.boundary_width + geometry.field_length / 2.0;

        // Check if outside field boundaries
        if position.x < -field_half_length + margin
            || position.x > field_half_length - margin
            || position.y < -field_half_width + margin
            || position.y > field_half_width - margin
        {
            return true;
        }

        // Check goal areas if avoidance is enabled
        if avoid_goal_area {
            let goal_area_depth = geometry.penalty_area_depth + margin;
            let goal_area_width = geometry.penalty_area_width + margin * 2.0;

            // Our goal area (negative x)
            if position.x < -field_half_length + goal_area_depth
                && position.y.abs() < goal_area_width / 2.0
            {
                return true;
            }

            // Enemy goal area (positive x)
            if position.x > field_half_length - goal_area_depth
                && position.y.abs() < goal_area_width / 2.0
            {
                return true;
            }
        }

        false
    }
}
