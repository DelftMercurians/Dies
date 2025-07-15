use std::time::{Duration, Instant};

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
    max_angular_velocity: f64,
    max_angular_acceleration: f64,

    fan_speed: f64,
    kick_speed: f64,
}

impl PlayerController {
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
            max_angular_velocity: settings.controller_settings.max_angular_velocity,
            max_angular_acceleration: settings.controller_settings.max_angular_acceleration,

            fan_speed: 0.0,
            kick_speed: 0.0,
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
        let robot_cmd = match self.kicker {
            KickerState::Kicking => {
                self.kicker = KickerState::Disarming;
                self.kick_counter = self.kick_counter.wrapping_add(1);
                RobotCmd::Kick
            }
            _ => RobotCmd::Arm,
        };

        let _local_cmd = untransformer
            .set_target_velocity(self.target_velocity_global)
            .set_w(self.w)
            .set_dribble_speed(self.dribble_speed * MAX_DRIBBLE_SPEED)
            .set_kick_speed(self.kick_speed)
            .set_robot_cmd(robot_cmd)
            .untransform_move_cmd(self.id, self.last_yaw);
        // player_context.debug_value("sx", local_cmd.sx);
        // player_context.debug_value("sy", local_cmd.sy);
        // player_context.debug_value("w", local_cmd.w);
        // player_context.debug_value("dribble_speed", local_cmd.dribble_speed);
        // player_context.debug_string("robot_cmd", format!("{:?}", local_cmd.robot_cmd));

        // let global_cmd = PlayerGlobalMoveCmd {
        //     id: self.id,
        //     global_x: self.target_velocity_global.x / 1000.0,
        //     global_y: self.target_velocity_global.y / 1000.0,
        //     heading_setpoint: 0.0,
        //     last_heading: self.last_yaw.radians(),
        //     dribble_speed: self.dribble_speed * MAX_DRIBBLE_SPEED,
        //     kick_counter: self.kick_counter,
        //     robot_cmd,
        // };
        let global_cmd = untransformer
            .set_target_velocity(self.target_velocity_global)
            .set_target_yaw(Angle::from_radians(self.target_z))
            .set_dribble_speed(self.dribble_speed * MAX_DRIBBLE_SPEED)
            .set_kick_speed(self.kick_speed)
            .set_kick_counter(self.kick_counter)
            .set_robot_cmd(RobotCmd::Arm)
            .set_max_yaw_rate()
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
        all_players: &[&PlayerData],
        player_context: &PlayerContext,
        avoid_goal_area: bool,
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
            let pos_u = if self.use_two_step_mtp {
                self.two_step_mtp.set_setpoint(pos_target);
                self.two_step_mtp.update(
                    self.last_pos,
                    state.velocity,
                    dt,
                    input.acceleration_limit.unwrap_or(self.max_accel),
                    input.speed_limit.unwrap_or(self.max_speed),
                    input.acceleration_limit.unwrap_or(self.max_decel),
                    input.care,
                    player_context,
                    world,
                    state,
                    avoid_goal_area,
                    obstacles,
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
            let controller_name = if self.use_two_step_mtp {
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
        self.target_velocity_global += add_vel;
        self.target_velocity_global = self
            .target_velocity_global
            .cap_magnitude(input.speed_limit.unwrap_or(self.max_speed));

        self.target_velocity_global = // constrain velocity to not go into prohibited zones

        /*
        if !is_manual_override {
            let obstacles = if input.avoid_ball {
                if let Some(ball) = world.ball.as_ref() {
                    let mut obstacles = obstacles;
                    obstacles.push(Obstacle::Circle {
                        center: ball.position.xy(),
                        radius: 150.0,
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
                super::rvo::VelocityObstacleType::VO,
                input.avoid_robots,
            );

            println!("Obstacle avoiding trickery wtf??")
        }
        */

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
                    .unwrap_or(self.max_angular_velocity),
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

    pub fn update_target_velocity_with_avoidance(
        &mut self,
        target_velocity: Vector2,
        player_context: &PlayerContext,
    ) {
        self.target_velocity_global = target_velocity;
        player_context.debug_line_colored(
            "target_velocity",
            self.last_pos,
            self.last_pos + self.target_velocity_global,
            dies_core::DebugColor::Green,
        );
    }
}

fn is_about_to_collide(player: &PlayerData, world: &TeamData, time_horizon: f64) -> bool {
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
