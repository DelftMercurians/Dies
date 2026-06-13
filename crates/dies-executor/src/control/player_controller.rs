use dies_core::{
    Angle, ControllerSettings, ExecutorSettings, PlayerCmd, PlayerCmdUntransformer, PlayerData,
    PlayerGlobalMoveCmd, PlayerId, RobotCmd, TeamData, Vector2,
};

use super::{
    avoidance::PlanStep,
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
    two_step_mtp: TwoStepMTP,
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
    max_jerk: f64,
    max_speed: f64,
    max_decel: f64,
    max_angular_velocity: f64,
    max_angular_acceleration: f64,
    last_yaw_rate_limit: Option<f64>,

    /// Jerk-limited output tracker state: the actually-commanded velocity and its
    /// acceleration, slewed toward the desired (post-ORCA) velocity so the
    /// command stays C1 (no acceleration steps → no lurch).
    cmd_velocity: Vector2,
    cmd_accel: Vector2,

    /// Most recent measured global-frame velocity, captured at the start of
    /// `update()`. Used by the output-side acceleration clamp in `command()`.
    last_velocity: Vector2,
    /// dt corresponding to `last_velocity`. Same purpose.
    last_dt: f64,

    fan_speed: f64,
    kick_speed: f64,
}

impl PlayerController {
    /// Create a new player controller with the given ID.
    pub fn new(id: PlayerId, settings: &ExecutorSettings) -> Self {
        let mut instance = Self {
            id,

            two_step_mtp: TwoStepMTP::new(),
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
            max_jerk: settings.controller_settings.max_jerk,
            max_speed: settings.controller_settings.max_velocity,
            max_decel: settings.controller_settings.max_deceleration,
            max_angular_velocity: settings.controller_settings.max_angular_velocity,
            max_angular_acceleration: settings.controller_settings.max_angular_acceleration,

            fan_speed: 0.0,
            kick_speed: 0.0,

            last_yaw_rate_limit: None,
            last_velocity: Vector2::zeros(),
            last_dt: 0.0,

            cmd_velocity: Vector2::zeros(),
            cmd_accel: Vector2::zeros(),
        };
        instance.update_settings(&settings.controller_settings);
        instance
    }

    /// Update the controller settings.
    pub fn update_settings(&mut self, settings: &ControllerSettings) {
        self.two_step_mtp.update_settings(
            settings.position_kp,
            settings.position_cutoff_distance,
            settings.thresh,
        );
        self.yaw_control
            .update_settings(settings.angle_kp, settings.angle_cutoff_distance);

        self.max_accel = settings.max_acceleration;
        self.max_jerk = settings.max_jerk;
        self.max_speed = settings.max_velocity;
        self.max_decel = settings.max_deceleration;
        self.max_angular_velocity = settings.max_angular_velocity;
        self.max_angular_acceleration = settings.max_angular_acceleration;
    }

    /// Get the ID of the player.
    pub fn id(&self) -> PlayerId {
        self.id
    }

    /// Set the target velocity (used when the team controller overrides the
    /// MTP output with the ORCA-filtered velocity). The output-side accel
    /// clamp in `command()` still applies to whatever is set here.
    pub fn set_target_velocity(&mut self, velocity: Vector2) {
        self.target_velocity_global = velocity;
    }

    /// Commit a desired (post-ORCA) velocity through the jerk-limited output
    /// tracker and store the smoothed result as the command velocity. Runs once
    /// per control tick. Per-axis S-curve: acceleration is slewed toward the
    /// value that lets it ramp back to zero exactly as the velocity error
    /// closes, so the command is acceleration-continuous (no lurch) and
    /// overshoot-free.
    pub fn commit_tracked_velocity(&mut self, desired: Vector2, dt: f64) {
        let dt = dt.clamp(1.0e-3, 0.5);
        let a_max = self.max_accel;
        let j_max = self.max_jerk;
        for i in 0..2 {
            let err = desired[i] - self.cmd_velocity[i];
            // Acceleration that can still decelerate to 0 right at err = 0 under
            // the jerk limit: a = sqrt(2·j·|err|), capped at a_max.
            let a_target = err.signum() * a_max.min((2.0 * j_max * err.abs()).sqrt());
            let da = (a_target - self.cmd_accel[i]).clamp(-j_max * dt, j_max * dt);
            self.cmd_accel[i] = (self.cmd_accel[i] + da).clamp(-a_max, a_max);
            self.cmd_velocity[i] += self.cmd_accel[i] * dt;
        }
        self.cmd_velocity = self.cmd_velocity.cap_magnitude(self.max_speed);
        // Snap to a clean stop once both the target and the residual command are
        // negligible, so a holding robot doesn't keep a sub-mm/s creep alive.
        if desired.norm() < 1.0e-3 && self.cmd_velocity.norm() < 1.0 {
            self.cmd_velocity = Vector2::zeros();
            self.cmd_accel = Vector2::zeros();
        }
        self.target_velocity_global = self.cmd_velocity;
    }

    /// Access to the configured maximum speed, used by the team controller to
    /// bound ORCA preferred velocities.
    pub fn get_max_speed(&self) -> f64 {
        self.max_speed
    }

    /// Last computed velocity setpoint in **global** frame (mm/s). Reflects the
    /// MTP / ORCA / manual-velocity output from the most recent `update`.
    pub fn target_velocity_global(&self) -> Vector2 {
        self.target_velocity_global
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
            .set_max_yaw_rate(
                self.last_yaw_rate_limit
                    .unwrap_or(self.max_angular_velocity),
            )
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
    ///
    /// `step` is the planner-chosen setpoint to steer toward (with its
    /// final/intermediate flag and corner speed), or `None` when the robot has
    /// no position target. Collision avoidance is applied separately: the team
    /// controller overrides the resulting velocity with the ORCA solution via
    /// `set_target_velocity`.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        state: &PlayerData,
        world: &TeamData,
        input: &PlayerControlInput,
        step: Option<PlanStep>,
        dt: f64,
        is_manual_override: bool,
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

        self.last_velocity = state.velocity;
        self.last_dt = dt;

        // MTP velocity toward the planner waypoint. The team controller then
        // overrides this with the ORCA-filtered velocity.
        if let Some(step) = step {
            self.two_step_mtp.set_setpoint(step.waypoint);
            let pos_u = self.two_step_mtp.update(
                self.last_pos,
                state.velocity,
                dt,
                input.acceleration_limit.unwrap_or(self.max_accel),
                input.speed_limit.unwrap_or(self.max_speed),
                input.acceleration_limit.unwrap_or(self.max_decel),
                input.care,
                input.aggressiveness,
                input.control_paramer_override.clone(),
                step.is_final,
                step.speed_frac,
            );
            self.target_velocity_global = pos_u;
        } else {
            self.target_velocity_global = Vector2::zeros();
            player_context.debug_remove("control.target");
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
            self.target_z = yaw.radians();

            self.yaw_control.set_setpoint(yaw);
            let head_u = self.yaw_control.update(
                self.last_yaw,
                state.angular_speed,
                dt,
                input.angular_speed_limit.unwrap_or(
                    self.last_yaw_rate_limit
                        .unwrap_or(self.max_angular_velocity),
                ),
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
