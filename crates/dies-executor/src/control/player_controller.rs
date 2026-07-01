use dies_core::{
    Angle, ControllerSettings, DebugColor, ExecutorSettings, PlayerCmd, PlayerCmdUntransformer,
    PlayerData, PlayerGlobalMoveCmd, PlayerId, RobotCmd, TeamData, Vector2, PLAYER_RADIUS,
};

use super::{
    avoidance::{DynamicAgent, OrcaAgent, OrcaSolver, StaticObstacle},
    path_follower::PathFollower,
    player_input::{KickerControlInput, PlayerControlInput},
    team_context::PlayerContext,
    yaw_control::YawController,
    Velocity,
};

const MISSING_FRAMES_THRESHOLD: usize = 50;
const MAX_DRIBBLE_SPEED: f64 = 1000.0;
/// Terminal active-braking gain added per unit of `aggressiveness` when no
/// explicit `brake_gain` override is supplied.
const AGGRESSIVENESS_BRAKE_SCALE: f64 = 1.0;
/// Safety margin (mm) the bounded-region velocity envelope pulls the effective
/// boundary in by, to absorb position noise and one-tick discretization.
const BOUNDS_MARGIN: f64 = 30.0;

/// When no heading is commanded, the robot is considered "stable" (and its
/// current measured heading is latched as the hold setpoint) while its linear
/// speed is below this threshold \[mm/s\]. The on-board stability assist drives
/// better against a fixed heading setpoint than against NaN, so we emit the last
/// stable heading rather than releasing heading control.
const STABLE_HOLD_LINEAR_SPEED: f64 = 50.0;
/// Companion angular-speed gate \[rad/s\] for latching the hold heading.
const STABLE_HOLD_ANGULAR_SPEED: f64 = 0.2;

#[derive(Debug, Clone, Copy, PartialEq)]
enum KickerState {
    Disarming,
    Arming,
    Kicking,
}

pub struct PlayerController {
    id: PlayerId,
    path_follower: PathFollower,
    last_pos: Vector2,

    /// Output velocity \[mm/s\]: the post-ORCA, acceleration-clamped command from
    /// the most recent `update`. Also serves as the previous-command reference
    /// for the next tick's first-order acceleration clamp.
    target_velocity_global: Vector2,

    yaw_control: YawController,
    last_yaw: Angle,

    /// Target heading
    target_z: f64,
    /// Last heading the robot held while stable (near-stationary), latched
    /// whenever no heading is commanded and the robot is slow. Used as the
    /// hold setpoint when driving to a point with no fixed heading, so the
    /// on-board stability assist has a real setpoint instead of NaN.
    held_heading: Option<Angle>,
    /// Local angular velocity
    w: f64,
    /// Kick counter
    kick_counter: u8,

    frame_misses: usize,

    /// Kicker control
    kicker: KickerState,
    /// Whether a reflex kick is being requested this frame (held mode, no counter
    /// increment — firmware fires on breakbeam). Recomputed every `update`.
    reflex_requested: bool,
    /// Whether magnet mode is being requested this frame (held mode, composed with
    /// the kicker state into the wire command). Recomputed every `update`.
    magnet_requested: bool,
    /// Dribble speed normalized to \[0, 1\]
    dribble_speed: f64,

    max_accel: f64,
    max_speed: f64,
    max_decel: f64,
    brake_gain: f64,
    max_angular_velocity: f64,
    max_angular_acceleration: f64,
    last_yaw_rate_limit: Option<f64>,

    fan_speed: f64,
    kick_speed: f64,
}

impl PlayerController {
    /// Create a new player controller with the given ID.
    pub fn new(id: PlayerId, settings: &ExecutorSettings) -> Self {
        let mut instance = Self {
            id,

            path_follower: PathFollower::new(&settings.controller_settings),
            last_pos: Vector2::new(0.0, 0.0),
            target_velocity_global: Vector2::new(0.0, 0.0),
            w: 0.0,
            yaw_control: YawController::new(
                settings.controller_settings.angle_kp,
                settings.controller_settings.angle_cutoff_distance,
            ),
            last_yaw: Angle::from_radians(0.0),
            target_z: 0.0,
            held_heading: None,
            kick_counter: 0,

            frame_misses: 0,
            kicker: KickerState::Disarming,
            reflex_requested: false,
            magnet_requested: false,
            dribble_speed: 0.0,

            max_accel: settings.controller_settings.max_acceleration,
            max_speed: settings.controller_settings.max_velocity,
            max_decel: settings.controller_settings.max_deceleration,
            brake_gain: settings.controller_settings.brake_gain,
            max_angular_velocity: settings.controller_settings.max_angular_velocity,
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
        self.path_follower.update_settings(settings);
        self.yaw_control
            .update_settings(settings.angle_kp, settings.angle_cutoff_distance);

        self.max_accel = settings.max_acceleration;
        self.max_speed = settings.max_velocity;
        self.max_decel = settings.max_deceleration;
        self.brake_gain = settings.brake_gain;
        self.max_angular_velocity = settings.max_angular_velocity;
        self.max_angular_acceleration = settings.max_angular_acceleration;
    }

    /// Get the ID of the player.
    pub fn id(&self) -> PlayerId {
        self.id
    }

    /// Whether this controller will emit a kick on this tick's `command()`. Used
    /// as an efference copy for the possession metric.
    pub fn is_kicking(&self) -> bool {
        matches!(self.kicker, KickerState::Kicking)
    }

    /// Last computed velocity setpoint in **global** frame (mm/s). Reflects the
    /// path-follower / ORCA / manual-velocity output from the most recent
    /// `update`.
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

        // Priority list: 1. Smart kick (counter increment), 2. Reflex arm, 3. Arm.
        // Smart kick stays exactly as before: incrementing `kick_counter` while
        // sending ARM_COUNTER_KICK (RobotCmd::Arm) is what fires it firmware-side.
        // Reflex is a held mode (ARM_REFLEX_KICK) with no counter change; the
        // firmware fires it on breakbeam.
        // Magnet mode rides on the kicker state: while engaged the firmware servos
        // the ball onto the dribbler. We keep the kicker armed/charging throughout
        // (ARM_COUNTER) so a shot is ready the moment capture completes; a reflex
        // request composes to fire-on-capture. Magnet is never requested while a
        // kick is being commanded this frame.
        let robot_cmd = match self.kicker {
            KickerState::Kicking => {
                self.kicker = KickerState::Disarming;
                self.kick_counter = self.kick_counter.wrapping_add(1);
                RobotCmd::Arm
            }
            _ if self.reflex_requested && self.magnet_requested => RobotCmd::ArmReflexMagnet,
            _ if self.reflex_requested => RobotCmd::ArmReflex,
            _ if self.magnet_requested => RobotCmd::ArmMagnet,
            _ => RobotCmd::Arm,
        };

        let global_cmd = untransformer
            .set_target_velocity(self.target_velocity_global)
            .set_target_yaw(Angle::from_radians(self.target_z))
            .set_w(self.w)
            .set_dribble_speed(self.dribble_speed * MAX_DRIBBLE_SPEED)
            .set_kick_speed(self.kick_speed)
            .set_kick_counter(self.kick_counter)
            .set_robot_cmd(robot_cmd)
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
    /// `path` is the planner's full remaining polyline to follow, or `None` when
    /// the robot has no position target. The path-follower turns it into a
    /// dynamically-shaped preferred velocity, ORCA deflects it around
    /// `neighbors` to a collision-free velocity, and a first-order acceleration
    /// clamp smooths the result — all in this one place. `orca` is `None` when
    /// reciprocal avoidance is disabled, in which case the preferred velocity
    /// passes through unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        state: &PlayerData,
        world: &TeamData,
        input: &PlayerControlInput,
        path: Option<Vec<Vector2>>,
        dt: f64,
        is_manual_override: bool,
        player_context: &PlayerContext,
        neighbors: &[DynamicAgent],
        statics: &[StaticObstacle],
        orca: Option<&OrcaSolver>,
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

        // Raw ToF telemetry (for tuning the ToF-backup breakbeam substitute) plus
        // the latched output, so raw vs. Schmitt-latched can be compared in a log.
        match state.tof_confidence {
            Some(c) => player_context.debug_value("tof_confidence", c as f64),
            None => player_context.debug_remove("tof_confidence"),
        }
        match state.tof_xy {
            Some([x, y]) => {
                player_context.debug_value("tof_x", x as f64);
                player_context.debug_value("tof_y", y as f64);
            }
            None => {
                player_context.debug_remove("tof_x");
                player_context.debug_remove("tof_y");
            }
        }
        if state.tof_backup_enabled {
            player_context.debug_value(
                "tof_backup_ball_detected",
                if state.tof_backup_ball_detected {
                    1.0
                } else {
                    0.0
                },
            );
        } else {
            player_context.debug_remove("tof_backup_ball_detected");
        }

        self.last_yaw = state.raw_yaw;
        self.last_pos = state.position;

        // Previous command (set last tick by the team controller to the post-ORCA
        // velocity) — the reference for this tick's acceleration clamp and the
        // speed used to scale the pure-pursuit lookahead.
        let last_cmd = self.target_velocity_global;

        // Path-following preferred velocity (cruise / cornering / braking-to-goal
        // are all intrinsic). The team controller reconciles it with ORCA.
        // `hard_brake` requests bypassing the acceleration slew clamp for the
        // terminal active-braking maneuver.
        let mut hard_brake = false;
        if let Some(path) = path.as_deref() {
            // Aggressiveness is a single snappiness dial: it scales the position
            // approach gain and supplies the default terminal braking gain. An
            // explicit `input.brake_gain` decouples braking from the dial.
            let aggressiveness = input.aggressiveness.max(0.0);
            let approach_kp = self.path_follower.base_approach_kp() * (1.0 + aggressiveness);
            let brake_gain = input
                .brake_gain
                .unwrap_or(self.brake_gain + aggressiveness * AGGRESSIVENESS_BRAKE_SCALE);
            let cmd = self.path_follower.follow(
                path,
                self.last_pos,
                state.velocity,
                last_cmd.norm(),
                brake_gain,
                approach_kp,
            );
            self.target_velocity_global = cmd.velocity;
            hard_brake = cmd.hard_brake;
        } else {
            self.target_velocity_global = Vector2::zeros();
            player_context.debug_remove("control.target");
        }
        let add_vel = match input.velocity {
            Velocity::Global(v) => v,
            Velocity::Local(v) => self.last_yaw.rotate_vector(&v),
        };

        // Field-boundary containment is enforced downstream by ORCA (wall
        // half-planes, gated per-robot by `avoid_wall`) and, for bounded skills,
        // by `MotionBounds::clamp_velocity`. The old prohibited-zone velocity
        // damping that used to live here is gone — it predated the ORCA/planner
        // avoidance stack and only throttled this feed-forward term.
        if !is_manual_override {
            self.target_velocity_global += add_vel;
            self.target_velocity_global = self
                .target_velocity_global
                .cap_magnitude(input.speed_limit.unwrap_or(self.max_speed));
        } else {
            self.target_velocity_global += add_vel;
        }

        // Reciprocal collision avoidance (ORCA). The velocity built above is the
        // *preferred* velocity; ORCA returns the closest collision-free velocity
        // given this robot's neighbours. Folded in here — rather than overridden
        // at the team level — so the acceleration clamp below applies to the ORCA
        // output instead of bypassing it. Each agent's solve is data-independent
        // (it reads neighbours' observed velocities, not their intent), so doing
        // it per-controller is equivalent to the old team-level batch.
        if let Some(orca) = orca {
            let pref = self.target_velocity_global;
            let agent = OrcaAgent {
                id: self.id,
                position: state.position,
                velocity: state.velocity,
                pref_velocity: pref,
                radius: PLAYER_RADIUS,
                max_speed: input.speed_limit.unwrap_or(self.max_speed),
                neighbors: neighbors.to_vec(),
                statics: statics.to_vec(),
            };
            let safe = orca.solve(&agent, dt);
            self.target_velocity_global = safe;

            // Debug: preferred (gray) vs safe (green) velocity hints, scaled down
            // so they read as direction hints rather than full-length vectors.
            const ORCA_VIZ_SCALE: f64 = 0.3;
            player_context.debug_line_colored(
                "orca.pref",
                state.position,
                state.position + pref * ORCA_VIZ_SCALE,
                DebugColor::Gray,
            );
            player_context.debug_line_colored(
                "orca.safe",
                state.position,
                state.position + safe * ORCA_VIZ_SCALE,
                DebugColor::Green,
            );
            player_context.debug_value("orca.deflection", (safe - pref).norm());
            player_context.debug_value("orca.neighbors", neighbors.len() as f64);
        }

        // Per-robot acceleration limits. `acceleration_limit` (when set) caps both
        // ramp directions; the goalkeeper raises it to accelerate harder than the
        // global default. The decel value also drives the bounded-region envelope
        // below, so the brake profile it assumes is exactly what the clamp applies.
        let eff_accel = input.acceleration_limit.unwrap_or(self.max_accel);
        let eff_decel = input.acceleration_limit.unwrap_or(self.max_decel);

        // Bounded-region velocity envelope: cap the outward component of the final
        // commanded velocity so the robot can always brake before crossing the
        // region boundary (no overshoot). Applied after ORCA so it bounds the
        // truly-final velocity regardless of source; keyed off position, so it is
        // robust to velocity-estimate lag.
        if let Some(bounds) = input.bounds.as_ref() {
            self.target_velocity_global = bounds.clamp_velocity(
                state.position,
                self.target_velocity_global,
                eff_decel,
                BOUNDS_MARGIN,
            );
        }

        // First-order asymmetric acceleration clamp toward the desired velocity
        // (accel when speeding up, decel when slowing). This is the only output
        // smoothing — no jerk stage. ORCA may step the command in an emergency;
        // that's intentional and rare. The terminal active-braking maneuver
        // (`hard_brake`) bypasses the clamp so the reverse command lands in one
        // tick and the firmware reverse-thrusts to a crisp stop.
        let desired = self.target_velocity_global;
        let mut new_cmd = if hard_brake {
            desired
        } else {
            let a = if desired.norm() >= last_cmd.norm() {
                eff_accel
            } else {
                eff_decel
            };
            last_cmd + cap_vec(desired - last_cmd, a * dt.max(1.0e-3))
        };
        if desired.norm() < 1.0 && new_cmd.norm() < 1.0 {
            new_cmd = Vector2::zeros();
        }
        self.target_velocity_global = new_cmd;

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
            // Keep the hold heading fresh while actively commanding one, so that
            // when the command is released mid-motion we hold the heading the
            // robot actually ended up at rather than a stale latched value.
            self.held_heading = Some(self.last_yaw);
        } else if input.angular_velocity.is_some() {
            // Caller is commanding a raw angular velocity (a spin). The firmware
            // ignores `w` when a heading setpoint is present, so we must release
            // heading control (NaN) to let the spin through. `self.w` is set from
            // `input.angular_velocity` further below. Keep the hold heading fresh
            // so releasing the spin later resumes holding near the current yaw.
            self.target_z = f64::NAN;
            self.w = 0.0;
            self.held_heading = Some(self.last_yaw);
        } else {
            // No commanded heading. Rather than releasing heading control
            // (emitting NaN), hold the last heading the robot had while stable.
            // The on-board stability assist drives more smoothly against a real
            // heading setpoint. We (re)latch the measured heading whenever the
            // robot is near-stationary, and keep emitting that latched value
            // while it drives to a point with no fixed heading.
            let is_stable = state.velocity.norm() < STABLE_HOLD_LINEAR_SPEED
                && state.angular_speed.abs() < STABLE_HOLD_ANGULAR_SPEED;
            if is_stable || self.held_heading.is_none() {
                self.held_heading = Some(self.last_yaw);
            }
            let hold = self.held_heading.unwrap_or(self.last_yaw);
            self.target_z = hold.radians();
            self.yaw_control.set_setpoint(hold);
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

        // Magnet is a held mode (level signal), recomputed each frame; composed
        // with the kicker state when the wire command is built.
        self.magnet_requested = input.magnet;

        // Set kicker control. Reflex is a held mode (level signal), recomputed
        // each frame; everything else drives the smart-kick state machine.
        self.reflex_requested = matches!(input.kicker, KickerControlInput::ReflexKick);
        match input.kicker {
            KickerControlInput::Arm => {
                self.kicker = KickerState::Arming;
            }
            // Reflex: keep the capacitor charging; the firmware fires on
            // breakbeam, so we never enter the smart-kick `Kicking` state.
            KickerControlInput::ReflexKick => {
                if self.kicker != KickerState::Kicking {
                    self.kicker = KickerState::Arming;
                }
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
}

/// Clamp a vector's magnitude to `cap` (preserving direction).
fn cap_vec(v: Vector2, cap: f64) -> Vector2 {
    let n = v.norm();
    if n > cap && n > f64::EPSILON {
        v * (cap / n)
    } else {
        v
    }
}

// Field-boundary containment moved out of this controller: it is now enforced by
// the ORCA wall half-planes (gated per-robot by `avoid_wall`) and, for bounded
// skills, `MotionBounds::clamp_velocity`. The old prohibited-zone damping and its
// `keeper_with_bounds_is_not_frozen_at_the_post` regression test were deleted with
// it; successor coverage lives in `avoidance::obstacle`/`orca` and
// `dies_core::geom` (`in_defense_area`) tests.
