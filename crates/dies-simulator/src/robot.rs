//! Robot command queues, movement limits, hardware state, and ball contacts.
use crate::{
    config::SimulationConfig,
    packets::SimOutput,
    physics::PhysicsWorld,
    realism::gaussian,
    referee::{AutoReferee, SimulationGameState},
};
use dies_core::{
    Angle, PlayerFeedbackMsg, PlayerGlobalMoveCmd, PlayerId, PlayerMoveCmd, ReflexKickState,
    RobotCmd, SysStatus, TeamColor, Vector2,
};
use rand::{rngs::StdRng, Rng};
use rapier3d_f64::prelude::*;
use std::{collections::HashMap, f64::consts::PI};

pub(crate) struct RobotStepContext<'a> {
    pub(crate) config: &'a SimulationConfig,
    pub(crate) now: f64,
    pub(crate) physics: &'a mut PhysicsWorld,
    pub(crate) referee: &'a mut AutoReferee,
    pub(crate) output: &'a mut SimOutput,
    pub(crate) rng: &'a mut StdRng,
}
pub(crate) struct RobotStep {
    pub(crate) dribble_claimants: Vec<(PlayerId, TeamColor)>,
    pub(crate) ball_kicked_this_frame: bool,
}

#[derive(Debug)]
pub(crate) struct Player {
    pub(crate) id: PlayerId,
    pub(crate) team_color: TeamColor,
    pub(crate) rigid_body_handle: RigidBodyHandle,
    pub(crate) _collider_handle: ColliderHandle,
    pub(crate) last_cmd_time: f64,
    pub(crate) target_velocity: Vector<f64>,
    pub(crate) w: f64,
    pub(crate) target_heading: f64,
    /// Per-command yaw-rate limit (rad/s) from the controller's `max_yaw_rate`.
    /// The body slews toward `target_heading` no faster than this (capped by the
    /// hardware `max_ang_vel`), mirroring the firmware — so it can't teleport and
    /// fling a dribbled ball out of the dribbler cone.
    pub(crate) target_yaw_rate: f64,
    pub(crate) current_dribble_speed: f64,
    pub(crate) breakbeam: bool,
    pub(crate) last_kick_counter: u8,
    /// Reflex kick armed (robot_cmd == ArmReflex). Fires once when the ball
    /// reaches the dribbler/breakbeam, mirroring the firmware.
    pub(crate) reflex_armed: bool,
    /// Whether the armed reflex has already fired (one-shot per arm).
    pub(crate) reflex_fired_this_arm: bool,
    /// Firmware reflex-kick counter mirror — incremented each time a reflex kick
    /// actually fires. Reported over feedback so the strike verify can confirm a
    /// fire without seeing the ball leave. Wraps at 255.
    pub(crate) reflex_kick_counter: u8,
    /// Firmware smart-kick counter mirror — incremented each time a commanded
    /// (smart) kick actually fires. Wraps at 255.
    pub(crate) smart_kick_counter: u8,
}

#[derive(Debug, Clone)]
pub(crate) enum MoveCmd {
    Local(PlayerMoveCmd),
    Global(PlayerGlobalMoveCmd),
}

#[derive(Debug, Clone)]
pub(crate) struct TimedPlayerCmd {
    pub(crate) id: PlayerId,
    pub(crate) team_color: TeamColor,
    pub(crate) execute_time: f64,
    pub(crate) player_cmd: MoveCmd,
}

pub(crate) struct RobotFleet {
    pub(crate) players: Vec<Player>,
    pub(crate) cmd_queue: Vec<TimedPlayerCmd>,
}

impl RobotFleet {
    pub(crate) fn new() -> Self {
        Self {
            players: Vec::new(),
            cmd_queue: Vec::new(),
        }
    }

    pub(crate) fn spawn_robot(
        &mut self,
        physics: &mut PhysicsWorld,
        config: &SimulationConfig,
        team_color: TeamColor,
        player_id: PlayerId,
        position: Vector2,
        yaw: Angle,
    ) {
        let player_radius = config.player_radius;
        let player_height = config.player_height;
        let pos = Vector::new(position.x, position.y, (player_height / 2.0) + 1.0);
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(pos)
            .rotation(Vector::z() * yaw.radians())
            .locked_axes(
                LockedAxes::TRANSLATION_LOCKED_Z
                    | LockedAxes::ROTATION_LOCKED_X
                    | LockedAxes::ROTATION_LOCKED_Y,
            )
            .build();
        let collider = ColliderBuilder::cylinder(player_height / 2.0, player_radius)
            .rotation(Vector::x() * std::f64::consts::FRAC_PI_2)
            .restitution(0.0)
            .restitution_combine_rule(CoefficientCombineRule::Min)
            .build();
        let rigid_body_handle = physics.rigid_body_set.insert(rigid_body);
        let collider_handle = physics.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut physics.rigid_body_set,
        );
        self.players.push(Player {
            id: player_id,
            team_color,
            rigid_body_handle,
            _collider_handle: collider_handle,
            last_cmd_time: 0.0,
            target_velocity: Vector::zeros(),
            w: 0.0,
            target_heading: f64::NAN,
            target_yaw_rate: f64::INFINITY,
            current_dribble_speed: 0.0,
            breakbeam: false,
            last_kick_counter: 0,
            reflex_armed: false,
            reflex_fired_this_arm: false,
            reflex_kick_counter: 0,
            smart_kick_counter: 0,
        });
    }

    pub(crate) fn add_robot(
        &mut self,
        physics: &mut PhysicsWorld,
        config: &SimulationConfig,
        team_color: TeamColor,
        player_id: PlayerId,
        position: Vector2,
        yaw: Angle,
    ) {
        self.remove_robot(physics, team_color, player_id);
        self.spawn_robot(physics, config, team_color, player_id, position, yaw);
    }

    pub(crate) fn remove_robot(
        &mut self,
        physics: &mut PhysicsWorld,
        team_color: TeamColor,
        player_id: PlayerId,
    ) {
        if let Some(idx) = self
            .players
            .iter()
            .position(|p| p.id == player_id && p.team_color == team_color)
        {
            let player = self.players.remove(idx);
            physics.rigid_body_set.remove(
                player.rigid_body_handle,
                &mut physics.island_manager,
                &mut physics.collider_set,
                &mut physics.impulse_joint_set,
                &mut physics.multibody_joint_set,
                true,
            );
        }
    }

    pub(crate) fn teleport_robot(
        &mut self,
        physics: &mut PhysicsWorld,
        team_color: TeamColor,
        player_id: PlayerId,
        position: Vector2,
        yaw: Angle,
    ) {
        if let Some(player) = self
            .players
            .iter()
            .find(|p| p.id == player_id && p.team_color == team_color)
        {
            if let Some(rigid_body) = physics.rigid_body_set.get_mut(player.rigid_body_handle) {
                let old_pos = rigid_body.position().translation.vector;
                let pos = Vector::new(position.x, position.y, old_pos.z);
                rigid_body.set_position(
                    Isometry::translation(pos.x, pos.y, pos.z)
                        * Isometry::rotation(Vector::z() * yaw.radians()),
                    true,
                );
                rigid_body.set_linvel(Vector::zeros(), true);
                rigid_body.set_angvel(Vector::zeros(), true);
            }
        }
    }

    /// Pushes a PlayerCmd onto the execution queue with the time delay specified in
    /// the config
    pub(crate) fn push_cmd(
        &mut self,
        now: f64,
        command_delay: f64,
        team_color: TeamColor,
        cmd: PlayerMoveCmd,
    ) {
        self.cmd_queue.push(TimedPlayerCmd {
            id: cmd.id,
            team_color,
            execute_time: now + command_delay,
            player_cmd: MoveCmd::Local(cmd),
        });
    }

    pub(crate) fn push_global_cmd(
        &mut self,
        now: f64,
        command_delay: f64,
        team_color: TeamColor,
        cmd: PlayerGlobalMoveCmd,
    ) {
        self.cmd_queue.push(TimedPlayerCmd {
            id: cmd.id,
            team_color,
            execute_time: now + command_delay,
            player_cmd: MoveCmd::Global(cmd),
        });
    }

    pub(crate) fn take_due_commands(
        &mut self,
        now: f64,
    ) -> HashMap<(TeamColor, PlayerId), MoveCmd> {
        let mut to_exec = HashMap::new();
        self.cmd_queue.retain(|cmd| {
            if cmd.execute_time <= now {
                to_exec.insert((cmd.team_color, cmd.id), cmd.player_cmd.clone());
                false
            } else {
                true
            }
        });
        to_exec
    }

    pub(crate) fn update(
        &mut self,
        ctx: &mut RobotStepContext<'_>,
        commands_to_exec: &HashMap<(TeamColor, PlayerId), MoveCmd>,
        dt: f64,
    ) -> RobotStep {
        let send_feedback = ctx.output.feedback_interval.trigger(ctx.now);
        // Ownership of the ball is resolved after the player loop from this
        // frame's claimants, so that two robots racing for the same ball don't
        // flip ownership every frame (last writer won) and deadlock.
        let mut dribble_claimants: Vec<(PlayerId, TeamColor)> = Vec::new();
        let mut ball_kicked_this_frame = false;
        for player in self.players.iter_mut() {
            // Only update players from controlled teams
            if (!ctx.config.blue_controlled && player.team_color == TeamColor::Blue)
                || (!ctx.config.yellow_controlled && player.team_color == TeamColor::Yellow)
            {
                let rigid_body = ctx
                    .physics
                    .rigid_body_set
                    .get_mut(player.rigid_body_handle)
                    .unwrap();
                rigid_body.set_linvel(Vector::zeros(), true);
                rigid_body.set_angvel(Vector::zeros(), true);
                continue;
            }

            if send_feedback {
                let mut feedback = PlayerFeedbackMsg::empty(player.id);
                feedback.breakbeam_ball_detected = Some(player.breakbeam);
                feedback.reflex_kick_state = Some(if player.reflex_armed {
                    ReflexKickState::Armed
                } else {
                    ReflexKickState::Off
                });
                feedback.reflex_kick_counter = Some(player.reflex_kick_counter);
                feedback.smart_kick_counter = Some(player.smart_kick_counter);
                feedback.kicker_status = Some(SysStatus::Ready);
                feedback.imu_status = if ctx.config.has_imu {
                    Some(SysStatus::Ready)
                } else {
                    Some(SysStatus::NotInstalled)
                };
                ctx.output
                    .feedback_queue
                    .insert((player.team_color, player.id), feedback);
            }

            let rigid_body = ctx
                .physics
                .rigid_body_set
                .get_mut(player.rigid_body_handle)
                .unwrap();

            let mut is_kicking = false;
            if let Some(command) = commands_to_exec.get(&(player.team_color, player.id)) {
                match command {
                    MoveCmd::Local(cmd) => {
                        // In the robot's local frame, +sx means forward, +sy means right and both are in m/s
                        // Angular velocity is in rad/s and +w means counter-clockwise
                        player.target_velocity = Vector::new(cmd.sx, -cmd.sy, 0.0) * 1000.0; // m/s to mm/s
                        player.w = -cmd.w;
                        player.current_dribble_speed = cmd.dribble_speed;
                        player.last_cmd_time = ctx.now;
                        is_kicking = matches!(cmd.robot_cmd, RobotCmd::Kick);

                        if let RobotCmd::Kick = cmd.robot_cmd {
                            is_kicking = true;
                        }
                    }
                    MoveCmd::Global(cmd) => {
                        // Rotate the velocity to the robot's frame
                        let yaw = Rotation::<f64>::new(
                            Vector::z() * rigid_body.rotation().euler_angles().2,
                        );
                        let target_velocity =
                            yaw.inverse() * Vector::new(cmd.global_x, cmd.global_y, 0.0);
                        player.target_velocity = target_velocity * 1000.0; // m/s to mm/s
                        player.target_heading = cmd.heading_setpoint;
                        player.target_yaw_rate = cmd.max_yaw_rate;
                        player.current_dribble_speed = cmd.dribble_speed;
                        player.last_cmd_time = ctx.now;
                        player.w = -cmd.w;
                        if cmd.kick_counter > player.last_kick_counter {
                            player.last_kick_counter = cmd.kick_counter;
                            is_kicking = true;
                        }
                        // Reflex kick: armed by the held mode (no counter). Reset
                        // the one-shot latch on a fresh arm (false -> true).
                        let want_reflex = matches!(cmd.robot_cmd, RobotCmd::ArmReflex);
                        if want_reflex && !player.reflex_armed {
                            player.reflex_fired_this_arm = false;
                        }
                        player.reflex_armed = want_reflex;
                    }
                }
            }

            if (ctx.now - player.last_cmd_time).abs() > ctx.config.player_cmd_timeout {
                player.target_velocity = Vector::zeros();
                player.w = 0.0;
                player.target_heading = f64::NAN;
                player.target_yaw_rate = f64::INFINITY;
            }

            let velocity = rigid_body.linvel();

            // Convert to global frame
            let target_velocity =
                Rotation::<f64>::new(Vector::z() * rigid_body.rotation().euler_angles().2)
                    * player.target_velocity;

            let vel_err = target_velocity - velocity;

            let new_vel = {
                // Asymmetric limit: braking (commanding a lower speed) is much
                // stronger than accelerating, matching the real robots. Without
                // this, the robot can't shed cruise speed fast enough near a
                // target and overshoots it.
                let limit = if target_velocity.norm() >= velocity.norm() {
                    ctx.config.max_accel
                } else {
                    ctx.config.max_decel
                };
                let acc = (vel_err / dt).cap_magnitude(limit);
                velocity + acc * dt
            };
            let new_vel = new_vel.cap_magnitude(ctx.config.max_vel);
            // Realism: general sluggishness (steady scale) plus occasional
            // per-step stutters (a brief actuator hitch). Both are velocity
            // multipliers applied for this step only; the controller still
            // commands normally, so this models the robot under-delivering, not
            // the strategy changing.
            let vel_scale = if ctx.config.realism.applies_to(player.team_color) {
                if ctx.config.realism.move_stutter_prob > 0.0
                    && ctx.rng.gen::<f64>() < ctx.config.realism.move_stutter_prob
                {
                    ctx.config.realism.move_stutter_factor
                } else {
                    ctx.config.realism.move_sluggish
                }
            } else {
                1.0
            };
            rigid_body.set_linvel(new_vel * vel_scale, true);

            if !player.target_heading.is_nan() {
                // Slew toward the target heading at a bounded rate, capped by the
                // controller's per-command `max_yaw_rate` and the hardware
                // `max_ang_vel`. (Previously this teleported to the target every
                // step — `5f64.to_degrees()` ≈ 286 rad could never be exceeded by a
                // radian error, and the cap used `max_ang_vel/dt` — so a large aim
                // rotation snapped the body in one step and flung a dribbled ball
                // out of the dribbler cone. The firmware rate-limits yaw; match it.)
                let current_yaw = rigid_body.rotation().euler_angles().2;
                let target_yaw = player.target_heading;
                // Shortest signed angular error in (-π, π], so we never slew the
                // long way around when the headings straddle ±π.
                let yaw_err = (target_yaw - current_yaw + PI).rem_euclid(2.0 * PI) - PI;
                let rate = player.target_yaw_rate.min(ctx.config.max_ang_vel);
                let max_step = rate * dt;
                let new_yaw = if yaw_err.abs() > max_step {
                    current_yaw + yaw_err.signum() * max_step
                } else {
                    target_yaw
                };
                /*
                println!(
                    "Turning {} {} to {}",
                    player.team_color,
                    player.id,
                    new_yaw.to_degrees()
                );
                */
                rigid_body.set_rotation(Rotation::from_euler_angles(0.0, 0.0, new_yaw), true);
            } else {
                let target_ang_vel = player.w;
                let new_ang_vel =
                    target_ang_vel.clamp(-ctx.config.max_ang_vel, ctx.config.max_ang_vel);
                rigid_body.set_angvel(Vector::z() * new_ang_vel, true);
            }

            // Check if the ball is in the dribbler
            player.breakbeam = false;
            let yaw = rigid_body.position().rotation * Vector::x();
            let player_position = rigid_body.position().translation.vector;
            let ball_handle = ctx
                .physics
                .ball
                .as_ref()
                .map(|ball| ball._rigid_body_handle);
            if let Some(ball_handle) = ball_handle {
                let ball_body = ctx.physics.rigid_body_set.get_mut(ball_handle).unwrap();
                let ball_position = ball_body.position().translation.vector;
                let ball_dir = ball_position - player_position;
                let distance = ball_dir.norm();
                let angle = yaw.angle(&ball_dir);
                if distance < ctx.config.player_radius + ctx.config.dribbler_radius + 20.0
                    && angle < ctx.config.dribbler_angle
                {
                    player.breakbeam = true;
                    // A reflex kick fires the instant the ball reaches the
                    // breakbeam, once per arm (matches firmware ARM_REFLEX_KICK).
                    let reflex_fire = player.reflex_armed && !player.reflex_fired_this_arm;
                    // Realism: a kicker dud — the solenoid doesn't fire, so the
                    // ball stays on the dribbler (falls through to a dribble claim).
                    let misfire = ctx.config.realism.applies_to(player.team_color)
                        && ctx.config.realism.shot_failure_prob > 0.0
                        && ctx.rng.gen::<f64>() < ctx.config.realism.shot_failure_prob;
                    if (is_kicking || reflex_fire) && !misfire {
                        // Bump the counter that matches the kicker that actually
                        // fired — mirrors the firmware, which increments a distinct
                        // reflex/smart counter on each discharge (never on a dud).
                        if reflex_fire {
                            player.reflex_fired_this_arm = true;
                            player.reflex_kick_counter = player.reflex_kick_counter.wrapping_add(1);
                        } else {
                            player.smart_kick_counter = player.smart_kick_counter.wrapping_add(1);
                        }
                        ball_kicked_this_frame = true;
                        // Realism: noisy kick heading + power. `kick_dir` is the
                        // nominal robot facing rotated by a Gaussian heading error;
                        // `power` scales the impulse by a Gaussian fraction.
                        let (kick_dir, power) = if ctx.config.realism.applies_to(player.team_color)
                        {
                            let dtheta =
                                gaussian(&mut ctx.rng, ctx.config.realism.shot_heading_noise);
                            let (s, c) = dtheta.sin_cos();
                            let dir =
                                Vector::new(yaw.x * c - yaw.y * s, yaw.x * s + yaw.y * c, 0.0);
                            let p = (1.0
                                + gaussian(&mut ctx.rng, ctx.config.realism.shot_power_noise))
                            .max(0.2);
                            (dir, p)
                        } else {
                            (yaw, 1.0)
                        };
                        // Move the ball away from the player
                        let new_ball_position = ball_position + kick_dir * 80.0;
                        ball_body.set_position(
                            Isometry::translation(
                                new_ball_position.x,
                                new_ball_position.y,
                                new_ball_position.z,
                            ),
                            true,
                        );

                        let force = kick_dir * ctx.config.kicker_strength * power;
                        ball_body.add_force(force, true);

                        // Set the ball_is_kicked to true
                        if let SimulationGameState::FreeKick { ball_is_kicked, .. } =
                            &mut ctx.referee.game_state
                        {
                            *ball_is_kicked = true;
                        }
                    } else if player.current_dribble_speed > 0.0 {
                        // Defer ownership to after the loop; record the claim.
                        dribble_claimants.push((player.id, player.team_color));
                    }
                }
            }
        }
        RobotStep {
            dribble_claimants,
            ball_kicked_this_frame,
        }
    }
}
