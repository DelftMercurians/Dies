use dies_core::{BallData, FieldGeometry, PlayerCmd, PlayerData, WorldData};
use dies_protos::{
    ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame, SSL_DetectionRobot},
    ssl_vision_geometry::{
        SSL_FieldCircularArc, SSL_FieldLineSegment, SSL_GeometryData, SSL_GeometryFieldSize,
    },
    ssl_vision_wrapper::SSL_WrapperPacket,
};
use rapier3d::{
    na::{SimdPartialOrd, Vector2},
    prelude::*,
};
use serde::Serialize;
use std::collections::HashMap;
use utils::IntervalTrigger;

mod utils;

const BALL_RADIUS: f32 = 43.0;
const PLAYER_RADIUS: f32 = 200.0;
const PLAYER_HEIGHT: f32 = 140.0;
const DRIBBLER_RADIUS: f32 = BALL_RADIUS + 60.0;
const DRIBBLER_ANGLE: f32 = std::f32::consts::PI / 6.0;
const DRIBBLER_STRENGHT: f32 = 0.6;
const GROUND_THICKNESS: f32 = 10.0;
const PLAYER_CMD_TIMEOUT: f64 = 1.0 / 20.0;
const GEOM_INTERVAL: f64 = 3.0;

#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub gravity: Vector<f32>,
    pub bias: f32,
    pub ball_damping: f32,    // angular damping (rolling friction) on the ball
    pub simulation_step: f64, // time between simulation steps
    pub vision_update_step: f64, // time between vision updates
    pub command_delay: f64,   // delay for the execution of the command
    pub max_accel: Vector<f32>, // max lateral acceleration
    pub max_vel: Vector<f32>, // max lateral velocity
    pub max_ang_accel: f32,   // max angular acceleration
    pub max_ang_vel: f32,     // max angular velocity
    pub velocity_treshold: f32, // max difference between target and current velocity
    pub angular_velocity_treshold: f32, // max difference between target and current angular velocity
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            gravity: Vector::z() * -9.81 * 1000.0,
            bias: 0.0,
            ball_damping: 1.4,
            command_delay: 110.0 / 1000.0, // 6 ms
            max_accel: Vector::new(1200.0, 1200.0, 0.0),
            max_vel: Vector::new(2000.0, 2000.0, 0.0),
            max_ang_accel: 2.0,
            max_ang_vel: 2.0,
            velocity_treshold: 1.0,
            angular_velocity_treshold: 0.1,
            simulation_step: 1.0 / 60.0,
            vision_update_step: 1.0 / 40.0,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SimulationState {
    players: Vec<SimulationPlayerState>,
    ball: Option<Vector<f32>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimulationPlayerState {
    position: Vector2<f32>,
    orientation: f32,
}

#[derive(Debug)]
struct Ball {
    _rigid_body_handle: RigidBodyHandle,
    _collider_handle: ColliderHandle,
}

#[derive(Debug)]
struct Player {
    id: u32,
    is_own: bool,
    rigid_body_handle: RigidBodyHandle,
    _collider_handle: ColliderHandle,
    last_cmd_time: f64,
    target_velocity: Vector<f32>,
    target_ang_velocity: f32,
    current_dribble_speed: f32,
}

#[derive(Debug)]
struct TimedPlayerCmd {
    execute_time: f64,
    player_cmd: PlayerCmd,
}

/// A complete simulator for testing strategies and robot control in silico.
///
/// ## Usage
///
/// ```
/// let mut simulator = Simulation::default();
///
/// let dt = std::time::Duration::from_millis(10);
/// loop {
///     simulator.step(dt);
///     
///     // Get the latest detection packet
///     if let Some(detection) = simulator.detection() {
///        // Do something with the detection packet
///     }
///
///     // Get a geometry packet
///     if let Some(geometry) = simulator.geometry() {
///       // Do something with the geometry packet
///     }
///
///     // Push commands for players
///     simulator.push_cmd(PlayerCmd::zero(0))
/// }
/// ```
pub struct Simulation {
    config: SimulationConfig,
    current_time: f64,
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    ball: Option<Ball>,
    players: Vec<Player>,
    cmd_queue: Vec<TimedPlayerCmd>,
    detection_interval: IntervalTrigger,
    last_detection_packet: Option<SSL_WrapperPacket>,
    geometry_interval: IntervalTrigger,
    geometry_packet: SSL_WrapperPacket,
}

impl Simulation {
    /// Createa a new instance of [`Simulation`]. After creation, the simulation is
    /// empty and needs to be populated with players and a ball. It is better to use
    /// [`SimulationBuilder`] to create a new simulation and add players and a ball.
    pub fn new(config: SimulationConfig) -> Simulation {
        let geom_config = FieldGeometry {
            field_length: 11000,
            field_width: 9000,
            goal_width: 1000,
            goal_depth: 200,
            boundary_width: 200,
            line_segments: Vec::new(),
            circular_arcs: Vec::new(),
        };
        let vision_update_step = config.vision_update_step;

        let mut simulation = Simulation {
            config,
            current_time: 0.0,
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            ball: None,
            players: Vec::new(),
            cmd_queue: Vec::new(),
            detection_interval: IntervalTrigger::new(vision_update_step),
            last_detection_packet: None,
            geometry_interval: IntervalTrigger::new(GEOM_INTERVAL),
            geometry_packet: geometry(&geom_config),
        };

        // Create the ground
        let ground_body = RigidBodyBuilder::fixed()
            // z=0.0 is the ground surface
            .translation(Vector::new(0.0, 0.0, -GROUND_THICKNESS / 2.0))
            .build();
        let ground_collider = ColliderBuilder::cuboid(5500.0, 4500.0, GROUND_THICKNESS).build();
        let ground_body_handle = simulation.rigid_body_set.insert(ground_body);
        simulation.collider_set.insert_with_parent(
            ground_collider,
            ground_body_handle,
            &mut simulation.rigid_body_set,
        );

        simulation
    }

    /// Pushes a PlayerCmd onto the execution queue with the time delay specified in
    /// the config
    pub fn push_cmd(&mut self, cmd: PlayerCmd) {
        self.cmd_queue.push(TimedPlayerCmd {
            execute_time: self.current_time + self.config.command_delay,
            player_cmd: cmd,
        });
    }

    /// Executes a PlayerCmd immediately.
    pub fn execute_cmd(&mut self, cmd: PlayerCmd) {
        self.cmd_queue.push(TimedPlayerCmd {
            execute_time: self.current_time,
            player_cmd: cmd,
        });
    }

    pub fn detection(&mut self) -> Option<SSL_WrapperPacket> {
        self.last_detection_packet.take()
    }

    pub fn geometry(&mut self) -> Option<SSL_WrapperPacket> {
        if self.geometry_interval.trigger(self.current_time) {
            Some(self.geometry_packet.clone())
        } else {
            None
        }
    }

    pub fn step(&mut self, dt: f64) {
        // Create detection update if it's time
        if self.detection_interval.trigger(self.current_time) {
            self.new_detection_packet();
        }

        // Select commands that are due to be executed
        let commands_to_exec = {
            let mut to_exec = HashMap::new();
            self.cmd_queue.retain(|cmd| {
                if cmd.execute_time <= self.current_time {
                    to_exec.insert(cmd.player_cmd.id, cmd.player_cmd.clone());
                    false
                } else {
                    true
                }
            });
            to_exec
        };

        // Update query pipeline
        self.query_pipeline
            .update(&self.rigid_body_set, &self.collider_set);

        // Reset ball forces
        if let Some(ball) = self.ball.as_ref() {
            let ball_body = self
                .rigid_body_set
                .get_mut(ball._rigid_body_handle)
                .unwrap();
            ball_body.reset_forces(true);
            ball_body.set_linear_damping(self.config.ball_damping);
        }

        // Update players
        for player in self.players.iter_mut() {
            if let Some(command) = commands_to_exec.get(&player.id) {
                // In the robot's local frame, +sy means forward, +sx means left and both are in m/s
                // Angular velocity is in rad/s and +w means counter-clockwise
                // To make things easier, we swap the x and y velocities so that they
                // correspond to the simulator's frame
                player.target_velocity = Vector::new(command.sy, command.sx, 0.0) * 1000.0; // m/s to mm/s
                player.target_ang_velocity = command.w;
                player.current_dribble_speed = command.dribble_speed;
                player.last_cmd_time = self.current_time;
            }

            let rigid_body = self
                .rigid_body_set
                .get_mut(player.rigid_body_handle)
                .unwrap();

            if (self.current_time - player.last_cmd_time).abs() > PLAYER_CMD_TIMEOUT {
                player.target_velocity = Vector::zeros();
                player.target_ang_velocity = 0.0;
            }

            let velocity = rigid_body.linvel();

            // Convert to global frame
            let target_velocity =
                Rotation::<f32>::new(Vector::z() * rigid_body.rotation().euler_angles().2)
                    * player.target_velocity;

            let vel_err = target_velocity - velocity;
            let new_vel = if vel_err.norm() > 10.0 {
                let acc = (2.0 * vel_err).simd_clamp(-self.config.max_accel, self.config.max_accel);
                velocity + acc * (dt as f32)
            } else {
                target_velocity
            };
            let new_vel = new_vel.simd_clamp(-self.config.max_vel, self.config.max_vel);
            rigid_body.set_linvel(new_vel, true);

            let target_ang_vel = player.target_ang_velocity;
            let ang_velocity = rigid_body.angvel().z;
            let delta = (target_ang_vel - ang_velocity).abs();
            let new_ang_vel = if delta > self.config.angular_velocity_treshold {
                let dir = (target_ang_vel - ang_velocity).signum();
                ang_velocity + (dir * self.config.max_ang_accel * (dt as f32))
            } else {
                target_ang_vel
            };
            let new_ang_vel = new_ang_vel.clamp(-self.config.max_ang_vel, self.config.max_ang_vel);
            rigid_body.set_angvel(Vector::z() * new_ang_vel, true);

            // Check if the ball is in the dribbler
            if player.current_dribble_speed > 0.0 {
                let heading = rigid_body.position().rotation * Vector::x();
                let player_position = rigid_body.position().translation.vector;
                let dribbler_position =
                    player_position + heading * (PLAYER_RADIUS + BALL_RADIUS + 20.0);
                let ball_handle = self.ball.as_ref().map(|ball| ball._rigid_body_handle);

                if let Some(ball_handle) = ball_handle {
                    let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
                    let ball_position = ball_body.position().translation.vector;
                    let ball_dir = ball_position - player_position;
                    let distance = ball_dir.norm();
                    let angle = heading.angle(&ball_dir);
                    if distance < PLAYER_RADIUS + DRIBBLER_RADIUS && angle < DRIBBLER_ANGLE {
                        let force = (player.current_dribble_speed * DRIBBLER_STRENGHT)
                            * (dribbler_position - ball_position);
                        // clamp the force to the max force
                        // let force = force.cap_magnitude(200.0);
                        ball_body.add_force(force, true);
                        // dampen the ball's velocity
                        ball_body.set_linear_damping(self.config.ball_damping * 2.0);
                    }
                }
            }
        }

        self.integration_parameters.dt = dt as f32;
        self.physics_pipeline.step(
            &self.config.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );

        self.current_time += dt as f64;
    }

    fn new_detection_packet(&mut self) {
        let mut detection = SSL_DetectionFrame::new();
        detection.set_t_capture(self.current_time);
        detection.set_t_sent(self.current_time);

        // Players
        for player in self.players.iter() {
            let rigid_body = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
            let position = rigid_body.position().translation.vector;
            let orientation = rigid_body.rotation().euler_angles().2;
            let mut robot = SSL_DetectionRobot::new();
            robot.set_robot_id(player.id as u32);
            robot.set_x(position.x);
            robot.set_y(position.y);
            robot.set_orientation(orientation);
            robot.set_confidence(1.0);
            if player.is_own {
                detection.robots_blue.push(robot);
            } else {
                detection.robots_yellow.push(robot);
            }
        }

        // Ball
        if let Some(ball) = self.ball.as_ref() {
            let mut ball_det = SSL_DetectionBall::new();
            let ball_body = self.rigid_body_set.get(ball._rigid_body_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;
            ball_det.set_x(ball_position.x);
            ball_det.set_y(ball_position.y);
            ball_det.set_z(ball_position.z);
            ball_det.set_confidence(1.0);
            detection.balls.push(ball_det);
        }

        let mut packet = SSL_WrapperPacket::new();
        packet.detection = Some(detection).into();

        self.last_detection_packet = Some(packet);
    }

    pub fn world_data(&self) -> WorldData {
        let mut own_players = Vec::new();
        let mut opp_players = Vec::new();
        self.players.iter().for_each(|player| {
            let rigid_body = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
            let position = rigid_body.position().translation.vector;
            let orientation = rigid_body.rotation().euler_angles().2;
            let data = PlayerData {
                id: player.id,
                timestamp: self.current_time,
                position: Vector2::new(position.x, position.y),
                velocity: Vector2::new(rigid_body.linvel().x, rigid_body.linvel().y),
                orientation,
                angular_speed: rigid_body.angvel().z,
                raw_position: Vector2::new(position.x, position.y),
            };

            if player.is_own {
                own_players.push(data);
            } else {
                opp_players.push(data);
            }
        });

        let ball = if let Some(ball) = self.ball.as_ref() {
            let ball_body = self.rigid_body_set.get(ball._rigid_body_handle).unwrap();
            let position = ball_body.position().translation.vector;
            Some(BallData {
                position: Vector::new(position.x, position.y, position.z),
                timestamp: self.current_time,
                velocity: Vector::new(
                    ball_body.linvel().x,
                    ball_body.linvel().y,
                    ball_body.linvel().z,
                ),
            })
        } else {
            None
        };

        WorldData {
            own_players,
            opp_players,
            ball,
            field_geom: Some(FieldGeometry::from_protobuf(
                &self
                    .geometry_packet
                    .geometry
                    .as_ref()
                    .unwrap_or_default()
                    .field,
            )),
            ..Default::default()
        }
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Simulation::new(SimulationConfig::default())
    }
}

pub struct SimulationBuilder {
    sim: Simulation,
    last_own_id: u32,
    last_opp_id: u32,
}

impl SimulationBuilder {
    pub fn new(config: SimulationConfig) -> Self {
        SimulationBuilder {
            sim: Simulation::new(config),
            last_own_id: 0,
            last_opp_id: 0,
        }
    }

    pub fn add_own_player_with_id(
        mut self,
        id: u32,
        position: Vector2<f32>,
        orientation: f32,
    ) -> Self {
        self.add_player(id, true, position, orientation);
        self
    }

    pub fn add_opp_player_with_id(
        mut self,
        id: u32,
        position: Vector2<f32>,
        orientation: f32,
    ) -> Self {
        self.add_player(id, false, position, orientation);
        self
    }

    pub fn add_own_player(mut self, position: Vector2<f32>, orientation: f32) -> Self {
        self.add_player(self.last_own_id, true, position, orientation);
        self.last_own_id += 1;
        self
    }

    pub fn add_opp_player(mut self, position: Vector2<f32>, orientation: f32) -> Self {
        self.add_player(self.last_opp_id, false, position, orientation);
        self.last_opp_id += 1;
        self
    }

    pub fn add_ball_with_velocity(mut self, position: Vector<f32>, velocity: Vector<f32>) -> Self {
        let sim = &mut self.sim;

        let mut ball_body = RigidBodyBuilder::dynamic()
            .can_sleep(false)
            .translation(position)
            .linear_damping(sim.config.ball_damping)
            .build();
        ball_body.set_linvel(velocity, true);
        let ball_collider = ColliderBuilder::ball(BALL_RADIUS)
            .mass(1.0)
            .restitution(0.0)
            .restitution_combine_rule(CoefficientCombineRule::Min)
            .build();
        let ball_body_handle = sim.rigid_body_set.insert(ball_body);
        let ball_collider_handle = sim.collider_set.insert_with_parent(
            ball_collider,
            ball_body_handle,
            &mut sim.rigid_body_set,
        );
        sim.ball = Some(Ball {
            _rigid_body_handle: ball_body_handle,
            _collider_handle: ball_collider_handle,
        });

        self
    }

    pub fn build(self) -> Simulation {
        self.sim
    }

    fn add_player(&mut self, id: u32, is_own: bool, position: Vector2<f32>, orientation: f32) {
        let sim = &mut self.sim;

        // Players have fixed z position - their bottom surface 1mm above the ground
        let position = Vector::new(position.x, position.y, (PLAYER_HEIGHT / 2.0) + 1.0);
        let rigid_body = RigidBodyBuilder::kinematic_velocity_based()
            .translation(position)
            .rotation(Vector::z() * orientation)
            .locked_axes(
                LockedAxes::TRANSLATION_LOCKED_Z
                    | LockedAxes::ROTATION_LOCKED_X
                    | LockedAxes::ROTATION_LOCKED_Y,
            )
            .build();
        let collider = ColliderBuilder::cylinder(PLAYER_HEIGHT / 2.0, PLAYER_RADIUS)
            .restitution(0.0)
            .restitution_combine_rule(CoefficientCombineRule::Min)
            .build();
        let rigid_body_handle = sim.rigid_body_set.insert(rigid_body);
        let collider_handle = sim.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut sim.rigid_body_set,
        );

        sim.players.push(Player {
            id,
            is_own,
            rigid_body_handle,
            _collider_handle: collider_handle,
            last_cmd_time: 0.0,
            target_velocity: Vector::zeros(),
            target_ang_velocity: 0.0,
            current_dribble_speed: 0.0,
        });
    }
}

impl Default for SimulationBuilder {
    fn default() -> Self {
        SimulationBuilder::new(SimulationConfig::default())
    }
}

fn geometry(config: &FieldGeometry) -> SSL_WrapperPacket {
    let mut geometry = SSL_GeometryData::new();
    let mut field = SSL_GeometryFieldSize::new();
    field.set_field_length(config.field_length);
    field.set_field_width(config.field_width);
    field.set_goal_width(config.goal_width);
    field.set_goal_depth(config.goal_depth);
    field.set_boundary_width(config.boundary_width);

    for line in &config.line_segments {
        let mut ssl_segment = SSL_FieldLineSegment::new();
        ssl_segment.set_name(line.name.clone());
        let mut p1 = dies_protos::ssl_vision_geometry::Vector2f::new();
        p1.set_x(line.p1.x);
        p1.set_y(line.p1.y);
        ssl_segment.p1 = Some(p1).into();
        let mut p2 = dies_protos::ssl_vision_geometry::Vector2f::new();
        p2.set_x(line.p2.x);
        p2.set_y(line.p2.y);
        ssl_segment.p2 = Some(p2).into();
        ssl_segment.set_thickness(line.thickness);
        field.field_lines.push(ssl_segment);
    }

    for arc in &config.circular_arcs {
        let mut ssl_arc = SSL_FieldCircularArc::new();
        ssl_arc.set_name(arc.name.clone());
        let mut center = dies_protos::ssl_vision_geometry::Vector2f::new();
        center.set_x(arc.center.x);
        center.set_y(arc.center.y);
        ssl_arc.center = Some(center).into();
        ssl_arc.set_radius(arc.radius);
        ssl_arc.set_a1(arc.a1);
        ssl_arc.set_a2(arc.a2);
        ssl_arc.set_thickness(arc.thickness);
        field.field_arcs.push(ssl_arc);
    }

    geometry.field = Some(field).into();
    let mut packet = SSL_WrapperPacket::new();
    packet.geometry = Some(geometry).into();
    packet
}
