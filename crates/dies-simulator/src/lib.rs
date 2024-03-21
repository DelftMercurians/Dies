use dies_core::{FieldGeometry, PlayerCmd};
use dies_protos::{
    ssl_vision_detection::{SSL_DetectionFrame, SSL_DetectionRobot},
    ssl_vision_geometry::{
        SSL_FieldCircularArc, SSL_FieldLineSegment, SSL_GeometryData, SSL_GeometryFieldSize,
    },
    ssl_vision_wrapper::SSL_WrapperPacket,
};
use dies_serial_client::SerialClientConfig;
use dies_ssl_client::VisionClientConfig;
use rapier3d::{na::Rotation, prelude::*};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, SystemTime},
};
use tokio::sync::mpsc;

const PLAYER_RADIUS: f32 = 200.0;
const PLAYER_HEIGHT: f32 = 140.0;
// const PLAYER_MASS: f32 = 1.0;
const PLAYER_CMD_TIMEOUT: f32 = 1.0 / 20.0;

pub struct SimulationConfig {
    pub gravity: Vector<f32>,
    pub bias: f32,
    pub delay: Duration,         // delay in the emited packets
    pub command_delay: Duration, // delay for the execution of the command
    // TODO addition: run simulator faster than real time => keep my own time stamps
    pub max_accel: Vector<f32>,         // max lateral acceleration
    pub max_ang_accel: f32,             // max angular acceleration
    pub velocity_treshold: f32,         // max difference between target and current velocity
    pub angular_velocity_treshold: f32, // max difference between target and current angular velocity
    pub simulation_step: f32,           // time between simulation steps
    pub vision_update_step: f32,        // time between vision updates
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            gravity: Vector::z() * -9.81 * 1000.0,
            bias: 0.0,
            delay: Duration::from_millis(100),       // 0.1 second
            command_delay: Duration::from_millis(6), // 6 ms
            max_accel: Vector::new(70.0, 70.0, 0.0),
            max_ang_accel: 0.1,
            velocity_treshold: 1.0,
            angular_velocity_treshold: 0.001,
            simulation_step: 1.0 / 60.0,
            vision_update_step: 1.0 / 40.0,
        }
    }
}

struct Player {
    id: u32,
    is_own: bool,
    rigid_body_handle: RigidBodyHandle,
    _collider_handle: ColliderHandle,
    last_cmd_time: SystemTime,
    target_velocity: Vector<f32>,
    target_ang_velocity: f32,
}

struct TimedPlayerCmd {
    execute_time: SystemTime,
    player_cmd: PlayerCmd,
}

pub struct Simulation {
    config: SimulationConfig,
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
    players: Vec<Player>,
}

impl Simulation {
    pub fn spawn(config: SimulationConfig) -> (VisionClientConfig, SerialClientConfig) {
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
        let (vision_tx, vision_rx) = mpsc::unbounded_channel();
        let queue = Arc::new(Mutex::new(Vec::new())); // can be optimized
        let simulation_step = config.simulation_step;
        let vision_delay = config.delay;
        let vision_update_step = config.vision_update_step;
        let simulation = Arc::new(Mutex::new(Simulation::new(config)));

        let _receiver_task = {
            let queue = queue.clone();
            let simulation = simulation.clone();
            tokio::spawn(async move {
                while let Some(cmd) = cmd_rx.recv().await {
                    let simulation = simulation.lock().unwrap();
                    let timed_cmd = TimedPlayerCmd {
                        execute_time: SystemTime::now()
                            .checked_add(simulation.config.command_delay)
                            .unwrap(),
                        player_cmd: cmd,
                    };
                    let mut q = queue.lock().unwrap();
                    q.push(timed_cmd);
                }
            })
        };

        let geom_config = FieldGeometry {
            field_length: 11000,
            field_width: 9000,
            goal_width: 1000,
            goal_depth: 200,
            boundary_width: 200,
            line_segments: Vec::new(),
            circular_arcs: Vec::new(),
        };
        let _geom_task = {
            let vision_tx = vision_tx.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(3));
                loop {
                    interval.tick().await;
                    let geometry = geometry(&geom_config);
                    let _ = vision_tx.send(geometry);
                }
            });
        };

        let _vision_task = {
            let simulation = simulation.clone();
            tokio::spawn(async move {
                let mut interval =
                    tokio::time::interval(Duration::from_secs_f32(vision_update_step));
                loop {
                    interval.tick().await;
                    let vision = {
                        let sim = simulation.lock().unwrap();
                        sim.get_vision()
                    };
                    tokio::time::sleep(vision_delay).await;
                    let _ = vision_tx.send(vision);
                }
            });
        };

        let _step_physics = {
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs_f32(simulation_step));
                loop {
                    interval.tick().await;

                    let mut commands_to_exec = Vec::new();
                    queue.lock().unwrap().retain(|cmd| {
                        if cmd.execute_time <= SystemTime::now() {
                            commands_to_exec.push(cmd.player_cmd.clone());
                            false
                        } else {
                            true
                        }
                    });

                    let mut simulation = simulation.lock().unwrap();
                    for cmd in commands_to_exec {
                        simulation.exec_cmd(cmd);
                    }
                    simulation.step(simulation_step);
                }
            })
        };

        (
            VisionClientConfig::InMemory(vision_rx),
            SerialClientConfig::Memory(cmd_tx),
        )
    }

    pub fn new(config: SimulationConfig) -> Simulation {
        let mut simulation = Simulation {
            config,
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
            players: Vec::new(),
        };

        // Add players
        for i in 0..3 {
            let rigid_body = RigidBodyBuilder::dynamic()
                .translation(Vector::new(i as f32 * 3.0 * PLAYER_RADIUS, 0.0, 0.0))
                .build();
            let collider = ColliderBuilder::cylinder(PLAYER_HEIGHT / 2.0, PLAYER_RADIUS).build();
            let rigid_body_handle = simulation.rigid_body_set.insert(rigid_body);
            let collider_handle = simulation.collider_set.insert_with_parent(
                collider,
                rigid_body_handle,
                &mut simulation.rigid_body_set,
            );
            simulation.players.push(Player {
                id: i,
                is_own: true,
                rigid_body_handle,
                _collider_handle: collider_handle,
                last_cmd_time: SystemTime::now(),
                target_velocity: Vector::zeros(),
                target_ang_velocity: 0.0,
            })
        }

        simulation
    }

    // driver: get state, add new commands, return wrapper packet
    // keep queue with player commmands with timestapms

    pub fn exec_cmd(&mut self, cmd: PlayerCmd) {
        let player = self.players.iter_mut().find(|p| p.id == cmd.id).unwrap();
        player.target_velocity = Vector::new(cmd.sx, cmd.sy, 0.0) * 1000.0; // m/s to mm/s
        player.target_ang_velocity = cmd.w;
        player.last_cmd_time = SystemTime::now();
    }

    pub fn get_vision(&self) -> SSL_WrapperPacket {
        let mut detection = SSL_DetectionFrame::new();
        let t = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();
        detection.set_t_capture(t);
        detection.set_t_sent(t);
        for player in self.players.iter() {
            let rigid_body = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
            let position = rigid_body.position().translation.vector;
            let orientation = rigid_body.position().rotation.angle();
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
        let mut packet = SSL_WrapperPacket::new();
        packet.detection = Some(detection).into();
        packet
    }

    pub fn step(&mut self, dt: f32) {
        // Update players
        for player in self.players.iter_mut() {
            let rigid_body = self
                .rigid_body_set
                .get_mut(player.rigid_body_handle)
                .unwrap();

            if player.last_cmd_time.elapsed().unwrap().as_secs_f32() > PLAYER_CMD_TIMEOUT {
                player.target_velocity = Vector::zeros();
                player.target_ang_velocity = 0.0;
            }

            // Convert to global frame
            let velocity = rigid_body.linvel();
            let target_velocity =
                Rotation::<f32, 3>::new(Vector::z() * rigid_body.position().rotation.angle())
                    * player.target_velocity;
            let delta = (target_velocity - velocity).norm();
            if delta > self.config.velocity_treshold {
                let dir = (target_velocity - velocity).normalize();
                let mut new_vel = velocity + dir.component_mul(&self.config.max_accel) * dt;
                new_vel.z = 0.0;
                if new_vel.norm() < self.config.velocity_treshold {
                    new_vel = Vector::zeros();
                }
                rigid_body.set_linvel(new_vel, true);
            }

            let target_ang_vel = player.target_ang_velocity;
            let ang_velocity = rigid_body.angvel().z;
            let delta = (target_ang_vel - ang_velocity).abs();
            if delta > self.config.angular_velocity_treshold {
                let dir = (target_ang_vel - ang_velocity).signum();
                let mut new_ang_vel = Vector::zeros();
                new_ang_vel.z = ang_velocity + (dir * self.config.max_ang_accel * dt);
                if new_ang_vel.z.abs() < self.config.angular_velocity_treshold {
                    new_ang_vel = Vector::zeros();
                }
                rigid_body.set_angvel(new_ang_vel, true);
            }
        }

        self.integration_parameters.dt = dt;
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
