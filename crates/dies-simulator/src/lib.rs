use atomic_float::AtomicF64;
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
use rapier3d::prelude::*;
use std::{
    sync::{atomic::Ordering, Arc, Mutex},
    time::{Duration, SystemTime},
};
use tokio::sync::mpsc::{self, error::TryRecvError};
use utils::IntervalTrigger;

mod utils;

const BALL_RADIUS: f32 = 43.0;
const BALL_USER_DATA: u128 = 8;
const DRIBBLER_USER_DATA: u128 = 4;
const PLAYER_RADIUS: f32 = 200.0;
const PLAYER_HEIGHT: f32 = 140.0;
const DRIBLER_WIDTH: f32 = 50.0;
const DRIBLER_HEIGHT: f32 = 10.0;
const DRIBLER_LENGTH: f32 = 10.0;
const BALL_COLLISION_GROUP: u32 = 0b1;
const PLAYER_COLLISION_GROUP: u32 = 0b10;
const DRIBLER_COLLISION_GROUP: u32 = 0b100;
const PLAYER_CMD_TIMEOUT: f64 = 1.0 / 20.0;
const GEOM_INTERVAL: f64 = 3.0;

pub struct SimulationConfig {
    pub gravity: Vector<f32>,
    pub bias: f32,
    pub simulation_step: f64,           // time between simulation steps
    pub vision_update_step: f64,        // time between vision updates
    pub command_delay: f64,             // delay for the execution of the command
    pub max_accel: Vector<f32>,         // max lateral acceleration
    pub max_ang_accel: f32,             // max angular acceleration
    pub velocity_treshold: f32,         // max difference between target and current velocity
    pub angular_velocity_treshold: f32, // max difference between target and current angular velocity
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            gravity: Vector::z() * -9.81 * 1000.0,
            bias: 0.0,
            command_delay: 110.0 / 1000.0, // 6 ms
            max_accel: Vector::new(70.0, 70.0, 0.0),
            max_ang_accel: 0.1,
            velocity_treshold: 1.0,
            angular_velocity_treshold: 0.001,
            simulation_step: 1.0 / 60.0,
            vision_update_step: 1.0 / 40.0,
        }
    }
}

#[derive(Debug, Default)]
struct Ball {
    _rigid_body_handle: RigidBodyHandle,
    _collider_handle: ColliderHandle,
    // collision groups: 0
}

impl Ball {
    fn default() -> Self {
        Ball {
            _rigid_body_handle: RigidBodyHandle::default(),
            _collider_handle: ColliderHandle::default(),
        }
    }
}

struct Player {
    id: u32,
    is_own: bool,
    rigid_body_handle: RigidBodyHandle,
    _collider_handle: ColliderHandle,
    _dribbler_collider_handle: ColliderHandle,
    last_cmd_time: f64,
    target_velocity: Vector<f32>,
    target_ang_velocity: f32,
}

struct TimedPlayerCmd {
    execute_time: f64,
    player_cmd: PlayerCmd,
}

pub struct Simulation {
    pub(crate) config: SimulationConfig,
    pub(crate) rigid_body_set: RigidBodySet,
    pub(crate) collider_set: ColliderSet,
    pub(crate) integration_parameters: IntegrationParameters,
    pub(crate) physics_pipeline: PhysicsPipeline,
    pub(crate) island_manager: IslandManager,
    pub(crate) broad_phase: BroadPhase,
    pub(crate) narrow_phase: NarrowPhase,
    pub(crate) impulse_joint_set: ImpulseJointSet,
    pub(crate) multibody_joint_set: MultibodyJointSet,
    pub(crate) ccd_solver: CCDSolver,
    pub(crate) query_pipeline: QueryPipeline,
    pub(crate) ball: Ball,
    pub(crate) players: Vec<Player>,
}

impl Simulation {
    pub fn spawn(config: SimulationConfig) -> (VisionClientConfig, SerialClientConfig) {
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
        let (vision_tx, vision_rx) = mpsc::unbounded_channel();
        let cmd_queue = Arc::new(Mutex::new(Vec::new()));
        let time = Arc::new(AtomicF64::new(0.0));

        let simulation_step = config.simulation_step;
        let vision_update_step = config.vision_update_step;
        let cmd_delay = config.command_delay;
        let geom_config = FieldGeometry {
            field_length: 11000,
            field_width: 9000,
            goal_width: 1000,
            goal_depth: 200,
            boundary_width: 200,
            line_segments: Vec::new(),
            circular_arcs: Vec::new(),
        };

        let _receiver_task = {
            let time = Arc::clone(&time);
            let cmd_queue = Arc::clone(&cmd_queue);
            tokio::spawn(async move {
                while let Some(cmd) = cmd_rx.recv().await {
                    let timed_cmd = TimedPlayerCmd {
                        execute_time: time.load(Ordering::Relaxed) + cmd_delay,
                        player_cmd: cmd,
                    };
                    cmd_queue.lock().unwrap().push(timed_cmd);
                }
            })
        };

        let _step_physics = {
            let time = Arc::clone(&time);
            let cmd_queue = Arc::clone(&cmd_queue);
            tokio::spawn(async move {
                let mut simulation = Simulation::new(config);
                let mut interval = tokio::time::interval(Duration::from_secs_f64(simulation_step));
                let mut geom_interval = IntervalTrigger::new(GEOM_INTERVAL);
                let mut det_interval = IntervalTrigger::new(vision_update_step);

                'main_loop: loop {
                    interval.tick().await;

                    let dt = simulation_step;
                    let current_time = time.fetch_add(dt, Ordering::Relaxed) + dt;
                    let commands_to_exec = {
                        let mut to_exec = Vec::new();
                        cmd_queue.lock().unwrap().retain(|cmd| {
                            if cmd.execute_time <= current_time {
                                to_exec.push(cmd.player_cmd.clone());
                                false
                            } else {
                                true
                            }
                        });
                        to_exec
                    };

                    // Send out geometry/detection updates
                    if geom_interval.trigger(current_time) {
                        if let Err(_) = vision_tx.send(geometry(&geom_config)) {
                            break 'main_loop;
                        }
                    }
                    if det_interval.trigger(current_time) {
                        if let Err(_) = vision_tx.send(simulation.get_vision()) {
                            break 'main_loop;
                        }
                    }

                    for cmd in commands_to_exec {
                        simulation.exec_cmd(cmd, current_time);
                    }
                    simulation.step(dt as f32, current_time);
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
            ball: Ball::default(),
            players: Vec::new(),
        };

        {
            // Add the ball
            let rigid_body = RigidBodyBuilder::dynamic()
                // .translation(Vector::new(0.0, 0.0, 0.0))
                .build();
            let collider = ColliderBuilder::ball(BALL_RADIUS)
                .user_data(BALL_USER_DATA) // only in group 0, interact with group 2
                .collision_groups(InteractionGroups::new(
                    BALL_COLLISION_GROUP.into(),
                    (DRIBLER_COLLISION_GROUP | PLAYER_COLLISION_GROUP | BALL_COLLISION_GROUP)
                        .into(),
                ))
                .build();
            let rigid_body_handle = simulation.rigid_body_set.insert(rigid_body);
            let collider_handle = simulation.collider_set.insert_with_parent(
                collider,
                rigid_body_handle,
                &mut simulation.rigid_body_set,
            );
            simulation.ball = Ball {
                _rigid_body_handle: rigid_body_handle,
                _collider_handle: collider_handle,
            };
        }

        // Add players
        for i in 0..3 {
            let pos = Vector::new(i as f32 * 3.0 * PLAYER_RADIUS, 0.0, 0.0);
            simulation.add_player(i, true, pos, 0.0);
        }

        simulation
    }

    fn exec_cmd(&mut self, cmd: PlayerCmd, time: f64) {
        let player = self.players.iter_mut().find(|p| p.id == cmd.id).unwrap();
        player.target_velocity = Vector::new(cmd.sx, cmd.sy, 0.0) * 1000.0; // m/s to mm/s
        player.target_ang_velocity = cmd.w;
        player.last_cmd_time = time;
    }

    fn get_vision(&self) -> SSL_WrapperPacket {
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

    fn step(&mut self, dt: f32, current_time: f64) {
        // Update players
        for player in self.players.iter_mut() {
            let rigid_body = self
                .rigid_body_set
                .get_mut(player.rigid_body_handle)
                .unwrap();

            if (current_time - player.last_cmd_time).abs() > PLAYER_CMD_TIMEOUT {
                player.target_velocity = Vector::zeros();
                player.target_ang_velocity = 0.0;
            }

            // Convert to global frame
            let velocity = rigid_body.linvel();
            let target_velocity =
                Rotation::<f32>::new(Vector::z() * rigid_body.position().rotation.angle())
                    * player.target_velocity;
            // println!("target: {}", target_velocity);
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
                println!("new_ang_vel: {}", new_ang_vel);
                rigid_body.set_angvel(new_ang_vel, true);
            }
        }

        let (collision_send, collision_recv) = crossbeam::channel::unbounded();
        let (contact_force_send, _) = crossbeam::channel::unbounded();
        let event_handler = ChannelEventCollector::new(collision_send, contact_force_send);

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
            &event_handler,
        );

        while let Ok(collision_event) = collision_recv.try_recv() {
            // Handle the collision event.
            println!("Received collision event: {:?}", collision_event);

            // if the ball colides with a dribbler of a robot, the ball should e given a force in the direction of the dribbler
            // go through all the  _dribler_collider_handle of all the players and check if it's equal to the collision_event.collider1 or collision_event.collider2

            let collider_1 = self.collider_set.get(collision_event.collider1()).unwrap(); // safe, we know it exists
            let collider_2 = self.collider_set.get(collision_event.collider2()).unwrap(); // safe, we know it exists

            let dribbler_collider = if collider_1.user_data == DRIBBLER_USER_DATA {
                collider_1
            } else if collider_2.user_data == DRIBBLER_USER_DATA {
                collider_2
            } else {
                continue;
            };

            let ball_collider = if collider_1.user_data == BALL_USER_DATA {
                collider_1
            } else if collider_2.user_data == BALL_USER_DATA {
                collider_2
            } else {
                continue;
            };

            let player_body = self
                .rigid_body_set
                .get(dribbler_collider.parent().unwrap())
                .unwrap();
            let player_position = player_body.position().translation.vector;
            let ball_body = self
                .rigid_body_set
                .get_mut(ball_collider.parent().unwrap())
                .unwrap();
            let ball_position = ball_body.position().translation.vector;

            if collision_event.clone().started() && ball_body.user_force().norm() < 1e-9 {
                let force_direction = player_position - ball_position;

                let force = force_direction.normalize() * 1000.0;

                ball_body.add_force(force, true);
            } else if collision_event.stopped() {
                ball_body.reset_forces(true);
            }
        }
    }

    fn add_player(&mut self, id: u32, is_own: bool, position: Vector<f32>, orientation: f32) {
        let rigid_body = RigidBodyBuilder::dynamic()
            .translation(position)
            .rotation(Vector::z() * orientation)
            .build();
        let collider = ColliderBuilder::cylinder(PLAYER_HEIGHT / 2.0, PLAYER_RADIUS)
            .collision_groups(InteractionGroups::new(
                PLAYER_COLLISION_GROUP.into(),
                (PLAYER_COLLISION_GROUP | BALL_COLLISION_GROUP).into(),
            ))
            .build(); // only in group 1, interacts with nobody
        let rigid_body_handle = self.rigid_body_set.insert(rigid_body);
        let collider_handle = self.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut self.rigid_body_set,
        );

        let dribler = ColliderBuilder::cuboid(DRIBLER_LENGTH, DRIBLER_WIDTH, DRIBLER_HEIGHT)
            .collision_groups(InteractionGroups::new(
                DRIBLER_COLLISION_GROUP.into(),
                BALL_COLLISION_GROUP.into(),
            ))
            .user_data(DRIBBLER_USER_DATA)
            .translation(Vector::new(PLAYER_RADIUS + DRIBLER_LENGTH / 2.0, 0.0, 0.0))
            .sensor(true)
            .active_events(ActiveEvents::COLLISION_EVENTS)
            .build();

        let dribler_colider_handle = self.collider_set.insert_with_parent(
            dribler,
            rigid_body_handle,
            &mut self.rigid_body_set,
        );

        self.players.push(Player {
            id,
            is_own,
            rigid_body_handle,
            _collider_handle: collider_handle,
            // colision group 1
            _dribbler_collider_handle: dribler_colider_handle,
            // colision group 2
            last_cmd_time: 0.0,
            target_velocity: Vector::zeros(),
            target_ang_velocity: 0.0,
        })
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
