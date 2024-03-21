use dies_protos::{ssl_vision_detection::{SSL_DetectionFrame, SSL_DetectionRobot}, ssl_vision_wrapper::SSL_WrapperPacket};
use dies_ssl_client::VisionClientConfig;
use dies_serial_client::SerialClientConfig;
use dies_core::PlayerCmd;
use rapier3d::prelude::*;
use std::{sync::{Arc, Mutex}, thread::sleep, time::{Duration, SystemTime}};
use tokio::sync::mpsc;

const PLAYER_RADIUS: f32 = 200.0;
const PLAYER_HEIGHT: f32 = 140.0;
const PLAYER_MASS: f32 = 1.0;

pub struct SimulationConfig {
    pub gravity: Vector<f32>,
    pub packets: u32, // packets per second: 50 hz
    pub bias: f32,
    pub velocity: f32, // TODO what is this?
    pub delay: Duration, // delay in the emited packets
    // TODO addition: run simulator faster than real time => keep my own time stamps
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            gravity: Vector::y() * -9.81 * 1000.0,
            packets: 50,
            bias: 0.0,
            velocity: 0.0,
            delay: Duration::from_millis(1000), // 1 second
        }
    }
}

struct Player {
    id: u32,
    is_own: bool,
    rigid_body_handle: RigidBodyHandle,
    collider_handle: ColliderHandle,
}

struct TimedPlayerCmd {
    received_time: SystemTime,
    player_cmd: PlayerCmd,
}

pub struct Simulation {
    gravity: Vector<f32>,
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
    pub fn spawn() -> (VisionClientConfig, SerialClientConfig) {
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
        let (vision_tx, vision_rx) = mpsc::unbounded_channel();
        let queue = Arc::new(Mutex::new(Vec::new())); // can be optimized
        let simulation = Arc::new(Mutex::new(Simulation::new(SimulationConfig::default())));

        let receiver_task = {
            let queue = queue.clone();
            tokio::spawn(async move {
                while let Some(cmd) = cmd_rx.recv().await {
                    let timed_cmd = TimedPlayerCmd {
                        received_time: SystemTime::now(),
                        player_cmd: cmd,
                    };
                    let mut q = queue.lock().unwrap();
                    q.push(timed_cmd);
                }
            })
        };
    
        let executor_task = {
            let queue = queue.clone();
            tokio::spawn(async move {
                loop {
                    sleep(Duration::from_millis(100)); // Check interval, adjust as needed
                    let mut q = queue.lock().unwrap();

                    if let Some(first_cmd) = q.first() {
                        // Adjust this condition based on your timing logic
                        if first_cmd.received_time <= SystemTime::now() {
                            if let timed_cmd = q.remove(0) {
                                // Execute the step function
                                let mut simulation = simulation.lock().unwrap();
                                let geometry = simulation.step(timed_cmd.player_cmd);
                                vision_tx.send(geometry).unwrap();
                            }
                        }
                    }
                }
            })
        };

        (VisionClientConfig::InMemory(vision_rx), SerialClientConfig::Memory(cmd_tx))
    
    }

    pub fn new(config: SimulationConfig) -> Simulation {
        let mut simulation = Simulation {
            gravity: config.gravity,
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
                .translation(Vector::new(i as f32 * 2.0, 5.0, 0.0))
                .build();
            let collider = ColliderBuilder::cuboid(1.0, 1.0, 1.0).build();
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
                collider_handle,
            })
        }

        simulation
    }

    // driver: get state, add new commands, return wrapper packet
    // keep queue with player commmands with timestapms
    
    pub fn step(&mut self, player_cmd: PlayerCmd) -> SSL_WrapperPacket {
        self.physics_pipeline.step(
            &self.gravity,
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
}
