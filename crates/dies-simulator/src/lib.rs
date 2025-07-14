use std::{
    collections::{HashMap, HashSet, VecDeque},
    f64::consts::PI,
};

use dies_core::{
    Angle, FieldGeometry, GcSimCommand, PlayerFeedbackMsg, PlayerId, PlayerMoveCmd, RobotCmd,
    SideAssignment, SysStatus, TeamColor, Vector2, Vector3, WorldInstant,
};
use dies_protos::{
    ssl_gc_referee_message::{referee, Referee},
    ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame, SSL_DetectionRobot},
    ssl_vision_geometry::{
        SSL_FieldCircularArc, SSL_FieldLineSegment, SSL_GeometryData, SSL_GeometryFieldSize,
    },
    ssl_vision_wrapper::SSL_WrapperPacket,
};
use rapier3d_f64::prelude::*;
use serde::Serialize;
use std::fmt;
use utils::IntervalTrigger;

mod utils;

// Simulation constants - these are in mm
const BALL_RADIUS: f64 = 21.45;
const GROUND_THICKNESS: f64 = 10.0;
const WALL_HEIGHT: f64 = 1000.0;
const WALL_THICKNESS: f64 = 1.0;

#[derive(Debug, Clone)]
pub struct SimulationConfig {
    // PHYSICAL CONSTANTS
    /// Gravity vector in mm/s^2
    pub gravity: Vector<f64>,
    /// Angular damping (rolling friction) on the ball
    pub ball_damping: f64,
    /// Time between vision updates
    pub vision_update_step: f64,

    // ROBOT MODEL PARAMETERS
    /// Radius of the player in mm
    pub player_radius: f64,
    /// Height of the player in mm
    pub player_height: f64,
    /// Maximum reach of the dribbler from the edge of the robot in mm
    pub dribbler_radius: f64,
    /// Maximum angle from the front of the robot where the dribbler can pick up the ball
    /// in radians
    pub dribbler_angle: f64,
    /// Maximum force exerted by the kicker
    pub kicker_strength: f64,
    /// Time after which the player defaults to zero velocity if no command is received
    pub player_cmd_timeout: f64,
    /// Strength of the dribbler in picking up the ball
    pub dribbler_strength: f64,
    /// Delay for the execution of the command
    pub command_delay: f64,
    /// Maximum lateral acceleration in mm/s^2
    pub max_accel: f64,
    /// Maximum lateral speed in mm/s
    pub max_vel: f64,
    /// Maximum angular acceleration in rad/s^2
    pub max_ang_accel: f64,
    /// Maximum angular velocity in rad/s
    pub max_ang_vel: f64,
    /// Maximum difference between target and current velocity
    pub velocity_treshold: f64,
    /// Maximum difference between target and current angular velocity
    pub angular_velocity_treshold: f64,
    /// Interval for sending feedback packets in seconds
    pub feedback_interval: f64,
    /// Whether the robot has an IMU and is capable of absolute orientation
    pub has_imu: bool,

    // FIELD GEOMRTY PARAMETERS
    /// Field configuration
    pub field_geometry: FieldGeometry,
    /// Interval for sending geometry packets
    pub geometry_interval: f64,

    // TEAM CONFIGURATION
    /// Which teams are controlled/simulated by the AI
    pub blue_controlled: bool,
    pub yellow_controlled: bool,
    pub initial_side_assignment: SideAssignment,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            // PHYSICAL CONSTANTS
            gravity: Vector::z() * -9.81 * 1000.0,
            ball_damping: 1.0,
            vision_update_step: 1.0 / 40.0,

            // ROBOT MODEL PARAMETERS
            player_radius: 90.0,
            player_height: 140.0,
            dribbler_radius: BALL_RADIUS + 120.0,
            dribbler_angle: PI / 6.0,
            kicker_strength: 300000.0,
            player_cmd_timeout: 0.1,
            dribbler_strength: 0.6,
            command_delay: 30.0 / 1000.0,
            max_accel: 15000.0,
            max_vel: 6000.0,
            max_ang_accel: 50.0 * 720.0f64.to_radians(),
            max_ang_vel: 0.5 * 720.0f64.to_radians(),
            velocity_treshold: 1.0,
            angular_velocity_treshold: 0.0,
            feedback_interval: 0.05,
            has_imu: true,

            // FIELD GEOMRTY PARAMETERS
            field_geometry: FieldGeometry::default(),
            geometry_interval: 3.0,

            // TEAM CONFIGURATION
            blue_controlled: true,
            yellow_controlled: false,
            initial_side_assignment: SideAssignment::BlueOnPositive,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub enum SimulationGameState {
    #[default]
    StartGame,
    Stop,
    Halt,
    StopAndKickOff {
        team_color: TeamColor,
        stop_timer: Timer,
    },
    StopAndForceStart {
        stop_timer: Timer,
    },
    PrepareKickOff {
        team_color: TeamColor,
    },
    Run {
        wait_timer: Timer,
        no_progress_timer: Timer,
        last_ball_position: Option<Vector2>,
    },
    FreeKick {
        kick_timer: Timer,
        ball_is_kicked: bool,
        team_color: TeamColor,
    },
    Penalty {
        team_color: TeamColor,
    },
    BallPlacement {
        team_color: TeamColor,
        position: Vector2,
    },
    Null,
}

impl SimulationGameState {
    fn run() -> Self {
        SimulationGameState::Run {
            wait_timer: Timer::new(0.2),
            no_progress_timer: Timer::new(10.0),
            last_ball_position: None,
        }
    }

    fn stop() -> Self {
        SimulationGameState::Stop
    }

    fn stop_and_kickoff(team_color: TeamColor) -> Self {
        SimulationGameState::StopAndKickOff {
            stop_timer: Timer::new(1.0),
            team_color,
        }
    }

    fn stop_and_force_start() -> Self {
        SimulationGameState::StopAndForceStart {
            stop_timer: Timer::new(1.0),
        }
    }

    fn prepare_kickoff(team_color: TeamColor) -> Self {
        SimulationGameState::PrepareKickOff { team_color }
    }

    fn free_kick(team_color: TeamColor) -> Self {
        SimulationGameState::FreeKick {
            kick_timer: Timer::new(10.0),
            ball_is_kicked: false,
            team_color,
        }
    }

    fn penalty(team_color: TeamColor) -> Self {
        SimulationGameState::Penalty { team_color }
    }

    fn ball_placement(team_color: TeamColor, position: Vector2) -> Self {
        SimulationGameState::BallPlacement {
            team_color,
            position,
        }
    }
}

impl PartialEq for SimulationGameState {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (SimulationGameState::StartGame, SimulationGameState::StartGame) => true,
            (SimulationGameState::Stop, SimulationGameState::Stop) => true,
            (SimulationGameState::Halt, SimulationGameState::Halt) => true,
            (
                SimulationGameState::StopAndKickOff { .. },
                SimulationGameState::StopAndKickOff { .. },
            ) => true,
            (
                SimulationGameState::PrepareKickOff { .. },
                SimulationGameState::PrepareKickOff { .. },
            ) => true,
            (SimulationGameState::Run { .. }, SimulationGameState::Run { .. }) => true,
            (SimulationGameState::FreeKick { .. }, SimulationGameState::FreeKick { .. }) => true,
            (SimulationGameState::Penalty { .. }, SimulationGameState::Penalty { .. }) => true,
            (
                SimulationGameState::BallPlacement { .. },
                SimulationGameState::BallPlacement { .. },
            ) => true,
            _ => false,
        }
    }
}

impl fmt::Display for SimulationGameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SimulationGameState::StartGame => write!(f, "Start game"),
            SimulationGameState::Stop => write!(f, "Stop"),
            SimulationGameState::Halt => write!(f, "Halt"),
            SimulationGameState::StopAndKickOff { .. } => write!(f, "Stop and kick off"),
            SimulationGameState::StopAndForceStart { .. } => write!(f, "Stop and force start"),
            SimulationGameState::PrepareKickOff { .. } => write!(f, "Prepare kick off"),
            SimulationGameState::Run { .. } => write!(f, "Run"),
            SimulationGameState::FreeKick { .. } => write!(f, "Free kick"),
            SimulationGameState::Penalty { .. } => write!(f, "Penalty"),
            SimulationGameState::BallPlacement { .. } => write!(f, "Ball placement"),
            SimulationGameState::Null => write!(f, "Null state"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub enum RefereeMessage {
    RobotTooCloseToOpponentDefenseArea, //8.4.1
    BoundaryCrossing,                   //8.4.1
    DefenderTooCloseToBall,             //8.4.3
    BallPlacementInterference,          //8.4.3
    RobotOutOfField,
    FreekickTimeExceeded,
    PlayerTooCloseToBall,
    KickoffPositionViolation,
    DoubleTouchViolation,
    PenaltyKeeperPositionViolation,
    PenaltyRobotPositionViolation,
    PenaltyBallDirectionViolation,
    PenaltyTimeExceeded,
}

impl fmt::Display for RefereeMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefereeMessage::RobotTooCloseToOpponentDefenseArea => {
                write!(f, "Robot too close to opponent defense area")
            }
            RefereeMessage::BoundaryCrossing => write!(f, "Boundary crossing"),
            RefereeMessage::DefenderTooCloseToBall => write!(f, "Defender too close to ball"),
            RefereeMessage::FreekickTimeExceeded => write!(f, "Free kick time exceeded"),
            RefereeMessage::BallPlacementInterference => write!(f, "Ball placement interference"),
            RefereeMessage::RobotOutOfField => write!(f, "Robot out of field"),
            RefereeMessage::PlayerTooCloseToBall => write!(f, "Player too close to ball"),
            RefereeMessage::KickoffPositionViolation => write!(f, "Kickoff position violation"),
            RefereeMessage::DoubleTouchViolation => write!(f, "Double touch violation"),
            RefereeMessage::PenaltyKeeperPositionViolation => {
                write!(f, "Penalty keeper position violation")
            }
            RefereeMessage::PenaltyRobotPositionViolation => {
                write!(f, "Penalty robot position violation")
            }
            RefereeMessage::PenaltyBallDirectionViolation => {
                write!(f, "Penalty ball direction violation")
            }
            RefereeMessage::PenaltyTimeExceeded => write!(f, "Penalty time exceeded"),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SimulationState {
    players: Vec<SimulationPlayerState>,
    ball: Option<Vector<f64>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SimulationPlayerState {
    position: Vector2,
    yaw: Angle,
}

#[derive(Debug, Clone)]
pub struct Timer {
    counter: f64,
    duration: f64, // in seconds
}

impl Timer {
    pub fn new(duration: f64) -> Self {
        Timer {
            counter: 0.0,
            duration: duration,
        }
    }

    pub fn reset(&mut self) {
        self.counter = 0.0;
    }

    pub fn tick(&mut self, dt: f64) -> bool {
        self.counter = self.counter + dt;
        if self.counter >= self.duration {
            self.reset();
            return true;
        }
        return false;
    }
}

#[derive(Debug)]
struct Ball {
    _rigid_body_handle: RigidBodyHandle,
    _collider_handle: ColliderHandle,
}

#[derive(Debug)]
struct Player {
    id: PlayerId,
    team_color: TeamColor,
    rigid_body_handle: RigidBodyHandle,
    _collider_handle: ColliderHandle,
    last_cmd_time: f64,
    target_velocity: Vector<f64>,
    target_z: f64,
    current_dribble_speed: f64,
    breakbeam: bool,
    heading_control: bool,
}

#[derive(Debug, Clone)]
struct TimedPlayerCmd {
    team_color: TeamColor,
    execute_time: f64,
    player_cmd: PlayerMoveCmd,
}

/// A complete simulator for testing strategies and robot control in silico.
///
/// ## Usage
///
/// ```ignore
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
    referee_message: VecDeque<Referee>,
    feedback_interval: IntervalTrigger,
    feedback_queue: HashMap<(TeamColor, PlayerId), PlayerFeedbackMsg>,
    game_state: SimulationGameState,
    designated_ball_position: Vector<f64>,
    last_touch_info: Option<(PlayerId, TeamColor)>,
    side_assignment: SideAssignment,
    ball_being_dribbled_by: Option<(PlayerId, TeamColor)>,
    // New fields for rule enforcement
    kick_start_time: f64,
    kick_ball_position: Option<Vector2>,
    last_kicker_info: Option<(PlayerId, TeamColor)>,
    kick_in_progress: bool,
}

impl Simulation {
    /// Createa a new instance of [`Simulation`]. After creation, the simulation is
    /// empty and needs to be populated with players and a ball. It is better to use
    /// [`SimulationBuilder`] to create a new simulation and add players and a ball.
    pub fn new(config: SimulationConfig) -> Simulation {
        let vision_update_step = config.vision_update_step;
        let geometry_interval = config.geometry_interval;
        let feedback_interval = config.feedback_interval;
        let field_length = config.field_geometry.field_length;
        let field_width = config.field_geometry.field_width;
        let geometry_packet = geometry(&config.field_geometry);

        let side_assignment = config.initial_side_assignment.clone();
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
            geometry_interval: IntervalTrigger::new(geometry_interval),
            geometry_packet,
            referee_message: VecDeque::new(),
            feedback_interval: IntervalTrigger::new(feedback_interval),
            feedback_queue: HashMap::new(),
            game_state: SimulationGameState::default(),
            designated_ball_position: Vector::new(0.0, 0.0, 20.0),
            last_touch_info: None,
            side_assignment,
            ball_being_dribbled_by: None,
            kick_start_time: 0.0,
            kick_ball_position: None,
            last_kicker_info: None,
            kick_in_progress: false,
        };

        // Create the ground
        let ground_body = RigidBodyBuilder::fixed()
            // z=0.0 is the ground surface
            .translation(Vector::new(0.0, 0.0, -GROUND_THICKNESS / 2.0))
            .build();
        let ground_collider =
            ColliderBuilder::cuboid(field_length / 2.0, field_width / 2.0, GROUND_THICKNESS)
                .build();
        let ground_body_handle = simulation.rigid_body_set.insert(ground_body);
        simulation.collider_set.insert_with_parent(
            ground_collider,
            ground_body_handle,
            &mut simulation.rigid_body_set,
        );

        simulation.add_wall(0.0, field_width / 2.0, field_length, WALL_THICKNESS);
        simulation.add_wall(0.0, -field_width / 2.0, field_length, WALL_THICKNESS);
        simulation.add_wall(field_length / 2.0, 0.0, WALL_THICKNESS, field_width);
        simulation.add_wall(-field_length / 2.0, 0.0, WALL_THICKNESS, field_width);

        simulation
    }

    pub fn time(&self) -> WorldInstant {
        WorldInstant::Simulated(self.current_time)
    }

    /// Get the controlled teams
    pub fn controlled_teams(&self) -> HashSet<TeamColor> {
        let mut teams = HashSet::new();
        if self.config.blue_controlled {
            teams.insert(TeamColor::Blue);
        }
        if self.config.yellow_controlled {
            teams.insert(TeamColor::Yellow);
        }
        teams
    }

    /// Set which teams are controlled by the simulation
    pub fn set_controlled_teams(&mut self, teams: &[TeamColor]) {
        if teams.contains(&TeamColor::Blue) {
            self.config.blue_controlled = true;
        } else {
            self.config.blue_controlled = false;
        }
        if teams.contains(&TeamColor::Yellow) {
            self.config.yellow_controlled = true;
        } else {
            self.config.yellow_controlled = false;
        }
    }

    /// Check if a team is controlled
    pub fn is_team_controlled(&self, team_color: TeamColor) -> bool {
        match team_color {
            TeamColor::Blue => self.config.blue_controlled,
            TeamColor::Yellow => self.config.yellow_controlled,
        }
    }

    fn add_wall(&mut self, x: f64, y: f64, width: f64, height: f64) {
        let wall_body = RigidBodyBuilder::fixed()
            .translation(Vector::new(x, y, 0.0))
            .build();
        let wall_collider = ColliderBuilder::cuboid(width, height, WALL_HEIGHT)
            .restitution(0.0)
            .restitution_combine_rule(CoefficientCombineRule::Min)
            .build();
        let wall_body_handle = self.rigid_body_set.insert(wall_body);
        self.collider_set.insert_with_parent(
            wall_collider,
            wall_body_handle,
            &mut self.rigid_body_set,
        );
    }

    pub fn apply_force_to_ball(&mut self, force: Vector<f64>) {
        if let Some(ball) = self.ball.as_ref() {
            let ball_body = self
                .rigid_body_set
                .get_mut(ball._rigid_body_handle)
                .unwrap();
            ball_body.apply_impulse(force, true);
        }
    }

    pub fn teleport_robot(
        &mut self,
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
            if let Some(rigid_body) = self.rigid_body_set.get_mut(player.rigid_body_handle) {
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

    pub fn add_robot(
        &mut self,
        team_color: TeamColor,
        player_id: PlayerId,
        position: Vector2,
        yaw: Angle,
    ) {
        // Remove any existing robot with the same id and color
        self.remove_robot(team_color, player_id);
        let player_radius = self.config.player_radius;
        let player_height = self.config.player_height;
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
        let rigid_body_handle = self.rigid_body_set.insert(rigid_body);
        let collider_handle = self.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut self.rigid_body_set,
        );
        self.players.push(Player {
            id: player_id,
            team_color,
            rigid_body_handle,
            _collider_handle: collider_handle,
            last_cmd_time: 0.0,
            target_velocity: Vector::zeros(),
            target_z: 0.0,
            current_dribble_speed: 0.0,
            breakbeam: false,
            heading_control: false,
        });
    }

    pub fn remove_robot(&mut self, team_color: TeamColor, player_id: PlayerId) {
        if let Some(idx) = self
            .players
            .iter()
            .position(|p| p.id == player_id && p.team_color == team_color)
        {
            let player = self.players.remove(idx);
            self.rigid_body_set.remove(
                player.rigid_body_handle,
                &mut self.island_manager,
                &mut self.collider_set,
                &mut self.impulse_joint_set,
                &mut self.multibody_joint_set,
                true,
            );
        }
    }

    /// Pushes a PlayerCmd onto the execution queue with the time delay specified in
    /// the config
    pub fn push_cmd(&mut self, team_color: TeamColor, cmd: PlayerMoveCmd) {
        self.cmd_queue.push(TimedPlayerCmd {
            team_color,
            execute_time: self.current_time + self.config.command_delay,
            player_cmd: cmd,
        });
    }

    pub fn detection(&mut self) -> Option<SSL_WrapperPacket> {
        self.last_detection_packet.take()
    }

    pub fn handle_gc_command(&mut self, command: GcSimCommand) {
        match command {
            GcSimCommand::KickOff { team_color } => {
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(referee::Command::PREPARE_KICKOFF_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(referee::Command::PREPARE_KICKOFF_YELLOW);
                    }
                }
                self.game_state = SimulationGameState::prepare_kickoff(team_color);
            }
            GcSimCommand::Penalty { team_color } => {
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(referee::Command::PREPARE_PENALTY_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(referee::Command::PREPARE_PENALTY_YELLOW);
                    }
                }
                self.game_state = SimulationGameState::penalty(team_color);
            }
            GcSimCommand::Stop => {
                self.update_referee_command(referee::Command::STOP);
                self.game_state = SimulationGameState::stop();
            }
            GcSimCommand::Halt => self.update_referee_command(referee::Command::HALT),
            GcSimCommand::NormalStart => {
                self.update_referee_command(referee::Command::NORMAL_START)
            }
            GcSimCommand::ForceStart => self.update_referee_command(referee::Command::FORCE_START),
            GcSimCommand::DirectFree { team_color } => {
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(referee::Command::DIRECT_FREE_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(referee::Command::DIRECT_FREE_YELLOW);
                    }
                }
                self.game_state = SimulationGameState::free_kick(team_color);
            }
            GcSimCommand::BallPlacement {
                team_color,
                position,
            } => {
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(referee::Command::BALL_PLACEMENT_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(referee::Command::BALL_PLACEMENT_YELLOW);
                    }
                }
                self.game_state = SimulationGameState::ball_placement(team_color, position);
            }
        }
    }

    fn update_referee_command(&mut self, command: referee::Command) {
        let mut msg = Referee::new();
        msg.set_command(command);
        msg.packet_timestamp = Some(0);
        msg.set_stage(referee::Stage::NORMAL_FIRST_HALF);
        msg.command_counter = Some(1);
        msg.command_timestamp = Some(0);
        if let SimulationGameState::BallPlacement { position, .. } = &self.game_state {
            let mut point = referee::Point::new();
            point.set_x(position.x as f32);
            point.set_y(position.y as f32);
            msg.designated_position = Some(point).into();
        }
        self.referee_message.push_back(msg);
    }

    pub fn gc_message(&mut self) -> Option<Referee> {
        self.referee_message.pop_front()
    }

    pub fn geometry(&mut self) -> Option<SSL_WrapperPacket> {
        if self.geometry_interval.trigger(self.current_time) {
            Some(self.geometry_packet.clone())
        } else {
            None
        }
    }

    pub fn feedback(&mut self) -> HashMap<(TeamColor, PlayerId), PlayerFeedbackMsg> {
        std::mem::take(&mut self.feedback_queue)
    }

    pub fn update_game_state(&mut self) {
        dies_core::debug_string("Auto Game State", self.game_state.to_string());

        let mut game_state = std::mem::take(&mut self.game_state);
        self.game_state = match game_state {
            SimulationGameState::StartGame => {
                self.update_referee_command(referee::Command::STOP);
                SimulationGameState::stop_and_kickoff(TeamColor::Blue)
            }
            SimulationGameState::Stop => game_state,
            SimulationGameState::Halt => game_state,
            SimulationGameState::StopAndKickOff {
                team_color,
                ref mut stop_timer,
            } => {
                // Wait for 1s before starting the game
                if stop_timer.tick(self.integration_parameters.dt) {
                    match team_color {
                        TeamColor::Blue => {
                            self.update_referee_command(referee::Command::PREPARE_KICKOFF_BLUE);
                        }
                        TeamColor::Yellow => {
                            self.update_referee_command(referee::Command::PREPARE_KICKOFF_YELLOW);
                        }
                    }
                    SimulationGameState::prepare_kickoff(team_color)
                } else {
                    game_state
                }
            }
            SimulationGameState::StopAndForceStart { ref mut stop_timer } => {
                // Wait for 1s before starting the game
                if stop_timer.tick(self.integration_parameters.dt) {
                    self.update_referee_command(referee::Command::FORCE_START);
                    SimulationGameState::run()
                } else {
                    game_state
                }
            }
            SimulationGameState::PrepareKickOff { team_color } => {
                if self.check_kickoff_positions(team_color) {
                    // Initialize kick tracking
                    self.kick_start_time = self.current_time;
                    if let Some(ball_handle) =
                        self.ball.as_ref().map(|ball| ball._rigid_body_handle)
                    {
                        let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
                        self.kick_ball_position =
                            Some(ball_body.position().translation.vector.xy());
                    }
                    self.kick_in_progress = true;
                    self.last_kicker_info = None; // Will be set when robot touches ball
                    self.update_referee_command(referee::Command::NORMAL_START);
                    SimulationGameState::run()
                } else {
                    game_state
                }
            }
            SimulationGameState::Run {
                ref mut no_progress_timer,
                last_ball_position,
                ..
            } => {
                // Check if we're in a kick and should clear tracking due to time or ball movement
                if self.kick_in_progress {
                    let ball_in_play = self.check_ball_in_play();
                    let time_limit = 10.0; // 10 seconds for kickoff
                    let time_exceeded = (self.current_time - self.kick_start_time) > time_limit;

                    if ball_in_play || time_exceeded {
                        self.kick_in_progress = false;
                        self.last_kicker_info = None;
                        self.kick_ball_position = None;
                    }
                }

                if self.goal() {
                    let ball_pos = {
                        let ball_handle = self.ball.as_mut().map(|ball| ball._rigid_body_handle);
                        if let Some(ball_handle) = ball_handle {
                            let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
                            ball_body.position().translation.vector.xy()
                        } else {
                            Vector2::new(0.0, 0.0)
                        }
                    };

                    // Clear kick tracking
                    self.kick_in_progress = false;
                    self.last_kicker_info = None;
                    self.kick_ball_position = None;

                    if ball_pos.x > 0.0 && self.side_assignment == SideAssignment::BlueOnPositive {
                        self.update_referee_command(referee::Command::PREPARE_KICKOFF_BLUE);
                        SimulationGameState::prepare_kickoff(TeamColor::Blue)
                    } else {
                        self.update_referee_command(referee::Command::PREPARE_KICKOFF_YELLOW);
                        SimulationGameState::prepare_kickoff(TeamColor::Yellow)
                    }
                } else if self.ball_out() {
                    // Clear kick tracking
                    self.kick_in_progress = false;
                    self.last_kicker_info = None;
                    self.kick_ball_position = None;

                    if let Some((_kicker_id, kicker_color)) = self.last_touch_info {
                        if kicker_color == TeamColor::Blue {
                            self.update_referee_command(referee::Command::DIRECT_FREE_YELLOW);
                            SimulationGameState::free_kick(TeamColor::Yellow)
                        } else {
                            self.update_referee_command(referee::Command::DIRECT_FREE_BLUE);
                            SimulationGameState::free_kick(TeamColor::Blue)
                        }
                    } else {
                        game_state
                    }
                } else {
                    // Check for no progress
                    let ball_handle = self.ball.as_mut().map(|ball| ball._rigid_body_handle);
                    if let Some(ball_handle) = ball_handle {
                        let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
                        let ball_position_2d = ball_body.position().translation.vector.xy();
                        let last_ball_position_unwrapped =
                            last_ball_position.as_ref().unwrap_or(&ball_position_2d);
                        if (ball_position_2d - last_ball_position_unwrapped).norm() < 10.0 {
                            if no_progress_timer.tick(self.integration_parameters.dt) {
                                self.update_referee_command(referee::Command::STOP);
                                SimulationGameState::stop_and_force_start()
                            } else {
                                game_state
                            }
                        } else {
                            game_state
                        }
                    } else {
                        game_state
                    }
                }
            }
            SimulationGameState::FreeKick {
                team_color,
                ref mut kick_timer,
                ball_is_kicked,
            } => {
                // Check if positions are valid
                if !self.check_free_kick_positions(team_color) {
                    // Reset timer when positions are invalid
                    kick_timer.reset();
                    game_state
                } else {
                    // Initialize kick tracking if not already done
                    if !self.kick_in_progress {
                        self.kick_start_time = self.current_time;
                        if let Some(ball_handle) =
                            self.ball.as_ref().map(|ball| ball._rigid_body_handle)
                        {
                            let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
                            self.kick_ball_position =
                                Some(ball_body.position().translation.vector.xy());
                        }
                        self.kick_in_progress = true;
                        self.last_kicker_info = None;
                    }

                    // Check if ball is in play
                    let ball_in_play = self.check_ball_in_play();

                    let time_limit = 10.0;
                    let time_exceeded = (self.current_time - self.kick_start_time) > time_limit;

                    let is_free_kick_time_exceeded =
                        kick_timer.tick(self.integration_parameters.dt);

                    if ball_in_play || time_exceeded {
                        // Ball is in play or time exceeded - transition to run
                        self.kick_in_progress = false;
                        SimulationGameState::run()
                    } else if is_free_kick_time_exceeded {
                        // Original timer logic - keep for compatibility
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::FreekickTimeExceeded.to_string(),
                        );
                        self.kick_in_progress = false;
                        SimulationGameState::run()
                    } else {
                        game_state
                    }
                }
            }
            SimulationGameState::Penalty { team_color } => {
                // Check if positions are valid
                if !self.check_penalty_positions(team_color) {
                    // Stay in penalty state until positions are correct
                    game_state
                } else {
                    // Initialize kick tracking if not already done
                    if !self.kick_in_progress {
                        self.kick_start_time = self.current_time;
                        if let Some(ball_handle) =
                            self.ball.as_ref().map(|ball| ball._rigid_body_handle)
                        {
                            let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
                            self.kick_ball_position =
                                Some(ball_body.position().translation.vector.xy());
                        }
                        self.kick_in_progress = true;
                        self.last_kicker_info = None;

                        // Issue normal start command
                        self.update_referee_command(referee::Command::NORMAL_START);
                    }

                    // Check if ball is in play
                    let ball_in_play = self.check_ball_in_play();

                    // Check 10 second time limit
                    let time_exceeded = (self.current_time - self.kick_start_time) > 10.0;

                    // Check if ball direction is valid (toward goal)
                    let direction_valid = self.check_penalty_ball_direction(team_color);

                    if time_exceeded {
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::PenaltyTimeExceeded.to_string(),
                        );
                        self.kick_in_progress = false;
                        // Award goal kick to defending team
                        let defending_team = if team_color == TeamColor::Blue {
                            TeamColor::Yellow
                        } else {
                            TeamColor::Blue
                        };
                        match team_color {
                            TeamColor::Blue => {
                                self.update_referee_command(referee::Command::DIRECT_FREE_BLUE);
                            }
                            TeamColor::Yellow => {
                                self.update_referee_command(referee::Command::DIRECT_FREE_YELLOW);
                            }
                        }
                        SimulationGameState::free_kick(defending_team)
                    } else if ball_in_play && !direction_valid {
                        // Ball moved but in wrong direction - award goal kick to defending team
                        self.kick_in_progress = false;
                        let defending_team = if team_color == TeamColor::Blue {
                            TeamColor::Yellow
                        } else {
                            TeamColor::Blue
                        };
                        match team_color {
                            TeamColor::Blue => {
                                self.update_referee_command(referee::Command::DIRECT_FREE_BLUE);
                            }
                            TeamColor::Yellow => {
                                self.update_referee_command(referee::Command::DIRECT_FREE_YELLOW);
                            }
                        }
                        SimulationGameState::free_kick(defending_team)
                    } else if ball_in_play {
                        // Ball in play and direction is valid - continue with run state
                        self.kick_in_progress = false;
                        SimulationGameState::run()
                    } else {
                        game_state
                    }
                }
            }
            SimulationGameState::BallPlacement {
                team_color,
                position,
            } => {
                match team_color {
                    TeamColor::Blue => {
                        self.update_referee_command(referee::Command::BALL_PLACEMENT_BLUE);
                    }
                    TeamColor::Yellow => {
                        self.update_referee_command(referee::Command::BALL_PLACEMENT_YELLOW);
                    }
                }
                let ball_pos = {
                    let ball_handle = self.ball.as_mut().map(|ball| ball._rigid_body_handle);
                    if let Some(ball_handle) = ball_handle {
                        let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
                        ball_body.position().translation.vector.xy()
                    } else {
                        Vector2::new(0.0, 0.0)
                    }
                };
                if (ball_pos - position).norm() < 100.0 {
                    let opposite_team = team_color.opposite();
                    match opposite_team {
                        TeamColor::Blue => {
                            self.update_referee_command(referee::Command::DIRECT_FREE_BLUE);
                        }
                        TeamColor::Yellow => {
                            self.update_referee_command(referee::Command::DIRECT_FREE_YELLOW);
                        }
                    }
                    SimulationGameState::free_kick(opposite_team)
                } else {
                    game_state
                }
            }
            SimulationGameState::Null => {
                // Do nothing, this is a placeholder state
                game_state
            }
        }
    }

    fn kickoff_ready(&mut self) -> bool {
        // Detect the ball's placement
        let ball_handle = self.ball.as_mut().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            // Check if the ball is close to the designated position & inside the field & 0.7m away from the defense area & stationary
            if (ball_position - self.designated_ball_position).norm() < 1000.0
                && ball_position.x.abs() < self.config.field_geometry.field_length / 2.0
                && ball_position.y.abs() < self.config.field_geometry.field_width / 2.0
                && ball_body.linvel().norm() < 0.001
            {
                dies_core::debug_string(
                    "RefereeMessage.ball",
                    "Ball in designated position, no need to move",
                );
            } else {
                // Reset the ball's position to the designated position
                dies_core::debug_string(
                    "RefereeMessage.ball",
                    RefereeMessage::BallPlacementInterference.to_string(),
                );
                // ball_body.set_position(
                //     Isometry::translation(
                //         self.designated_ball_position.x,
                //         self.designated_ball_position.y,
                //         self.designated_ball_position.z,
                //     ),
                //     true,
                // );
                // ball_body.set_linvel(Vector::zeros(), true);
            }
        }

        // Check if all team players in position
        for player in self.players.iter_mut() {
            let rigid_body = self
                .rigid_body_set
                .get_mut(player.rigid_body_handle)
                .unwrap();
            let player_position = rigid_body.position().translation.vector;
            // Check if the player is close to the ball
            if (player_position - self.designated_ball_position).norm() < 500.0 {
                dies_core::debug_string(
                    "RefereeMessage",
                    format!("Defender {} too close to ball", player.id.as_u32()),
                );
                return false;
            }
            // Check if the player is inside the field
            if player_position.y.abs() > self.config.field_geometry.field_width / 2.0 {
                dies_core::debug_string(
                    "RefereeMessage",
                    format!("Player {} out of field", player.id.as_u32()),
                );
                return false;
            }
            // Check if the player is in the own half, default to the left side
            // let team_color = player.team_color;
            // let out_of_field = self
            //     .side_assignment
            //     .is_on_opp_side_vec3(team_color, &player_position);
            // if out_of_field {
            //     dies_core::debug_string(
            //         "RefereeMessage",
            //         format!("Player {} out of field", player.id.as_u32()),
            //     );
            //     return false;
            // }
        }
        true
    }

    fn ball_out(&mut self) -> bool {
        let ball_handle = self.ball.as_mut().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            let test_boundary: f64 = 100.0; // When test in simulation, should not be 0.0 otherwise it will never be triggered
            if ball_position.x.abs() < self.config.field_geometry.field_length / 2.0 - test_boundary
                && ball_position.y.abs()
                    < self.config.field_geometry.field_width / 2.0 - test_boundary
            {
                let ball_velocity: Vector<f64> = *ball_body.linvel(); // ball's current velocity as Vector3
                let next_ball_position =
                    ball_position + ball_velocity * self.integration_parameters.dt;
                if next_ball_position.x.abs()
                    > self.config.field_geometry.field_length / 2.0 - test_boundary
                    || next_ball_position.y.abs()
                        > self.config.field_geometry.field_width / 2.0 - test_boundary
                {
                    dies_core::debug_string(
                        "RefereeMessage.BallOut",
                        &format!("Ball out of bounds at position: {:?}", next_ball_position),
                    );
                    // In next frame, Ball will be out of bounds
                    // Do linear interpolation to find the free kick position
                    let mut designated = ball_position + (next_ball_position - ball_position) * 0.5;
                    // Constrain designated position to the field
                    designated.x = designated.x.clamp(
                        -(self.config.field_geometry.field_length / 2.0 - test_boundary),
                        self.config.field_geometry.field_length / 2.0 - test_boundary,
                    );
                    designated.y = designated.y.clamp(
                        -(self.config.field_geometry.field_width / 2.0 - test_boundary),
                        self.config.field_geometry.field_width / 2.0 - test_boundary,
                    );
                    self.designated_ball_position = designated;
                }
                return false;
            } else {
                dies_core::debug_string(
                    "RefereeMessage",
                    format!("Ball out of bounds at position: {:?}", ball_position),
                );
                // Wait a few frames before resetting the ball's position (default: 0.2s)
                if let SimulationGameState::Run { wait_timer, .. } = &mut self.game_state {
                    if wait_timer.tick(self.integration_parameters.dt) {
                        // Reset the ball's position to the free kick position
                        ball_body.set_position(
                            Isometry::translation(
                                self.designated_ball_position.x,
                                self.designated_ball_position.y,
                                self.designated_ball_position.z,
                            ),
                            true,
                        );
                        ball_body.set_linvel(Vector::zeros(), true);
                    }
                }

                return true;
            }
        }
        false
    }

    fn goal(&mut self) -> bool {
        let ball_handle = self.ball.as_mut().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector; // Center of the ball

            let goal_height = 150.0;
            if ball_position.x.abs()
                > self.config.field_geometry.field_length / 2.0
                    - self.config.field_geometry.goal_depth
                && ball_position.x.abs() < self.config.field_geometry.field_length / 2.0
                && ball_position.y.abs() < self.config.field_geometry.goal_width / 2.0
                && ball_position.z < goal_height
            {
                // Log which side scored
                let scoring_side = if ball_position.x > 0.0 {
                    "positive x side (blue goal)"
                } else {
                    "negative x side (yellow goal)"
                };
                dies_core::debug_string(
                    "RefereeMessage.Goal",
                    &format!("Goal scored at {}", scoring_side),
                );

                // Wait a few frames before resetting the ball's position (default: 0.2s)
                if let SimulationGameState::Run { wait_timer, .. } = &mut self.game_state {
                    if wait_timer.tick(self.integration_parameters.dt) {
                        wait_timer.reset();
                    }
                }

                // Reset the ball's position to the kick off position (center)
                // ball_body.set_position(Isometry::translation(0.0, 0.0, 20.0), true);
                // ball_body.set_linvel(Vector::zeros(), true);
                return true;
            }
        }
        false
    }

    fn players_too_close_to_ball(&self) -> bool {
        let ball_handle = self.ball.as_ref().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            // Check all players regardless of team - referee decisions should be team-neutral
            for player in self.players.iter() {
                let rigid_body = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
                let player_position = rigid_body.position().translation.vector;
                // Check if the player is close to the ball
                if (player_position - ball_position).norm() < 500.0 {
                    dies_core::debug_string(
                        "RefereeMessage",
                        format!("Player {} too close to ball", player.id.as_u32()),
                    );
                    return true;
                }
            }
        }
        false
    }

    fn num_players_too_close_to_ball(&self) -> usize {
        let ball_handle = self.ball.as_ref().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            let mut count = 0;
            for player in self.players.iter() {
                let rigid_body = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
                let player_position = rigid_body.position().translation.vector;
                if (player_position - ball_position).norm() < 500.0 {
                    count += 1;
                }
            }
            count
        } else {
            0
        }
    }

    fn check_kickoff_positions(&self, attacking_team: TeamColor) -> bool {
        let ball_handle = self.ball.as_ref().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            let center_circle_radius = 500.0; // 1m diameter = 0.5m radius
            let touch_distance = self.config.player_radius + BALL_RADIUS;
            let mut attacking_robots_in_circle = 0;

            for player in self.players.iter() {
                let rigid_body = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
                let player_position = rigid_body.position().translation.vector;

                // Check if robot is touching the ball (not allowed)
                if (player_position - ball_position).norm() < touch_distance {
                    dies_core::debug_string(
                        "RefereeMessage",
                        RefereeMessage::KickoffPositionViolation.to_string(),
                    );
                    return false;
                }

                // Check if robot is in own half (excluding center circle)
                let is_on_opp_side = self
                    .side_assignment
                    .is_on_opp_side_vec3(player.team_color, &player_position);
                if is_on_opp_side {
                    // Allow attacking team robot in center circle
                    let distance_to_center = (player_position.xy() - Vector2::new(0.0, 0.0)).norm();
                    if player.team_color == attacking_team
                        && distance_to_center <= center_circle_radius
                    {
                        attacking_robots_in_circle += 1;
                        if attacking_robots_in_circle > 1 {
                            dies_core::debug_string(
                                "RefereeMessage",
                                RefereeMessage::KickoffPositionViolation.to_string(),
                            );
                            return false;
                        }
                    }
                } else {
                    dies_core::debug_string(
                        "RefereeMessage",
                        RefereeMessage::KickoffPositionViolation.to_string(),
                    );
                    return false;
                }
            }
        }
        true
    }

    fn check_free_kick_positions(&self, attacking_team: TeamColor) -> bool {
        let ball_handle = self.ball.as_ref().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            let required_distance = 500.0; // 0.5m

            for player in self.players.iter() {
                // Only check defending team robots
                if player.team_color != attacking_team {
                    let rigid_body = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
                    let player_position = rigid_body.position().translation.vector;

                    if (player_position - ball_position).norm() < required_distance {
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::DefenderTooCloseToBall.to_string(),
                        );
                        return false;
                    }
                }
            }
        }
        true
    }

    fn check_penalty_positions(&self, attacking_team: TeamColor) -> bool {
        let ball_handle = self.ball.as_ref().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            let defending_team = if attacking_team == TeamColor::Blue {
                TeamColor::Yellow
            } else {
                TeamColor::Blue
            };
            let goal_x = if self
                .side_assignment
                .is_on_opp_side_vec3(defending_team, &ball_position)
            {
                self.config.field_geometry.field_length / 2.0
            } else {
                -self.config.field_geometry.field_length / 2.0
            };

            let mut keeper_found = false;
            let mut keeper_position_valid = false;

            for player in self.players.iter() {
                let rigid_body = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
                let player_position = rigid_body.position().translation.vector;

                // Check if this is the keeper (for now, assume robot 0 is keeper)
                if player.team_color == defending_team && player.id.as_u32() == 0 {
                    keeper_found = true;
                    // Check if keeper is on goal line between goal posts
                    let goal_line_tolerance = 20.0; // 2cm tolerance
                    let goal_width = self.config.field_geometry.goal_width;

                    if (player_position.x - goal_x).abs() < goal_line_tolerance
                        && player_position.y.abs() < goal_width / 2.0
                    {
                        keeper_position_valid = true;
                    } else {
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::PenaltyKeeperPositionViolation.to_string(),
                        );
                        return false;
                    }
                } else {
                    // Check if other robots are 1m behind ball
                    let required_distance = 1000.0; // 1m
                    let ball_to_goal_direction = if goal_x > ball_position.x { 1.0 } else { -1.0 };
                    let robot_behind_ball =
                        (player_position.x - ball_position.x) * ball_to_goal_direction < 0.0;

                    if !robot_behind_ball
                        || (player_position - ball_position).norm() < required_distance
                    {
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::PenaltyRobotPositionViolation.to_string(),
                        );
                        return false;
                    }
                }
            }

            if !keeper_found || !keeper_position_valid {
                dies_core::debug_string(
                    "RefereeMessage",
                    RefereeMessage::PenaltyKeeperPositionViolation.to_string(),
                );
                return false;
            }
        }
        true
    }

    fn check_ball_in_play(&self) -> bool {
        if let Some(kick_ball_pos) = self.kick_ball_position {
            let ball_handle = self.ball.as_ref().map(|ball| ball._rigid_body_handle);
            if let Some(ball_handle) = ball_handle {
                let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
                let current_ball_pos = ball_body.position().translation.vector.xy();

                // Check if ball moved 0.05m (50mm)
                if (current_ball_pos - kick_ball_pos).norm() >= 50.0 {
                    return true;
                }
            }
        }
        false
    }

    fn check_double_touch(&self, touching_robot: (PlayerId, TeamColor)) -> bool {
        if let Some(last_kicker) = self.last_kicker_info {
            if last_kicker.0 == touching_robot.0 && last_kicker.1 == touching_robot.1 {
                dies_core::debug_string(
                    "RefereeMessage",
                    RefereeMessage::DoubleTouchViolation.to_string(),
                );
                return true;
            }
        }
        false
    }

    fn check_penalty_ball_direction(&self, attacking_team: TeamColor) -> bool {
        let ball_handle = self.ball.as_ref().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
            let ball_velocity = ball_body.linvel();

            // Check if ball is moving toward opponent goal
            let defending_team = if attacking_team == TeamColor::Blue {
                TeamColor::Yellow
            } else {
                TeamColor::Blue
            };
            let ball_position = ball_body.position().translation.vector;
            let goal_x = if self
                .side_assignment
                .is_on_opp_side_vec3(defending_team, &ball_position)
            {
                self.config.field_geometry.field_length / 2.0
            } else {
                -self.config.field_geometry.field_length / 2.0
            };

            let toward_goal = (goal_x - ball_position.x).signum();
            let ball_direction = ball_velocity.x.signum();

            if ball_velocity.norm() > 0.1 && ball_direction != toward_goal {
                dies_core::debug_string(
                    "RefereeMessage",
                    RefereeMessage::PenaltyBallDirectionViolation.to_string(),
                );
                return false;
            }
        }
        true
    }

    pub fn step(&mut self, dt: f64) {
        // Update last touch info
        if matches!(self.game_state, SimulationGameState::Run { .. }) {
            if let Some(ball_handle) = self.ball.as_ref().map(|b| b._rigid_body_handle) {
                let ball_body = self.rigid_body_set.get(ball_handle).unwrap();
                let ball_position = ball_body.position().translation.vector;
                let touch_threshold = self.config.player_radius + BALL_RADIUS + 5.0;

                // Find the closest player touching the ball in this step
                let mut closest_touching_player: Option<(PlayerId, TeamColor, f64)> = None;

                for player in self.players.iter() {
                    let player_body = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
                    let player_position = player_body.position().translation.vector;
                    let distance = (player_position - ball_position).norm();

                    if distance < touch_threshold {
                        if closest_touching_player.is_none()
                            || distance < closest_touching_player.unwrap().2
                        {
                            closest_touching_player =
                                Some((player.id, player.team_color, distance));
                        }
                    }
                }

                if let Some((id, team_color, _)) = closest_touching_player {
                    // Check for double touch during kicks
                    if self.kick_in_progress {
                        if self.check_double_touch((id, team_color)) {
                            // Double touch violation - stop game and award free kick to opponent
                            self.kick_in_progress = false;
                            self.last_kicker_info = None;
                            self.kick_ball_position = None;

                            let opponent_team = if team_color == TeamColor::Blue {
                                TeamColor::Yellow
                            } else {
                                TeamColor::Blue
                            };
                            self.game_state = SimulationGameState::free_kick(opponent_team);
                            self.update_referee_command(if opponent_team == TeamColor::Blue {
                                referee::Command::DIRECT_FREE_BLUE
                            } else {
                                referee::Command::DIRECT_FREE_YELLOW
                            });
                        } else if self.last_kicker_info.is_none() {
                            // Set the first kicker
                            self.last_kicker_info = Some((id, team_color));
                        }
                    }

                    self.last_touch_info = Some((id, team_color));
                }
            }
        } else {
            self.last_touch_info = None;
        }

        // Update the game state
        self.update_game_state();

        // Create detection update if it's time
        if self.detection_interval.trigger(self.current_time) {
            self.new_detection_packet();
        }

        // Select commands that are due to be executed
        let commands_to_exec = {
            let mut to_exec = HashMap::new();
            self.cmd_queue.retain(|cmd| {
                if cmd.execute_time <= self.current_time {
                    to_exec.insert((cmd.team_color, cmd.player_cmd.id), cmd.player_cmd);
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
        let send_feedback = self.feedback_interval.trigger(self.current_time);
        for player in self.players.iter_mut() {
            // Only update players from controlled teams
            if (!self.config.blue_controlled && player.team_color == TeamColor::Blue)
                || (!self.config.yellow_controlled && player.team_color == TeamColor::Yellow)
            {
                let rigid_body = self
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
                feedback.kicker_status = Some(SysStatus::Ready);
                feedback.imu_status = if self.config.has_imu {
                    Some(SysStatus::Ready)
                } else {
                    Some(SysStatus::NotInstalled)
                };
                self.feedback_queue
                    .insert((player.team_color, player.id), feedback);
            }

            let mut is_kicking = false;
            if let Some(command) = commands_to_exec.get(&(player.team_color, player.id)) {
                // In the robot's local frame, +sx means forward, +sy means right and both are in m/s
                // Angular velocity is in rad/s and +w means counter-clockwise
                player.target_velocity = Vector::new(command.sx, -command.sy, 0.0) * 1000.0; // mm/s to m/s
                player.target_z = -command.w;
                player.current_dribble_speed = command.dribble_speed;
                player.last_cmd_time = self.current_time;
                is_kicking = matches!(command.robot_cmd, RobotCmd::Kick);

                match command.robot_cmd {
                    RobotCmd::Kick => {
                        is_kicking = true;
                    }
                    RobotCmd::YawRateControl => {
                        player.heading_control = false;
                    }
                    RobotCmd::HeadingControl => {
                        player.heading_control = true;
                    }
                    _ => {}
                }
            }

            let rigid_body = self
                .rigid_body_set
                .get_mut(player.rigid_body_handle)
                .unwrap();

            if (self.current_time - player.last_cmd_time).abs() > self.config.player_cmd_timeout {
                player.target_velocity = Vector::zeros();
                player.target_z = 0.0;
            }

            let velocity = rigid_body.linvel();

            // Convert to global frame
            let target_velocity =
                Rotation::<f64>::new(Vector::z() * rigid_body.rotation().euler_angles().2)
                    * player.target_velocity;

            let vel_err = target_velocity - velocity;

            let new_vel = {
                let acc = (vel_err / dt).cap_magnitude(self.config.max_accel);
                velocity + acc * dt
            };
            let new_vel = new_vel.cap_magnitude(self.config.max_vel);
            rigid_body.set_linvel(new_vel, true);

            if player.heading_control {
                // Move towards the target heading with config.max_ang_vel
                let current_yaw = rigid_body.rotation().euler_angles().2;
                let target_yaw = player.target_z;
                let yaw_err = target_yaw - current_yaw;
                let new_yaw = if yaw_err.abs() > 5f64.to_degrees() {
                    let ang_vel = (yaw_err / dt).min(self.config.max_ang_vel / dt);
                    current_yaw + ang_vel * dt
                } else {
                    target_yaw
                };
                rigid_body.set_rotation(Rotation::from_euler_angles(0.0, 0.0, new_yaw), true);
            } else {
                let target_ang_vel = player.target_z;
                let new_ang_vel =
                    target_ang_vel.clamp(-self.config.max_ang_vel, self.config.max_ang_vel);
                rigid_body.set_angvel(Vector::z() * new_ang_vel, true);
            }

            // Check if the ball is in the dribbler
            let yaw = rigid_body.position().rotation * Vector::x();
            let player_position = rigid_body.position().translation.vector;
            let ball_handle = self.ball.as_ref().map(|ball| ball._rigid_body_handle);
            if let Some(ball_handle) = ball_handle {
                let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
                let ball_position = ball_body.position().translation.vector;
                let ball_dir = ball_position - player_position;
                let distance = ball_dir.norm();
                let angle = yaw.angle(&ball_dir);
                let is_player_dribbling = matches!(self.ball_being_dribbled_by, Some((player_id, team_color)) if player_id == player.id && team_color == player.team_color);
                if distance < self.config.player_radius + self.config.dribbler_radius + 20.0
                    && angle < self.config.dribbler_angle
                {
                    player.breakbeam = true;
                    dies_core::debug_string(
                        format!("breakbeam_{}_{}", player.team_color, player.id),
                        format!("{} {}", player.team_color, player.id),
                    );
                    if is_kicking {
                        self.ball_being_dribbled_by = None;
                        let force = yaw * self.config.kicker_strength;
                        ball_body.add_force(force, true);
                        ball_body.set_linear_damping(self.config.ball_damping * 2.0);
                        // Set the ball_is_kicked to true
                        if let SimulationGameState::FreeKick { ball_is_kicked, .. } =
                            &mut self.game_state
                        {
                            *ball_is_kicked = true;
                        }
                    } else if player.current_dribble_speed > 0.0 {
                        self.ball_being_dribbled_by = Some((player.id, player.team_color));
                    } else {
                        if is_player_dribbling {
                            // If the player is dribbling, we need to stop the dribbling
                            self.ball_being_dribbled_by = None;
                        }
                    }
                }
            }
        }

        if let Some((player_id, team_color)) = self.ball_being_dribbled_by {
            // println!("Dribbling ball by {} {}", team_color, player_id);
            dies_core::debug_string("dribbling_ball", format!("{} {}", team_color, player_id));
            let player = self
                .players
                .iter()
                .find(|p| p.id == player_id && p.team_color == team_color)
                .unwrap();
            let rigid_body = self
                .rigid_body_set
                .get_mut(player.rigid_body_handle)
                .unwrap();
            let yaw = rigid_body.position().rotation * Vector::x();
            let player_position = rigid_body.position().translation.vector;
            let ball_body = self
                .rigid_body_set
                .get_mut(self.ball.as_ref().unwrap()._rigid_body_handle)
                .unwrap();

            // Fix the ball position to the dribbler
            let dribbler_position = player_position
                + yaw * (self.config.player_radius + self.config.dribbler_radius - 90.0);
            ball_body.set_position(
                Isometry::translation(
                    dribbler_position.x,
                    dribbler_position.y,
                    dribbler_position.z,
                ),
                true,
            );
            ball_body.set_linvel(Vector3::zeros(), true);
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

        // Clamp ball position to the field
        if let Some(ball) = self.ball.as_ref() {
            let ball_body = self
                .rigid_body_set
                .get_mut(ball._rigid_body_handle)
                .unwrap();
            let position = ball_body.position().translation.vector;
            let x = position.x.clamp(
                -(self.config.field_geometry.field_width
                    + self.config.field_geometry.boundary_width)
                    / 2.0,
                (self.config.field_geometry.field_width
                    + self.config.field_geometry.boundary_width)
                    / 2.0,
            );
            let y = position.y.clamp(
                -(self.config.field_geometry.field_length
                    + self.config.field_geometry.boundary_width)
                    / 2.0,
                (self.config.field_geometry.field_length
                    + self.config.field_geometry.boundary_width)
                    / 2.0,
            );
            ball_body.set_position(Isometry::translation(x, y, position.z), true);
        }

        self.current_time += dt;
    }

    fn new_detection_packet(&mut self) {
        let mut detection = SSL_DetectionFrame::new();
        detection.set_frame_number(1);
        detection.set_t_capture(self.current_time);
        detection.set_t_sent(self.current_time);
        detection.set_camera_id(1);

        // Players
        for player in self.players.iter() {
            let rigid_body = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
            let position = rigid_body.position().translation.vector;
            // Add high frequency noise to the position
            // let position = position
            //     + Vector::new(
            //         (rand::random::<f64>() - 0.5) * 5.0,
            //         (rand::random::<f64>() - 0.5) * 5.0,
            //         0.0,
            //     );

            let yaw = rigid_body.rotation().euler_angles().2;
            // Add high frequency noise to the yaw
            // let yaw = yaw + ((rand::random::<f64>() - 0.5) * 4f64).to_radians();

            let mut robot = SSL_DetectionRobot::new();
            robot.set_robot_id(player.id.as_u32());
            robot.set_x(position.x as f32);
            robot.set_y(position.y as f32);
            robot.set_pixel_x(0.0);
            robot.set_pixel_y(0.0);
            robot.set_orientation(yaw as f32);
            robot.set_confidence(1.0);
            if player.team_color == TeamColor::Blue {
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
            ball_det.set_x(ball_position.x as f32);
            ball_det.set_y(ball_position.y as f32);
            ball_det.set_z(ball_position.z as f32);
            ball_det.set_confidence(1.0);
            ball_det.set_pixel_x(0.0);
            ball_det.set_pixel_y(0.0);
            detection.balls.push(ball_det);
        }

        let mut packet = SSL_WrapperPacket::new();
        packet.detection = Some(detection).into();

        self.last_detection_packet = Some(packet);
    }
}

impl Default for Simulation {
    fn default() -> Self {
        Simulation::new(SimulationConfig::default())
    }
}

pub struct SimulationBuilder {
    sim: Simulation,
    last_blue_id: u32,
    last_yellow_id: u32,
}

impl SimulationBuilder {
    pub fn new(config: SimulationConfig) -> Self {
        SimulationBuilder {
            sim: Simulation::new(config),
            last_blue_id: 0,
            last_yellow_id: 0,
        }
    }

    pub fn add_blue_player_with_id(mut self, id: u32, position: Vector2, yaw: Angle) -> Self {
        self.add_player(id, TeamColor::Blue, position, yaw);
        self
    }

    pub fn add_yellow_player_with_id(mut self, id: u32, position: Vector2, yaw: Angle) -> Self {
        self.add_player(id, TeamColor::Yellow, position, yaw);
        self
    }

    pub fn add_blue_player(mut self, position: Vector2, yaw: Angle) -> Self {
        self.add_player(self.last_blue_id, TeamColor::Blue, position, yaw);
        self.last_blue_id += 1;
        self
    }

    pub fn add_yellow_player(mut self, position: Vector2, yaw: Angle) -> Self {
        self.add_player(self.last_yellow_id, TeamColor::Yellow, position, yaw);
        self.last_yellow_id += 1;
        self
    }

    /// Set which teams should be controlled by the simulation
    pub fn with_controlled_teams(mut self, blue_controlled: bool, yellow_controlled: bool) -> Self {
        self.sim.config.blue_controlled = blue_controlled;
        self.sim.config.yellow_controlled = yellow_controlled;
        self
    }

    pub fn add_ball(mut self, position: Vector<f64>) -> Self {
        let sim = &mut self.sim;

        let ball_body = RigidBodyBuilder::dynamic()
            .can_sleep(false)
            .translation(position)
            .linear_damping(sim.config.ball_damping)
            .build();
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

    fn add_player(&mut self, id: u32, team_color: TeamColor, position: Vector2, yaw: Angle) {
        let sim = &mut self.sim;

        // Players have fixed z position - their bottom surface 1mm above the ground
        let player_radius = sim.config.player_radius;
        let player_height = sim.config.player_height;
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
        let rigid_body_handle = sim.rigid_body_set.insert(rigid_body);
        let collider_handle = sim.collider_set.insert_with_parent(
            collider,
            rigid_body_handle,
            &mut sim.rigid_body_set,
        );

        sim.players.push(Player {
            id: PlayerId::new(id),
            team_color,
            rigid_body_handle,
            _collider_handle: collider_handle,
            last_cmd_time: 0.0,
            target_velocity: Vector::zeros(),
            target_z: 0.0,
            current_dribble_speed: 0.0,
            breakbeam: false,
            heading_control: false,
        });
    }
}

impl Default for SimulationBuilder {
    fn default() -> Self {
        // By default we add 6 players on each side, lined up on their respective
        // half of the field
        let mut builder = SimulationBuilder::new(SimulationConfig::default());
        let field_width = builder.sim.config.field_geometry.field_width / 2.0;
        let field_length = builder.sim.config.field_geometry.field_length / 2.0;
        let boundary_width = builder.sim.config.field_geometry.boundary_width;
        let player_radius = builder.sim.config.player_radius;
        let player_margin = 0.75 * player_radius;
        let sides = builder.sim.config.initial_side_assignment;

        for i in 0..6 {
            let position = Vector2::new(
                field_length - player_margin - i as f64 * (2.0 * player_radius + player_margin),
                field_width - player_radius - boundary_width,
            );
            builder = builder.add_blue_player(
                sides.transform_vec2(TeamColor::Blue, &position),
                Angle::from_radians(0.0),
            );
        }
        for i in 0..6 {
            let position = Vector2::new(
                field_length - player_margin - i as f64 * (2.0 * player_radius + player_margin),
                -(field_width - player_radius - boundary_width),
            );
            builder = builder.add_yellow_player(
                sides.transform_vec2(TeamColor::Yellow, &position),
                Angle::from_radians(0.0),
            );
        }

        builder = builder.add_ball(Vector::new(0.0, 0.0, 20.0));

        builder
    }
}

fn geometry(config: &FieldGeometry) -> SSL_WrapperPacket {
    let mut geometry = SSL_GeometryData::new();
    let mut field = SSL_GeometryFieldSize::new();
    field.set_field_length(config.field_length as i32);
    field.set_field_width(config.field_width as i32);
    field.set_goal_width(config.goal_width as i32);
    field.set_goal_depth(config.goal_depth as i32);
    field.set_boundary_width(config.boundary_width as i32);
    field.set_penalty_area_depth(config.penalty_area_depth as i32);
    field.set_penalty_area_width(config.penalty_area_width as i32);
    field.set_center_circle_radius(config.center_circle_radius as i32);
    field.set_goal_center_to_penalty_mark(config.goal_line_to_penalty_mark as i32);
    field.set_ball_radius(config.ball_radius as f32);

    for line in &config.line_segments {
        let mut ssl_segment = SSL_FieldLineSegment::new();
        ssl_segment.set_name(line.name.clone());
        let mut p1 = dies_protos::ssl_vision_geometry::Vector2f::new();
        p1.set_x(line.p1.x as f32);
        p1.set_y(line.p1.y as f32);
        ssl_segment.p1 = Some(p1).into();
        let mut p2 = dies_protos::ssl_vision_geometry::Vector2f::new();
        p2.set_x(line.p2.x as f32);
        p2.set_y(line.p2.y as f32);
        ssl_segment.p2 = Some(p2).into();
        ssl_segment.set_thickness(line.thickness as f32);
        field.field_lines.push(ssl_segment);
    }

    for arc in &config.circular_arcs {
        let mut ssl_arc = SSL_FieldCircularArc::new();
        ssl_arc.set_name(arc.name.clone());
        let mut center = dies_protos::ssl_vision_geometry::Vector2f::new();
        center.set_x(arc.center.x as f32);
        center.set_y(arc.center.y as f32);
        ssl_arc.center = Some(center).into();
        ssl_arc.set_radius(arc.radius as f32);
        ssl_arc.set_a1(arc.a1 as f32);
        ssl_arc.set_a2(arc.a2 as f32);
        ssl_arc.set_thickness(arc.thickness as f32);
        field.field_arcs.push(ssl_arc);
    }

    geometry.field = Some(field).into();
    let mut packet = SSL_WrapperPacket::new();
    packet.geometry = Some(geometry).into();
    packet
}
