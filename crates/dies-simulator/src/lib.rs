use std::{
    collections::{HashMap, HashSet, VecDeque},
    f64::consts::PI,
};

use dies_core::{
    Angle, FieldGeometry, FieldSnapshot, GameState, GcSimCommand, PlayerFeedbackMsg,
    PlayerGlobalMoveCmd, PlayerId, PlayerMoveCmd, RobotCmd, SideAssignment, SysStatus, TeamColor,
    Vector2, Vector3, WorldInstant,
};
use dies_protos::{
    ssl_gc_referee_message::{
        referee::{self, TeamInfo},
        Referee,
    },
    ssl_vision_detection::{SSL_DetectionBall, SSL_DetectionFrame, SSL_DetectionRobot},
    ssl_vision_geometry::{
        SSL_FieldCircularArc, SSL_FieldLineSegment, SSL_GeometryData, SSL_GeometryFieldSize,
    },
    ssl_vision_wrapper::SSL_WrapperPacket,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rapier3d_f64::prelude::*;
use serde::Serialize;
use std::fmt;
use utils::IntervalTrigger;

mod utils;

// Simulation constants - these are in mm
const BALL_RADIUS: f64 = 21.45;
/// How far inside the field lines the ball is placed after it leaves the field.
/// Must be large enough that a robot can fully get behind the ball to take the
/// free kick (otherwise it would stay pinned against the boundary wall).
/// Doubles as the rule 5.3.3 "0.2 m from all field lines" placement constraint.
const FREE_KICK_PLACEMENT_MARGIN: f64 = 200.0;
/// Minimum distance a free-kick ball must be placed from either defense area
/// (SSL rule 5.3.3). Keeping this clear is what lets the kicker get behind the
/// ball without violating the 0.2 m opponent-defense-area keep-out (rule 8.4.1).
const FREE_KICK_DEFENSE_AREA_DISTANCE: f64 = 1000.0;
/// How long the game sits in a `Stop` (robots slowed, ball being placed) before
/// resuming after the ball leaves the field, a goal, or a no-progress stoppage.
/// Tune this to make self-play matches more/less leisurely.
const STOP_DURATION: f64 = 2.0;
/// How far the ball must travel from its last reference point to count as
/// "progress" and reset the no-progress timer (rule 8.1). Measured against a
/// sticky reference over the whole 10 s window, not frame-to-frame, so a slow
/// dribble still counts as progress and only a genuinely stuck ball trips the
/// timer.
const NO_PROGRESS_DISTANCE: f64 = 200.0;
/// Speed (mm/s) at which a contested ball is squirted out from between two
/// robots that both try to dribble it, so it escapes instead of staying pinned.
const CONTESTED_BALL_POP_SPEED: f64 = 1200.0;
/// Snatch peel: how fast a contesting robot's rotation drags a held ball around
/// the holder's dribbler, as a fraction of the contester's own yaw rate. `1.0` =
/// the ball sweeps as fast as the contester spins; lower makes the strip slower.
const SNATCH_PEEL_GAIN: f64 = 0.6;
/// Speed (mm/s) the ball is flung along the sweep direction when it finally pops
/// out of the holder's dribbler cone. Gentle — it's a strip for a teammate, not a
/// shot.
const SNATCH_RELEASE_SPEED: f64 = 500.0;
/// Max centre-to-ball distance (mm) at which a rotating robot counts as pressing
/// (contesting) a held ball. Distance-gated, not cone-gated, so a robot spinning
/// in place still registers as pressing while its facing sweeps.
const SNATCH_PRESS_RANGE: f64 = 250.0;
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
    /// Maximum lateral acceleration in mm/s^2 (speeding up — sluggish).
    pub max_accel: f64,
    /// Maximum lateral deceleration in mm/s^2 (braking — much harder than
    /// acceleration on the real robots, which is why they stop without
    /// overshoot even when starting is slow).
    pub max_decel: f64,
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

    /// Seed for the simulation RNG. Drives initial pose jitter (and any other
    /// seeded variation) so headless self-play matches are reproducible: same
    /// seed → identical match, different seed → different-but-reproducible match.
    pub seed: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            // PHYSICAL CONSTANTS
            gravity: Vector::z() * -9.81 * 1000.0,
            ball_damping: 0.8,
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
            // Match real-hardware limits (measured): ~2.8 m/s top speed,
            // ~1.5 m/s^2 to accelerate but much harder braking (~6 m/s^2), so
            // the robot is sluggish to start yet stops without overshoot. The
            // controller may *command* more aggressively; the simulated robot,
            // like the real one, can't exceed these.
            max_accel: 1500.0,
            max_decel: 6000.0,
            max_vel: 2800.0,
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
            // Must match what the sim reports via `blue_team_on_positive_half`
            // (and therefore how the executor/strategies place and play the
            // teams), otherwise goal attribution and penalty checks are computed
            // against the opposite side from reality.
            initial_side_assignment: SideAssignment::YellowOnPositive,

            seed: 0,
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
    StopAndFreeKick {
        team_color: TeamColor,
        stop_timer: Timer,
    },
    PrepareKickOff {
        team_color: TeamColor,
        prepare_timer: Timer,
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
            stop_timer: Timer::new(STOP_DURATION),
            team_color,
        }
    }

    fn stop_and_force_start() -> Self {
        SimulationGameState::StopAndForceStart {
            stop_timer: Timer::new(STOP_DURATION),
        }
    }

    fn stop_and_free_kick(team_color: TeamColor) -> Self {
        SimulationGameState::StopAndFreeKick {
            stop_timer: Timer::new(STOP_DURATION),
            team_color,
        }
    }

    fn prepare_kickoff(team_color: TeamColor) -> Self {
        SimulationGameState::PrepareKickOff {
            team_color,
            // Hard ceiling so a kickoff can never stall the match if positions
            // never become valid; matches the referee NORMAL_START schedule.
            prepare_timer: Timer::new(10.0),
        }
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
                SimulationGameState::StopAndFreeKick { .. },
                SimulationGameState::StopAndFreeKick { .. },
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
            SimulationGameState::StopAndFreeKick { .. } => write!(f, "Stop and free kick"),
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
    NoProgress,
    Goal,
}

/// A simulator-internal referee event together with the team it concerns (if
/// any). Surfaced to the executor so sim matches narrate the same class of
/// events (fouls, violations, goals) that a real autoref would report.
#[derive(Debug, Clone)]
pub struct SimRefereeEvent {
    pub kind: RefereeMessage,
    pub team: Option<TeamColor>,
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
            RefereeMessage::NoProgress => write!(f, "No progress"),
            RefereeMessage::Goal => write!(f, "Goal"),
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
            duration,
        }
    }

    pub fn reset(&mut self) {
        self.counter = 0.0;
    }

    pub fn tick(&mut self, dt: f64) -> bool {
        self.counter += dt;
        if self.counter >= self.duration {
            self.reset();
            return true;
        }
        false
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
    w: f64,
    target_heading: f64,
    /// Per-command yaw-rate limit (rad/s) from the controller's `max_yaw_rate`.
    /// The body slews toward `target_heading` no faster than this (capped by the
    /// hardware `max_ang_vel`), mirroring the firmware — so it can't teleport and
    /// fling a dribbled ball out of the dribbler cone.
    target_yaw_rate: f64,
    current_dribble_speed: f64,
    breakbeam: bool,
    last_kick_counter: u8,
    /// Reflex kick armed (robot_cmd == ArmReflex). Fires once when the ball
    /// reaches the dribbler/breakbeam, mirroring the firmware.
    reflex_armed: bool,
    /// Whether the armed reflex has already fired (one-shot per arm).
    reflex_fired_this_arm: bool,
}

#[derive(Debug, Clone)]
enum MoveCmd {
    Local(PlayerMoveCmd),
    Global(PlayerGlobalMoveCmd),
}

#[derive(Debug, Clone)]
struct TimedPlayerCmd {
    id: PlayerId,
    team_color: TeamColor,
    execute_time: f64,
    player_cmd: MoveCmd,
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
    /// Monotonic counter advertised in the referee `command_counter` field so
    /// consumers can tell distinct commands apart.
    command_counter: u32,
    /// Simulator-internal referee events (fouls/violations/goals) awaiting drain
    /// into the announcer feed.
    referee_events: VecDeque<SimRefereeEvent>,
    /// Running match score, incremented on each goal and reported via `TeamInfo`.
    blue_score: u32,
    yellow_score: u32,
    feedback_interval: IntervalTrigger,
    feedback_queue: HashMap<(TeamColor, PlayerId), PlayerFeedbackMsg>,
    game_state: SimulationGameState,
    designated_ball_position: Vector<f64>,
    last_touch_info: Option<(PlayerId, TeamColor)>,
    side_assignment: SideAssignment,
    ball_being_dribbled_by: Option<(PlayerId, TeamColor)>,
    /// Accumulated peel angle (rad) of a held ball being stripped: how far a
    /// contesting robot's rotation has dragged the ball around the holder's
    /// dribbler. Reset to 0 whenever the ball isn't being actively peeled; once it
    /// exceeds the holder's dribbler cone the ball pops loose.
    ball_peel_angle: f64,
    // New fields for rule enforcement
    kick_start_time: f64,
    kick_ball_position: Option<Vector2>,
    last_kicker_info: Option<(PlayerId, TeamColor)>,
    kick_in_progress: bool,
    /// Seeded RNG for deterministic variation (initial pose jitter, etc).
    rng: StdRng,
}

impl Simulation {
    /// Createa a new instance of [`Simulation`]. After creation, the simulation is
    /// empty and needs to be populated with players and a ball. It is better to use
    /// [`SimulationBuilder`] to create a new simulation and add players and a ball.
    pub fn new(config: SimulationConfig) -> Simulation {
        let vision_update_step = config.vision_update_step;
        let geometry_interval = config.geometry_interval;
        let feedback_interval = config.feedback_interval;
        let field_length =
            config.field_geometry.field_length + 2.0 * config.field_geometry.boundary_width;
        let field_width =
            config.field_geometry.field_width + 2.0 * config.field_geometry.boundary_width;
        let goal_width = config.field_geometry.goal_width;
        let goal_depth =
            config.field_geometry.field_length / 2.0 + config.field_geometry.goal_depth;
        let boundary_width = config.field_geometry.boundary_width;
        let geometry_packet = geometry(&config.field_geometry);

        let side_assignment = config.initial_side_assignment;
        let rng = StdRng::seed_from_u64(config.seed);
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
            command_counter: 0,
            referee_events: VecDeque::new(),
            blue_score: 0,
            yellow_score: 0,
            feedback_interval: IntervalTrigger::new(feedback_interval),
            feedback_queue: HashMap::new(),
            game_state: SimulationGameState::default(),
            designated_ball_position: Vector::new(0.0, 0.0, 20.0),
            last_touch_info: None,
            side_assignment,
            ball_being_dribbled_by: None,
            ball_peel_angle: 0.0,
            kick_start_time: 0.0,
            kick_ball_position: None,
            last_kicker_info: None,
            kick_in_progress: false,
            rng,
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

        // Create the field walls
        simulation.add_wall(0.0, field_width / 2.0, field_length / 2.0, WALL_THICKNESS);
        simulation.add_wall(0.0, -field_width / 2.0, field_length / 2.0, WALL_THICKNESS);
        simulation.add_wall(field_length / 2.0, 0.0, WALL_THICKNESS, field_width / 2.0);
        simulation.add_wall(-field_length / 2.0, 0.0, WALL_THICKNESS, field_width / 2.0);
        // Create the goal walls
        simulation.add_wall(
            -field_length / 2.0 + boundary_width / 2.0,
            goal_width / 2.0,
            boundary_width / 2.0,
            WALL_THICKNESS,
        );
        simulation.add_wall(
            -field_length / 2.0 + boundary_width / 2.0,
            -goal_width / 2.0,
            boundary_width / 2.0,
            WALL_THICKNESS,
        );
        simulation.add_wall(
            field_length / 2.0 - boundary_width / 2.0,
            goal_width / 2.0,
            boundary_width / 2.0,
            WALL_THICKNESS,
        );
        simulation.add_wall(
            field_length / 2.0 - boundary_width / 2.0,
            -goal_width / 2.0,
            boundary_width / 2.0,
            WALL_THICKNESS,
        );
        simulation.add_wall(-goal_depth, 0.0, WALL_THICKNESS, goal_width / 2.0);
        simulation.add_wall(goal_depth, 0.0, WALL_THICKNESS, goal_width / 2.0);

        simulation
    }

    pub fn time(&self) -> WorldInstant {
        WorldInstant::Simulated(self.current_time)
    }

    /// Current simulated time in seconds.
    pub fn time_secs(&self) -> f64 {
        self.current_time
    }

    /// Current match score as `(blue, yellow)`.
    pub fn score(&self) -> (u32, u32) {
        (self.blue_score, self.yellow_score)
    }

    /// The field geometry this simulation was built with (logged into the log's
    /// `meta.json` so analytics know the goal/field coordinates).
    pub fn field_geometry(&self) -> &FieldGeometry {
        &self.config.field_geometry
    }

    /// A deterministic 64-bit fingerprint of the current physical state: the
    /// ball position plus every player's pose, in player-vector order. Folding
    /// this each frame yields a trajectory hash that is byte-identical across
    /// runs with the same `(seed, strategies)` on the same binary, and diverges
    /// as soon as the trajectories differ — used to verify determinism and to
    /// confirm that the seed actually varies the match. Uses raw float bits
    /// (FNV-1a) for exact same-binary equality.
    pub fn state_fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
        let mut mix = |x: f64, h: &mut u64| {
            for b in x.to_bits().to_le_bytes() {
                *h ^= b as u64;
                *h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        if let Some(ball) = &self.ball {
            if let Some(rb) = self.rigid_body_set.get(ball._rigid_body_handle) {
                let p = rb.translation();
                mix(p.x, &mut h);
                mix(p.y, &mut h);
                mix(p.z, &mut h);
            }
        }
        for player in &self.players {
            if let Some(rb) = self.rigid_body_set.get(player.rigid_body_handle) {
                let t = rb.translation();
                let yaw = rb.rotation().euler_angles().2;
                mix(t.x, &mut h);
                mix(t.y, &mut h);
                mix(yaw, &mut h);
            }
        }
        h
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
        self.config.blue_controlled = teams.contains(&TeamColor::Blue);
        self.config.yellow_controlled = teams.contains(&TeamColor::Yellow);
    }

    /// Check if a team is controlled
    pub fn is_team_controlled(&self, team_color: TeamColor) -> bool {
        match team_color {
            TeamColor::Blue => self.config.blue_controlled,
            TeamColor::Yellow => self.config.yellow_controlled,
        }
    }

    fn add_wall(&mut self, x: f64, y: f64, half_width: f64, half_length: f64) {
        let wall_body = RigidBodyBuilder::fixed()
            .translation(Vector::new(x, y, 0.0))
            .build();
        let wall_collider = ColliderBuilder::cuboid(half_width, half_length, WALL_HEIGHT)
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

    /// Teleport the ball to a field position (z held at rest height) and zero
    /// its velocity. Used by scenarios to reset to a clean state between drills.
    pub fn teleport_ball(&mut self, position: Vector2) {
        if let Some(ball) = self.ball.as_ref() {
            let handle = ball._rigid_body_handle;
            if let Some(ball_body) = self.rigid_body_set.get_mut(handle) {
                ball_body.set_position(
                    Isometry::translation(position.x, position.y, BALL_RADIUS),
                    true,
                );
                ball_body.set_linvel(Vector::zeros(), true);
                ball_body.set_angvel(Vector::zeros(), true);
            }
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
            w: 0.0,
            target_heading: f64::NAN,
            target_yaw_rate: f64::INFINITY,
            current_dribble_speed: 0.0,
            breakbeam: false,
            last_kick_counter: 0,
            reflex_armed: false,
            reflex_fired_this_arm: false,
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
            id: cmd.id,
            team_color,
            execute_time: self.current_time + self.config.command_delay,
            player_cmd: MoveCmd::Local(cmd),
        });
    }

    pub fn push_global_cmd(&mut self, team_color: TeamColor, cmd: PlayerGlobalMoveCmd) {
        self.cmd_queue.push(TimedPlayerCmd {
            id: cmd.id,
            team_color,
            execute_time: self.current_time + self.config.command_delay,
            player_cmd: MoveCmd::Global(cmd),
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
            GcSimCommand::Halt => {
                // Make Halt authoritative over the auto-ref, not just a broadcast,
                // so a seeded/forced Halt actually freezes the state machine.
                self.update_referee_command(referee::Command::HALT);
                self.game_state = SimulationGameState::Halt;
            }
            GcSimCommand::NormalStart => {
                self.update_referee_command(referee::Command::NORMAL_START)
            }
            GcSimCommand::ForceStart => {
                // Drop straight into free play and make the auto-ref agree, so a
                // forced/seeded start isn't immediately overridden by a pending
                // kickoff in the state machine.
                self.update_referee_command(referee::Command::FORCE_START);
                self.game_state = SimulationGameState::run();
            }
            GcSimCommand::DirectFree { team_color } => {
                // Relocate the ball to a rule 5.3.3-valid free-kick position. A real
                // referee/auto-ref always places the ball ≥1 m from either defense
                // area before a free kick; without this a manual `DirectFree` can
                // leave the ball jammed against a defense area where the kicker
                // cannot legally get behind it, and the restart just times out.
                if let Some(ball) = self.ball.as_ref() {
                    let current = self
                        .rigid_body_set
                        .get(ball._rigid_body_handle)
                        .unwrap()
                        .position()
                        .translation
                        .vector
                        .xy();
                    let placement = self.valid_free_kick_position(current);
                    self.place_ball(placement.x, placement.y);
                }
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

    /// Apply a saved field snapshot: teleport existing robots to the snapshot's
    /// poses, add ones it has that the field doesn't, remove ones it omits, and
    /// place the ball. Mirrors the Web UI snapshot-load flow but runs directly
    /// against the simulator (used to seed a headless rollout). Game state is
    /// seeded separately via [`Simulation::seed_game_state`].
    pub fn apply_snapshot(&mut self, snapshot: &FieldSnapshot) {
        for (color, robots) in [
            (TeamColor::Blue, &snapshot.blue),
            (TeamColor::Yellow, &snapshot.yellow),
        ] {
            let snap_ids: HashSet<PlayerId> = robots.iter().map(|r| r.id).collect();
            for r in robots {
                let exists = self
                    .players
                    .iter()
                    .any(|p| p.id == r.id && p.team_color == color);
                if exists {
                    self.teleport_robot(color, r.id, r.position, r.yaw);
                } else {
                    self.add_robot(color, r.id, r.position, r.yaw);
                }
            }
            // Remove robots not present in the snapshot so the setup reproduces exactly.
            let to_remove: Vec<PlayerId> = self
                .players
                .iter()
                .filter(|p| p.team_color == color && !snap_ids.contains(&p.id))
                .map(|p| p.id)
                .collect();
            for id in to_remove {
                self.remove_robot(color, id);
            }
        }
        if let Some(ball) = snapshot.ball {
            self.teleport_ball(ball);
        }
    }

    /// Seed the auto-referee game state directly, e.g. when starting a headless
    /// rollout from a snapshot. Maps the strategy-facing [`GameState`] onto the
    /// equivalent GC command; `Run`/`Kickoff`/`PenaltyRun` drop the match straight
    /// into free play via `ForceStart` (skipping the kickoff cycle). `operating_team`
    /// selects the team for asymmetric states (kickoff/free-kick/penalty/placement)
    /// and is ignored for symmetric ones; it defaults to Blue when unknown.
    pub fn seed_game_state(&mut self, state: GameState, operating_team: Option<TeamColor>) {
        let team = operating_team.unwrap_or(TeamColor::Blue);
        match state {
            GameState::Run | GameState::Kickoff | GameState::PenaltyRun => {
                self.handle_gc_command(GcSimCommand::ForceStart)
            }
            GameState::Stop => self.handle_gc_command(GcSimCommand::Stop),
            GameState::Halt | GameState::Timeout => self.handle_gc_command(GcSimCommand::Halt),
            GameState::PrepareKickoff => {
                self.handle_gc_command(GcSimCommand::KickOff { team_color: team })
            }
            GameState::FreeKick => {
                self.handle_gc_command(GcSimCommand::DirectFree { team_color: team })
            }
            GameState::Penalty | GameState::PreparePenalty => {
                self.handle_gc_command(GcSimCommand::Penalty { team_color: team })
            }
            GameState::BallReplacement(position) => {
                self.handle_gc_command(GcSimCommand::BallPlacement {
                    team_color: team,
                    position,
                })
            }
            GameState::Unknown => {}
        }
    }

    fn update_referee_command(&mut self, command: referee::Command) {
        self.update_referee_command_full(command, None, None);
    }

    /// Like [`update_referee_command`], but also advertises the command that will
    /// resume play (`next_command`) and how long the current action has left
    /// (`action_time_s`). This is what lets the UI's game-state panel show a live
    /// "Next: … in N s" countdown in sim, where there is no real game controller.
    fn update_referee_command_full(
        &mut self,
        command: referee::Command,
        next_command: Option<referee::Command>,
        action_time_s: Option<f64>,
    ) {
        let mut msg = Referee::new();
        msg.set_command(command);
        msg.packet_timestamp = Some(0);
        msg.set_stage(referee::Stage::NORMAL_FIRST_HALF);
        // Increment so consumers can detect distinct commands (the real GC does).
        self.command_counter = self.command_counter.wrapping_add(1);
        msg.command_counter = Some(self.command_counter);
        msg.command_timestamp = Some(0);
        if let Some(next) = next_command {
            msg.set_next_command(next);
        }
        if let Some(t) = action_time_s {
            msg.current_action_time_remaining = Some((t * 1_000_000.0) as i64);
        }
        if let SimulationGameState::BallPlacement { position, .. } = &self.game_state {
            let mut point = referee::Point::new();
            point.set_x(position.x as f32);
            point.set_y(position.y as f32);
            msg.designated_position = Some(point).into();
        }
        let mut blue_team_info = TeamInfo::new();
        blue_team_info.set_max_allowed_bots(6);
        blue_team_info.set_goalkeeper(1);
        blue_team_info.set_score(self.blue_score);
        msg.blue = Some(blue_team_info).into();
        let mut yellow_team_info = TeamInfo::new();
        yellow_team_info.set_max_allowed_bots(6);
        yellow_team_info.set_goalkeeper(1);
        yellow_team_info.set_score(self.yellow_score);
        msg.yellow = Some(yellow_team_info).into();
        // Report the sim's true side so the executor/strategies and the sim's own
        // goal/penalty checks all agree on which team defends which half.
        msg.blue_team_on_positive_half =
            Some(self.side_assignment == SideAssignment::BlueOnPositive);
        self.referee_message.push_back(msg);
    }

    /// Record a simulator-internal referee event (foul, violation, goal, …) so the
    /// executor can surface it in the announcer feed. Drained via
    /// [`take_referee_events`].
    fn push_referee_event(&mut self, kind: RefereeMessage, team: Option<TeamColor>) {
        dies_core::debug_string("RefereeMessage", kind.to_string());
        self.referee_events
            .push_back(SimRefereeEvent { kind, team });
    }

    /// Drain the simulator-internal referee events accumulated since the last call.
    pub fn take_referee_events(&mut self) -> Vec<SimRefereeEvent> {
        self.referee_events.drain(..).collect()
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
                self.update_referee_command_full(
                    referee::Command::STOP,
                    Some(referee::Command::PREPARE_KICKOFF_BLUE),
                    Some(STOP_DURATION),
                );
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
                    let cmd = match team_color {
                        TeamColor::Blue => referee::Command::PREPARE_KICKOFF_BLUE,
                        TeamColor::Yellow => referee::Command::PREPARE_KICKOFF_YELLOW,
                    };
                    self.update_referee_command_full(
                        cmd,
                        Some(referee::Command::NORMAL_START),
                        Some(10.0),
                    );
                    SimulationGameState::prepare_kickoff(team_color)
                } else {
                    game_state
                }
            }
            SimulationGameState::StopAndForceStart { ref mut stop_timer } => {
                // Wait out the stop period before resuming the game.
                if stop_timer.tick(self.integration_parameters.dt) {
                    self.update_referee_command(referee::Command::FORCE_START);
                    SimulationGameState::run()
                } else {
                    game_state
                }
            }
            SimulationGameState::StopAndFreeKick {
                team_color,
                ref mut stop_timer,
            } => {
                // Wait out the stop period, then award the free kick.
                if stop_timer.tick(self.integration_parameters.dt) {
                    let cmd = match team_color {
                        TeamColor::Blue => referee::Command::DIRECT_FREE_BLUE,
                        TeamColor::Yellow => referee::Command::DIRECT_FREE_YELLOW,
                    };
                    self.update_referee_command_full(cmd, None, Some(10.0));
                    SimulationGameState::free_kick(team_color)
                } else {
                    game_state
                }
            }
            SimulationGameState::PrepareKickOff {
                team_color,
                ref mut prepare_timer,
            } => {
                // If positions never become valid (e.g. the scoring team's
                // robots don't retreat to their own half), don't hang forever —
                // force the kickoff after the timer so the match keeps running.
                if prepare_timer.tick(self.integration_parameters.dt) {
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
                    self.update_referee_command(referee::Command::NORMAL_START);
                    SimulationGameState::run()
                } else if self.check_kickoff_positions(team_color) {
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
                ref mut last_ball_position,
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

                    // Attribute the goal. The goal the ball entered belongs to the
                    // team defending that side; that team conceded, the opponent
                    // scored, and the conceding team restarts with a kick-off.
                    let positive_defender = match self.side_assignment {
                        SideAssignment::BlueOnPositive => TeamColor::Blue,
                        SideAssignment::YellowOnPositive => TeamColor::Yellow,
                    };
                    let conceding = if ball_pos.x > 0.0 {
                        positive_defender
                    } else {
                        positive_defender.opposite()
                    };
                    let scorer = conceding.opposite();
                    let kickoff_team = conceding;

                    match scorer {
                        TeamColor::Blue => self.blue_score += 1,
                        TeamColor::Yellow => self.yellow_score += 1,
                    }
                    self.push_referee_event(RefereeMessage::Goal, Some(scorer));

                    // Re-center the ball for the kick-off so the match keeps
                    // running (otherwise it stays in the goal and re-triggers
                    // the goal check every frame). Go through a Stop period
                    // first so robots slow down before the kick-off.
                    self.place_ball(0.0, 0.0);
                    let next = match kickoff_team {
                        TeamColor::Blue => referee::Command::PREPARE_KICKOFF_BLUE,
                        TeamColor::Yellow => referee::Command::PREPARE_KICKOFF_YELLOW,
                    };
                    self.update_referee_command_full(
                        referee::Command::STOP,
                        Some(next),
                        Some(STOP_DURATION),
                    );
                    SimulationGameState::stop_and_kickoff(kickoff_team)
                } else if self.ball_out() {
                    // Clear kick tracking
                    self.kick_in_progress = false;
                    self.last_kicker_info = None;
                    self.kick_ball_position = None;

                    // Snap the ball back onto a reachable spot inside the field
                    // so the free kick can actually be taken. Without this the
                    // ball stays pinned against the boundary wall and the match
                    // gets stuck. Also enforce the rule 5.3.3 defense-area
                    // clearance so the kicker can legally get behind the ball.
                    // Go through a Stop period first so robots slow down and back
                    // off before the free kick.
                    let placement =
                        self.valid_free_kick_position(self.designated_ball_position.xy());
                    self.place_ball(placement.x, placement.y);

                    if let Some((_kicker_id, kicker_color)) = self.last_touch_info {
                        // Free kick is awarded to the opponent of the last toucher.
                        let awarded = kicker_color.opposite();
                        let next = match awarded {
                            TeamColor::Blue => referee::Command::DIRECT_FREE_BLUE,
                            TeamColor::Yellow => referee::Command::DIRECT_FREE_YELLOW,
                        };
                        self.update_referee_command_full(
                            referee::Command::STOP,
                            Some(next),
                            Some(STOP_DURATION),
                        );
                        SimulationGameState::stop_and_free_kick(awarded)
                    } else {
                        // No known last toucher — neutral restart with a force start.
                        self.update_referee_command_full(
                            referee::Command::STOP,
                            Some(referee::Command::FORCE_START),
                            Some(STOP_DURATION),
                        );
                        SimulationGameState::stop_and_force_start()
                    }
                } else {
                    // Check for no progress (rule 8.1): the timer only fires if the
                    // ball stays within NO_PROGRESS_DISTANCE of a sticky reference
                    // point for the whole window. As soon as the ball travels far
                    // enough from that reference, we move the reference up and reset
                    // the timer, so any real progress keeps the game running.
                    let ball_handle = self.ball.as_mut().map(|ball| ball._rigid_body_handle);
                    if let Some(ball_handle) = ball_handle {
                        let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
                        let ball_position_2d = ball_body.position().translation.vector.xy();
                        match last_ball_position {
                            Some(reference)
                                if (ball_position_2d - *reference).norm()
                                    < NO_PROGRESS_DISTANCE =>
                            {
                                // Ball hasn't made progress — count down.
                                if no_progress_timer.tick(self.integration_parameters.dt) {
                                    // tick() resets the timer on fire; advance the
                                    // reference so the next window starts cleanly.
                                    *last_ball_position = Some(ball_position_2d);
                                    self.push_referee_event(RefereeMessage::NoProgress, None);
                                    self.update_referee_command_full(
                                        referee::Command::STOP,
                                        Some(referee::Command::FORCE_START),
                                        Some(STOP_DURATION),
                                    );
                                    SimulationGameState::stop_and_force_start()
                                } else {
                                    game_state
                                }
                            }
                            _ => {
                                // First frame, or the ball moved far enough to count
                                // as progress: advance the reference and reset.
                                *last_ball_position = Some(ball_position_2d);
                                no_progress_timer.reset();
                                game_state
                            }
                        }
                    } else {
                        game_state
                    }
                }
            }
            SimulationGameState::FreeKick {
                team_color,
                ref mut kick_timer,
                ball_is_kicked: _,
            } => {
                // Check if positions are valid
                if !self.check_free_kick_positions(team_color) {
                    // Defending robots are crowding the ball (announcer dedupes
                    // the repeated frames into a single line).
                    self.push_referee_event(
                        RefereeMessage::DefenderTooCloseToBall,
                        Some(team_color.opposite()),
                    );
                    // Don't wait forever: tick the kick timer as a hard ceiling
                    // and force the game back to running if they never clear out,
                    // so self-play can't stall on a free kick.
                    if kick_timer.tick(self.integration_parameters.dt) {
                        self.update_referee_command(referee::Command::FORCE_START);
                        self.kick_in_progress = false;
                        SimulationGameState::run()
                    } else {
                        game_state
                    }
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
                        self.push_referee_event(RefereeMessage::FreekickTimeExceeded, None);
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
                        self.push_referee_event(RefereeMessage::PenaltyTimeExceeded, None);
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

    /// Place the ball at rest at the given (x, y) position. Used to reset the
    /// ball onto a reachable spot after it leaves the field or after a goal, so
    /// the match keeps running in self-play.
    fn place_ball(&mut self, x: f64, y: f64) {
        if let Some(ball) = self.ball.as_ref() {
            let ball_body = self
                .rigid_body_set
                .get_mut(ball._rigid_body_handle)
                .unwrap();
            ball_body.set_position(Isometry::translation(x, y, BALL_RADIUS), true);
            ball_body.set_linvel(Vector::zeros(), true);
            // Also kill the spin: a ball that rolled out carries angular velocity,
            // and ground friction would convert that residual spin back into
            // linear motion, dragging the freshly-placed ball off its spot and
            // back over the line during the Stop period — which re-triggers
            // ball-out and collapses the free-kick restart into a force-start.
            ball_body.set_angvel(Vector::zeros(), true);
        }
    }

    /// Nudge a desired free-kick ball position to the closest spot that satisfies
    /// the SSL rule 5.3.3 placement constraints: at least 0.2 m from every field
    /// line and at least 1 m from either defense area. A position that is already
    /// valid is returned unchanged.
    fn valid_free_kick_position(&self, desired: Vector2) -> Vector2 {
        let geom = &self.config.field_geometry;
        let hl = geom.field_length / 2.0;
        let hw = geom.field_width / 2.0;
        let half_pa_w = geom.penalty_area_width / 2.0;

        let mut pos = desired;

        // Push out of the 1 m keep-out zone around each defense area (one per
        // goal line). The areas are 2× penalty_area_depth apart in x, so their
        // keep-out zones never overlap and the pushes don't fight each other.
        //
        // Extend each rectangle *behind* its goal line (well off the field) so the
        // goal-line face is never the nearest exit. Otherwise a ball that went out
        // over the goal line — closer to the goal line than to the front edge or
        // the sides — would be pushed back through the goal (off the field), and
        // the field-line clamp below would drag it straight back inside the box,
        // leaving the free kick at an illegal in-area position. The only valid
        // exits are the front (field-facing) edge and the two sides.
        for &goal_x in &[-hl, hl] {
            let inner_x = goal_x - geom.penalty_area_depth * goal_x.signum();
            let outer_x = goal_x + 10_000.0 * goal_x.signum();
            let min = Vector2::new(inner_x.min(outer_x), -half_pa_w);
            let max = Vector2::new(inner_x.max(outer_x), half_pa_w);
            pos = push_out_of_rect(pos, min, max, FREE_KICK_DEFENSE_AREA_DISTANCE);
        }

        // Keep the ball at least 0.2 m inside every field line.
        pos.x = pos.x.clamp(
            -(hl - FREE_KICK_PLACEMENT_MARGIN),
            hl - FREE_KICK_PLACEMENT_MARGIN,
        );
        pos.y = pos.y.clamp(
            -(hw - FREE_KICK_PLACEMENT_MARGIN),
            hw - FREE_KICK_PLACEMENT_MARGIN,
        );
        pos
    }

    fn ball_out(&mut self) -> bool {
        let ball_handle = self.ball.as_mut().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector;

            let half_length = self.config.field_geometry.field_length / 2.0;
            let half_width = self.config.field_geometry.field_width / 2.0;

            // Clamp a candidate free-kick spot to the reachable field margin so
            // the placement is always inside the playable area.
            let clamp_designated = |mut p: Vector<f64>| {
                p.x = p.x.clamp(
                    -(half_length - FREE_KICK_PLACEMENT_MARGIN),
                    half_length - FREE_KICK_PLACEMENT_MARGIN,
                );
                p.y = p.y.clamp(
                    -(half_width - FREE_KICK_PLACEMENT_MARGIN),
                    half_width - FREE_KICK_PLACEMENT_MARGIN,
                );
                p
            };

            // A ball is out only once its centre has actually crossed a real
            // field line. The out-of-bounds line must be the true field edge,
            // not a shrunk one: the placement margin (FREE_KICK_PLACEMENT_MARGIN)
            // already keeps a placed free-kick ball inside the field, so testing
            // against any smaller boundary would flag a legally-placed corner
            // free kick as immediately out and re-stop the game.
            if ball_position.x.abs() > half_length || ball_position.y.abs() > half_width {
                dies_core::debug_string(
                    "RefereeMessage",
                    format!("Ball out of bounds at position: {:?}", ball_position),
                );
                // Recompute the designated position from where the ball actually
                // is, so a ball that is already out (e.g. came to rest just past
                // the line, or crossed in a single fast frame) does not reuse a
                // stale designated position from an earlier out-of-bounds on the
                // other side of the field.
                self.designated_ball_position = clamp_designated(ball_position);
                // The actual reposition happens in `update_game_state` once the
                // free kick is awarded (see `place_ball`); here we only report
                // that the ball has left the field.
                return true;
            }

            // Still inside: predictively record where it *will* leave so the
            // free-kick placement is ready, but keep the game running this frame.
            let ball_velocity: Vector<f64> = *ball_body.linvel(); // ball's current velocity as Vector3
            let next_ball_position = ball_position + ball_velocity * self.integration_parameters.dt;
            if next_ball_position.x.abs() > half_length || next_ball_position.y.abs() > half_width {
                dies_core::debug_string(
                    "RefereeMessage.BallOut",
                    format!("Ball out of bounds at position: {:?}", next_ball_position),
                );
                // Linear interpolation toward where it crosses the line.
                let designated = ball_position + (next_ball_position - ball_position) * 0.5;
                self.designated_ball_position = clamp_designated(designated);
            }
            false
        } else {
            false
        }
    }

    fn goal(&mut self) -> bool {
        let ball_handle = self.ball.as_mut().map(|ball| ball._rigid_body_handle);
        if let Some(ball_handle) = ball_handle {
            let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
            let ball_position = ball_body.position().translation.vector; // Center of the ball

            let goal_height = 150.0;
            // A goal is scored once the ball crosses the goal line (field_length/2)
            // into the net, which extends back to field_length/2 + goal_depth.
            if ball_position.x.abs() > self.config.field_geometry.field_length / 2.0
                && ball_position.x.abs()
                    < self.config.field_geometry.field_length / 2.0
                        + self.config.field_geometry.goal_depth
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
                    format!("Goal scored at {}", scoring_side),
                );

                // The ball is re-centered for the kick-off in `update_game_state`
                // (see `place_ball`); here we only report that a goal was scored.
                return true;
            }
        }
        false
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

                // Every robot must be on its own half. The only exception is a
                // single attacking robot allowed inside the center circle (which
                // straddles the halfway line) so it can take the kick.
                let is_on_own_side = self
                    .side_assignment
                    .is_on_own_side_vec3(player.team_color, &player_position);
                if !is_on_own_side {
                    let distance_to_center = (player_position.xy() - Vector2::new(0.0, 0.0)).norm();
                    let is_kicker_in_circle = player.team_color == attacking_team
                        && distance_to_center <= center_circle_radius;
                    if is_kicker_in_circle {
                        attacking_robots_in_circle += 1;
                    }
                    if !is_kicker_in_circle || attacking_robots_in_circle > 1 {
                        dies_core::debug_string(
                            "RefereeMessage",
                            RefereeMessage::KickoffPositionViolation.to_string(),
                        );
                        return false;
                    }
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

                    if distance < touch_threshold
                        && (closest_touching_player.is_none()
                            || distance < closest_touching_player.unwrap().2)
                    {
                        closest_touching_player = Some((player.id, player.team_color, distance));
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

                            self.push_referee_event(
                                RefereeMessage::DoubleTouchViolation,
                                Some(team_color),
                            );

                            let opponent_team = if team_color == TeamColor::Blue {
                                TeamColor::Yellow
                            } else {
                                TeamColor::Blue
                            };
                            self.game_state = SimulationGameState::free_kick(opponent_team);
                            let cmd = if opponent_team == TeamColor::Blue {
                                referee::Command::DIRECT_FREE_BLUE
                            } else {
                                referee::Command::DIRECT_FREE_YELLOW
                            };
                            self.update_referee_command_full(cmd, None, Some(10.0));
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
                    to_exec.insert((cmd.team_color, cmd.id), cmd.player_cmd.clone());
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
        // Ownership of the ball is resolved after the player loop from this
        // frame's claimants, so that two robots racing for the same ball don't
        // flip ownership every frame (last writer won) and deadlock.
        let mut dribble_claimants: Vec<(PlayerId, TeamColor)> = Vec::new();
        let mut ball_kicked_this_frame = false;
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

            let rigid_body = self
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
                        player.last_cmd_time = self.current_time;
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
                        player.last_cmd_time = self.current_time;
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

            if (self.current_time - player.last_cmd_time).abs() > self.config.player_cmd_timeout {
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
                    self.config.max_accel
                } else {
                    self.config.max_decel
                };
                let acc = (vel_err / dt).cap_magnitude(limit);
                velocity + acc * dt
            };
            let new_vel = new_vel.cap_magnitude(self.config.max_vel);
            rigid_body.set_linvel(new_vel, true);

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
                let rate = player.target_yaw_rate.min(self.config.max_ang_vel);
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
                    target_ang_vel.clamp(-self.config.max_ang_vel, self.config.max_ang_vel);
                rigid_body.set_angvel(Vector::z() * new_ang_vel, true);
            }

            // Check if the ball is in the dribbler
            player.breakbeam = false;
            let yaw = rigid_body.position().rotation * Vector::x();
            let player_position = rigid_body.position().translation.vector;
            let ball_handle = self.ball.as_ref().map(|ball| ball._rigid_body_handle);
            if let Some(ball_handle) = ball_handle {
                let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
                let ball_position = ball_body.position().translation.vector;
                let ball_dir = ball_position - player_position;
                let distance = ball_dir.norm();
                let angle = yaw.angle(&ball_dir);
                if distance < self.config.player_radius + self.config.dribbler_radius + 20.0
                    && angle < self.config.dribbler_angle
                {
                    player.breakbeam = true;
                    // A reflex kick fires the instant the ball reaches the
                    // breakbeam, once per arm (matches firmware ARM_REFLEX_KICK).
                    let reflex_fire = player.reflex_armed && !player.reflex_fired_this_arm;
                    if is_kicking || reflex_fire {
                        if reflex_fire {
                            player.reflex_fired_this_arm = true;
                        }
                        ball_kicked_this_frame = true;
                        // Move the ball away from the player
                        let new_ball_position = ball_position + yaw * 80.0;
                        ball_body.set_position(
                            Isometry::translation(
                                new_ball_position.x,
                                new_ball_position.y,
                                new_ball_position.z,
                            ),
                            true,
                        );

                        let force = yaw * self.config.kicker_strength;
                        ball_body.add_force(force, true);

                        // Set the ball_is_kicked to true
                        if let SimulationGameState::FreeKick { ball_is_kicked, .. } =
                            &mut self.game_state
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

        // Resolve ball ownership from this frame's claimants. A ball is only
        // cleanly captured when exactly one robot is actively dribbling it.
        //
        // Before the normal resolution we check for an active *snatch*: an
        // established holder (last frame's owner) still gripping the ball, plus a
        // distinct robot pressing its dribbler against the ball and rotating. The
        // contester's rotation drags the ball around the holder's dribbler
        // (`ball_peel_angle` accumulates from the contester's yaw rate) until it
        // leaves the holder's cone and pops loose (handled in the pin block).
        //
        // The contester is detected by *contact distance*, not the dribbler cone:
        // a robot rotating in place sweeps its cone off the ball, but it's still
        // pressing — so cone-gating would drop it mid-spin. Touching without
        // rotating does nothing — only ω peels.
        //
        // With no established holder, several claimants are a genuine loose-ball
        // 50/50: the contact pops it loose (`pop_contested_ball`) as before, which
        // is what stops ownership flipping every frame and deadlocking self-play.
        let prior_holder = self.ball_being_dribbled_by;
        let ball_xy = self
            .ball
            .as_ref()
            .map(|b| {
                let v = self
                    .rigid_body_set
                    .get(b._rigid_body_handle)
                    .unwrap()
                    .position()
                    .translation
                    .vector;
                Vector2::new(v.x, v.y)
            })
            .unwrap_or_else(Vector2::zeros);

        // Fastest-rotating presser contesting the holder, if any.
        let snatch_omega = prior_holder
            .filter(|h| !ball_kicked_this_frame && dribble_claimants.contains(h))
            .map(|holder| {
                let mut best = 0.0_f64;
                for p in self.players.iter() {
                    if (p.id, p.team_color) == holder || p.current_dribble_speed <= 0.0 {
                        continue;
                    }
                    let rb = self.rigid_body_set.get(p.rigid_body_handle).unwrap();
                    let ppos = rb.position().translation.vector;
                    if (ball_xy - Vector2::new(ppos.x, ppos.y)).norm() > SNATCH_PRESS_RANGE {
                        continue;
                    }
                    // The contester's rotation is commanded as angular velocity
                    // (a spin, no heading setpoint), already applied to the body
                    // via `set_angvel` this frame — read it directly. (The body's
                    // orientation itself only integrates in the physics step that
                    // runs after this resolution, so a yaw delta would read zero.)
                    let omega = rb.angvel().z;
                    if omega.abs() > best.abs() {
                        best = omega;
                    }
                }
                (holder, best)
            });

        if ball_kicked_this_frame {
            self.ball_being_dribbled_by = None;
            self.ball_peel_angle = 0.0;
        } else if let Some((holder, omega)) = snatch_omega {
            // Pure-ω: no centerline bias — the skill picks the spin direction to aim
            // where the ball pops loose.
            self.ball_peel_angle += SNATCH_PEEL_GAIN * omega * dt;
            self.ball_being_dribbled_by = Some(holder);
        } else if dribble_claimants.len() == 1 {
            self.ball_being_dribbled_by = Some(dribble_claimants[0]);
            self.ball_peel_angle = 0.0;
        } else {
            if dribble_claimants.len() > 1 {
                self.pop_contested_ball(&dribble_claimants);
            }
            self.ball_being_dribbled_by = None;
            self.ball_peel_angle = 0.0;
        }

        if let Some((player_id, team_color)) = self.ball_being_dribbled_by {
            // Read the holder's pose into locals so the borrow ends before we mutate
            // the ball body and the peel state.
            let (player_position, yaw_angle) = {
                let player = self
                    .players
                    .iter()
                    .find(|p| p.id == player_id && p.team_color == team_color)
                    .unwrap();
                let rb = self.rigid_body_set.get(player.rigid_body_handle).unwrap();
                (
                    rb.position().translation.vector,
                    rb.rotation().euler_angles().2,
                )
            };

            // Pin the ball to the dribbler, rotated by the accumulated peel angle.
            // At peel 0 this is the holder's dribbler centerline (the normal hold).
            let base_offset = self.config.player_radius + self.config.dribbler_radius - 80.0;
            let hold_angle = yaw_angle + self.ball_peel_angle;
            let dir = Vector::new(hold_angle.cos(), hold_angle.sin(), 0.0);
            let dribbler_position = player_position + dir * base_offset;
            let released = self.ball_peel_angle.abs() > self.config.dribbler_angle;

            let ball_body = self
                .rigid_body_set
                .get_mut(self.ball.as_ref().unwrap()._rigid_body_handle)
                .unwrap();
            ball_body.set_position(
                Isometry::translation(
                    dribbler_position.x,
                    dribbler_position.y,
                    dribbler_position.z,
                ),
                true,
            );
            if released {
                // Ball swept past the holder's dribbler cone: it pops loose, flung
                // along the sweep (the direction the contester aimed it) plus a
                // radial-outward component so it clearly leaves the holder's grip
                // and isn't re-claimed on the next frame.
                let tangential = Vector::new(-dir.y, dir.x, 0.0) * self.ball_peel_angle.signum();
                let release_dir = (tangential + dir).normalize();
                ball_body.set_linvel(release_dir * SNATCH_RELEASE_SPEED, true);
            } else {
                ball_body.set_linvel(Vector3::zeros(), true);
            }

            if released {
                self.ball_being_dribbled_by = None;
                self.ball_peel_angle = 0.0;
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

        // Clamp ball position to the field
        if let Some(ball) = self.ball.as_ref() {
            let ball_body = self
                .rigid_body_set
                .get_mut(ball._rigid_body_handle)
                .unwrap();
            let position = ball_body.position().translation.vector;
            let x = position.x.clamp(
                -self.config.field_geometry.field_length / 2.0
                    - self.config.field_geometry.boundary_width,
                self.config.field_geometry.field_length / 2.0
                    + self.config.field_geometry.boundary_width,
            );
            let y = position.y.clamp(
                -self.config.field_geometry.field_width / 2.0
                    - self.config.field_geometry.boundary_width,
                self.config.field_geometry.field_width / 2.0
                    + self.config.field_geometry.boundary_width,
            );
            let z = position.z.clamp(24.0, 100.0);
            ball_body.set_position(Isometry::translation(x, y, z), true);
        }

        self.current_time += dt;
    }

    /// Resolve a contested ball: when two or more robots try to dribble the same
    /// ball, none can hold it. Squirt the ball out perpendicular to the line
    /// joining the first two contenders so it escapes the pile instead of
    /// staying pinned between them (which would deadlock a 50/50).
    fn pop_contested_ball(&mut self, claimants: &[(PlayerId, TeamColor)]) {
        let ball_handle = match self.ball.as_ref() {
            Some(ball) => ball._rigid_body_handle,
            None => return,
        };

        // Look up the first two contenders' positions.
        let mut positions: Vec<Vector3> = Vec::with_capacity(2);
        for &(id, team) in claimants.iter().take(2) {
            if let Some(player) = self
                .players
                .iter()
                .find(|p| p.id == id && p.team_color == team)
            {
                if let Some(body) = self.rigid_body_set.get(player.rigid_body_handle) {
                    positions.push(body.position().translation.vector);
                }
            }
        }
        if positions.len() < 2 {
            return;
        }
        let (p0, p1) = (positions[0], positions[1]);

        let ball_position = match self.rigid_body_set.get(ball_handle) {
            Some(body) => body.position().translation.vector,
            None => return,
        };

        // Escape perpendicular to the squeeze axis between the two robots.
        let axis = p1 - p0;
        let mut escape = Vector3::new(-axis.y, axis.x, 0.0);
        if escape.norm() < 1e-6 {
            escape = Vector3::new(1.0, 0.0, 0.0);
        }
        let mut escape = escape.normalize();
        // Send it toward whichever side the ball already leans, so the pop is
        // stable frame-to-frame instead of fighting itself.
        let midpoint = (p0 + p1) * 0.5;
        if escape.dot(&(ball_position - midpoint)) < 0.0 {
            escape = -escape;
        }

        let ball_body = self.rigid_body_set.get_mut(ball_handle).unwrap();
        ball_body.set_linvel(escape * CONTESTED_BALL_POP_SPEED, true);
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

/// If `p` lies within `margin` of the axis-aligned rectangle `[min, max]`, move
/// it to the closest point exactly `margin` outside the rectangle; otherwise
/// return it unchanged.
fn push_out_of_rect(p: Vector2, min: Vector2, max: Vector2, margin: f64) -> Vector2 {
    let inside_x = p.x > min.x && p.x < max.x;
    let inside_y = p.y > min.y && p.y < max.y;

    if inside_x && inside_y {
        // Strictly inside: exit through the nearest face.
        let to_left = p.x - min.x;
        let to_right = max.x - p.x;
        let to_bottom = p.y - min.y;
        let to_top = max.y - p.y;
        let nearest = to_left.min(to_right).min(to_bottom).min(to_top);
        let mut out = p;
        if nearest == to_left {
            out.x = min.x - margin;
        } else if nearest == to_right {
            out.x = max.x + margin;
        } else if nearest == to_bottom {
            out.y = min.y - margin;
        } else {
            out.y = max.y + margin;
        }
        return out;
    }

    let nearest = Vector2::new(p.x.clamp(min.x, max.x), p.y.clamp(min.y, max.y));
    let delta = p - nearest;
    let dist = delta.norm();
    if dist >= margin {
        return p;
    }
    if dist > 1e-6 {
        nearest + delta / dist * margin
    } else {
        // On an edge: push along +y or -y, away from the rectangle centre.
        let dir = if p.y >= (min.y + max.y) / 2.0 {
            Vector2::new(0.0, 1.0)
        } else {
            Vector2::new(0.0, -1.0)
        };
        nearest + dir * margin
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
            w: 0.0,
            target_heading: f64::NAN,
            target_yaw_rate: f64::INFINITY,
            current_dribble_speed: 0.0,
            breakbeam: false,
            last_kick_counter: 0,
            reflex_armed: false,
            reflex_fired_this_arm: false,
        });
    }
}

impl SimulationBuilder {
    /// Like [`SimulationBuilder::default`] but with the given config, and with
    /// each robot's initial pose deterministically jittered by `config.seed`.
    /// The ball stays centered (kickoff repositions it anyway). Same seed →
    /// identical layout, so headless self-play matches are reproducible while
    /// different seeds produce different-but-reproducible openings.
    ///
    /// Draws are made in a fixed order (blue 0..6, then yellow 0..6) so the
    /// perturbation is a pure function of the seed. Jitter is bounded so robots
    /// never spawn overlapping: the line runs along x (tight spacing), so x
    /// jitter is kept small while y (toward the open center) is generous.
    pub fn default_seeded(config: SimulationConfig) -> Self {
        // Position jitter bounds (mm) and yaw jitter (rad). x is constrained by
        // the inter-robot spacing along the line; y has the open field.
        const JITTER_X: f64 = 30.0;
        const JITTER_Y: f64 = 150.0;
        const JITTER_YAW: f64 = 0.3;

        let mut builder = SimulationBuilder::new(config);
        let field_width = builder.sim.config.field_geometry.field_width / 2.0;
        let field_length = builder.sim.config.field_geometry.field_length / 2.0;
        let boundary_width = builder.sim.config.field_geometry.boundary_width;
        let player_radius = builder.sim.config.player_radius;
        let player_margin = 0.75 * player_radius;
        let sides = builder.sim.config.initial_side_assignment;

        let mut jitter = |builder: &mut SimulationBuilder| {
            let rng = &mut builder.sim.rng;
            (
                rng.gen_range(-JITTER_X..=JITTER_X),
                rng.gen_range(-JITTER_Y..=JITTER_Y),
                rng.gen_range(-JITTER_YAW..=JITTER_YAW),
            )
        };

        // Base x is negative so robots line up on their OWN half (team-relative
        // +x points at the enemy goal, so own half is x < 0); transform_vec2 then
        // maps that to the correct absolute side per the side assignment.
        for i in 0..6 {
            let (dx, dy, dyaw) = jitter(&mut builder);
            let position = Vector2::new(
                -(field_length - player_margin - i as f64 * (2.0 * player_radius + player_margin))
                    + dx,
                field_width - player_radius - boundary_width + dy,
            );
            builder = builder.add_blue_player(
                sides.transform_vec2(TeamColor::Blue, &position),
                Angle::from_radians(dyaw),
            );
        }
        for i in 0..6 {
            let (dx, dy, dyaw) = jitter(&mut builder);
            let position = Vector2::new(
                -(field_length - player_margin - i as f64 * (2.0 * player_radius + player_margin))
                    + dx,
                -(field_width - player_radius - boundary_width) + dy,
            );
            builder = builder.add_yellow_player(
                sides.transform_vec2(TeamColor::Yellow, &position),
                Angle::from_radians(dyaw),
            );
        }

        builder = builder.add_ball(Vector::new(0.0, 0.0, 20.0));

        builder
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

        // Base x is negative so robots line up on their OWN half (team-relative
        // +x points at the enemy goal); transform_vec2 maps to the absolute side.
        for i in 0..6 {
            let position = Vector2::new(
                -(field_length - player_margin - i as f64 * (2.0 * player_radius + player_margin)),
                field_width - player_radius - boundary_width,
            );
            builder = builder.add_blue_player(
                sides.transform_vec2(TeamColor::Blue, &position),
                Angle::from_radians(0.0),
            );
        }
        for i in 0..6 {
            let position = Vector2::new(
                -(field_length - player_margin - i as f64 * (2.0 * player_radius + player_margin)),
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

#[cfg(test)]
mod reflex_kick_tests {
    use super::*;

    /// Build a sim with one blue robot at the origin facing +x and a ball seated
    /// in its dribbler, then apply `cmd` and step a few frames.
    fn sim_with_ball_at_dribbler(cmd: PlayerGlobalMoveCmd) -> Simulation {
        let mut sim = SimulationBuilder::new(SimulationConfig::default())
            .with_controlled_teams(true, true)
            .add_blue_player_with_id(0, Vector2::new(0.0, 0.0), Angle::from_radians(0.0))
            .add_ball(Vector::new(150.0, 0.0, BALL_RADIUS))
            .build();
        sim.push_global_cmd(TeamColor::Blue, cmd);
        for _ in 0..10 {
            sim.step(0.016);
        }
        sim
    }

    fn ball_vx(sim: &Simulation) -> f64 {
        let handle = sim.ball.as_ref().unwrap()._rigid_body_handle;
        sim.rigid_body_set.get(handle).unwrap().linvel().x
    }

    #[test]
    fn reflex_kick_fires_on_ball_contact() {
        // ARM_REFLEX_KICK with no counter increment must fire once the ball is at
        // the breakbeam — the strike-through restart release.
        let mut cmd = PlayerGlobalMoveCmd::zero(PlayerId::new(0));
        cmd.heading_setpoint = 0.0;
        cmd.robot_cmd = RobotCmd::ArmReflex;
        let sim = sim_with_ball_at_dribbler(cmd);
        assert!(
            ball_vx(&sim) > 100.0,
            "reflex kick should have driven the ball forward, vx={}",
            ball_vx(&sim)
        );
    }

    #[test]
    fn smart_kick_fires_on_counter_increment() {
        // Unchanged original path: ARM_COUNTER_KICK (RobotCmd::Arm) fires when the
        // kick counter increments past the robot's last seen value.
        let mut cmd = PlayerGlobalMoveCmd::zero(PlayerId::new(0));
        cmd.heading_setpoint = 0.0;
        cmd.robot_cmd = RobotCmd::Arm;
        cmd.kick_counter = 1;
        let sim = sim_with_ball_at_dribbler(cmd);
        assert!(
            ball_vx(&sim) > 100.0,
            "smart kick should have driven the ball forward, vx={}",
            ball_vx(&sim)
        );
    }

    #[test]
    fn no_kick_without_trigger() {
        // Plain arm with no counter change and no reflex must not kick.
        let mut cmd = PlayerGlobalMoveCmd::zero(PlayerId::new(0));
        cmd.heading_setpoint = 0.0;
        cmd.robot_cmd = RobotCmd::Arm;
        let sim = sim_with_ball_at_dribbler(cmd);
        assert!(
            ball_vx(&sim).abs() < 50.0,
            "ball should be (near) stationary without a kick, vx={}",
            ball_vx(&sim)
        );
    }
}

#[cfg(test)]
mod free_kick_placement_tests {
    use super::*;

    /// Distance from a point to an axis-aligned rectangle (0 if inside).
    fn dist_to_rect(p: Vector2, min: Vector2, max: Vector2) -> f64 {
        let dx = (min.x - p.x).max(0.0).max(p.x - max.x);
        let dy = (min.y - p.y).max(0.0).max(p.y - max.y);
        (dx * dx + dy * dy).sqrt()
    }

    /// Assert a position obeys rule 5.3.3: inside the field margin and ≥1 m from
    /// both defense areas.
    fn assert_valid(sim: &Simulation, p: Vector2) {
        let g = &sim.config.field_geometry;
        let (hl, hw) = (g.field_length / 2.0, g.field_width / 2.0);
        let half_pa_w = g.penalty_area_width / 2.0;
        assert!(
            p.x.abs() <= hl - FREE_KICK_PLACEMENT_MARGIN + 1e-6
                && p.y.abs() <= hw - FREE_KICK_PLACEMENT_MARGIN + 1e-6,
            "outside field margin: {p:?}"
        );
        for &goal_x in &[-hl, hl] {
            let inner_x = goal_x - g.penalty_area_depth * goal_x.signum();
            let min = Vector2::new(goal_x.min(inner_x), -half_pa_w);
            let max = Vector2::new(goal_x.max(inner_x), half_pa_w);
            let d = dist_to_rect(p, min, max);
            assert!(
                d >= FREE_KICK_DEFENSE_AREA_DISTANCE - 1e-6,
                "too close to defense area: {p:?} d={d}"
            );
        }
    }

    #[test]
    fn relocates_ball_jammed_against_defense_area() {
        // The exact position from the stuck free kick: 63 mm outside the
        // opponent defense-area corner, where the kicker could not get behind it.
        let sim = Simulation::default();
        let fixed = sim.valid_free_kick_position(Vector2::new(-4300.0, 1063.1));
        assert!(
            (fixed - Vector2::new(-4300.0, 1063.1)).norm() > 1.0,
            "should have moved the ball"
        );
        assert_valid(&sim, fixed);
    }

    #[test]
    fn ball_out_over_goal_line_is_not_left_inside_the_box() {
        // Regression: a ball that left over the goal line beside the goal lands
        // inside the defense area, closer to the goal line than to the front edge
        // or the sides. The keep-out must push it out a *field-facing* face, not
        // back through the goal line (which the field clamp would then undo,
        // stranding the free kick at an illegal in-area position).
        let sim = Simulation::default();
        for desired in [
            Vector2::new(4300.0, 693.0),   // the exact stuck case from the log
            Vector2::new(-4300.0, -693.0), // mirror at the other goal
            Vector2::new(4400.0, 0.0),     // dead centre, hard against the goal line
        ] {
            let fixed = sim.valid_free_kick_position(desired);
            assert_valid(&sim, fixed);
        }
    }

    #[test]
    fn leaves_already_valid_position_untouched() {
        let sim = Simulation::default();
        for desired in [
            Vector2::new(0.0, 0.0),
            Vector2::new(1500.0, -500.0),
            Vector2::new(-2000.0, 2000.0),
        ] {
            let fixed = sim.valid_free_kick_position(desired);
            assert!(
                (fixed - desired).norm() < 1e-6,
                "moved valid pos {desired:?}"
            );
            assert_valid(&sim, fixed);
        }
    }

    #[test]
    fn relocates_near_both_goals() {
        let sim = Simulation::default();
        for desired in [
            Vector2::new(-4300.0, 1063.1),
            Vector2::new(4300.0, -1063.1),
            Vector2::new(4000.0, 0.0),
            Vector2::new(-4000.0, 900.0),
        ] {
            assert_valid(&sim, sim.valid_free_kick_position(desired));
        }
    }
}

#[cfg(test)]
mod ball_out_tests {
    use super::*;

    fn sim_with_ball() -> Simulation {
        SimulationBuilder::new(SimulationConfig::default())
            .add_ball(Vector::new(0.0, 0.0, BALL_RADIUS))
            .build()
    }

    #[test]
    fn ball_inside_real_field_is_not_out() {
        // Regression: a corner free-kick ball resting ~62 mm inside the goal line
        // (the exact stuck position from the log) must NOT count as out. The old
        // 100 mm test boundary shrank the field below the 200 mm placement margin,
        // so it flagged a legally-placed corner free kick as out and instantly
        // re-stopped the game.
        let mut sim = sim_with_ball();
        sim.place_ball(-4437.86, 2738.58);
        assert!(!sim.ball_out(), "ball inside the real field flagged as out");
    }

    #[test]
    fn placed_free_kick_ball_is_never_out() {
        // Any ball snapped to a valid free-kick spot sits within the placement
        // margin and must stay in play.
        let mut sim = sim_with_ball();
        for desired in [
            Vector2::new(-4300.0, 2900.0), // top-left corner
            Vector2::new(4300.0, -2900.0), // bottom-right corner
            Vector2::new(4400.0, 0.0),
        ] {
            let p = sim.valid_free_kick_position(desired);
            sim.place_ball(p.x, p.y);
            assert!(
                !sim.ball_out(),
                "placed free-kick ball {p:?} flagged as out"
            );
        }
    }

    #[test]
    fn ball_past_the_line_is_out() {
        let mut sim = sim_with_ball();
        sim.place_ball(-4600.0, 0.0); // past the goal line
        assert!(sim.ball_out(), "ball past the goal line not flagged out");
        sim.place_ball(0.0, 3100.0); // past the touch line
        assert!(sim.ball_out(), "ball past the touch line not flagged out");
    }

    fn set_ball_vel(sim: &mut Simulation, vx: f64, vy: f64) {
        let h = sim.ball.as_ref().unwrap()._rigid_body_handle;
        sim.rigid_body_set
            .get_mut(h)
            .unwrap()
            .set_linvel(Vector::new(vx, vy, 0.0), true);
    }

    fn ball_xy(sim: &Simulation) -> Vector2 {
        let h = sim.ball.as_ref().unwrap()._rigid_body_handle;
        sim.rigid_body_set
            .get(h)
            .unwrap()
            .position()
            .translation
            .vector
            .xy()
    }

    /// Regression for the botched free-kick sequence (log frame 813): a ball that
    /// rolls out of bounds while in `Run` must be teleported back to a legal
    /// free-kick spot *and stay there* through the Stop + FreeKick period, so the
    /// free kick can actually be taken. The original bug let the ball keep rolling
    /// out to the boundary wall; the free kick was awarded with the ball still out,
    /// `check_ball_in_play` mistook the ball's drift along the wall for the kick
    /// being taken, and the sequence collapsed back into a force-start.
    #[test]
    fn rolling_ball_out_is_relocated_in_bounds_for_free_kick() {
        let mut sim = sim_with_ball();
        // Last toucher set so a free kick (not a neutral force-start) is awarded.
        sim.last_touch_info = Some((PlayerId::new(0), TeamColor::Blue));
        sim.game_state = SimulationGameState::run();

        // Seat the ball near the negative goal line, still in bounds, flying out
        // fast and at an angle (mirrors the logged corner roll-out).
        sim.place_ball(-4400.0, -1700.0);
        set_ball_vel(&mut sim, -1500.0, 350.0);

        let hl = sim.config.field_geometry.field_length / 2.0;
        let hw = sim.config.field_geometry.field_width / 2.0;

        // Step through the whole restart at 60 Hz: Stop (2 s) + free kick, with
        // enough headroom for the 10 s free-kick timeout to resume play since this
        // test has no kicker robot. Once the autoref has awarded the free kick
        // (StopAndFreeKick / FreeKick), the ball must be sitting at a legal
        // in-bounds spot — never pinned against the boundary wall.
        let mut saw_free_kick = false;
        let mut reached_run_after = false;
        for _ in 0..780 {
            sim.step(1.0 / 60.0);
            let b = ball_xy(&sim);
            match sim.game_state {
                SimulationGameState::StopAndFreeKick { .. }
                | SimulationGameState::FreeKick { .. } => {
                    saw_free_kick = true;
                    assert!(
                        b.x.abs() <= hl + 1e-3 && b.y.abs() <= hw + 1e-3,
                        "free kick awarded with the ball still out of bounds: {b:?}"
                    );
                }
                SimulationGameState::Run { .. } if saw_free_kick => {
                    reached_run_after = true;
                }
                // The free-kick restart must not collapse back into a neutral
                // force-start because the ball was left out of bounds.
                SimulationGameState::StopAndForceStart { .. } if saw_free_kick => {
                    panic!("free-kick restart collapsed into a force-start (ball left out)");
                }
                _ => {}
            }
        }

        assert!(saw_free_kick, "free kick was never awarded");
        assert!(
            reached_run_after,
            "game never resumed (Run) after the free kick"
        );
        let b = ball_xy(&sim);
        assert!(
            b.x.abs() <= hl && b.y.abs() <= hw,
            "ball still out of bounds at end of restart: {b:?}"
        );
    }
}

#[cfg(test)]
mod determinism_tests {
    use super::*;

    /// Build the standard seeded 6v6 layout and step it `n` frames at 60 Hz with
    /// no commands (robots settle, ball rests), returning the trajectory
    /// fingerprint folded over every frame.
    fn run(seed: u64, n: usize) -> u64 {
        let config = SimulationConfig {
            seed,
            ..Default::default()
        };
        let mut sim = SimulationBuilder::default_seeded(config)
            .with_controlled_teams(true, true)
            .build();
        let mut hash: u64 = 0;
        for _ in 0..n {
            sim.step(1.0 / 60.0);
            hash =
                hash.rotate_left(7).wrapping_mul(0x0000_0100_0000_01b3) ^ sim.state_fingerprint();
        }
        hash
    }

    #[test]
    fn same_seed_is_deterministic() {
        // The whole point of headless self-play: identical seed → identical run.
        assert_eq!(run(42, 300), run(42, 300));
    }

    #[test]
    fn different_seeds_diverge() {
        // Pose jitter must actually vary the match, otherwise sweeping seeds for
        // A-vs-B evaluation is pointless.
        assert_ne!(run(1, 300), run(2, 300));
    }
}

#[cfg(test)]
mod snatch_peel_tests {
    use super::*;

    const BLUE: TeamColor = TeamColor::Blue;
    const YELLOW: TeamColor = TeamColor::Yellow;

    /// Blue holder at the origin facing +x with the ball seated in its dribbler,
    /// and a yellow contester pressed against the ball from the far (+x) side.
    fn setup() -> Simulation {
        SimulationBuilder::new(SimulationConfig::default())
            .with_controlled_teams(true, true)
            .add_blue_player_with_id(0, Vector2::new(0.0, 0.0), Angle::from_radians(0.0))
            .add_yellow_player_with_id(0, Vector2::new(340.0, 0.0), Angle::from_radians(PI))
            .add_ball(Vector::new(150.0, 0.0, BALL_RADIUS))
            .build()
    }

    fn hold(dribble: f64) -> PlayerMoveCmd {
        let mut c = PlayerMoveCmd::zero(PlayerId::new(0));
        c.dribble_speed = dribble;
        c
    }

    fn press(dribble: f64, w: f64) -> PlayerMoveCmd {
        let mut c = PlayerMoveCmd::zero(PlayerId::new(0));
        c.dribble_speed = dribble;
        c.w = w;
        c
    }

    fn holder(sim: &Simulation) -> Option<(PlayerId, TeamColor)> {
        sim.ball_being_dribbled_by
    }

    fn ball_speed(sim: &Simulation) -> f64 {
        let h = sim.ball.as_ref().unwrap()._rigid_body_handle;
        let v = sim.rigid_body_set.get(h).unwrap().linvel();
        (v.x * v.x + v.y * v.y).sqrt()
    }

    /// A contester that presses *and rotates* peels the ball off the holder.
    #[test]
    fn rotating_press_strips_the_ball() {
        let mut sim = setup();

        // Phase A: blue establishes possession, yellow idle.
        for _ in 0..30 {
            sim.push_cmd(BLUE, hold(0.5));
            sim.step(1.0 / 60.0);
        }
        assert_eq!(
            holder(&sim),
            Some((PlayerId::new(0), BLUE)),
            "blue should hold the ball before the strip"
        );

        // Phase B: yellow presses with its dribbler on and rotates in place.
        let mut stripped_at = None;
        for f in 0..120 {
            sim.push_cmd(BLUE, hold(0.5));
            sim.push_cmd(YELLOW, press(0.8, 2.0));
            sim.step(1.0 / 60.0);
            if holder(&sim).is_none() {
                stripped_at = Some(f);
                break;
            }
        }

        let f = stripped_at.expect("rotating press should strip the ball within 2s");
        assert!(
            ball_speed(&sim) > 100.0,
            "stripped ball should be flung loose, speed={}",
            ball_speed(&sim)
        );
        // Slow peel, not instant: it should take a meaningful fraction of a second.
        assert!(f > 10, "strip happened suspiciously fast at frame {f}");
    }

    /// Pressing without rotating must NOT strip the ball — only ω peels.
    #[test]
    fn static_press_does_not_strip() {
        let mut sim = setup();
        for _ in 0..30 {
            sim.push_cmd(BLUE, hold(0.5));
            sim.step(1.0 / 60.0);
        }
        assert_eq!(holder(&sim), Some((PlayerId::new(0), BLUE)));

        for _ in 0..120 {
            sim.push_cmd(BLUE, hold(0.5));
            sim.push_cmd(YELLOW, press(0.8, 0.0)); // dribbler on, no rotation
            sim.step(1.0 / 60.0);
        }
        assert_eq!(
            holder(&sim),
            Some((PlayerId::new(0), BLUE)),
            "a non-rotating press should not peel the ball off the holder"
        );
    }

    /// Two robots reaching for a loose ball on the same frame (no prior holder)
    /// must still trip the pop guard that squirts the ball out, rather than the
    /// snatch peel path silently handing possession to one of them. The peel path
    /// only applies once a robot has an *established* possession.
    #[test]
    fn fresh_5050_still_pops_loose() {
        let mut sim = setup();
        // Both reach for the ball on the same frame, neither rotating.
        let mut max_speed = 0.0_f64;
        for _ in 0..20 {
            sim.push_cmd(BLUE, hold(0.8));
            sim.push_cmd(YELLOW, press(0.8, 0.0));
            sim.step(1.0 / 60.0);
            max_speed = max_speed.max(ball_speed(&sim));
        }
        assert!(
            max_speed > 800.0,
            "a fresh 50/50 should pop the ball loose (guard fires); max speed={max_speed}"
        );
    }
}
