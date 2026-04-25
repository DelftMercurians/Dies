use std::{collections::HashMap, path::PathBuf, time::Duration};

use anyhow::Result;
use dies_basestation_client::BasestationHandle;
use dies_core::{
    ExecutorInfo, ExecutorSettings, PlayerCmd, PlayerFeedbackMsg, PlayerId, PlayerOverrideCommand,
    SideAssignment, SimulatorCmd, TeamColor, TeamPlayerId, Vector3, WorldInstant, WorldUpdate,
};
use dies_logger::{log_referee, log_vision, log_world};
use dies_protos::{
    ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket,
    ssl_vision_wrapper_tracked::TrackerWrapperPacket,
};
use dies_simulator::Simulation;
use dies_ssl_client::{SslMessage, VisionClient};
use dies_test_driver::{LogBus, PlayerControlSlot, TestDriver, TestEnv, TestStatus};
use dies_world::WorldTracker;
use gc_client::GcClient;
pub use handle::{ControlMsg, ExecutorHandle};
use tokio::sync::{broadcast, mpsc, oneshot, watch};

pub mod control;
mod gc_client;
mod handle;
pub mod skills;
pub mod strategy_host;

pub use control::*;

/// Translate a test-driver `PlayerControlSlot` into the executor's
/// `PlayerControlInput`. Global velocity wins over local if both set.
fn slot_to_control_input(slot: &PlayerControlSlot) -> PlayerControlInput {
    let mut input = PlayerControlInput::default();
    input.position = slot.position;
    input.yaw = slot.yaw;
    input.velocity = if let Some(v) = slot.vel_global {
        Velocity::Global(v)
    } else if let Some(v) = slot.vel_local {
        Velocity::Local(v)
    } else {
        Velocity::Local(dies_core::Vector2::zeros())
    };
    input.angular_velocity = slot.angular_velocity;
    input.dribbling_speed = slot.dribble;
    input.fan_speed = slot.fan;
    input.kick_speed = slot.kick_speed;
    input.kicker = if let Some(force) = slot.kick_force {
        KickerControlInput::Kick { force }
    } else if slot.disarm_kicker {
        KickerControlInput::Disarm
    } else {
        KickerControlInput::Idle
    };
    input
}

const SIMULATION_DT: Duration = Duration::from_micros(1_000_000 / 60); // 60 Hz
const CMD_INTERVAL: Duration = Duration::from_micros(1_000_000 / 40); // 40 Hz

enum Environment {
    Live {
        ssl_client: VisionClient,
        bs_client: BasestationHandle,
    },
    Simulation {
        simulator: Simulation,
    },
}

struct TeamMap {
    blue_team: Option<TeamController>,
    yellow_team: Option<TeamController>,
    side_assignment: SideAssignment,
}

impl TeamMap {
    fn new(side_assignment: SideAssignment) -> Self {
        Self {
            blue_team: None,
            yellow_team: None,
            side_assignment,
        }
    }

    fn activate_team(&mut self, team_color: TeamColor, settings: &ExecutorSettings) {
        if team_color == TeamColor::Blue {
            if self.blue_team.is_none() {
                self.blue_team = Some(TeamController::new(settings, TeamColor::Blue));
            }
        } else {
            if self.yellow_team.is_none() {
                self.yellow_team = Some(TeamController::new(settings, TeamColor::Yellow));
            }
        }
    }

    fn update_settings(&mut self, settings: &ExecutorSettings) {
        if let Some(controller) = &mut self.blue_team {
            controller.update_controller_settings(settings);
        }
        if let Some(controller) = &mut self.yellow_team {
            controller.update_controller_settings(settings);
        }
    }

    fn swap_teams(&mut self) {
        std::mem::swap(&mut self.blue_team, &mut self.yellow_team);
    }

    /// Get all player commands from active controllers
    fn get_all_commands(&mut self) -> Vec<(TeamColor, PlayerCmd)> {
        let mut commands = Vec::new();

        if let Some(controller) = &mut self.blue_team {
            commands.extend(
                controller
                    .commands(self.side_assignment, TeamColor::Blue)
                    .into_iter()
                    .map(|c| (TeamColor::Blue, c)),
            );
        }
        if let Some(controller) = &mut self.yellow_team {
            commands.extend(
                controller
                    .commands(self.side_assignment, TeamColor::Yellow)
                    .into_iter()
                    .map(|c| (TeamColor::Yellow, c)),
            );
        }

        commands
    }

    /// Get list of active team colors
    fn active_teams(&self) -> Vec<TeamColor> {
        let mut teams = Vec::new();

        // Check if team_a is active and get its color
        if self.blue_team.is_some() {
            teams.push(TeamColor::Blue);
        }

        // Check if team_b is active and get its color
        if self.yellow_team.is_some() {
            teams.push(TeamColor::Yellow);
        }

        teams
    }

    /// Update all active team controllers with team-specific data
    fn update(
        &mut self,
        world_data: &dies_core::WorldData,
        manual_override: HashMap<(TeamColor, PlayerId), PlayerControlInput>,
    ) {
        // Update blue team controller if active
        if let Some(controller) = &mut self.blue_team {
            let team_data = world_data.get_team_data(TeamColor::Blue);
            controller.update(
                TeamColor::Blue,
                world_data.side_assignment,
                team_data,
                manual_override
                    .iter()
                    .filter_map(|((color, id), input)| {
                        if *color == TeamColor::Blue {
                            Some((*id, input.clone()))
                        } else {
                            None
                        }
                    })
                    .collect(),
            );
        }

        // Update yellow team controller if active
        if let Some(controller) = &mut self.yellow_team {
            let team_data = world_data.get_team_data(TeamColor::Yellow);
            controller.update(
                TeamColor::Yellow,
                world_data.side_assignment,
                team_data,
                manual_override
                    .iter()
                    .filter_map(|((color, id), input)| {
                        if *color == TeamColor::Yellow {
                            Some((*id, input.clone()))
                        } else {
                            None
                        }
                    })
                    .collect(),
            );
        }
    }
}

struct PlayerOverrideState {
    frame_counter: u32,
    current_command: PlayerOverrideCommand,
}

impl PlayerOverrideState {
    const VELOCITY_TIMEOUT: u32 = 10;

    pub fn new() -> Self {
        Self {
            frame_counter: 0,
            current_command: PlayerOverrideCommand::Stop,
        }
    }

    pub fn set_cmd(&mut self, cmd: PlayerOverrideCommand) {
        self.current_command = cmd;
        self.frame_counter = 0;
    }

    pub fn advance(&mut self) -> PlayerControlInput {
        let input = match self.current_command {
            PlayerOverrideCommand::Stop => PlayerControlInput::new(),
            PlayerOverrideCommand::MoveTo {
                position,
                yaw,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                position: Some(position),
                yaw: Some(yaw),
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::LocalVelocity {
                velocity,
                yaw,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                velocity: Velocity::local(velocity),
                yaw,
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::GlobalVelocity {
                velocity,
                yaw,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                velocity: Velocity::global(velocity),
                yaw,
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::Kick { speed } => PlayerControlInput {
                kicker: KickerControlInput::Kick { force: speed },
                kick_speed: Some(speed),
                ..Default::default()
            },
            PlayerOverrideCommand::DischargeKicker => PlayerControlInput {
                kicker: KickerControlInput::Disarm,
                ..Default::default()
            },
            PlayerOverrideCommand::SetFanSpeed { speed } => PlayerControlInput {
                fan_speed: Some(speed),
                ..Default::default()
            },
        };

        // Advance the frame counter
        self.frame_counter += 1;
        match self.current_command {
            PlayerOverrideCommand::LocalVelocity { .. }
            | PlayerOverrideCommand::GlobalVelocity { .. } => {
                if self.frame_counter > Self::VELOCITY_TIMEOUT {
                    self.current_command = PlayerOverrideCommand::Stop;
                    self.frame_counter = 0;
                }
            }
            PlayerOverrideCommand::Kick { .. } => {
                self.current_command = PlayerOverrideCommand::Stop
            }
            _ => {}
        };

        input
    }
}

/// The central component of the framework. It contains all state and logic needed to
/// run a match -- processing vision and referee messages, executing the strategy, and
/// sending commands to the robots.
///
/// The executor can be used in 3 different regimes: externally driven, automatic, and
/// simulation. Now supports team-agnostic operation with 0, 1, or 2 active team controllers.
///
/// At runtime, the executor's control flow can be switched between **strategy** mode
/// (the default) and **scenario** mode via [`ControlMsg::StartScenario`] /
/// [`ControlMsg::StopScenario`]. When a scenario is loaded, the test driver takes
/// over for its declared team; the strategy host is preserved but skipped each tick.
pub struct Executor {
    tracker: WorldTracker,
    /// Strategy host. Always present even when a scenario is running — it's just
    /// not ticked. None only if the executor was constructed without strategy support
    /// (currently never).
    strategy_host: Option<strategy_host::StrategyHost>,
    /// Active scenario test driver. When `Some`, replaces the strategy update path
    /// for the scenario's declared team.
    test_driver: Option<TestDriver>,
    /// Persistent log bus shared with every test driver and exposed via the handle.
    log_bus: LogBus,
    /// Broadcasts the latest scenario status (Idle / Running / Completed / etc.).
    scenario_status_tx: watch::Sender<TestStatus>,
    team_controllers: TeamMap,
    gc_client: GcClient,
    environment: Option<Environment>,
    manual_override: HashMap<(TeamColor, PlayerId), PlayerOverrideState>,
    /// Players whose override input is being driven by the test driver (direct path).
    /// Separate from manual_override so operator keyboard overrides don't collide.
    test_manual: HashMap<(TeamColor, PlayerId), PlayerControlInput>,
    /// Queue of simulator commands produced by the test driver this tick —
    /// drained by `run_rt_sim` each frame.
    pending_sim_cmds: Vec<SimulatorCmd>,
    update_tx: broadcast::Sender<WorldUpdate>,
    command_tx: mpsc::UnboundedSender<ControlMsg>,
    command_rx: mpsc::UnboundedReceiver<ControlMsg>,
    paused_tx: watch::Sender<bool>,
    info_channel_rx: mpsc::UnboundedReceiver<oneshot::Sender<ExecutorInfo>>,
    info_channel_tx: mpsc::UnboundedSender<oneshot::Sender<ExecutorInfo>>,
    settings: ExecutorSettings,
}

impl Executor {
    /// Create a new executor.
    pub fn new_live(
        settings: ExecutorSettings,
        ssl_client: VisionClient,
        bs_client: BasestationHandle,
    ) -> Self {
        Self::build(
            settings,
            Environment::Live {
                ssl_client,
                bs_client,
            },
        )
    }

    /// Create a new simulation executor.
    pub fn new_simulation(settings: ExecutorSettings, simulator: Simulation) -> Self {
        Self::build(settings, Environment::Simulation { simulator })
    }

    fn build(settings: ExecutorSettings, mut environment: Environment) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (update_tx, _) = broadcast::channel(16);
        let (paused_tx, _) = watch::channel(false);
        let (info_channel_tx, info_channel_rx) = mpsc::unbounded_channel();
        let (scenario_status_tx, _) = watch::channel(TestStatus::Idle);

        let mut strategy_host =
            strategy_host::StrategyHost::new(strategy_host::StrategyHostConfig {
                strategies_dir: std::path::PathBuf::from("target/debug"),
                blue_strategy: settings.team_configuration.blue_strategy.clone(),
                yellow_strategy: settings.team_configuration.yellow_strategy.clone(),
                side_assignment: settings.team_configuration.side_assignment,
            });
        if let Some(ref name) = settings.team_configuration.blue_strategy {
            strategy_host.set_strategy(TeamColor::Blue, Some(name.clone()));
        }
        if let Some(ref name) = settings.team_configuration.yellow_strategy {
            strategy_host.set_strategy(TeamColor::Yellow, Some(name.clone()));
        }

        let mut team_controllers = TeamMap::new(settings.team_configuration.side_assignment);
        let mut controlled_teams = Vec::new();
        if settings.team_configuration.blue_active {
            team_controllers.activate_team(TeamColor::Blue, &settings);
            controlled_teams.push(TeamColor::Blue);
        }
        if settings.team_configuration.yellow_active {
            team_controllers.activate_team(TeamColor::Yellow, &settings);
            controlled_teams.push(TeamColor::Yellow);
        }
        if let Environment::Simulation { simulator } = &mut environment {
            simulator.set_controlled_teams(&controlled_teams);
        }

        Self {
            tracker: WorldTracker::new(&settings, &controlled_teams, settings.allow_no_vision),
            strategy_host: Some(strategy_host),
            test_driver: None,
            log_bus: LogBus::new(1024),
            scenario_status_tx,
            team_controllers,
            gc_client: GcClient::new(),
            environment: Some(environment),
            manual_override: HashMap::new(),
            test_manual: HashMap::new(),
            pending_sim_cmds: Vec::new(),
            command_tx,
            command_rx,
            update_tx,
            paused_tx,
            info_channel_rx,
            info_channel_tx,
            settings,
        }
    }

    /// Update the executor with a vision message.
    pub fn update_from_vision_msg(&mut self, message: SSL_WrapperPacket, time: WorldInstant) {
        log_vision(&message);
        self.tracker.update_from_vision(&message, time);
        log_world(&self.tracker.get());
        self.update_team_controller();
    }

    /// Update the executor with a referee message.
    pub fn update_from_gc_msg(&mut self, message: Referee) {
        log_referee(&message);
        self.tracker.update_from_referee(&message);

        // Detect side assignment from referee message
        if let Some(blue_on_positive) = message.blue_team_on_positive_half {
            self.team_controllers.side_assignment = if blue_on_positive {
                SideAssignment::BlueOnPositive
            } else {
                SideAssignment::YellowOnPositive
            };
            self.tracker
                .set_side_assignment(self.team_controllers.side_assignment);
        }

        self.update_team_controller();
    }

    pub fn update_from_tracker_msg(&mut self, mut message: TrackerWrapperPacket) {
        if let Some(tracked_frame) = message.tracked_frame.take() {
            self.tracker.update_from_tracker(tracked_frame);
        }
    }

    /// Update the executor with a feedback message from the robots.
    pub fn update_from_bs_msg(
        &mut self,
        team_color: TeamColor,
        message: PlayerFeedbackMsg,
        time: WorldInstant,
    ) {
        self.tracker
            .update_from_feedback(team_color, &message, time);
    }

    /// Get the currently active player commands.
    pub fn player_commands(&mut self) -> Vec<(TeamColor, PlayerCmd)> {
        self.team_controllers.get_all_commands()
    }

    /// Get the GC messages that need to be sent. This will remove the messages from the
    /// internal queue.
    pub fn gc_messages(&mut self) -> Vec<Referee> {
        self.gc_client.messages()
    }

    /// Get runtime information about the executor.
    pub fn info(&self) -> ExecutorInfo {
        ExecutorInfo {
            paused: *self.paused_tx.borrow(),
            manual_controlled_players: self
                .manual_override
                .keys()
                .map(|(team_color, player_id)| TeamPlayerId {
                    team_color: *team_color,
                    player_id: *player_id,
                })
                .collect(),
            active_teams: self.team_controllers.active_teams(),
        }
    }

    /// Run the executor in real time, either with real clients or with a simulator,
    /// depending on the configuration.
    pub async fn run_real_time(mut self) -> Result<()> {
        match self.environment.take() {
            Some(Environment::Live {
                ssl_client,
                bs_client,
            }) => self.run_rt_live(ssl_client, bs_client).await?,
            Some(Environment::Simulation { simulator }) => {
                self.run_rt_sim(simulator, SIMULATION_DT).await?
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    pub fn step_simulation(&mut self, simulator: &mut Simulation, dt: Duration) -> Result<()> {
        simulator.step(dt.as_secs_f64());
        if let Some(geom) = simulator.geometry() {
            self.update_from_vision_msg(geom, simulator.time());
        }
        if let Some(det) = simulator.detection() {
            self.update_from_vision_msg(det, simulator.time());
        }
        if let Some(gc_message) = simulator.gc_message() {
            self.update_from_gc_msg(gc_message);
        }
        for ((team_color, _), feedback) in simulator.feedback().into_iter() {
            self.update_from_bs_msg(team_color, feedback, simulator.time());
        }

        Ok(())
    }

    /// Run the executor in real time on the simulator
    async fn run_rt_sim(&mut self, mut simulator: Simulation, dt: Duration) -> Result<()> {
        let mut update_interval = tokio::time::interval(dt);
        update_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut cmd_interval = tokio::time::interval(CMD_INTERVAL);
        cmd_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut paused_rx = self.paused_tx.subscribe();
        loop {
            let paused = { *paused_rx.borrow_and_update() };
            if !paused {
                tokio::select! {
                    Some(msg) = self.command_rx.recv() => {
                        match msg {
                            ControlMsg::Stop => break,
                            ControlMsg::GcCommand { command } => {
                                simulator.handle_gc_command(command);
                            }
                            ControlMsg::SimulatorCmd(cmd) => self.handle_simulator_cmd(&mut simulator, cmd),
                            ControlMsg::SetActiveTeams { .. } => {
                                log::warn!("Setting active teams is not supported mid run");
                            }
                            msg => self.handle_control_msg(msg)
                        }
                    }
                    Some(tx) = self.info_channel_rx.recv() => {
                        let _  = tx.send(self.info());
                    }
                    _ = update_interval.tick() => {
                        self.step_simulation(&mut simulator, dt)?;
                    }
                    _ = cmd_interval.tick() => {
                        // Drain any simulator commands queued by the test driver.
                        let pending = std::mem::take(&mut self.pending_sim_cmds);
                        for cmd in pending {
                            self.handle_simulator_cmd(&mut simulator, cmd);
                        }
                        for (team_color, cmd) in self.player_commands() {
                            match cmd {
                                PlayerCmd::Move(cmd) => {
                                    simulator.push_cmd(team_color, cmd);
                                }
                                PlayerCmd::GlobalMove(cmd) => {
                                    simulator.push_global_cmd(team_color, cmd);
                                }
                            }
                        }
                    }
                }
            } else {
                tokio::select! {
                    Some(ControlMsg::Stop) = self.command_rx.recv() => break,
                    Some(tx) = self.info_channel_rx.recv() => {
                        let _  = tx.send(self.info());
                    }
                    res = paused_rx.changed() => res?
                }
            }
        }

        if let Some(h) = &mut self.strategy_host {
            h.shutdown();
        }
        Ok(())
    }

    /// Run the executor in real time, processing vision and referee messages and sending
    /// commands to the robots.
    async fn run_rt_live(
        &mut self,
        mut ssl_client: VisionClient,
        mut bs_client: BasestationHandle,
    ) -> Result<()> {
        let mut cmd_interval = tokio::time::interval(CMD_INTERVAL);

        let mut last_bs_msg = tokio::time::Instant::now();
        loop {
            tokio::select! {
                Some(msg) = self.command_rx.recv() => {
                    match msg {
                        ControlMsg::Stop => break,
                        ControlMsg::SetActiveTeams { .. } => {
                            log::warn!("Setting active teams is not supported mid run");
                        }
                        msg => self.handle_control_msg(msg)
                    }
                }
                ssl_msg = ssl_client.recv() => {
                    match ssl_msg {
                        Ok(SslMessage::Vision(vision_msg)) => {
                            self.update_from_vision_msg(vision_msg, WorldInstant::now_real());
                        }
                        Ok(SslMessage::Referee(gc_msg)) => {
                            self.update_from_gc_msg(gc_msg);
                        }
                        Ok(SslMessage::Tracker(tracker_msg)) => {
                            self.update_from_tracker_msg(tracker_msg);
                        }
                        Err(err) => {
                            if !self.settings.allow_no_vision {
                                log::error!("Failed to receive vision/gc msg: {}", err);
                            } else {
                                self.update_team_controller();
                            }
                        }
                    }
                }
                Some(tx) = self.info_channel_rx.recv() => {
                    let _  = tx.send(self.info());
                }
                bs_msg = bs_client.recv() => {
                    dies_core::debug_value("exec_bs msg_elapsed", last_bs_msg.elapsed().as_secs_f64() * 1000.0);
                    last_bs_msg = tokio::time::Instant::now();
                    match bs_msg {
                        Ok((team_color, bs_msg)) => {
                            let team_color = team_color.or(self.team_controllers.active_teams().first().copied());
                            if let Some(team_color) = team_color {
                                self.update_from_bs_msg(
                                    team_color,
                                    bs_msg,
                                    WorldInstant::now_real(),
                                );
                            }
                        }
                        Err(err) => {
                            log::error!("Failed to receive BS msg: {}", err);
                        }
                    }
                }
                _ = cmd_interval.tick() => {
                    for (team_color, cmd) in self.player_commands() {
                        bs_client.send_no_wait(team_color, cmd);
                    }
                }
            }
        }
        if let Some(h) = &mut self.strategy_host {
            h.shutdown();
        }
        Ok(())
    }

    fn handle_simulator_cmd(&self, sim: &mut Simulation, cmd: SimulatorCmd) {
        match cmd {
            SimulatorCmd::ApplyBallForce { force } => {
                sim.apply_force_to_ball(Vector3::new(force.x, force.y, 0.0));
            }
            SimulatorCmd::TeleportRobot {
                team_color,
                player_id,
                position,
                yaw,
            } => {
                sim.teleport_robot(team_color, player_id, position, yaw);
            }
            SimulatorCmd::AddRobot {
                team_color,
                player_id,
                position,
                yaw,
            } => {
                sim.add_robot(team_color, player_id, position, yaw);
            }
            SimulatorCmd::RemoveRobot {
                team_color,
                player_id,
            } => {
                sim.remove_robot(team_color, player_id);
            }
        }
    }

    pub fn handle(&self) -> ExecutorHandle {
        ExecutorHandle {
            control_tx: self.command_tx.clone(),
            update_rx: self.update_tx.subscribe(),
            info_channel: self.info_channel_tx.clone(),
            log_bus: self.log_bus.clone(),
            scenario_status_rx: self.scenario_status_tx.subscribe(),
        }
    }

    fn handle_control_msg(&mut self, msg: ControlMsg) {
        match msg {
            ControlMsg::SetPause(pause) => {
                self.paused_tx.send(pause).ok();
            }
            ControlMsg::SetPlayerOverride {
                team_color,
                player_id,
                override_active,
            } => {
                if override_active {
                    self.manual_override
                        .insert((team_color, player_id), PlayerOverrideState::new());
                } else {
                    self.manual_override.remove(&(team_color, player_id));
                }
            }
            ControlMsg::PlayerOverrideCommand {
                team_color,
                player_id,
                command,
            } => {
                if let Some(state) = self.manual_override.get_mut(&(team_color, player_id)) {
                    state.set_cmd(command);
                }
            }
            ControlMsg::UpdateSettings(settings) => {
                self.team_controllers.update_settings(&settings);
                self.tracker.update_settings(&settings);
                self.settings = settings;
            }
            ControlMsg::SetSideAssignment(side_assignment) => {
                self.team_controllers.side_assignment = side_assignment;
                self.tracker.set_side_assignment(side_assignment);
                if let Some(h) = &mut self.strategy_host {
                    h.set_side_assignment(side_assignment);
                }
            }
            ControlMsg::SetTeamConfiguration(_) => {
                log::warn!("Setting team configuration is not supported mid run");
            }
            ControlMsg::SwapTeamColors => {
                self.team_controllers.swap_teams();
            }
            ControlMsg::SwapTeamSides => {
                self.team_controllers.side_assignment = match self.team_controllers.side_assignment
                {
                    SideAssignment::BlueOnPositive => SideAssignment::YellowOnPositive,
                    SideAssignment::YellowOnPositive => SideAssignment::BlueOnPositive,
                };
                self.tracker
                    .set_side_assignment(self.team_controllers.side_assignment);
                if let Some(h) = &mut self.strategy_host {
                    h.set_side_assignment(self.team_controllers.side_assignment);
                }
            }
            ControlMsg::StartScenario { path, team } => {
                self.handle_start_scenario(path, team);
            }
            ControlMsg::StopScenario => {
                self.handle_stop_scenario();
            }
            ControlMsg::Stop => {}

            ControlMsg::GcCommand { .. }
            | ControlMsg::SimulatorCmd(_)
            | ControlMsg::SetActiveTeams { .. } => {
                unreachable!();
            }
        }
    }

    fn current_test_env(&self) -> TestEnv {
        match self.environment.as_ref() {
            Some(Environment::Simulation { .. }) => TestEnv::Sim,
            Some(Environment::Live { .. }) => TestEnv::Real,
            None => TestEnv::Either,
        }
    }

    fn handle_start_scenario(&mut self, path: PathBuf, team_hint: Option<TeamColor>) {
        // Stop any running scenario first; replacing in-place mid-tick is messy.
        if self.test_driver.is_some() {
            self.handle_stop_scenario();
        }
        let env = self.current_test_env();
        // Default starting team if scenario meta doesn't override it.
        let initial_team = team_hint
            .or_else(|| self.team_controllers.active_teams().first().copied())
            .unwrap_or(TeamColor::Blue);

        let mut driver = match TestDriver::new(initial_team, env, self.log_bus.clone()) {
            Ok(d) => d,
            Err(e) => {
                log::error!("scenario driver creation failed: {:?}", e);
                let _ = self.scenario_status_tx.send(TestStatus::Failed {
                    error: e.to_string(),
                });
                return;
            }
        };
        let _ = self.scenario_status_tx.send(TestStatus::Starting);
        match driver.load_and_start(&path) {
            Ok(meta) => {
                log::info!("scenario started: {} (team={:?})", meta.name, meta.team);
                self.test_driver = Some(driver);
                // Disable game-state compliance for the team being driven; scenarios
                // own safety. Re-enabled in handle_stop_scenario.
                self.set_active_team_comply(meta.team, false);
                let _ = self
                    .scenario_status_tx
                    .send(TestStatus::Running { name: meta.name });
            }
            Err(e) => {
                log::error!("scenario start failed: {:?}", e);
                let _ = self.scenario_status_tx.send(TestStatus::Failed {
                    error: e.to_string(),
                });
            }
        }
    }

    fn handle_stop_scenario(&mut self) {
        if let Some(mut driver) = self.test_driver.take() {
            driver.abort();
            // Re-enable compliance on whichever team it had been driving.
            let team = driver.team_color();
            self.set_active_team_comply(team, true);
            self.test_manual.clear();
            self.pending_sim_cmds.clear();
            log::info!("scenario stopped");
            let _ = self.scenario_status_tx.send(TestStatus::Aborted);
        }
    }

    fn set_active_team_comply(&mut self, team: TeamColor, enabled: bool) {
        let controller = match team {
            TeamColor::Blue => self.team_controllers.blue_team.as_mut(),
            TeamColor::Yellow => self.team_controllers.yellow_team.as_mut(),
        };
        if let Some(c) = controller {
            c.comply_enabled = enabled;
        }
    }

    fn update_team_controller(&mut self) {
        let world_data = self.tracker.get();

        let update = WorldUpdate {
            world_data: world_data.clone(),
        };
        if let Err(err) = self.update_tx.send(update) {
            log::error!("Failed to broadcast world update: {}", err);
        }

        let blue_team_data = if self.team_controllers.blue_team.is_some() {
            Some(world_data.get_team_data(TeamColor::Blue))
        } else {
            None
        };
        let yellow_team_data = if self.team_controllers.yellow_team.is_some() {
            Some(world_data.get_team_data(TeamColor::Yellow))
        } else {
            None
        };

        if let Some(driver) = self.test_driver.as_mut() {
            // Scenario takes over for its declared team. The other team (if any)
            // continues to run its strategy as usual — strategies coexist with scenarios.
            let driver_team = driver.team_color();
            let team_data_for_driver = match driver_team {
                TeamColor::Blue => blue_team_data.as_ref(),
                TeamColor::Yellow => yellow_team_data.as_ref(),
            };
            let active_controller = match driver_team {
                TeamColor::Blue => self.team_controllers.blue_team.as_ref(),
                TeamColor::Yellow => self.team_controllers.yellow_team.as_ref(),
            };
            if let Some(ctrl) = active_controller {
                driver.set_skill_statuses(ctrl.get_skill_statuses());
                driver.set_actual_cmds_global(ctrl.target_velocities_global());
            }

            let mut applied = false;
            if let Some(td) = team_data_for_driver {
                let frame = driver.tick(td);
                if let Some(controller) = match driver_team {
                    TeamColor::Blue => self.team_controllers.blue_team.as_mut(),
                    TeamColor::Yellow => self.team_controllers.yellow_team.as_mut(),
                } {
                    controller.set_strategy_input(StrategyInput {
                        skill_commands: frame.skill_commands,
                        player_roles: frame.player_roles,
                    });
                }
                self.test_manual.clear();
                for (pid, slot) in frame.direct_inputs {
                    self.test_manual
                        .insert((driver_team, pid), slot_to_control_input(&slot));
                }
                self.pending_sim_cmds.extend(frame.sim_commands);
                applied = true;
            }

            // Strategy host still drives the *other* team if one is configured.
            let other_team = match driver_team {
                TeamColor::Blue => TeamColor::Yellow,
                TeamColor::Yellow => TeamColor::Blue,
            };
            if let Some(strategy_host) = self.strategy_host.as_mut() {
                let other_team_data = match other_team {
                    TeamColor::Blue => blue_team_data.as_ref(),
                    TeamColor::Yellow => yellow_team_data.as_ref(),
                };
                if other_team_data.is_some() {
                    if let Some(controller) = match other_team {
                        TeamColor::Blue => self.team_controllers.blue_team.as_ref(),
                        TeamColor::Yellow => self.team_controllers.yellow_team.as_ref(),
                    } {
                        strategy_host
                            .update_skill_statuses(other_team, controller.get_skill_statuses());
                    }
                    let blue_arg = match other_team {
                        TeamColor::Blue => other_team_data,
                        TeamColor::Yellow => None,
                    };
                    let yellow_arg = match other_team {
                        TeamColor::Yellow => other_team_data,
                        TeamColor::Blue => None,
                    };
                    let frame_output = strategy_host.update(blue_arg, yellow_arg);
                    if let Some(controller) = self.team_controllers.blue_team.as_mut() {
                        if other_team == TeamColor::Blue {
                            controller.set_strategy_input(StrategyInput {
                                skill_commands: frame_output.blue_commands,
                                player_roles: frame_output.blue_roles,
                            });
                        }
                    }
                    if let Some(controller) = self.team_controllers.yellow_team.as_mut() {
                        if other_team == TeamColor::Yellow {
                            controller.set_strategy_input(StrategyInput {
                                skill_commands: frame_output.yellow_commands,
                                player_roles: frame_output.yellow_roles,
                            });
                        }
                    }
                }
            }

            // Detect terminal state and broadcast it. The driver itself transitions
            // to Completed/Failed inside `tick`.
            if applied {
                let status = driver.status();
                if !matches!(
                    status,
                    TestStatus::Idle | TestStatus::Starting | TestStatus::Running { .. }
                ) {
                    let _ = self.scenario_status_tx.send(status);
                    // Scenario has ended on its own — clean up.
                    let team = driver.team_color();
                    self.test_driver = None;
                    self.set_active_team_comply(team, true);
                    self.test_manual.clear();
                }
            }
        } else if let Some(strategy_host) = self.strategy_host.as_mut() {
            if let Some(ref controller) = self.team_controllers.blue_team {
                let statuses = controller.get_skill_statuses();
                strategy_host.update_skill_statuses(TeamColor::Blue, statuses);
            }
            if let Some(ref controller) = self.team_controllers.yellow_team {
                let statuses = controller.get_skill_statuses();
                strategy_host.update_skill_statuses(TeamColor::Yellow, statuses);
            }

            let frame_output =
                strategy_host.update(blue_team_data.as_ref(), yellow_team_data.as_ref());

            if let Some(ref mut controller) = self.team_controllers.blue_team {
                controller.set_strategy_input(StrategyInput {
                    skill_commands: frame_output.blue_commands,
                    player_roles: frame_output.blue_roles,
                });
            }
            if let Some(ref mut controller) = self.team_controllers.yellow_team {
                controller.set_strategy_input(StrategyInput {
                    skill_commands: frame_output.yellow_commands,
                    player_roles: frame_output.yellow_roles,
                });
            }
            self.test_manual.clear();
        }

        // Merge real manual_override with test_manual, test has priority for same key.
        let mut manual_override: HashMap<(TeamColor, PlayerId), PlayerControlInput> = self
            .manual_override
            .iter_mut()
            .map(|(id, s)| (*id, s.advance()))
            .collect();
        for (k, v) in &self.test_manual {
            manual_override.insert(*k, v.clone());
        }
        self.team_controllers.update(&world_data, manual_override);
    }
}
