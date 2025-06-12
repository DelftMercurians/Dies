use std::{collections::HashMap, time::Duration};

use anyhow::Result;
use dies_basestation_client::BasestationHandle;
use dies_core::{
    ExecutorInfo, ExecutorSettings, PlayerCmd, PlayerFeedbackMsg, PlayerId, PlayerOverrideCommand,
    SideAssignment, SimulatorCmd, TeamColor, TeamPlayerId, Vector3, WorldInstant, WorldUpdate,
};
use dies_logger::{log_referee, log_vision, log_world};
use dies_protos::{ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket};
use dies_simulator::Simulation;
use dies_ssl_client::{SslMessage, VisionClient};
use dies_world::WorldTracker;
use gc_client::GcClient;
pub use handle::{ControlMsg, ExecutorHandle};
use tokio::sync::{broadcast, mpsc, oneshot, watch};

mod behavior_tree;
mod control;
mod gc_client;
mod handle;
mod roles;
pub mod scenarios;

pub use control::{KickerControlInput, PlayerControlInput, PlayerInputs};
use control::{TeamController, Velocity};

const SIMULATION_DT: Duration = Duration::from_micros(1_000_000 / 60); // 60 Hz
const CMD_INTERVAL: Duration = Duration::from_micros(1_000_000 / 30); // 30 Hz

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
                self.blue_team = Some(TeamController::new(team_color, settings));
            }
        } else {
            if self.yellow_team.is_none() {
                self.yellow_team = Some(TeamController::new(team_color, settings));
            }
        }
    }

    fn deactivate_team(&mut self, team_color: TeamColor) {
        log::info!("deactivate_team called with team_color: {}", team_color);
        if team_color == TeamColor::Blue {
            log::info!(
                "Deactivating blue team controller for team_color: {}",
                team_color
            );
            self.blue_team = None;
        } else if team_color == TeamColor::Yellow {
            log::info!(
                "Deactivating yellow team controller for team_color: {}",
                team_color
            );
            self.yellow_team = None;
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

    /// Get all player commands from active controllers
    fn get_all_commands(&mut self) -> Vec<(TeamColor, PlayerCmd)> {
        let mut commands = Vec::new();

        if let Some(controller) = &mut self.blue_team {
            commands.extend(
                controller
                    .commands()
                    .iter()
                    .map(|c| {
                        self.side_assignment
                            .untransform_player_cmd(TeamColor::Blue, c)
                    })
                    .map(|c| (TeamColor::Blue, c)),
            );
        }
        if let Some(controller) = &mut self.yellow_team {
            commands.extend(
                controller
                    .commands()
                    .iter()
                    .map(|c| {
                        self.side_assignment
                            .untransform_player_cmd(TeamColor::Yellow, c)
                    })
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
                team_data,
                manual_override
                    .iter()
                    .filter_map(|(color, input)| {
                        if color.0 == TeamColor::Blue {
                            Some((color.1, input.clone()))
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
                team_data,
                manual_override
                    .iter()
                    .filter_map(|(color, input)| {
                        if color.0 == TeamColor::Yellow {
                            Some((color.1, input.clone()))
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
    const VELOCITY_TIMEOUT: u32 = 5;

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
                angular_velocity,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                velocity: Velocity::local(velocity),
                angular_velocity: Some(angular_velocity),
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
                angular_velocity,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                velocity: Velocity::global(velocity),
                angular_velocity: Some(angular_velocity),
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::Kick { speed } => PlayerControlInput {
                kicker: KickerControlInput::Kick,
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
pub struct Executor {
    tracker: WorldTracker,
    team_controllers: TeamMap,
    gc_client: GcClient,
    environment: Option<Environment>,
    manual_override: HashMap<(TeamColor, PlayerId), PlayerOverrideState>,
    update_tx: broadcast::Sender<WorldUpdate>,
    command_tx: mpsc::UnboundedSender<ControlMsg>,
    command_rx: mpsc::UnboundedReceiver<ControlMsg>,
    paused_tx: watch::Sender<bool>,
    info_channel_rx: mpsc::UnboundedReceiver<oneshot::Sender<ExecutorInfo>>,
    info_channel_tx: mpsc::UnboundedSender<oneshot::Sender<ExecutorInfo>>,
    settings: ExecutorSettings,
}

impl Executor {
    pub fn new_live(
        settings: ExecutorSettings,
        ssl_client: VisionClient,
        bs_client: BasestationHandle,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (update_tx, _) = broadcast::channel(16);
        let (paused_tx, _) = watch::channel(false);
        let (info_channel_tx, info_channel_rx) = mpsc::unbounded_channel();

        // Start with blue team active by default for backwards compatibility
        let mut team_controllers = TeamMap::new(SideAssignment::YellowOnPositive);
        team_controllers.activate_team(TeamColor::Blue, &settings);

        let controlled_teams = vec![TeamColor::Blue];
        Self {
            tracker: WorldTracker::new(&settings, &controlled_teams),
            team_controllers,
            gc_client: GcClient::new(),
            environment: Some(Environment::Live {
                ssl_client,
                bs_client,
            }),
            manual_override: HashMap::new(),
            command_tx,
            command_rx,
            update_tx,
            paused_tx,
            info_channel_rx,
            info_channel_tx,
            settings,
        }
    }

    pub fn new_simulation(settings: ExecutorSettings, mut simulator: Simulation) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (update_tx, _) = broadcast::channel(16);
        let (paused_tx, _) = watch::channel(false);
        let (info_channel_tx, info_channel_rx) = mpsc::unbounded_channel();

        // Start with blue team active by default for backwards compatibility
        let mut team_controllers = TeamMap::new(SideAssignment::YellowOnPositive);
        team_controllers.activate_team(TeamColor::Blue, &settings);
        simulator.set_controlled_teams((vec![TeamColor::Blue]).into_iter().collect());

        let controlled_teams = vec![TeamColor::Blue];
        Self {
            tracker: WorldTracker::new(&settings, &controlled_teams),
            team_controllers,
            gc_client: GcClient::new(),
            environment: Some(Environment::Simulation { simulator }),
            manual_override: HashMap::new(),
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
        self.update_team_controller();
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
        if let Some((team_color, feedback)) = simulator.feedback() {
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
                                simulator.update_referee_command(command);
                            }
                            ControlMsg::SimulatorCmd(cmd) => self.handle_simulator_cmd(&mut simulator, cmd),
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
                        for (team_color, cmd) in self.player_commands() {
                            if let PlayerCmd::Move(cmd) = cmd {
                                simulator.push_cmd(team_color, cmd);
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

        Ok(())
    }

    /// Run the executor in real time, processing vision and referee messages and sending
    /// commands to the robots.
    async fn run_rt_live(
        &mut self,
        mut ssl_client: VisionClient,
        mut bs_client: BasestationHandle,
    ) -> Result<()> {
        // Check that we have ssl and bs clients

        let mut cmd_interval = tokio::time::interval(CMD_INTERVAL);

        loop {
            tokio::select! {
                Some(msg) = self.command_rx.recv() => {
                    match msg {
                        ControlMsg::Stop => break,
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
                        Err(err) => {
                            log::error!("Failed to receive vision/gc msg: {}", err);
                        }
                    }
                }
                Some(tx) = self.info_channel_rx.recv() => {
                    let _  = tx.send(self.info());
                }
                bs_msg = bs_client.recv() => {
                    match bs_msg {
                        Ok(bs_msg) => {
                            if let Some(controlled_team_color) = self.team_controllers.active_teams().get(0) {
                                self.update_from_bs_msg(
                                    *controlled_team_color,
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
                    let paused = { *self.paused_tx.borrow() };
                    if !paused {
                        for (_, cmd) in self.player_commands() {
                            bs_client.send_no_wait(cmd);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn handle_simulator_cmd(&self, sim: &mut Simulation, cmd: SimulatorCmd) {
        match cmd {
            SimulatorCmd::ApplyBallForce { force } => {
                sim.apply_force_to_ball(Vector3::new(force.x, force.y, 0.0));
            }
        }
    }

    pub fn handle(&self) -> ExecutorHandle {
        ExecutorHandle {
            control_tx: self.command_tx.clone(),
            update_rx: self.update_tx.subscribe(),
            info_channel: self.info_channel_tx.clone(),
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
            ControlMsg::SetActiveTeams {
                blue_active,
                yellow_active,
            } => {
                log::info!(
                    "SetActiveTeams received: blue_active={}, yellow_active={}",
                    blue_active,
                    yellow_active
                );

                if blue_active {
                    log::info!("Activating blue team ({})", TeamColor::Blue);
                    self.team_controllers
                        .activate_team(TeamColor::Blue, &self.settings);
                } else {
                    log::info!("Deactivating blue team ({})", TeamColor::Blue);
                    self.team_controllers.deactivate_team(TeamColor::Blue);
                }

                if yellow_active {
                    log::info!("Activating yellow team ({})", TeamColor::Yellow);
                    self.team_controllers
                        .activate_team(TeamColor::Yellow, &self.settings);
                } else {
                    log::info!("Deactivating yellow team ({})", TeamColor::Yellow);
                    self.team_controllers.deactivate_team(TeamColor::Yellow);
                }
            }
            ControlMsg::SetSideAssignment(side_assignment) => {
                self.team_controllers.side_assignment = side_assignment;
            }
            ControlMsg::Stop => {}
            ControlMsg::GcCommand { .. } | ControlMsg::SimulatorCmd(_) => {
                // This should be handled by the caller
            }
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

        let manual_override = self
            .manual_override
            .iter_mut()
            .map(|(id, s)| (*id, s.advance()))
            .collect();
        self.team_controllers.update(&world_data, manual_override);
    }
}
