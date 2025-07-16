use std::{collections::HashMap, time::Duration};

use anyhow::Result;
use dies_basestation_client::BasestationHandle;
use dies_core::{
    ExecutorInfo, ExecutorSettings, PlayerCmd, PlayerFeedbackMsg, PlayerId, PlayerOverrideCommand,
    ScriptError, SideAssignment, SimulatorCmd, TeamColor, TeamPlayerId, Vector3, WorldInstant,
    WorldUpdate,
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
pub mod control;
mod gc_client;
mod handle;
pub mod skills;

pub use control::*;
use control::{TeamController, Velocity};

pub use crate::behavior_tree::Strategy;

// Export behavior tree API for native strategies
pub mod behavior_tree_api {
    // Re-export core types
    pub use crate::behavior_tree::{
        Argument, BehaviorTree, BtContext, GameContext, RobotSituation, RoleAssignmentProblem,
        RoleAssignmentSolver, RoleBuilder, Strategy,
    };

    // Re-export node types
    pub use crate::behavior_tree::bt_node::*;

    // Re-export callback types
    pub use crate::behavior_tree::BtCallback;

    // Re-export skills
    pub use crate::skills::{Skill, SkillCtx, SkillProgress, SkillResult};

    // Re-export role assignment types
    pub use crate::behavior_tree::role_assignment::Role;
}

const SIMULATION_DT: Duration = Duration::from_micros(1_000_000 / 60); // 60 Hz
const CMD_INTERVAL: Duration = Duration::from_micros(1_000_000 / 20); // 20 Hz

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
    blue_script_path: Option<String>,
    yellow_script_path: Option<String>,
}

impl TeamMap {
    fn new(side_assignment: SideAssignment) -> Self {
        Self {
            blue_team: None,
            yellow_team: None,
            side_assignment,
            blue_script_path: None,
            yellow_script_path: None,
        }
    }

    fn activate_team(
        &mut self,
        team_color: TeamColor,
        settings: &ExecutorSettings,
        strategy: Strategy,
    ) {
        if team_color == TeamColor::Blue {
            if self.blue_team.is_none() {
                self.blue_team = Some(TeamController::new(settings, TeamColor::Blue, strategy));
            }
        } else {
            if self.yellow_team.is_none() {
                self.yellow_team = Some(TeamController::new(settings, TeamColor::Yellow, strategy));
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

    /// Get and clear script errors from all active team controllers
    fn take_script_errors(&mut self) -> Vec<ScriptError> {
        let mut errors = Vec::new();

        if let Some(controller) = &mut self.blue_team {
            errors.extend(controller.take_script_errors());
        }

        if let Some(controller) = &mut self.yellow_team {
            errors.extend(controller.take_script_errors());
        }

        errors
    }
}

struct PlayerOverrideState {
    frame_counter: u32,
    current_command: PlayerOverrideCommand,
}

impl PlayerOverrideState {
    const VELOCITY_TIMEOUT: u32 = 30;

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
    script_error_tx: broadcast::Sender<ScriptError>,
    settings: ExecutorSettings,
}

impl Executor {
    pub fn new_live(
        settings: ExecutorSettings,
        ssl_client: VisionClient,
        bs_client: BasestationHandle,
        strategy: Strategy,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (update_tx, _) = broadcast::channel(16);
        let (paused_tx, _) = watch::channel(false);
        let (info_channel_tx, info_channel_rx) = mpsc::unbounded_channel();
        let (script_error_tx, _) = broadcast::channel(16);

        // Use team configuration from settings
        let mut team_controllers = TeamMap::new(settings.team_configuration.side_assignment);

        // Activate teams based on configuration
        let mut controlled_teams = Vec::new();
        if settings.team_configuration.blue_active {
            team_controllers.activate_team(TeamColor::Blue, &settings, strategy);
            controlled_teams.push(TeamColor::Blue);
        }
        if settings.team_configuration.yellow_active {
            team_controllers.activate_team(TeamColor::Yellow, &settings, strategy);
            controlled_teams.push(TeamColor::Yellow);
        }

        Self {
            tracker: WorldTracker::new(&settings, &controlled_teams, settings.allow_no_vision),
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
            script_error_tx,
            settings,
        }
    }

    pub fn new_simulation(
        settings: ExecutorSettings,
        mut simulator: Simulation,
        strategy: Strategy,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (update_tx, _) = broadcast::channel(16);
        let (paused_tx, _) = watch::channel(false);
        let (info_channel_tx, info_channel_rx) = mpsc::unbounded_channel();
        let (script_error_tx, _) = broadcast::channel(16);

        // Use team configuration from settings
        let mut team_controllers = TeamMap::new(settings.team_configuration.side_assignment);

        // Activate teams based on configuration
        let mut controlled_teams = Vec::new();
        if settings.team_configuration.blue_active {
            team_controllers.activate_team(TeamColor::Blue, &settings, strategy);
            controlled_teams.push(TeamColor::Blue);
        }
        if settings.team_configuration.yellow_active {
            team_controllers.activate_team(TeamColor::Yellow, &settings, strategy);
            controlled_teams.push(TeamColor::Yellow);
        }

        simulator.set_controlled_teams(&controlled_teams);

        Self {
            tracker: WorldTracker::new(&settings, &controlled_teams, settings.allow_no_vision),
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
            script_error_tx,
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
        let mut last_ssl_msg = tokio::time::Instant::now();
        let mut last_cmd_time = tokio::time::Instant::now();
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
                    last_ssl_msg = tokio::time::Instant::now();
                    match ssl_msg {
                        Ok(SslMessage::Vision(vision_msg)) => {
                            self.update_from_vision_msg(vision_msg, WorldInstant::now_real());
                        }
                        Ok(SslMessage::Referee(gc_msg)) => {
                            self.update_from_gc_msg(gc_msg);
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
                            let team_color = team_color.or(self.team_controllers.active_teams().get(0).copied());
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
                    dies_core::debug_value("exec_cmd msg_elapsed", last_cmd_time.elapsed().as_secs_f64() * 1000.0);
                    let paused = { *self.paused_tx.borrow() };
                    if !paused {
                        for (team_color, cmd) in self.player_commands() {
                            bs_client.send_no_wait(team_color, cmd);
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
            script_error_rx: self.script_error_tx.subscribe(),
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
            }
            ControlMsg::SetTeamScriptPaths { .. } => {
                log::warn!("Setting team script paths is not supported mid run");
            }
            ControlMsg::SetTeamConfiguration(_) => {
                log::warn!("Setting team configuration is not supported mid run");
                // Apply complete team configuration
                // self.team_controllers.side_assignment = config.side_assignment;
                // self.tracker.set_side_assignment(config.side_assignment);

                // // Update active teams
                // let mut controlled_teams = Vec::new();

                // if config.blue_active {
                //     if !self.team_controllers.blue_team.is_some() {
                //         self.team_controllers
                //             .activate_team(TeamColor::Blue, &self.settings);
                //     }
                //     controlled_teams.push(TeamColor::Blue);
                // } else {
                //     self.team_controllers.deactivate_team(TeamColor::Blue);
                // }

                // if config.yellow_active {
                //     if !self.team_controllers.yellow_team.is_some() {
                //         self.team_controllers
                //             .activate_team(TeamColor::Yellow, &self.settings);
                //     }
                //     controlled_teams.push(TeamColor::Yellow);
                // } else {
                //     self.team_controllers.deactivate_team(TeamColor::Yellow);
                // }

                // self.tracker.set_controlled_teams(&controlled_teams);
            }
            ControlMsg::SwapTeamColors => {
                self.team_controllers.swap_teams();
            }
            ControlMsg::SwapTeamSides => {
                // Swap side assignment
                self.team_controllers.side_assignment = match self.team_controllers.side_assignment
                {
                    SideAssignment::BlueOnPositive => SideAssignment::YellowOnPositive,
                    SideAssignment::YellowOnPositive => SideAssignment::BlueOnPositive,
                };
                self.tracker
                    .set_side_assignment(self.team_controllers.side_assignment);
            }
            ControlMsg::ScriptError(_) => {
                // Script errors are sent from executor to UI, not handled by executor
                // This should not happen in normal operation
                log::warn!(
                    "Received ScriptError control message, but these should only be sent to UI"
                );
            }
            ControlMsg::Stop => {}

            ControlMsg::GcCommand { .. }
            | ControlMsg::SimulatorCmd(_)
            | ControlMsg::SetActiveTeams { .. } => {
                unreachable!();
            }
        }
    }

    fn update_team_controller(&mut self) {
        let world_data = self.tracker.get();

        let update = WorldUpdate {
            world_data: world_data.clone(),
        };
        dies_core::debug_string(
            "update_team_controller",
            format!("update_tx: {:?}", update.world_data.t_received),
        );
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
