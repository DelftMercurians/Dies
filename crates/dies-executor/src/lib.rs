use std::{collections::HashMap, time::Duration};

use anyhow::Result;
use dies_basestation_client::BasestationHandle;
use dies_core::{
    ExecutorInfo, ExecutorSettings, GameState, PlayerCmd, PlayerFeedbackMsg, PlayerId,
    PlayerMoveCmd, PlayerOverrideCommand, StrategyGameStateMacther, WorldUpdate,
};
use dies_core::{SimulatorCmd, Vector3, WorldInstant};
use dies_logger::{log_referee, log_vision, log_world};
use dies_protos::{ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket};
use dies_simulator::Simulation;
use dies_ssl_client::{SslMessage, VisionClient};
use dies_world::WorldTracker;
use gc_client::GcClient;
pub use handle::{ControlMsg, ExecutorHandle};
use strategy::Strategy;
use tokio::sync::{broadcast, mpsc, oneshot, watch};

mod control;
mod gc_client;
mod handle;
mod roles;
pub mod scenarios;
pub mod strategy;

pub use control::{KickerControlInput, PlayerControlInput, PlayerInputs};
use control::{TeamController, Velocity};

const SIMULATION_DT: Duration = Duration::from_micros(1_000_000 / 60); // 60 Hz
const CMD_INTERVAL: Duration = Duration::from_micros(1_000_000 / 30); // 30 Hz

pub struct StrategyMap(Vec<(StrategyGameStateMacther, Box<dyn Strategy>)>);

impl StrategyMap {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn insert(&mut self, matcher: StrategyGameStateMacther, strategy: Box<dyn Strategy>) {
        self.0.push((matcher, strategy));
    }

    pub fn get_strategy(&mut self, state: &GameState) -> Option<&mut dyn Strategy> {
        if let Some((_, strategy)) = self
            .0
            .iter_mut()
            .find(|(matcher, _)| matcher.matches(state))
        {
            Some(strategy.as_mut())
        } else {
            None
        }
    }
}

enum Environment {
    Live {
        ssl_client: VisionClient,
        bs_client: BasestationHandle,
    },
    Simulation {
        simulator: Simulation,
    },
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
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::Kick { .. } => PlayerControlInput {
                kicker: KickerControlInput::Kick,
                ..Default::default()
            },
            PlayerOverrideCommand::DischargeKicker => PlayerControlInput {
                kicker: KickerControlInput::Disarm,
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
/// simulation.
pub struct Executor {
    tracker: WorldTracker,
    controller: TeamController,
    gc_client: GcClient,
    environment: Option<Environment>,
    manual_override: HashMap<PlayerId, PlayerOverrideState>,
    update_tx: broadcast::Sender<WorldUpdate>,
    command_tx: mpsc::UnboundedSender<ControlMsg>,
    command_rx: mpsc::UnboundedReceiver<ControlMsg>,
    paused_tx: watch::Sender<bool>,
    info_channel_rx: mpsc::UnboundedReceiver<oneshot::Sender<ExecutorInfo>>,
    info_channel_tx: mpsc::UnboundedSender<oneshot::Sender<ExecutorInfo>>,
}

impl Executor {
    pub fn new_live(
        settings: ExecutorSettings,
        strategy: StrategyMap,
        ssl_client: VisionClient,
        bs_client: BasestationHandle,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (update_tx, _) = broadcast::channel(16);
        let (paused_tx, _) = watch::channel(false);
        let (info_channel_tx, info_channel_rx) = mpsc::unbounded_channel();

        Self {
            tracker: WorldTracker::new(&settings),
            controller: TeamController::new(strategy, &settings),
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
        }
    }

    pub fn new_simulation(
        settings: ExecutorSettings,
        strategy: StrategyMap,
        simulator: Simulation,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (update_tx, _) = broadcast::channel(16);
        let (paused_tx, _) = watch::channel(false);
        let (info_channel_tx, info_channel_rx) = mpsc::unbounded_channel();

        Self {
            tracker: WorldTracker::new(&settings),
            controller: TeamController::new(strategy, &settings),
            gc_client: GcClient::new(),
            environment: Some(Environment::Simulation { simulator }),
            manual_override: HashMap::new(),
            command_tx,
            command_rx,
            update_tx,
            paused_tx,
            info_channel_rx,
            info_channel_tx,
        }
    }

    pub fn set_opp_goal_sign(&mut self, opp_goal_sign: f64) {
        self.tracker.set_play_dir_x(opp_goal_sign);
        self.controller.set_opp_goal_sign(opp_goal_sign);
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
    pub fn update_from_bs_msg(&mut self, message: PlayerFeedbackMsg, time: WorldInstant) {
        self.tracker.update_from_feedback(&message, time);
    }

    /// Get the currently active player commands.
    pub fn player_commands(&mut self) -> Vec<PlayerCmd> {
        self.controller.commands()
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
            manual_controlled_players: self.manual_override.keys().copied().collect(),
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
        if let Some(feedback) = simulator.feedback() {
            self.update_from_bs_msg(feedback, simulator.time());
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
                        for cmd in self.player_commands() {
                            if let PlayerCmd::Move(cmd) = cmd {
                                simulator.push_cmd(cmd);
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
                            log::error!("Failed to receive vision msg: {}", err);
                        }
                    }
                }
                Some(tx) = self.info_channel_rx.recv() => {
                    let _  = tx.send(self.info());
                }
                bs_msg = bs_client.recv() => {
                    match bs_msg {
                        Ok(bs_msg) => {
                            self.update_from_bs_msg(bs_msg, WorldInstant::now_real());
                        }
                        Err(err) => {
                            log::error!("Failed to receive BS msg: {}", err);
                        }
                    }
                }
                _ = cmd_interval.tick() => {
                    let paused = { *self.paused_tx.borrow() };
                    if !paused {
                        for cmd in self.player_commands() {
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
                player_id,
                override_active,
            } => {
                if override_active {
                    self.manual_override
                        .insert(player_id, PlayerOverrideState::new());
                } else {
                    self.manual_override.remove(&player_id);
                }
            }
            ControlMsg::PlayerOverrideCommand(player_id, cmd) => {
                if let Some(state) = self.manual_override.get_mut(&player_id) {
                    state.set_cmd(cmd);
                }
            }
            ControlMsg::UpdateSettings(settings) => {
                self.controller.update_controller_settings(&settings);
                self.tracker.update_settings(&settings);
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
        self.controller.update(world_data, manual_override);
    }
}
