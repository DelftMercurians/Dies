use std::time::Duration;

use anyhow::{bail, Result};

use dies_core::{PlayerCmd, PlayerFeedbackMsg, WorldUpdate};
use dies_protos::{ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket};
use dies_serial_client::SerialClient;
use dies_simulator::Simulation;
use dies_ssl_client::VisionClient;
use dies_world::{WorldConfig, WorldTracker};
use gc_client::GcClient;
use handle::{ControlMsg, ExecutorHandle};
use strategy::Strategy;
use tokio::sync::{
    broadcast::{self},
    mpsc, watch,
};

mod control;
mod gc_client;
mod handle;
pub mod strategy;

use control::TeamController;
pub use control::{KickerControlInput, PlayerControlInput, PlayerInputs};

const CMD_INTERVAL: Duration = Duration::from_millis(1000 / 30);

#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    pub world_config: WorldConfig,
}

enum Environment {
    Live {
        ssl_client: VisionClient,
        bs_client: SerialClient,
    },
    Simulation {
        simulator: Simulation,
        dt: Duration,
    },
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
    update_tx: broadcast::Sender<WorldUpdate>,
    command_tx: mpsc::UnboundedSender<ControlMsg>,
    command_rx: mpsc::UnboundedReceiver<ControlMsg>,
    paused_tx: watch::Sender<bool>,
}

impl Executor {
    pub fn new_live(
        config: ExecutorConfig,
        strategy: Box<dyn Strategy>,
        ssl_client: VisionClient,
        bs_client: SerialClient,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (update_tx, _) = broadcast::channel(16);
        let (paused_tx, _) = watch::channel(false);

        Self {
            tracker: WorldTracker::new(config.world_config),
            controller: TeamController::new(strategy),
            gc_client: GcClient::new(),
            environment: Some(Environment::Live {
                ssl_client,
                bs_client,
            }),
            command_tx,
            command_rx,
            update_tx,
            paused_tx,
        }
    }

    pub fn new_simulation(
        config: ExecutorConfig,
        strategy: Box<dyn Strategy>,
        simulator: Simulation,
        dt: Duration,
    ) -> Self {
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (update_tx, _) = broadcast::channel(16);
        let (paused_tx, _) = watch::channel(false);

        Self {
            tracker: WorldTracker::new(config.world_config),
            controller: TeamController::new(strategy),
            gc_client: GcClient::new(),
            environment: Some(Environment::Simulation { simulator, dt }),
            command_tx,
            command_rx,
            update_tx,
            paused_tx,
        }
    }

    /// Update the executor with a vision message.
    pub fn update_from_vision_msg(&mut self, message: SSL_WrapperPacket) {
        self.tracker.update_from_vision(&message);
        self.update_team_controller();
    }

    /// Update the executor with a referee message.
    pub fn update_from_gc_msg(&mut self, message: Referee) {
        self.tracker.update_from_referee(&message);
        self.update_team_controller();
    }

    /// Update the executor with a feedback message from the robots.
    #[allow(dead_code)]
    pub fn update_from_bs_msg(&mut self, _message: PlayerFeedbackMsg) {
        todo!()
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

    /// Run the executor in real time, either with real clients or with a simulator,
    /// depending on the configuration.
    pub async fn run_real_time(mut self) -> Result<()> {
        match self.environment.take() {
            Some(Environment::Live {
                ssl_client,
                bs_client,
            }) => self.run_rt_live(ssl_client, bs_client).await?,
            Some(Environment::Simulation { simulator, dt }) => {
                self.run_rt_sim(simulator, dt).await?
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    pub fn step_simulation(&mut self) -> Result<()> {
        let packet = if let Some(Environment::Simulation { simulator, dt }) = &mut self.environment
        {
            simulator.step(dt.as_secs_f64());
            simulator.detection().or(simulator.geometry())
        } else {
            bail!("Simulator not set");
        };

        packet.map(|p| self.update_from_vision_msg(p));

        Ok(())
    }

    /// Run the executor in real time on the simulator
    async fn run_rt_sim(&mut self, mut simulator: Simulation, dt: Duration) -> Result<()> {
        let mut update_interval = tokio::time::interval(dt);
        let mut cmd_interval = tokio::time::interval(CMD_INTERVAL);
        let mut paused_rx = self.paused_tx.subscribe();
        loop {
            let paused = { *paused_rx.borrow_and_update() };
            if paused {
                tokio::select! {
                    Some(msg) = self.command_rx.recv() => {
                        match msg {
                            ControlMsg::Stop => break,
                            msg => self.handle_control_msg(msg)
                        }
                    }
                    _ = update_interval.tick() => {
                        self.step_simulation()?;
                    }
                    _ = cmd_interval.tick() => {
                        for cmd in self.player_commands() {
                            simulator.push_cmd(cmd);
                        }
                    }
                }
            } else {
                paused_rx.changed().await?;
            }
        }

        Ok(())
    }

    /// Run the executor in real time, processing vision and referee messages and sending
    /// commands to the robots.
    async fn run_rt_live(
        &mut self,
        mut ssl_client: VisionClient,
        mut bs_client: SerialClient,
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
                vision_msg = ssl_client.recv() => {
                    match vision_msg {
                        Ok(vision_msg) => {
                            self.update_from_vision_msg(vision_msg);
                        }
                        Err(err) => {
                            log::error!("Failed to receive vision msg: {}", err);
                        }
                    }
                }
                // bs_msg = bs_client.recv() => {
                //     match bs_msg {
                //         Ok(bs_msg) => {
                //             self.update_from_bs_msg(bs_msg);
                //         }
                //         Err(err) => {
                //             log::error!("Failed to receive BS msg: {}", err);
                //         }
                //     }
                // }
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

    pub fn handle(&self) -> ExecutorHandle {
        ExecutorHandle {
            control_tx: self.command_tx.clone(),
            update_rx: self.update_tx.subscribe(),
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
            } => todo!(),
            ControlMsg::Stop => {}
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

        self.controller.update(world_data);
    }
}
