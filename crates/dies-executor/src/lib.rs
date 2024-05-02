use std::time::Duration;

use anyhow::{bail, Result};

use dies_control::TeamController;
use dies_core::{PlayerCmd, PlayerFeedbackMsg, WorldData};
use dies_protos::{ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket};
use dies_serial_client::SerialClient;
use dies_simulator::Simulation;
use dies_ssl_client::VisionClient;
use dies_world::{WorldConfig, WorldTracker};
use gc_client::GcClient;
use tokio::sync::{broadcast, watch};

mod gc_client;

#[derive(Debug, Clone)]
pub struct WorldUpdate {
    pub world_data: WorldData,
}

/// The central component of the framework. It contains all state and logic needed to
/// run a match -- processing vision and referee messages, executing the strategy, and
/// sending commands to the robots.
///
/// The executor can be used in 3 different regimes: externally driven, automatic, and
/// simulation.
///
/// To construct an executor, use the [`Executor::builder`] method.
pub struct Executor {
    tracker: WorldTracker,
    team_controller: TeamController,
    gc_client: GcClient,
    ssl_client: Option<VisionClient>,
    bs_client: Option<SerialClient>,
    simulator: Option<Simulation>,
    simulator_dt: Duration,
    update_broadcast: watch::Sender<WorldUpdate>,
}

impl Executor {
    /// Start constructing an executor with the [`ExecutorBuilder`].
    pub fn builder() -> ExecutorBuilder {
        ExecutorBuilder::default()
    }

    /// Set the play direction for the executor. This is used to determine which side of
    /// the field the team is playing on.
    pub fn set_play_dir_x(&mut self, opp_x_sign: f32) {
        self.tracker.set_play_dir_x(opp_x_sign);
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
    pub fn player_commands(&self) -> Vec<PlayerCmd> {
        self.team_controller.commands()
    }

    /// Get the GC messages that need to be sent. This will remove the messages from the
    /// internal queue.
    pub fn gc_messages(&mut self) -> Vec<Referee> {
        self.gc_client.messages()
    }

    /// Run the executor in real time, either with real clients or with a simulator,
    /// depending on the configuration.
    pub async fn run_real_time(mut self, stop_rx: broadcast::Receiver<()>) -> Result<()> {
        if self.ssl_client.is_some() && self.bs_client.is_some() {
            self.run_rt_irl(stop_rx).await?;
        } else if self.simulator.is_some() {
            self.run_rt_sim(stop_rx).await?;
        } else {
            bail!("No clients or simulator set");
        }

        Ok(())
    }

    pub fn step_simulation(&mut self) -> Result<()> {
        if let Some(_simulator) = &mut self.simulator {
            todo!()
        } else {
            bail!("Simulator not set");
        }
    }

    /// Run the executor in real time on the simulator
    async fn run_rt_sim(&mut self, stop_rx: broadcast::Receiver<()>) -> Result<()> {
        // Check that we have a simulator
        if self.simulator.is_none() {
            bail!("Simulator not set");
        }

        let mut interval = tokio::time::interval(self.simulator_dt);
        tokio::pin!(stop_rx);
        loop {
            tokio::select! {
                _ = stop_rx.recv() => {
                    break;
                }
                _ = interval.tick() => {
                    self.step_simulation()?;
                }
            }
        }
        Ok(())
    }

    /// Run the executor in real time, processing vision and referee messages and sending
    /// commands to the robots.
    async fn run_rt_irl(&mut self, stop_rx: broadcast::Receiver<()>) -> Result<()> {
        // Check that we have ssl and bs clients
        let mut ssl_client = self
            .ssl_client
            .take()
            .ok_or(anyhow::anyhow!("SSL client not set"))?;
        let mut bs_client = self
            .bs_client
            .take()
            .ok_or(anyhow::anyhow!("BS client not set"))?;

        tokio::pin!(stop_rx);
        loop {
            tokio::select! {
                _ = stop_rx.recv() => {
                    break;
                }
                vision_msg = ssl_client.recv() => {
                    match vision_msg {
                        Ok(vision_msg) => {
                            self.update_from_vision_msg(vision_msg);
                        }
                        Err(err) => {
                            tracing::error!("Failed to receive vision msg: {}", err);
                        }
                    }
                }
                bs_msg = bs_client.recv() => {
                    match bs_msg {
                        Ok(bs_msg) => {
                            self.update_from_bs_msg(bs_msg);
                        }
                        Err(err) => {
                            tracing::error!("Failed to receive BS msg: {}", err);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn subscribe(&self) -> watch::Receiver<WorldUpdate> {
        self.update_broadcast.subscribe()
    }

    fn update_team_controller(&mut self) {
        let world_data = self.tracker.get().unwrap();
        self.team_controller.update(world_data);
    }
}

/// A builder for the [`Executor`]. Valid combinations of fields are:
///
/// - `ssl_client` and `bs_client` for real-time execution
/// - `simulator` for simulation
///
/// The `world_config` field is required in all cases.
#[derive(Default)]
pub struct ExecutorBuilder {
    ssl_client: Option<VisionClient>,
    bs_client: Option<SerialClient>,
    simulator: Option<Simulation>,
    world_config: Option<WorldConfig>,
    simulator_dt: Duration,
}

impl ExecutorBuilder {
    pub fn with_ssl_client(&mut self, ssl_client: VisionClient) -> &mut Self {
        self.ssl_client = Some(ssl_client);
        self
    }

    pub fn with_bs_client(&mut self, bs_client: SerialClient) -> &mut Self {
        self.bs_client = Some(bs_client);
        self
    }

    pub fn with_simulator(&mut self, simulator: Simulation, dt: Duration) -> &mut Self {
        self.simulator = Some(simulator);
        self.simulator_dt = dt;
        self
    }

    pub fn with_world_config(&mut self, world_config: WorldConfig) -> &mut Self {
        self.world_config = Some(world_config);
        self
    }

    /// Build the executor from the current configuration.
    ///
    /// # Errors
    ///
    /// This function will return an error if the world config is not set, or if neither
    /// the simulator nor the ssl_client and bs_client are set.
    pub fn build(self) -> Result<Executor> {
        // Ensure either simulator or ssl_client and bs_client is set
        if self.simulator.is_none() && (self.ssl_client.is_none() || self.bs_client.is_none()) {
            bail!("Either simulator or ssl_client and bs_client must be set");
        }

        Ok(Executor {
            tracker: WorldTracker::new(
                self.world_config
                    .ok_or(anyhow::anyhow!("World config not set"))?,
            ),
            team_controller: TeamController::new(1.0),
            gc_client: GcClient::new(),
            ssl_client: self.ssl_client,
            bs_client: self.bs_client,
            simulator: self.simulator,
            update_broadcast: watch::channel(WorldUpdate {
                world_data: WorldData::default(),
            })
            .0,
            simulator_dt: self.simulator_dt,
        })
    }
}
