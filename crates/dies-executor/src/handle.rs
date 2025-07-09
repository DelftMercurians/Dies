use std::time::Duration;

use anyhow::{anyhow, Result};
use dies_core::{
    ExecutorInfo, ExecutorSettings, PlayerId, PlayerOverrideCommand, SideAssignment, SimulatorCmd,
    TeamColor, TeamConfiguration, WorldUpdate,
};
use dies_protos::ssl_gc_referee_message::referee::Command;
use tokio::sync::{broadcast, mpsc, oneshot};

#[derive(Debug)]
pub enum ControlMsg {
    SetPlayerOverride {
        team_color: TeamColor,
        player_id: PlayerId,
        override_active: bool,
    },
    PlayerOverrideCommand {
        team_color: TeamColor,
        player_id: PlayerId,
        command: PlayerOverrideCommand,
    },
    SetPause(bool),
    UpdateSettings(ExecutorSettings),
    GcCommand {
        command: Command,
    },
    SimulatorCmd(SimulatorCmd),
    /// Control which teams are active
    SetActiveTeams {
        blue_active: bool,
        yellow_active: bool,
    },
    /// Set script paths for teams
    SetTeamScriptPaths {
        blue_script_path: Option<String>,
        yellow_script_path: Option<String>,
    },
    SetSideAssignment(SideAssignment),
    /// Set complete team configuration (for pre-executor setup)
    SetTeamConfiguration(TeamConfiguration),
    /// Swap team colors (Blue <-> Yellow)
    SwapTeamColors,
    /// Swap team sides (BlueOnPositive <-> YellowOnPositive)
    SwapTeamSides,
    Stop,
}

pub struct ExecutorInfoReceiver(oneshot::Receiver<ExecutorInfo>);

impl ExecutorInfoReceiver {
    /// Wait for the executor info with a timeout of 500ms.
    ///
    /// If the timeout is reached,
    /// `None` is returned. This can happen if the executor is not running or stopped
    /// mid-request.
    pub async fn recv(self) -> Option<ExecutorInfo> {
        match tokio::time::timeout(Duration::from_millis(500), self.0).await {
            Ok(Ok(info)) => Some(info),
            _ => None,
        }
    }
}
#[derive(Debug)]
pub struct ExecutorHandle {
    pub control_tx: mpsc::UnboundedSender<ControlMsg>,
    pub update_rx: broadcast::Receiver<WorldUpdate>,
    pub info_channel: mpsc::UnboundedSender<oneshot::Sender<ExecutorInfo>>,
}

impl ExecutorHandle {
    pub async fn recv(&mut self) -> Option<WorldUpdate> {
        self.update_rx
            .recv()
            .await
            .map_err(|err| {
                log::error!("Error receiving world update: {:?}", err);
            })
            .ok()
    }

    // messages are handeled in the executor run_rt_live
    pub fn send(&self, msg: ControlMsg) {
        self.control_tx
            .send(msg)
            .map_err(|err| {
                log::error!("Error sending control message: {:?}", err);
            })
            .ok();
    }

    /// Request the current executor info.
    pub fn info(&self) -> Result<ExecutorInfoReceiver> {
        let (tx, rx) = oneshot::channel();
        self.info_channel
            .send(tx)
            .map_err(|err| anyhow!("Error sending info request: {:?}", err))?;
        Ok(ExecutorInfoReceiver(rx))
    }
}

impl Clone for ExecutorHandle {
    fn clone(&self) -> Self {
        Self {
            control_tx: self.control_tx.clone(),
            update_rx: self.update_rx.resubscribe(),
            info_channel: self.info_channel.clone(),
        }
    }
}
