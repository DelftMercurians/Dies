use tokio::sync::{broadcast, mpsc};

use dies_core::{PlayerId, WorldUpdate};

pub enum ControlMsg {
    SetPlayerOverride {
        player_id: PlayerId,
        override_active: bool,
    },
    SetPause(bool),
    Stop,
}

#[derive(Debug)]
pub struct ExecutorHandle {
    pub control_tx: mpsc::UnboundedSender<ControlMsg>,
    pub update_rx: broadcast::Receiver<WorldUpdate>,
}

impl ExecutorHandle {
    pub async fn recv(&mut self) -> Option<WorldUpdate> {
        self.update_rx
            .recv()
            .await
            .map_err(|err| {
                tracing::error!("Error receiving world update: {:?}", err);
            })
            .ok()
    }

    pub fn send(&mut self, msg: ControlMsg) {
        self.control_tx
            .send(msg)
            .map_err(|err| {
                tracing::error!("Error sending control message: {:?}", err);
            })
            .ok();
    }
}

impl Clone for ExecutorHandle {
    fn clone(&self) -> Self {
        Self {
            control_tx: self.control_tx.clone(),
            update_rx: self.update_rx.resubscribe(),
        }
    }
}
