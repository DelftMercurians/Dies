use anyhow::Result;
use dies_core::PlayerFeedbackMsg;
use dies_ssl_client::{SslMessage, VisionClient};
use std::sync::{Arc, RwLock};

use dies_basestation_client::BasestationHandle;
use tokio::{
    sync::{broadcast, mpsc, oneshot},
    task::JoinHandle,
};

use crate::ui_common::{UiCmd, UiPushMsg, UiState};

pub struct Handle {
    cmd_tx: mpsc::UnboundedSender<UiCmd>,
    msg_rx: broadcast::Receiver<UiPushMsg>,
    state: Arc<RwLock<UiState>>,
    task: JoinHandle<()>,
    shutdown_tx: oneshot::Sender<()>,
}

impl Handle {
    pub fn send_ui_cmd(&self, cmd: UiCmd) {
        self.cmd_tx.send(cmd);
    }

    pub fn ui_state(&self) -> UiState {
        self.state.read().unwrap().clone()
    }

    pub async fn shudown(self) -> anyhow::Result<()> {
        self.shutdown_tx.send(());
        self.task.await.map_err(|err| err.into())
    }

    pub fn subscribe_messages(&self) -> broadcast::Receiver<UiPushMsg> {
        self.msg_rx.resubscribe()
    }
}

pub struct InteractiveMainLoop {
    cmd_rx: mpsc::UnboundedReceiver<UiCmd>,
    msg_tx: broadcast::Sender<UiPushMsg>,
    state: Arc<RwLock<UiState>>,
    shutdown_rx: oneshot::Receiver<()>,

    ssl_client: Option<dies_ssl_client::VisionClient>,
    bs_client: Option<BasestationHandle>,
    teams: Option<TeamMap>,
}

impl InteractiveMainLoop {
    pub fn spawn() -> Handle {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (msg_tx, msg_rx) = broadcast::channel(16);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let state = Arc::new(RwLock::new(UiState::default()));
        let task = {
            let state = Arc::clone(&state);
            tokio::spawn(async move {
                let main_loop = InteractiveMainLoop {
                    cmd_rx,
                    msg_tx,
                    state,
                    shutdown_rx,
                    ssl_client: None,
                    bs_client: None,
                };
                main_loop.run().await;
            })
        };
        Handle {
            cmd_tx,
            msg_rx,
            state,
            task,
            shutdown_tx,
        }
    }

    async fn run(mut self) {
        loop {
            tokio::select! {
                _ = &mut self.shutdown_rx => break,
                result = self.cmd_rx.recv() => match result {
                    Some(msg) => self.handle_ui_cmd(msg),
                    None => break
                },
                Some(result) = recv_ssl(self.ssl_client.as_mut()) => self.handle_ssl(result),
                Some(result) = recv_bs(self.bs_client.as_mut()) => self.handle_bs(result),
            }
        }
    }

    fn handle_ui_cmd(&mut self, cmd: UiCmd) {
        match cmd {
            UiCmd::StartGame(game_setup) => todo!(),
            UiCmd::StopGame => todo!(),
            UiCmd::PauseGame => todo!(),
            UiCmd::ConnectVision => todo!(),
            UiCmd::ConnectGc => todo!(),
            UiCmd::ConnectBasestation => todo!(),
        }
    }

    fn handle_ssl(&mut self, msg: Result<SslMessage>) {
        let msg = match msg {
            Ok(msg) => msg,
            Err(err) => {
                log::error!("Failed receiving SSL message: {err}");
                return;
            }
        };
    }

    fn handle_bs(&mut self, msg: Result<PlayerFeedbackMsg>) {
        let msg = match msg {
            Ok(msg) => msg,
            Err(err) => {
                log::error!("Failed receiving basestation message: {err}");
                return;
            }
        };
    }
}

async fn recv_ssl(ssl_client: Option<&mut VisionClient>) -> Option<Result<SslMessage>> {
    if let Some(client) = ssl_client {
        Some(client.recv().await)
    } else {
        None
    }
}

async fn recv_bs(bs_client: Option<&mut BasestationHandle>) -> Option<Result<PlayerFeedbackMsg>> {
    if let Some(handle) = bs_client {
        Some(handle.recv().await)
    } else {
        None
    }
}
