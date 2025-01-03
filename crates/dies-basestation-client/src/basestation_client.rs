use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use dies_core::{ColoredPlayerId, RobotCmd, RobotFeedback, SysStatus, TeamColor};
use tokio::sync::{broadcast, mpsc, oneshot};

use glue::{Monitor, Serial};

use crate::{
    glue_convert::{convert_cmd, convert_sys_status, convert_sys_status_opt},
    robot_router::RobotRouter,
};

/// List available serial ports. The port names can be used to create a
/// [`BasestationClient`].
pub fn list_serial_ports() -> Vec<String> {
    Serial::list_ports(true)
}

#[derive(Debug, Clone)]
/// Configuration for the serial client.
pub struct BasestationClientConfig {
    /// The initial team color to use for populating the robot router
    default_team_color: TeamColor,
    /// The maximum message frequency in Hz
    max_msg_freq: f64,
}

impl Default for BasestationClientConfig {
    fn default() -> Self {
        Self {
            default_team_color: TeamColor::Blue,
            max_msg_freq: 60.0,
        }
    }
}

enum Message {
    RobotCmd((ColoredPlayerId, RobotCmd, oneshot::Sender<Result<()>>)),
    SetPort(String),
    GetConnectionStatus(oneshot::Sender<BasestationConnectionStatus>),
    SetIdMap(Vec<(u32, Option<ColoredPlayerId>)>),
    GetIdMap(oneshot::Sender<Vec<(u32, Option<ColoredPlayerId>)>>),
}

#[derive(Debug, Clone)]
pub enum BasestationConnectionStatus {
    Connected(String),
    Disconnected,
    Error(String),
}

struct BasestationClient {
    cmd_rx: mpsc::UnboundedReceiver<Message>,
    feedback_tx: broadcast::Sender<Vec<(ColoredPlayerId, RobotFeedback)>>,
    monitor: Option<Monitor>,
    connection_status: BasestationConnectionStatus,
    robot_router: RobotRouter,
    max_msg_freq: f64,
}

impl BasestationClient {
    fn spawn(config: BasestationClientConfig) -> BasestationHandle {
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (feedback_tx, feedback_rx) =
            broadcast::channel::<Vec<(ColoredPlayerId, RobotFeedback)>>(16);

        let client = Self {
            cmd_rx,
            feedback_tx: feedback_tx.clone(),
            monitor: None,
            connection_status: BasestationConnectionStatus::Disconnected,
            robot_router: RobotRouter::new(config.default_team_color),
            max_msg_freq: config.max_msg_freq,
        };

        tokio::task::spawn_blocking(move || client.run());

        BasestationHandle {
            cmd_tx,
            feedback_rx,
        }
    }

    fn run(mut self) {
        // Attempt to connect to the first port
        if let Some(port) = list_serial_ports().first() {
            self.connect(port);
        }

        let mut last_ts = Instant::now();
        let interval = Duration::from_secs_f64(1.0 / self.max_msg_freq);
        loop {
            // Handle messages
            match self.cmd_rx.try_recv() {
                Ok(Message::SetPort(port)) => {
                    if let Some(monitor) = self.monitor.take() {
                        monitor.stop();
                    }
                    self.connect(&port);
                }
                Ok(Message::RobotCmd((id, cmd, tx))) => {
                    self.send_cmd(id, cmd);
                    tx.send(Ok(())).ok();
                }
                Ok(Message::SetIdMap(id_map)) => {
                    self.robot_router.set_id_map(id_map);
                }
                Ok(Message::GetConnectionStatus(sender)) => {
                    sender.send(self.connection_status.clone()).ok();
                }
                Ok(Message::GetIdMap(sender)) => {
                    sender.send(self.robot_router.id_map()).ok();
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    log::info!("Shutting down basestation client");
                    if let Some(monitor) = self.monitor.take() {
                        monitor.stop();
                    }
                    break;
                }
                Err(_) => {}
            }

            // Handle feedback from monitor
            if let Some(monitor) = &mut self.monitor {
                if let Some(feedback) = monitor.get_robots() {
                    // Update robot IDs from feedback
                    let active_robot_ids: Vec<_> = feedback
                        .iter()
                        .enumerate()
                        .filter_map(|(i, r)| {
                            if matches!(r.primary_status(), Some(glue::HG_Status::OK)) {
                                Some(i as u32)
                            } else {
                                None
                            }
                        })
                        .collect();
                    self.robot_router.update_robot_ids(active_robot_ids);

                    // Send feedback
                    let feedback = feedback
                        .into_iter()
                        .enumerate()
                        .map(|(i, f)| {
                            // Convert feedback
                            let robot_feedback = RobotFeedback {
                                primary_status: convert_sys_status_opt(f.primary_status()),
                                kicker_status: convert_sys_status_opt(f.kicker_status()),
                                imu_status: convert_sys_status_opt(f.imu_status()),
                                fan_status: convert_sys_status_opt(f.fan_status()),
                                kicker_cap_voltage: f.kicker_cap_voltage(),
                                kicker_temp: f.kicker_temperature(),
                                motor_statuses: f.motor_statuses().map(|m| {
                                    let mut statuses = [SysStatus::NotInstalled; 5];
                                    for (i, s) in statuses.iter_mut().enumerate() {
                                        *s = convert_sys_status(m[i]);
                                    }
                                    statuses
                                }),
                                motor_speeds: f.motor_speeds(),
                                motor_temps: f.motor_temperatures(),
                                breakbeam_ball_detected: f.breakbeam_ball_detected(),
                                breakbeam_sensor_ok: f.breakbeam_sensor_ok(),
                                pack_voltages: f.pack_voltages(),
                            };
                            let colored_id = self.robot_router.get_by_robot_id(i as u32);
                            (colored_id, robot_feedback)
                        })
                        .collect::<Vec<_>>();

                    self.feedback_tx.send(feedback).ok();
                }
            }

            // Sleep
            let elapsed = last_ts.elapsed();
            last_ts = Instant::now();
            if elapsed < interval {
                std::thread::sleep(interval - elapsed);
            } else {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    fn connect(&mut self, port: &str) {
        let monitor = Monitor::start();
        match monitor.connect_to(&port) {
            Ok(_) => {
                self.connection_status = BasestationConnectionStatus::Connected(port.to_string());
                self.monitor = Some(monitor);
            }
            Err(_) => {
                log::error!("Failed to connect to port {}", port);
                self.connection_status = BasestationConnectionStatus::Error(format!(
                    "Failed to connect to port {}",
                    port
                ));
            }
        }
    }

    fn send_cmd(&mut self, id: ColoredPlayerId, cmd: RobotCmd) {
        if let Some(monitor) = self.monitor.as_mut() {
            let robot_id = self.robot_router.get_by_player_id(id);
            monitor.send_single(robot_id as u8, convert_cmd(cmd));
        }
    }
}

/// Async client for the serial port.
#[derive(Debug)]
pub struct BasestationHandle {
    cmd_tx: mpsc::UnboundedSender<Message>,
    feedback_rx: broadcast::Receiver<Vec<(ColoredPlayerId, RobotFeedback)>>,
}

impl BasestationHandle {
    /// Create a new basestation client with the given configuration.
    pub fn new(config: BasestationClientConfig) -> Self {
        BasestationClient::spawn(config)
    }

    /// Send a command to a robot.
    pub async fn send_cmd(&self, id: ColoredPlayerId, cmd: RobotCmd) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(Message::RobotCmd((id, cmd, tx)))
            .context("Failed to send command")?;
        rx.await?
    }

    /// Update the id map
    pub fn update_id_map(&self, id_map: Vec<(u32, Option<ColoredPlayerId>)>) {
        self.cmd_tx.send(Message::SetIdMap(id_map)).ok();
    }

    /// Get the current connection status
    pub async fn get_connection_status(&self) -> Result<BasestationConnectionStatus> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(Message::GetConnectionStatus(tx))
            .context("Failed to get connection status")?;
        Ok(rx.await?)
    }

    /// Get the current id map
    pub async fn get_id_map(&self) -> Result<Vec<(u32, Option<ColoredPlayerId>)>> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(Message::GetIdMap(tx))
            .context("Failed to get id map")?;
        Ok(rx.await?)
    }

    /// Set the port to connect to
    pub fn set_port(&self, port: String) {
        self.cmd_tx.send(Message::SetPort(port)).ok();
    }

    /// Receive feedback from robots.
    pub async fn recv_feedback(&mut self) -> Result<Vec<(ColoredPlayerId, RobotFeedback)>> {
        self.feedback_rx
            .recv()
            .await
            .map_err(|e| anyhow!(e.to_string()))
    }
}

impl Clone for BasestationHandle {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            feedback_rx: self.feedback_rx.resubscribe(),
        }
    }
}
