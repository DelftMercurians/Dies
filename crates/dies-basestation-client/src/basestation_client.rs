use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use dies_core::{PlayerCmd, PlayerFeedbackMsg, PlayerId, SysStatus, TeamColor};
use glue::{Monitor, Serial};
use tokio::sync::{broadcast, mpsc, oneshot};

const MAX_MSG_FREQ: f64 = 50.0;
const BASE_STATION_READ_FREQ: f64 = 50.0;

/// List available serial ports. The port names can be used to create a
/// [`BasestationClient`].
pub fn list_serial_ports() -> Vec<String> {
    Serial::list_ports(true)
}

/// Protocol version for the base station communication.
#[derive(Debug, Clone)]
pub enum BaseStationProtocol {
    V0,
    V1,
}

#[derive(Debug, Clone)]
/// Configuration for the serial client.
pub struct BasestationClientConfig {
    /// The name of the serial port. Use [`list_serial_ports`] to get a list of
    /// available ports.
    port_name: String,
    /// Map of player IDs to robot ids
    robot_id_map: HashMap<(TeamColor, PlayerId), u32>,
    /// Protocol version
    protocol: BaseStationProtocol,
}

impl BasestationClientConfig {
    pub fn new(port: String, protocol: BaseStationProtocol) -> Self {
        BasestationClientConfig {
            port_name: port,
            robot_id_map: HashMap::new(),
            protocol,
        }
    }

    // pub fn set_robot_id_map_from_string(&mut self, map: &str) {
    //     // Parse string "<id1>:<id1>;..."
    //     self.robot_id_map = map
    //         .split(';')
    //         .filter(|s| !s.is_empty())
    //         .map(|s| {
    //             let mut parts = s.split(':');
    //             let player_id = parts
    //                 .next()
    //                 .expect("Failed to parse player id")
    //                 .parse::<u32>()
    //                 .expect("Failed to parse player id");
    //             let robot_id = parts
    //                 .next()
    //                 .expect("Failed to parse robot id")
    //                 .parse::<u32>()
    //                 .expect("Failed to parse robot id");
    //             (PlayerId::new(player_id), robot_id)
    //         })
    //         .collect();
    // }
}

impl Default for BasestationClientConfig {
    fn default() -> Self {
        Self {
            #[cfg(target_os = "windows")]
            port_name: "COM3".to_string(),
            #[cfg(not(target_os = "windows"))]
            port_name: "/dev/ttyACM0".to_string(),
            robot_id_map: HashMap::new(),
            protocol: BaseStationProtocol::V1,
        }
    }
}

enum Connection {
    V0(Serial),
    V1(Monitor),
}

enum Message {
    PlayerCmd((TeamColor, PlayerCmd, oneshot::Sender<Result<()>>)),
    ChangeIdMap(HashMap<(TeamColor, PlayerId), u32>),
}

/// Async client for the serial port.
#[derive(Debug)]
pub struct BasestationHandle {
    cmd_tx: mpsc::UnboundedSender<Message>,
    info_rx: broadcast::Receiver<(Option<TeamColor>, PlayerFeedbackMsg)>,
}

impl BasestationHandle {
    /// Spawn a new basestation client in a new thread.
    pub fn spawn(config: BasestationClientConfig) -> Result<Self> {
        let BasestationClientConfig { port_name, .. } = config;

        // Launch a blocking thread for writing to the serial port
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel::<Message>();
        let (info_tx, info_rx) = broadcast::channel::<(Option<TeamColor>, PlayerFeedbackMsg)>(16);

        let mut connection = match config.protocol {
            BaseStationProtocol::V0 => {
                let serial = Serial::new(&port_name)?;
                Connection::V0(serial)
            }
            BaseStationProtocol::V1 => {
                let monitor = Monitor::start();
                monitor
                    .connect_to(&port_name)
                    .map_err(|_| anyhow!("Failed to connect to base station"))?;
                Connection::V1(monitor)
            }
        };

        let result = tokio::task::spawn_blocking(move || {
            let mut last_send = std::time::Instant::now();
            let interval = Duration::from_secs_f64(1.0 / BASE_STATION_READ_FREQ);
            let mut id_map = config.robot_id_map;

            let mut all_ids = HashSet::new();
            'outer: loop {
                // Send commands
                'inner: loop {
                    match cmd_rx.try_recv() {
                        Ok(Message::PlayerCmd((team_color, cmd, resp))) => match cmd {
                            PlayerCmd::Move(cmd) => {
                                let robot_id = id_map
                                    .get(&(team_color, cmd.id))
                                    .copied()
                                    .unwrap_or_else(|| cmd.id.as_u32())
                                    as usize;
                                match &mut connection {
                                    Connection::V1(monitor) => {
                                        let glue_cmd: glue::Radio_Command = cmd.into();
                                        resp.send(
                                            monitor
                                                .send_single(cmd.id.as_u32() as u8, glue_cmd)
                                                .map_err(|_| anyhow!("Failed to send command")),
                                        )
                                        .ok();
                                    }
                                    Connection::V0(serial) => {
                                        all_ids.insert(robot_id);
                                        let cmd_str = cmd.into_proto_v0_with_id(robot_id);
                                        serial.send(&cmd_str);
                                        resp.send(Ok(())).ok();
                                    }
                                }
                            }
                            PlayerCmd::GlobalMove(cmd) => {
                                let robot_id = id_map
                                    .get(&(team_color, cmd.id))
                                    .copied()
                                    .unwrap_or_else(|| cmd.id.as_u32())
                                    as usize;
                                match &mut connection {
                                    Connection::V1(monitor) => {
                                        let glue_cmd: glue::Radio_GlobalCommand = cmd.into();
                                        resp.send(
                                            monitor
                                                .send_single_global(robot_id as u8, glue_cmd)
                                                .map_err(|_| anyhow!("Failed to send command")),
                                        )
                                        .ok();
                                    }
                                    Connection::V0(_) => {
                                        log::error!("Global move commands are not supported in V0");
                                        resp.send(Err(anyhow!(
                                            "Global move commands are not supported in V0"
                                        )))
                                        .ok();
                                    }
                                }
                            }
                        },
                        Ok(Message::ChangeIdMap(new_id_map)) => {
                            id_map = new_id_map;
                        }
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            if let Connection::V1(monitor) = connection {
                                monitor.stop();
                            }
                            break 'outer;
                        }
                        Err(mpsc::error::TryRecvError::Empty) => break 'inner,
                    }
                }

                // Receive feedback
                if let Connection::V1(monitor) = &mut connection {
                    if !monitor.is_connected() {
                        panic!(
                            "Base station disconnected: {:?}",
                            monitor.has_error().unwrap()
                        );
                    }

                    if let Some(robots) = monitor.get_robots() {
                        let feedback = robots
                            .iter()
                            .enumerate()
                            .filter_map(|(id, msg)| {
                                if msg
                                    .time_since_status_lf_update()
                                    .unwrap_or(Duration::from_secs(600))
                                    < Duration::from_millis(600)
                                {
                                    all_ids.insert(id);
                                    let (color, player_id) = id_map
                                        .iter()
                                        .find_map(|(k, v)| {
                                            if *v == id as u32 {
                                                Some((Some(k.0), k.1))
                                            } else {
                                                None
                                            }
                                        })
                                        .unwrap_or_else(|| (None, PlayerId::new(id as u32)));

                                    Some((
                                        color,
                                        PlayerFeedbackMsg {
                                            id: player_id,
                                            primary_status: SysStatus::from_option(
                                                msg.primary_status(),
                                            ),
                                            kicker_status: SysStatus::from_option(
                                                msg.kicker_status(),
                                            ),
                                            imu_status: SysStatus::from_option(msg.imu_status()),
                                            fan_status: SysStatus::from_option(msg.fan_status()),
                                            kicker_cap_voltage: msg.kicker_cap_voltage(),
                                            kicker_temp: msg.kicker_temperature(),
                                            motor_statuses: msg
                                                .motor_statuses()
                                                .map(|v| v.map(Into::into)),
                                            motor_speeds: msg.motor_speeds(),
                                            motor_temps: msg.motor_temperatures(),
                                            breakbeam_ball_detected: msg.breakbeam_ball_detected(),
                                            breakbeam_sensor_ok: msg.breakbeam_sensor_ok(),
                                            pack_voltages: msg.pack_voltages(),
                                        },
                                    ))
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>();

                        for msg in feedback {
                            info_tx.send(msg).ok();
                        }
                    }
                }

                // Sleep
                let elapsed = last_send.elapsed();
                last_send = std::time::Instant::now();
                if elapsed < interval {
                    std::thread::sleep(interval - elapsed);
                } else {
                    std::thread::sleep(Duration::from_secs_f64(1.0 / MAX_MSG_FREQ));
                }
            }
        });

        tokio::spawn(async move {
            let result = result.await;
            if let Err(e) = result {
                log::error!("Error in basestation client: {:?}", e);
                std::process::exit(1);
            }
        });

        Ok(Self { cmd_tx, info_rx })
    }

    /// Send a message to the serial port.
    pub async fn send(&mut self, team_color: TeamColor, msg: PlayerCmd) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(Message::PlayerCmd((team_color, msg, tx)))
            .context("Failed to send message to serial port")?;
        rx.await?
    }

    pub fn send_no_wait(&mut self, team_color: TeamColor, msg: PlayerCmd) {
        let (tx, _) = oneshot::channel();
        self.cmd_tx
            .send(Message::PlayerCmd((team_color, msg, tx)))
            .ok();
    }

    /// Update the id map
    pub fn update_id_map(&mut self, id_map: HashMap<(TeamColor, PlayerId), u32>) {
        self.cmd_tx.send(Message::ChangeIdMap(id_map)).ok();
    }

    /// Receive a message from the serial port.
    pub async fn recv(&mut self) -> Result<(Option<TeamColor>, PlayerFeedbackMsg)> {
        self.info_rx.recv().await.map_err(|e| e.into())
    }
}

impl Clone for BasestationHandle {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            info_rx: self.info_rx.resubscribe(),
        }
    }
}
