use std::{collections::HashMap, io::Write, time::Duration};

use anyhow::{anyhow, Context, Result};
use glue::{Monitor, Serial};
use tokio::sync::{broadcast, mpsc, oneshot};

use dies_core::{PlayerCmd, PlayerFeedbackMsg, PlayerId, SysStatus};

const MAX_MSG_FREQ: f64 = 60.0;
const BASE_STATION_READ_FREQ: f64 = 20.0;
const TIMEOUT: Duration = Duration::from_millis(20);

/// List available serial ports. The port names can be used to create a
/// [`BasestationClient`].
pub fn list_serial_ports() -> Vec<String> {
    Serial::list_ports(true)
}

/// Configuration for the serial client.
pub struct BasestationClientConfig {
    /// The name of the serial port. Use [`list_serial_ports`] to get a list of
    /// available ports.
    port_name: String,
    /// Map of player IDs to robot ids
    robot_id_map: HashMap<PlayerId, u32>,
}

impl BasestationClientConfig {
    pub fn new(port: String) -> Self {
        BasestationClientConfig {
            port_name: port,
            robot_id_map: HashMap::new(),
        }
    }

    pub fn new_with_id_map(port: String, robot_id_map: HashMap<PlayerId, u32>) -> Self {
        BasestationClientConfig {
            port_name: port,
            robot_id_map,
        }
    }

    pub fn set_robot_id_map_from_string(&mut self, map: &str) {
        // Parse string "<id1>:<id1>;..."
        self.robot_id_map = map
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let mut parts = s.split(':');
                let player_id = parts
                    .next()
                    .expect("Failed to parse player id")
                    .parse::<u32>()
                    .expect("Failed to parse player id");
                let robot_id = parts
                    .next()
                    .expect("Failed to parse robot id")
                    .parse::<u32>()
                    .expect("Failed to parse robot id");
                (PlayerId::new(player_id), robot_id)
            })
            .collect();
    }
}

impl Default for BasestationClientConfig {
    fn default() -> Self {
        Self {
            #[cfg(target_os = "windows")]
            port_name: "COM3".to_string(),
            #[cfg(not(target_os = "windows"))]
            port_name: "/dev/ttyACM0".to_string(),
            robot_id_map: HashMap::new(),
        }
    }
}

/// Async client for the serial port.
pub struct BasestationClient {
    cmd_tx: mpsc::UnboundedSender<(PlayerCmd, oneshot::Sender<Result<()>>)>,
    info_rx: broadcast::Receiver<PlayerFeedbackMsg>,
}

impl BasestationClient {
    /// Create a new `BasestationClient`.
    pub fn new(config: BasestationClientConfig) -> Result<Self> {
        let BasestationClientConfig {
            port_name,
            robot_id_map,
        } = config;

        // Launch a blocking thread for writing to the serial port
        let (cmd_tx, mut cmd_rx) =
            mpsc::unbounded_channel::<(PlayerCmd, oneshot::Sender<Result<()>>)>();
        let (info_tx, info_rx) = broadcast::channel::<PlayerFeedbackMsg>(1);
        let monitor = Monitor::start();
        monitor
            .connect_to(&port_name)
            .map_err(|_| anyhow!("Failed to connect to base station"))?;

        tokio::task::spawn_blocking(move || {
            let mut last_send = std::time::Instant::now();
            let interval = Duration::from_secs_f64(1.0 / BASE_STATION_READ_FREQ);

            loop {
                // Send commands
                if let Ok((cmd, resp)) = cmd_rx.try_recv() {
                    let mut commands = [None; glue::MAX_NUM_ROBOTS];
                    commands[cmd.id.as_u32() as usize] = Some(cmd.into());
                    resp.send(
                        monitor
                            .send(commands)
                            .map_err(|_| anyhow!("Failed to send command")),
                    )
                    .ok();
                }

                // Receive feedback
                if let Some(robots) = monitor.get_robots() {
                    let feedback = robots
                        .iter()
                        .enumerate()
                        .map(|(id, msg)| {
                            let player_id = PlayerId::new(id as u32);
                            let robot_id = robot_id_map.get(&player_id).copied();
                            PlayerFeedbackMsg {
                                id: player_id,
                                primary_status: SysStatus::from_option(msg.primary_status()),
                                kicker_status: SysStatus::from_option(msg.kicker_status()),
                                imu_status: SysStatus::from_option(msg.imu_status()),
                                fan_status: SysStatus::from_option(msg.fan_status()),
                                kicker_cap_voltage: msg.kicker_cap_voltage().map(|v| v as f64),
                                kicker_temp: msg.kicker_temperature().map(|v| v as f64),
                                motor_statuses: 
                                motor_speeds: msg.motor_speeds().map(|v| v as f64),
                                motor_temps: msg.motor_temps().map(|v| v as f64),
                                breakbeam_ball_detected: msg.breakbeam_ball_detected(),
                                breakbeam_sensor_ok: msg.breakbeam_sensor_ok(),
                                pack_voltages: msg.pack_voltages().map(|v| v as f64),
                            }
                        })
                        .collect::<Vec<_>>();
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

        Ok(Self { cmd_tx, info_rx })
    }

    /// Send a message to the serial port.
    pub async fn send(&mut self, msg: PlayerCmd) -> Result<()> {
        std::io::stdout().flush().unwrap();
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send((msg, tx))
            .context("Failed to send message to serial port")?;
        rx.await?
    }

    pub fn send_no_wait(&mut self, msg: PlayerCmd) {
        let (tx, _) = oneshot::channel();
        self.cmd_tx.send((msg, tx)).ok();
    }

    /// Receive a message from the serial port.
    pub async fn recv(&mut self) -> Result<PlayerFeedbackMsg> {
        Err(anyhow::anyhow!("Not implemented"))
    }
}
