use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use anyhow::{anyhow, Context, Result};
use dies_core::{
    BaseStationInfo, CommandEcho, FirmwareVersion, PlayerCmd, PlayerFeedbackMsg, PlayerId,
    RobotCmd, SysStatus, TeamColor,
};
use glue::{Monitor, Serial};
use tokio::sync::{broadcast, mpsc, oneshot, watch};

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
    pub fn new(
        port: String,
        protocol: BaseStationProtocol,
        robot_id_map: HashMap<(TeamColor, PlayerId), u32>,
    ) -> Self {
        BasestationClientConfig {
            port_name: port,
            robot_id_map,
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
    Bench(BenchOp),
}

/// A low-level bench operation, addressing robots by raw robot id and bypassing
/// the `(team, player) -> robot_id` map. Used only by the webui test bench.
enum BenchOp {
    /// Stream a movement setpoint to a raw robot id.
    SendRaw(u8, PlayerCmd),
    /// Fire a one-shot robot command (zero speed) at a raw robot id.
    RobotCommand {
        robot_id: u8,
        cmd: RobotCmd,
        /// Raw firmware kick duration (`kick_time_i`, u16 ms); 0 for non-kick.
        kick_time: u16,
    },
    /// Broadcast a one-shot robot command to all robots.
    Broadcast(RobotCmd),
    /// Override the robot's heading estimate \[rad].
    SetHeading(u8, f32),
    /// Set the base station radio channel (0-125, i.e. 2400-2525 MHz).
    SetBaseChannel(u8),
    /// Set a single robot's radio channel.
    SetRobotChannel(u8, u8),
    /// Request the firmware version of a robot (READ MCM).
    GetVersion(u8),
}

/// Async client for the serial port.
#[derive(Debug)]
pub struct BasestationHandle {
    cmd_tx: mpsc::Sender<Message>,
    info_rx: broadcast::Receiver<(Option<TeamColor>, PlayerFeedbackMsg)>,
    base_info_rx: watch::Receiver<Option<BaseStationInfo>>,
}

impl BasestationHandle {
    /// Spawn a new basestation client in a new thread.
    pub fn spawn(config: BasestationClientConfig) -> Result<Self> {
        let BasestationClientConfig { port_name, .. } = config;

        // Launch a blocking thread for writing to the serial port
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<Message>(64);
        let (info_tx, info_rx) = broadcast::channel::<(Option<TeamColor>, PlayerFeedbackMsg)>(16);
        let (base_info_tx, base_info_rx) = watch::channel::<Option<BaseStationInfo>>(None);

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
                let send_start = std::time::Instant::now();
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
                        Ok(Message::Bench(op)) => {
                            if let Connection::V1(monitor) = &mut connection {
                                dispatch_bench_op(monitor, op);
                            } else {
                                log::warn!("Bench commands require the V1 protocol");
                            }
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
                let send_elapsed = send_start.elapsed();
                dies_core::debug_string("bs_send_took", send_elapsed.as_millis().to_string());

                // Receive feedback
                if let Connection::V1(monitor) = &mut connection {
                    if !monitor.is_connected() {
                        println!(
                            "Base station disconnected: {:?}",
                            monitor.has_error().unwrap_or_default()
                        );
                        std::process::exit(1);
                    }

                    let fw_versions = read_firmware_versions(monitor);

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
                                            tof_status: SysStatus::from_option(msg.tof_status()),
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
                                            imu_readings: msg.imu_reading().map(|r| {
                                                [
                                                    r.ang_x, r.ang_y, r.ang_z, r.ang_wx, r.ang_wy,
                                                    r.ang_wz,
                                                ]
                                            }),
                                            motor_currents: msg.motor_currents(),
                                            main_board_current: msg.main_board_current(),
                                            avg_loop_time_us: msg.avg_loop_time(),
                                            max_loop_time_us: msg.max_loop_time(),
                                            smart_kick_counter: msg.smart_kick_counter(),
                                            kick_ok_flag: msg.kick_ok_flag(),
                                            reflex_kick_state: msg
                                                .reflex_kick_state()
                                                .map(Into::into),
                                            reflex_kick_counter: msg.reflex_kick_counter(),
                                            breakbeam_raw: msg.breakbeam_raw(),
                                            tof_ball_detected: msg.tof_ball_detected(),
                                            tof_xy: msg.tof_xy().map(|(x, y)| [x as i32, y as i32]),
                                            last_command: msg.global_command().map(|c| {
                                                CommandEcho {
                                                    global_speed_x: c.global_speed_x,
                                                    global_speed_y: c.global_speed_y,
                                                    heading_setpoint: c.heading_setpoint,
                                                    heading_last_measurement: c
                                                        .heading_last_measurement,
                                                    dribbler_speed: c.gen_command.dribbler_speed_i,
                                                    kick_time: c.gen_command.kick_time_i,
                                                    robot_command: c.gen_command.robot_command
                                                        as u8,
                                                }
                                            }),
                                            firmware_version: fw_versions[id],
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

                    // Publish the latest base station info for the bench UI.
                    if let glue::Stamped::Have(_, bi) = monitor.get_base_info() {
                        let radios_online = (0..bi.num_radios)
                            .map(|i| (bi.radios_online >> i) & 1 == 1)
                            .collect();
                        base_info_tx.send_replace(Some(BaseStationInfo {
                            connected: true,
                            protocol_ok: bi.version.protcol_version_matches(),
                            version: bi.version.version_to_string(),
                            protocol_version: bi.version.protocol_version_to_string(),
                            channel_mhz: bi.channel as u32 + 2400,
                            num_radios: bi.num_radios,
                            radios_online,
                            max_robots: bi.max_robots,
                        }));
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

        Ok(Self {
            cmd_tx,
            info_rx,
            base_info_rx,
        })
    }

    /// Send a message to the serial port.
    pub async fn send(&mut self, team_color: TeamColor, msg: PlayerCmd) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(Message::PlayerCmd((team_color, msg, tx)))
            .await
            .context("Failed to send message to serial port")?;
        rx.await?
    }

    pub fn send_no_wait(&mut self, team_color: TeamColor, msg: PlayerCmd) {
        let (tx, _) = oneshot::channel();
        self.cmd_tx
            .try_send(Message::PlayerCmd((team_color, msg, tx)))
            .map_err(|e| eprintln!("Failed to send message to serial port: {:?}", e))
            .ok();
    }

    /// Update the id map
    pub fn update_id_map(&mut self, id_map: HashMap<(TeamColor, PlayerId), u32>) {
        self.cmd_tx.try_send(Message::ChangeIdMap(id_map)).ok();
    }

    /// Receive a message from the serial port.
    pub async fn recv(&mut self) -> Result<(Option<TeamColor>, PlayerFeedbackMsg)> {
        self.info_rx.recv().await.map_err(|e| e.into())
    }

    /// The latest basestation info, if any has been received.
    pub fn base_info(&self) -> Option<BaseStationInfo> {
        self.base_info_rx.borrow().clone()
    }

    // --- Bench (test-bench) API: addresses robots by raw robot id, bypassing
    // the (team, player) -> robot_id map. V1 protocol only. ---

    fn send_bench(&self, op: BenchOp) {
        self.cmd_tx.try_send(Message::Bench(op)).ok();
    }

    /// Stream a movement setpoint to a raw robot id (the 50 Hz bench path).
    pub fn bench_send_raw(&self, robot_id: u8, cmd: PlayerCmd) {
        self.send_bench(BenchOp::SendRaw(robot_id, cmd));
    }

    /// Fire a one-shot robot command (zero speed) at a raw robot id. `kick_time`
    /// is the raw firmware kick duration (`kick_time_i`, u16 ms); 0 for non-kick.
    pub fn bench_robot_command(&self, robot_id: u8, cmd: RobotCmd, kick_time: u16) {
        self.send_bench(BenchOp::RobotCommand {
            robot_id,
            cmd,
            kick_time,
        });
    }

    /// Broadcast a one-shot robot command to all robots.
    pub fn bench_broadcast(&self, cmd: RobotCmd) {
        self.send_bench(BenchOp::Broadcast(cmd));
    }

    /// Override a robot's heading estimate \[rad].
    pub fn bench_set_heading(&self, robot_id: u8, heading_rad: f32) {
        self.send_bench(BenchOp::SetHeading(robot_id, heading_rad));
    }

    /// Set the base station radio channel (0-125 maps to 2400-2525 MHz).
    pub fn bench_set_base_channel(&self, channel: u8) {
        self.send_bench(BenchOp::SetBaseChannel(channel));
    }

    /// Set a single robot's radio channel.
    pub fn bench_set_robot_channel(&self, robot_id: u8, channel: u8) {
        self.send_bench(BenchOp::SetRobotChannel(robot_id, channel));
    }

    /// Request a robot's firmware version (populated into feedback once it replies).
    pub fn bench_get_version(&self, robot_id: u8) {
        self.send_bench(BenchOp::GetVersion(robot_id));
    }
}

impl Clone for BasestationHandle {
    fn clone(&self) -> Self {
        Self {
            cmd_tx: self.cmd_tx.clone(),
            info_rx: self.info_rx.resubscribe(),
            base_info_rx: self.base_info_rx.clone(),
        }
    }
}

/// Build a zero-velocity command carrying a single robot command (for bench
/// one-shots like beep/kick/calibrate).
fn build_robot_command(cmd: RobotCmd, kick_time: u16) -> glue::Radio_Command {
    glue::Radio_Command {
        speed: glue::HG_Pose {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        gen_command: glue::Radio_GenericCommand {
            dribbler_speed_i: 0,
            kick_time_i: kick_time,
            time_to_kick: 0,
            smart_kick_couter: 0,
            robot_command: cmd.into(),
        },
        _pad: [0, 0, 0, 0, 0, 0, 0, 0],
    }
}

/// Dispatch a bench operation directly to the glue monitor.
fn dispatch_bench_op(monitor: &mut Monitor, op: BenchOp) {
    match op {
        BenchOp::SendRaw(robot_id, cmd) => match cmd {
            PlayerCmd::Move(c) => {
                let g: glue::Radio_Command = c.into();
                monitor.send_single(robot_id, g).ok();
            }
            PlayerCmd::GlobalMove(c) => {
                let g: glue::Radio_GlobalCommand = c.into();
                monitor.send_single_global(robot_id, g).ok();
            }
        },
        BenchOp::RobotCommand {
            robot_id,
            cmd,
            kick_time,
        } => {
            monitor
                .send_single(robot_id, build_robot_command(cmd, kick_time))
                .ok();
        }
        BenchOp::Broadcast(cmd) => {
            monitor.send_broadcast(build_robot_command(cmd, 0)).ok();
        }
        BenchOp::SetHeading(robot_id, heading) => {
            monitor.set_current_heading(robot_id, heading).ok();
        }
        BenchOp::SetBaseChannel(channel) => {
            monitor.set_channel(channel).ok();
        }
        BenchOp::SetRobotChannel(robot_id, channel) => {
            let mcm = glue::Radio_MultiConfigMessage {
                operation: glue::HG_ConfigOperation::WRITE,
                vars: [
                    glue::HG_Variable::RADIO_CHANNEL,
                    glue::HG_Variable::NONE,
                    glue::HG_Variable::NONE,
                    glue::HG_Variable::NONE,
                    glue::HG_Variable::NONE,
                ],
                type_: glue::HG_VariableType::VOID,
                _pad: 0,
                values: [channel as u32, 0, 0, 0, 0],
            };
            monitor.send_mcm(robot_id, mcm).ok();
        }
        BenchOp::GetVersion(robot_id) => {
            let mcm = glue::Radio_MultiConfigMessage {
                operation: glue::HG_ConfigOperation::READ,
                vars: [
                    glue::HG_Variable::FW_VERSION_MAJOR,
                    glue::HG_Variable::FW_VERSION_MINOR,
                    glue::HG_Variable::FW_VERSION_PATCH,
                    glue::HG_Variable::FW_PROTOCOL_VERSION_MAJOR,
                    glue::HG_Variable::FW_PROTOCOL_VERSION_MINOR,
                ],
                type_: glue::HG_VariableType::VOID,
                _pad: 0,
                values: [0, 0, 0, 0, 0],
            };
            monitor.send_mcm(robot_id, mcm).ok();
        }
    }
}

/// Read the per-robot firmware version from the debug mux (populated after a
/// `GetVersion` request). Returns `None` for robots that haven't reported.
fn read_firmware_versions(monitor: &Monitor) -> [Option<FirmwareVersion>; glue::MAX_NUM_ROBOTS] {
    let mut out = [None; glue::MAX_NUM_ROBOTS];
    let Some(mux) = monitor.get_debug_mux() else {
        return out;
    };
    let read = |id: usize, var: glue::HG_Variable| -> Option<u32> {
        match mux.config_variable_returns[id][var as usize] {
            glue::Stamped::Have(t, v) if t.elapsed() < Duration::from_secs(30) => Some(v),
            _ => None,
        }
    };
    for (id, slot) in out.iter_mut().enumerate() {
        if let (Some(major), Some(minor), Some(patch)) = (
            read(id, glue::HG_Variable::FW_VERSION_MAJOR),
            read(id, glue::HG_Variable::FW_VERSION_MINOR),
            read(id, glue::HG_Variable::FW_VERSION_PATCH),
        ) {
            *slot = Some(FirmwareVersion {
                major: major as u16,
                minor: minor as u16,
                patch: patch as u16,
            });
        }
    }
    out
}
