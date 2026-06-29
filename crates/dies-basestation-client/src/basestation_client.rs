use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};

use anyhow::Result;
use dies_core::{
    BaseStationInfo, CommandEcho, FirmwareVersion, PlayerCmd, PlayerFeedbackMsg, PlayerId,
    RobotCmd, SysStatus, TeamColor,
};
use glue::{BaseStation, Serial};
use tokio::sync::{broadcast, mpsc, watch};

const DEFAULT_BASE_STATION_READ_FREQ: f64 = 800.0;

/// A robot's latest command is re-sent every loop until superseded; if no fresh
/// command arrives within this window the robot is dropped from the poll set, so
/// a stale command can't keep driving a robot after control stops.
const COMMAND_MAX_AGE: Duration = Duration::from_millis(200);

/// How long the link must keep failing to read before we treat it as
/// genuinely down.
const DISCONNECT_GRACE: Duration = Duration::from_millis(300);
/// Minimum spacing between reconnect attempts while the link is down.
const RECONNECT_INTERVAL: Duration = Duration::from_millis(500);

/// Sliding window over which per-robot feedback rate is measured.
const FEEDBACK_RATE_WINDOW: Duration = Duration::from_secs(1);

#[derive(Default)]
struct FeedbackRateTracker {
    /// Instant of the last fresh HF sample we observed, per robot index.
    last_update: HashMap<usize, Instant>,
    /// Instants of fresh frames within the current window, per robot index.
    hits: HashMap<usize, VecDeque<Instant>>,
}

impl FeedbackRateTracker {
    /// Record this read's HF-status age for `id` and return the current
    /// feedback rate (Hz). `hf_age` is `time_since_status_hf_update()`.
    fn observe(&mut self, id: usize, hf_age: Option<Duration>, now: Instant) -> f32 {
        if let Some(age) = hf_age {
            let update_instant = now.checked_sub(age).unwrap_or(now);
            let is_fresh = self.last_update.get(&id).map_or(true, |prev| {
                update_instant > *prev + Duration::from_micros(500)
            });
            if is_fresh {
                self.last_update.insert(id, update_instant);
                self.hits.entry(id).or_default().push_back(now);
            }
        }
        let hits = self.hits.entry(id).or_default();
        let cutoff = now.checked_sub(FEEDBACK_RATE_WINDOW).unwrap_or(now);
        while hits.front().is_some_and(|t| *t < cutoff) {
            hits.pop_front();
        }
        hits.len() as f32 / FEEDBACK_RATE_WINDOW.as_secs_f32()
    }
}

/// Sliding-window event-rate counter (Hz over [`FEEDBACK_RATE_WINDOW`]). Used
/// for the client's own tx / loop rates surfaced to the UI.
#[derive(Default)]
struct RateCounter {
    hits: VecDeque<Instant>,
}

impl RateCounter {
    fn tick(&mut self, now: Instant) {
        self.hits.push_back(now);
    }

    fn rate(&mut self, now: Instant) -> f32 {
        let cutoff = now.checked_sub(FEEDBACK_RATE_WINDOW).unwrap_or(now);
        while self.hits.front().is_some_and(|t| *t < cutoff) {
            self.hits.pop_front();
        }
        self.hits.len() as f32 / FEEDBACK_RATE_WINDOW.as_secs_f32()
    }
}

/// List available serial ports. The port names can be used to create a
/// [`BasestationClient`].
pub fn list_serial_ports() -> Vec<String> {
    Serial::list_ports(true)
}

#[derive(Debug, Clone)]
/// Configuration for the serial client.
pub struct BasestationClientConfig {
    /// The name of the serial port. Use [`list_serial_ports`] to get a list of
    /// available ports.
    port_name: String,
    /// Map of player IDs to robot ids
    robot_id_map: HashMap<(TeamColor, PlayerId), u32>,
}

impl BasestationClientConfig {
    pub fn new(port: String, robot_id_map: HashMap<(TeamColor, PlayerId), u32>) -> Self {
        BasestationClientConfig {
            port_name: port,
            robot_id_map,
        }
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

enum Message {
    PlayerCmd((TeamColor, PlayerCmd)),
    ChangeIdMap(HashMap<(TeamColor, PlayerId), u32>),
    Bench(BenchOp),
}

/// A low-level bench operation, addressing robots by raw robot id and bypassing
/// the `(team, player) -> robot_id` map. Used only by the webui test bench.
enum BenchOp {
    /// Stream a movement setpoint to a raw robot id. The trailing `u16` is the
    /// raw kick duration (`kick_time_i`, ms) stamped onto the packet, so a
    /// streamed reflex-arm carries the kick strength to fire with.
    SendRaw(u8, PlayerCmd, u16),
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

enum GlueRadioCommand {
    Local(glue::Radio_Command),
    Global(glue::Radio_GlobalCommand),
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

        let mut bs = BaseStation::new(&port_name)?;
        let result = tokio::task::spawn_blocking(move || {
            let interval = Duration::from_secs_f64(1.0 / DEFAULT_BASE_STATION_READ_FREQ);
            let mut id_map = config.robot_id_map;

            // Scratch buffer for `read_and_parse` (holds the config-variable
            // returns we read firmware versions out of).
            let mut debug = glue::Debug::new();
            let mut last_connected = std::time::Instant::now();
            let mut reconnecting = false;
            let mut last_reconnect_attempt: Option<std::time::Instant> = None;
            let mut rate_tracker = FeedbackRateTracker::default();
            // Latest command per robot + when it arrived (for age-out). Persistent
            // across loops; one robot is polled per loop, round-robin.
            let mut commands_by_robot: HashMap<u8, (Instant, GlueRadioCommand)> =
                HashMap::with_capacity(6);
            // The client's own link metrics, surfaced to the UI via base info.
            let mut tx_counter = RateCounter::default();
            let mut loop_counter = RateCounter::default();
            let mut current_base_info = BaseStationInfo::disconnected();
            // Round-robin cursor so every robot gets polled across successive loops.
            let mut rr_cursor = 0usize;

            'outer: loop {
                // Send commands
                let send_start = std::time::Instant::now();
                loop_counter.tick(send_start);

                'inner: loop {
                    match cmd_rx.try_recv() {
                        Ok(Message::PlayerCmd((team_color, cmd))) => match cmd {
                            PlayerCmd::Move(cmd) => {
                                let id = id_map
                                    .get(&(team_color, cmd.id))
                                    .copied()
                                    .unwrap_or_else(|| cmd.id.as_u32())
                                    as u8;
                                let glue_cmd: glue::Radio_Command = cmd.into();
                                commands_by_robot
                                    .insert(id, (Instant::now(), GlueRadioCommand::Local(glue_cmd)));
                            }
                            PlayerCmd::GlobalMove(cmd) => {
                                let id = id_map
                                    .get(&(team_color, cmd.id))
                                    .copied()
                                    .unwrap_or_else(|| cmd.id.as_u32())
                                    as u8;
                                let glue_cmd: glue::Radio_GlobalCommand = cmd.into();
                                commands_by_robot.insert(
                                    id,
                                    (Instant::now(), GlueRadioCommand::Global(glue_cmd)),
                                );
                            }
                        },
                        Ok(Message::ChangeIdMap(new_id_map)) => {
                            id_map = new_id_map;
                        }
                        Ok(Message::Bench(op)) => dispatch_bench_op(&mut bs, op),
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            break 'outer;
                        }
                        Err(mpsc::error::TryRecvError::Empty) => break 'inner,
                    }
                }

                // Drop robots whose latest command has gone stale, then poll ONE
                // robot per loop, round-robin, from the persistent map (the newest
                // command per robot is re-sent until superseded). One poll + one
                // read per loop keeps the half-duplex (old-firmware) link healthy;
                // on fast firmware the loop just runs much faster.
                commands_by_robot.retain(|_, (t, _)| t.elapsed() < COMMAND_MAX_AGE);
                if !commands_by_robot.is_empty() {
                    let mut ids: Vec<u8> = commands_by_robot.keys().copied().collect();
                    ids.sort_unstable();
                    let id = ids[rr_cursor % ids.len()];
                    let _ = match &commands_by_robot[&id].1 {
                        GlueRadioCommand::Local(c) => bs.serial.send_command(id, *c),
                        GlueRadioCommand::Global(c) => bs.serial.send_global_command(id, *c),
                    };
                    tx_counter.tick(std::time::Instant::now());
                    rr_cursor = rr_cursor.wrapping_add(1);
                }

                // Receive feedback / maintain the link.
                {
                    // Pull in any pending packets; a read error means the link
                    // dropped (USB unplug, base station power-cycle, ...).
                    let mut connected = bs.read_and_parse(Some(&mut debug)).is_ok();

                    if !connected {
                        // Link is (or has just gone) down. Instead of killing the
                        // whole process, throttle reconnect attempts and keep the
                        // thread alive so we recover transparently once the base
                        // station comes back.
                        if last_connected.elapsed() >= DISCONNECT_GRACE {
                            if !reconnecting {
                                log::warn!("Base station disconnected; attempting to reconnect...");
                                reconnecting = true;
                            }
                            let due = last_reconnect_attempt
                                .map_or(true, |t| t.elapsed() >= RECONNECT_INTERVAL);
                            if due {
                                last_reconnect_attempt = Some(std::time::Instant::now());
                                if let Some(new_bs) = reconnect(&port_name) {
                                    bs = new_bs;
                                    connected = true;
                                    log::info!("Reconnected to base station");
                                } else {
                                    log::debug!(
                                        "Base station reconnect attempt failed; retrying in {:?}",
                                        RECONNECT_INTERVAL
                                    );
                                }
                            }
                        }
                        // Keep publishing link metrics while down, so the UI shows
                        // the client is alive even with no base station attached.
                        if !connected {
                            let metric_now = std::time::Instant::now();
                            current_base_info = BaseStationInfo::disconnected();
                            current_base_info.tx_hz = Some(tx_counter.rate(metric_now));
                            current_base_info.rx_hz = Some(0.0);
                            current_base_info.loop_hz = Some(loop_counter.rate(metric_now));
                            base_info_tx.send_replace(Some(current_base_info.clone()));
                        }
                        // Skip the feedback read while the link is down.
                        let elapsed = send_start.elapsed();
                        if elapsed < interval {
                            std::thread::sleep(interval - elapsed);
                        } else {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        continue 'outer;
                    }

                    // Connected: note the time and clear any reconnect state.
                    last_connected = std::time::Instant::now();
                    if reconnecting {
                        log::info!("Base station link re-established");
                        reconnecting = false;
                        last_reconnect_attempt = None;
                    }

                    let fw_versions = read_firmware_versions(&debug);

                    let rx_hz = {
                        let now = std::time::Instant::now();
                        let feedback = bs
                            .robots
                            .iter()
                            .enumerate()
                            .filter_map(|(id, msg)| {
                                if msg
                                    .time_since_status_lf_update()
                                    .unwrap_or(Duration::from_secs(600))
                                    < Duration::from_millis(600)
                                {
                                    let hf_age = msg.time_since_status_hf_update();
                                    let feedback_hz = rate_tracker.observe(id, hf_age, now);
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
                                            feedback_hz: Some(feedback_hz),
                                            feedback_age_ms: hf_age.map(|d| d.as_millis() as u32),
                                            online: Some(msg.is_online()),
                                        },
                                    ))
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>();

                        // Aggregate receive rate = sum of per-robot feedback rates.
                        let feedback_rates = feedback
                            .iter()
                            .filter_map(|(_, m)| m.feedback_hz)
                            .collect::<Vec<_>>();
                        let rx_hz =
                            feedback_rates.iter().sum::<f32>() / (feedback_rates.len() as f32);
                        for msg in feedback {
                            info_tx.send(msg).ok();
                        }
                        rx_hz
                    };

                    // Refresh the cached hardware view from the latest
                    // Base_Information packet (these arrive less often than the
                    // loop runs), then publish with fresh client-link metrics.
                    if let glue::Stamped::Have(_, bi) = &bs.base_info {
                        let radios_online = (0..bi.num_radios)
                            .map(|i| (bi.radios_online >> i) & 1 == 1)
                            .collect();
                        current_base_info.connected = true;
                        current_base_info.protocol_ok = bi.version.protcol_version_matches();
                        current_base_info.version = bi.version.version_to_string();
                        current_base_info.protocol_version =
                            bi.version.protocol_version_to_string();
                        current_base_info.channel_mhz = bi.channel as u32 + 2400;
                        current_base_info.num_radios = bi.num_radios;
                        current_base_info.radios_online = radios_online;
                        current_base_info.max_robots = bi.max_robots;
                    } else {
                        current_base_info.connected = true;
                    }
                    let metric_now = std::time::Instant::now();
                    current_base_info.tx_hz = Some(tx_counter.rate(metric_now));
                    current_base_info.rx_hz = Some(rx_hz);
                    current_base_info.loop_hz = Some(loop_counter.rate(metric_now));
                    base_info_tx.send_replace(Some(current_base_info.clone()));
                }

                // Cap the loop at the target rate, timing only this iteration's
                // work (from send_start) so the period is exactly `interval` when
                // there's slack and runs flat-out when the work alone exceeds it.
                let elapsed = send_start.elapsed();
                if elapsed < interval {
                    std::thread::sleep(interval - elapsed);
                } else {
                    std::thread::sleep(Duration::from_millis(1));
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

    pub fn send_no_wait(&mut self, team_color: TeamColor, msg: PlayerCmd) {
        self.cmd_tx
            .try_send(Message::PlayerCmd((team_color, msg)))
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
    // the (team, player) -> robot_id map. ---

    fn send_bench(&self, op: BenchOp) {
        self.cmd_tx.try_send(Message::Bench(op)).ok();
    }

    /// Stream a movement setpoint to a raw robot id (the 50 Hz bench path).
    /// `kick_time` is stamped onto every packet (`kick_time_i`, ms).
    pub fn bench_send_raw(&self, robot_id: u8, cmd: PlayerCmd, kick_time: u16) {
        self.send_bench(BenchOp::SendRaw(robot_id, cmd, kick_time));
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

/// Attempt to (re)open the base station link. Prefers the configured port, then
/// falls back to the first detected base station — the device path can change
/// across a USB re-enumeration, so the original `port_name` may no longer exist.
/// Returns the reopened link, or `None` if nothing could be opened.
fn reconnect(port_name: &str) -> Option<BaseStation> {
    if let Ok(bs) = BaseStation::new(port_name) {
        return Some(bs);
    }
    let ports = Serial::list_ports(true);
    ports.first().and_then(|p| BaseStation::new(p).ok())
}

/// Dispatch a bench operation directly over the serial link.
fn dispatch_bench_op(bs: &mut BaseStation, op: BenchOp) {
    match op {
        BenchOp::SendRaw(robot_id, cmd, kick_time) => match cmd {
            PlayerCmd::Move(c) => {
                let mut g: glue::Radio_Command = c.into();
                g.gen_command.kick_time_i = kick_time;
                bs.serial.send_command(robot_id, g).ok();
            }
            PlayerCmd::GlobalMove(c) => {
                let mut g: glue::Radio_GlobalCommand = c.into();
                g.gen_command.kick_time_i = kick_time;
                bs.serial.send_global_command(robot_id, g).ok();
            }
        },
        BenchOp::RobotCommand {
            robot_id,
            cmd,
            kick_time,
        } => {
            let cmd = build_robot_command(cmd, kick_time);
            bs.serial.send_command(robot_id, cmd).ok();
        }
        BenchOp::Broadcast(cmd) => {
            let cmd = build_robot_command(cmd, 0);
            bs.serial.send_command(glue::Radio_Broadcast_ID, cmd).ok();
        }
        BenchOp::SetHeading(robot_id, heading) => {
            let over_odo = glue::Radio_OverrideOdometry {
                _pad: [0; 12],
                _pad0: 0,
                pos_x: 0.0,
                pos_y: 0.0,
                ang_z: heading,
                set_pos_x: false,
                set_pos_y: false,
                set_ang_z: true,
            };
            bs.serial.send_over_odo(robot_id, over_odo).ok();
        }
        BenchOp::SetBaseChannel(channel) => {
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
            bs.serial.send_mcm(glue::Radio_BaseStation_ID, mcm).ok();
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
            bs.serial.send_mcm(robot_id, mcm).ok();
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
            bs.serial.send_mcm(robot_id, mcm).ok();
        }
    }
}

/// Read the per-robot firmware version from the debug buffer (populated after a
/// `GetVersion` request). Returns `None` for robots that haven't reported.
fn read_firmware_versions(debug: &glue::Debug) -> [Option<FirmwareVersion>; glue::MAX_NUM_ROBOTS] {
    let mut out = [None; glue::MAX_NUM_ROBOTS];
    let read = |id: usize, var: glue::HG_Variable| -> Option<u32> {
        match debug.config_variable_returns[id][var as usize] {
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
