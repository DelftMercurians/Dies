//! Test-bench task: drives physical robots straight through the basestation,
//! bypassing the executor, tracker, controllers and vision entirely. Robots are
//! addressed by raw robot id. Sending is gated off while a Live executor runs;
//! telemetry caching always runs.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use dies_basestation_client::BasestationHandle;
use dies_core::{
    BenchCommand, BenchMotionMode, BenchOneShot, PlayerCmd, PlayerGlobalMoveCmd, PlayerId,
    PlayerMoveCmd, RobotCmd, RotationDirection,
};
use tokio::sync::broadcast;

use crate::{server::ServerState, ExecutorStatus, UiCommand, UiEnvironment, UiMode};

/// Streaming rate for taken robots.
const STREAM_HZ: f64 = 50.0;
/// If a taken robot's setpoint hasn't been refreshed within this window, stream
/// zero velocity (watchdog against a stuck/closed UI).
const SETPOINT_TIMEOUT: Duration = Duration::from_millis(300);
/// Yaw-rate cap for global-frame bench driving (maps to 10000 in firmware units).
const BENCH_MAX_YAW_RATE: f64 = 1000.0;

/// Per-robot streaming state.
struct RobotBench {
    taken: bool,
    setpoint: PlayerCmd,
    last_update: Instant,
}

impl RobotBench {
    fn new(robot_id: u32) -> Self {
        Self {
            taken: false,
            setpoint: PlayerCmd::Move(PlayerMoveCmd::zero(PlayerId::new(robot_id))),
            last_update: Instant::now(),
        }
    }
}

pub async fn run(
    state: Arc<ServerState>,
    env: UiEnvironment,
    mut cmd_rx: broadcast::Receiver<UiCommand>,
    mut shutdown_rx: broadcast::Receiver<()>,
) {
    let mut bs_handle = match env {
        UiEnvironment::WithLive { bs_handle, .. } => bs_handle,
        UiEnvironment::SimulationOnly => {
            // No hardware: nothing to stream. Just wait for shutdown so the join
            // in `start` completes cleanly.
            let _ = shutdown_rx.recv().await;
            return;
        }
    };

    let mut robots: HashMap<u32, RobotBench> = HashMap::new();
    let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / STREAM_HZ));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            // Telemetry from robots -> cache (always, even when sending is gated).
            msg = bs_handle.recv() => match msg {
                Ok((color, msg)) => {
                    state
                        .basestation_feedback
                        .write()
                        .unwrap()
                        .insert((color, msg.id), msg);
                }
                Err(_) => {
                    log::info!("Shutdown: bench task basestation channel closed");
                    break;
                }
            },
            // UI commands. Only `Bench` variants are handled here.
            cmd = cmd_rx.recv() => match cmd {
                Ok(UiCommand::Bench(bench_cmd)) => {
                    handle_bench_command(&bs_handle, &state, &mut robots, bench_cmd);
                }
                Ok(_) => {}
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(_)) => {}
            },
            // Stream the current setpoints to all taken robots.
            _ = tick.tick() => {
                // Refresh the cached base info regardless of gating.
                if let Some(info) = bs_handle.base_info() {
                    *state.base_info.write().unwrap() = Some(info);
                }
                if sending_blocked(&state) {
                    continue;
                }
                for (robot_id, rb) in robots.iter() {
                    if !rb.taken {
                        continue;
                    }
                    let cmd = if rb.last_update.elapsed() > SETPOINT_TIMEOUT {
                        zero_like(&rb.setpoint, *robot_id)
                    } else {
                        rb.setpoint
                    };
                    bs_handle.bench_send_raw(*robot_id as u8, cmd);
                }
            },
            _ = shutdown_rx.recv() => {
                log::info!("Shutdown: bench task got stop signal, exiting");
                break;
            }
        }
    }
}

/// True if bench sending must be suppressed (a Live executor is running).
fn sending_blocked(state: &ServerState) -> bool {
    let status = state.ui_status();
    status.ui_mode == UiMode::Live && matches!(status.executor, ExecutorStatus::RunningExecutor)
}

fn handle_bench_command(
    bs_handle: &BasestationHandle,
    state: &ServerState,
    robots: &mut HashMap<u32, RobotBench>,
    cmd: BenchCommand,
) {
    // Setpoint bookkeeping (TakeControl/Stop/StopAll/SetMotion) is allowed even
    // while gated so the UI state stays consistent; actually emitting to the
    // wire is gated in the stream tick and the one-shot dispatch below.
    match cmd {
        BenchCommand::TakeControl { robot_id, taken } => {
            let rb = robots
                .entry(robot_id)
                .or_insert_with(|| RobotBench::new(robot_id));
            rb.taken = taken;
            rb.setpoint = PlayerCmd::Move(PlayerMoveCmd::zero(PlayerId::new(robot_id)));
            rb.last_update = Instant::now();
            if !taken && !sending_blocked(state) {
                // Push a final zero so the robot halts immediately on release.
                bs_handle.bench_send_raw(
                    robot_id as u8,
                    PlayerCmd::Move(PlayerMoveCmd::zero(PlayerId::new(robot_id))),
                );
            }
        }
        BenchCommand::SetMotion {
            robot_id,
            mode,
            vx,
            vy,
            w_or_heading,
            dribble_speed,
        } => {
            let rb = robots
                .entry(robot_id)
                .or_insert_with(|| RobotBench::new(robot_id));
            rb.setpoint = build_setpoint(robot_id, mode, vx, vy, w_or_heading, dribble_speed);
            rb.last_update = Instant::now();
        }
        BenchCommand::Stop { robot_id } => {
            if let Some(rb) = robots.get_mut(&robot_id) {
                rb.setpoint = zero_like(&rb.setpoint, robot_id);
                rb.last_update = Instant::now();
            }
            if !sending_blocked(state) {
                bs_handle.bench_send_raw(
                    robot_id as u8,
                    PlayerCmd::Move(PlayerMoveCmd::zero(PlayerId::new(robot_id))),
                );
            }
        }
        BenchCommand::StopAll => {
            for (robot_id, rb) in robots.iter_mut() {
                rb.taken = false;
                rb.setpoint = PlayerCmd::Move(PlayerMoveCmd::zero(PlayerId::new(*robot_id)));
                if !sending_blocked(state) {
                    bs_handle.bench_send_raw(
                        *robot_id as u8,
                        PlayerCmd::Move(PlayerMoveCmd::zero(PlayerId::new(*robot_id))),
                    );
                }
            }
        }
        BenchCommand::OneShot { robot_id, kind } => {
            if sending_blocked(state) {
                return;
            }
            dispatch_one_shot(bs_handle, robot_id as u8, kind);
        }
        BenchCommand::Broadcast { kind } => {
            if sending_blocked(state) {
                return;
            }
            if let Some(robot_cmd) = one_shot_robot_cmd(kind) {
                bs_handle.bench_broadcast(robot_cmd);
            } else {
                log::warn!("Bench broadcast unsupported for {:?}", kind);
            }
        }
        BenchCommand::SetChannel { robot_id, channel } => {
            if sending_blocked(state) {
                return;
            }
            match robot_id {
                Some(id) => bs_handle.bench_set_robot_channel(id as u8, channel),
                None => bs_handle.bench_set_base_channel(channel),
            }
        }
    }
}

/// Build a streamed setpoint from a `SetMotion` command.
fn build_setpoint(
    robot_id: u32,
    mode: BenchMotionMode,
    vx: f64,
    vy: f64,
    w_or_heading: f64,
    dribble_speed: f64,
) -> PlayerCmd {
    let id = PlayerId::new(robot_id);
    match mode {
        BenchMotionMode::Local => PlayerCmd::Move(PlayerMoveCmd {
            id,
            sx: vx,
            sy: vy,
            w: w_or_heading,
            dribble_speed,
            robot_cmd: RobotCmd::None,
            fan_speed: 0.0,
            kick_speed: 0.0,
        }),
        BenchMotionMode::Global => PlayerCmd::GlobalMove(PlayerGlobalMoveCmd {
            id,
            global_x: vx,
            global_y: vy,
            heading_setpoint: w_or_heading,
            last_heading: f64::NAN,
            dribble_speed,
            kick_counter: 0,
            robot_cmd: RobotCmd::None,
            w: 0.0,
            max_yaw_rate: BENCH_MAX_YAW_RATE,
            preferred_rotation_direction: RotationDirection::NoPreference,
        }),
    }
}

/// A zeroed setpoint of the same frame as `cmd`.
fn zero_like(cmd: &PlayerCmd, robot_id: u32) -> PlayerCmd {
    let id = PlayerId::new(robot_id);
    match cmd {
        PlayerCmd::Move(_) => PlayerCmd::Move(PlayerMoveCmd::zero(id)),
        PlayerCmd::GlobalMove(_) => PlayerCmd::GlobalMove(PlayerGlobalMoveCmd::zero(id)),
    }
}

fn dispatch_one_shot(bs_handle: &BasestationHandle, robot_id: u8, kind: BenchOneShot) {
    match kind {
        BenchOneShot::Kick { speed } => {
            bs_handle.bench_robot_command(robot_id, RobotCmd::Kick, speed)
        }
        BenchOneShot::GetVersion => bs_handle.bench_get_version(robot_id),
        BenchOneShot::ZeroHeading => bs_handle.bench_set_heading(robot_id, 0.0),
        BenchOneShot::SetHeading { angle } => bs_handle.bench_set_heading(robot_id, angle as f32),
        other => {
            if let Some(robot_cmd) = one_shot_robot_cmd(other) {
                bs_handle.bench_robot_command(robot_id, robot_cmd, 0.0);
            }
        }
    }
}

/// Map the `RobotCmd`-expressible one-shots. Returns `None` for actions that go
/// through a dedicated path (kick/version/heading).
fn one_shot_robot_cmd(kind: BenchOneShot) -> Option<RobotCmd> {
    Some(match kind {
        BenchOneShot::Beep => RobotCmd::Beep,
        BenchOneShot::Reboot => RobotCmd::Reboot,
        BenchOneShot::Shutdown => RobotCmd::PowerBoardOff,
        BenchOneShot::Coast => RobotCmd::Coast,
        BenchOneShot::Arm => RobotCmd::Arm,
        BenchOneShot::Disarm => RobotCmd::Disarm,
        BenchOneShot::Discharge => RobotCmd::Discharge,
        BenchOneShot::ArmReflex => RobotCmd::ArmReflex,
        BenchOneShot::CalibrateBreakbeam => RobotCmd::CalibrateBreakbeam,
        BenchOneShot::CalibrateImu => RobotCmd::CalibrateImu,
        BenchOneShot::Kick { .. }
        | BenchOneShot::GetVersion
        | BenchOneShot::ZeroHeading
        | BenchOneShot::SetHeading { .. } => return None,
    })
}
