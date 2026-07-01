use std::collections::HashMap;

use anyhow::Result;
use dies_basestation_client::{BasestationClientConfig, BasestationHandle};
use dies_core::{
    Angle, PlayerFeedbackMsg, PlayerGlobalMoveCmd, PlayerId, RobotCmd, RotationDirection,
};
use tokio::time::{Duration, Instant};

use crate::cli::SerialPort;

/// Per-robot feedback integrity tally, accumulated from the feedback stream.
#[derive(Default)]
struct RobotStat {
    /// Feedback frames forwarded to us since the last 1 s report (→ the
    /// basestation *forwarding* rate, which re-emits the cached record every
    /// loop). This is NOT the true radio frame rate.
    interval_count: u32,
    /// Frames since the last report where `feedback_age_ms` *dropped* — a reset
    /// of the age toward 0 is an unambiguous new-frame edge (age is monotonic
    /// non-decreasing while the same frame is re-forwarded). This IS the true
    /// per-robot RF frame rate — compare it against `interval_count`.
    fresh_count: u32,
    /// Previous `feedback_age_ms`, to detect the reset edge.
    prev_age_ms: Option<u32>,
    total_count: u64,
    last_recv: Option<Instant>,
    /// Worst / best inter-arrival gap seen this window (ms).
    max_gap_ms: u128,
    min_gap_ms: Option<u128>,
    /// Worst / best `feedback_age_ms` seen this window (ms) — staleness range.
    max_age_ms: u32,
    min_age_ms: Option<u32>,
    online: bool,
    /// Whether the robot was ever reported offline (link flap).
    saw_offline: bool,
    /// Latest full feedback frame, for the per-robot telemetry dump.
    last: Option<PlayerFeedbackMsg>,
}

fn opt<T: std::fmt::Debug>(v: &Option<T>) -> String {
    v.as_ref()
        .map_or_else(|| "—".to_string(), |x| format!("{:?}", x))
}

fn optf(v: &Option<f32>) -> String {
    v.map_or_else(|| "—".to_string(), |x| format!("{:.2}", x))
}

fn optu(v: &Option<u32>) -> String {
    v.map_or_else(|| "—".to_string(), |x| x.to_string())
}

/// Compact fixed-precision formatter for optional f32 arrays.
fn farr<const N: usize>(v: &Option<[f32; N]>) -> String {
    v.as_ref().map_or_else(
        || "—".to_string(),
        |a| {
            let parts: Vec<String> = a.iter().map(|x| format!("{:.1}", x)).collect();
            format!("[{}]", parts.join(","))
        },
    )
}

pub async fn test_radio(
    port: SerialPort,
    ids: Vec<u32>,
    duration: f64,
    w: Option<f64>,
    sx: Option<f64>,
    sy: Option<f64>,
    max_yaw_rate: f64,
    preferred_rotation_direction: f64,
    kick: bool,
) -> Result<()> {
    // test-radio has no logger of its own (unlike start-ui / test-vision), so
    // install one here — otherwise the basestation client's log::* (reconnect
    // warnings etc.) and the report below are silently dropped. Honours RUST_LOG.
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    let port = port.select().await?;
    let bs_config = BasestationClientConfig::new(port, HashMap::new());
    let mut bs_handle = BasestationHandle::spawn(bs_config)?;
    // Separate clone for draining feedback so the send/report arms can still use
    // `bs_handle` without a borrow conflict in the select.
    let mut reader = bs_handle.clone();

    assert!(!ids.is_empty(), "No IDs provided");

    let mut send_tick = tokio::time::interval(Duration::from_secs_f64(1.0 / 30.0));
    let mut report_tick = tokio::time::interval(Duration::from_secs(1));
    report_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let start = Instant::now();
    let mut last_report = Instant::now();
    let mut stats: HashMap<u32, RobotStat> = HashMap::new();

    loop {
        tokio::select! {
            // Command stream to every target robot at 30 Hz.
            _ = send_tick.tick() => {
                if start.elapsed().as_secs_f64() >= duration {
                    break;
                }
                let elapsed = start.elapsed().as_secs_f64();
                for id in ids.iter() {
                    let mut cmd = PlayerGlobalMoveCmd::zero(PlayerId::new(*id));
                    if let Some(w) = w {
                        cmd.heading_setpoint = Angle::from_degrees(w).radians();
                    }
                    if let Some(sx) = sx {
                        cmd.global_x = sx;
                    }
                    if let Some(sy) = sy {
                        cmd.global_y = sy;
                    }
                    cmd.max_yaw_rate = max_yaw_rate;
                    cmd.preferred_rotation_direction =
                        RotationDirection::from_f64(preferred_rotation_direction);
                    cmd.last_heading = f64::NAN;
                    cmd.kick_counter = 0;
                    cmd.robot_cmd = RobotCmd::Arm;
                    if kick && elapsed > 2.0 {
                        cmd.kick_counter = 1;
                    }
                    bs_handle.send_no_wait(
                        dies_core::TeamColor::Blue,
                        dies_core::PlayerCmd::GlobalMove(cmd),
                    );
                }
            }

            // Drain feedback and tally per-robot integrity.
            msg = reader.recv() => {
                if let Ok((_, fb)) = msg {
                    let now = Instant::now();
                    let s = stats.entry(fb.id.as_u32()).or_default();
                    if let Some(last) = s.last_recv {
                        let gap = (now - last).as_millis();
                        s.max_gap_ms = s.max_gap_ms.max(gap);
                        s.min_gap_ms = Some(s.min_gap_ms.map_or(gap, |m| m.min(gap)));
                    }
                    s.last_recv = Some(now);
                    s.interval_count += 1;
                    s.total_count += 1;
                    if let Some(age) = fb.feedback_age_ms {
                        s.max_age_ms = s.max_age_ms.max(age);
                        s.min_age_ms = Some(s.min_age_ms.map_or(age, |m| m.min(age)));
                        // Fresh-frame detection by reset edge: the basestation
                        // re-forwards the cached record with a monotonically
                        // growing age until a new RF frame lands and resets it.
                        // Each downward step in age is exactly one new frame.
                        // (Reconstructing recv-age instead double-counts, because
                        // ms-quantized age crosses any sub-ms gate ~once/frame.)
                        if s.prev_age_ms.is_some_and(|pa| age < pa) {
                            s.fresh_count += 1;
                        }
                        s.prev_age_ms = Some(age);
                    }
                    s.online = fb.online.unwrap_or(false);
                    if !s.online {
                        s.saw_offline = true;
                    }
                    s.last = Some(fb);
                }
            }

            // Per-second report: link rates + per-robot feedback health.
            _ = report_tick.tick() => {
                let dt = last_report.elapsed().as_secs_f64();
                last_report = Instant::now();
                let t = start.elapsed().as_secs_f64();
                match bs_handle.base_info() {
                    Some(b) => {
                        let radios_up = b.radios_online.iter().filter(|x| **x).count();
                        log::info!(
                            "--- t={:.0}s | conn={} proto_ok={} fw={} proto={} | ch={}MHz radios={}/{}{:?} max_robots={} | tx={:.0} rx={:.0} loop={:.0} Hz ---",
                            t, b.connected, b.protocol_ok, b.version, b.protocol_version,
                            b.channel_mhz, radios_up, b.num_radios, b.radios_online, b.max_robots,
                            b.tx_hz.unwrap_or(0.0), b.rx_hz.unwrap_or(0.0), b.loop_hz.unwrap_or(0.0),
                        );
                    }
                    None => log::info!("--- t={:.0}s | link: <no base_info / disconnected> ---", t),
                }
                let mut robot_ids: Vec<u32> = stats.keys().copied().collect();
                robot_ids.sort_unstable();
                for id in robot_ids {
                    let s = stats.get_mut(&id).unwrap();
                    let recv_hz = s.interval_count as f64 / dt;
                    let fresh_hz = s.fresh_count as f64 / dt;
                    let bs_hz = s.last.and_then(|m| m.feedback_hz).unwrap_or(0.0);
                    log::info!(
                        "  robot {:>2}: recv {:>5.0} Hz | fresh {:>4.0} Hz (bs {:>4.0}) | gap {:>3}-{:<4} ms | age {:>3}-{:<4} ms | online={} | total={}",
                        id, recv_hz, fresh_hz, bs_hz,
                        s.min_gap_ms.unwrap_or(0), s.max_gap_ms,
                        s.min_age_ms.unwrap_or(0), s.max_age_ms,
                        s.online, s.total_count,
                    );
                    // if let Some(m) = s.last {
                    //     log::info!(
                    //         "       fw={} status prim={} kick={} imu={} tof={} | cap={}V temp={}C pack={}V mainI={}A | rloop avg={}us max={}us",
                    //         opt(&m.firmware_version),
                    //         opt(&m.primary_status), opt(&m.kicker_status), opt(&m.imu_status), opt(&m.tof_status),
                    //         optf(&m.kicker_cap_voltage), optf(&m.kicker_temp), farr(&m.pack_voltages), optf(&m.main_board_current),
                    //         optu(&m.avg_loop_time_us), optu(&m.max_loop_time_us),
                    //     );
                    //     log::info!(
                    //         "       motors st={} spd={} temp={} curr={}",
                    //         opt(&m.motor_statuses), farr(&m.motor_speeds), farr(&m.motor_temps), farr(&m.motor_currents),
                    //     );
                    //     log::info!(
                    //         "       bb det={} ok={} raw={} | tof det={} xy={} | imu={} | smartkick={} kickok={} reflex={} cnt={} | lastcmd={}",
                    //         opt(&m.breakbeam_ball_detected), opt(&m.breakbeam_sensor_ok), opt(&m.breakbeam_raw),
                    //         opt(&m.tof_ball_detected), opt(&m.tof_xy), farr(&m.imu_readings),
                    //         opt(&m.smart_kick_counter), opt(&m.kick_ok_flag), opt(&m.reflex_kick_state), opt(&m.reflex_kick_counter),
                    //         opt(&m.last_command),
                    //     );
                    // }
                    s.interval_count = 0;
                    s.fresh_count = 0;
                }

                // Clear per-window range trackers.
                for s in stats.values_mut() {
                    s.max_gap_ms = 0;
                    s.min_gap_ms = None;
                    s.max_age_ms = 0;
                    s.min_age_ms = None;
                }
            }
        }
    }

    // Final summary.
    let total_s = start.elapsed().as_secs_f64();
    log::info!("=== stress summary ({:.1}s) ===", total_s);
    let mut robot_ids: Vec<u32> = stats.keys().copied().collect();
    robot_ids.sort_unstable();
    for id in &robot_ids {
        let s = &stats[id];
        log::info!(
            "  robot {:>2}: total={} | avg {:>5.1} Hz | worst_gap {} ms | worst_age {} ms | ever_offline={}",
            id, s.total_count, s.total_count as f64 / total_s, s.max_gap_ms, s.max_age_ms, s.saw_offline,
        );
    }
    for id in ids.iter() {
        if !stats.contains_key(id) {
            log::warn!("  robot {:>2}: NO FEEDBACK EVER RECEIVED", id);
        }
    }

    Ok(())
}
