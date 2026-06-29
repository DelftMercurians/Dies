use std::collections::HashMap;

use anyhow::Result;
use dies_basestation_client::{BasestationClientConfig, BasestationHandle};
use dies_core::{Angle, PlayerGlobalMoveCmd, PlayerId, RobotCmd, RotationDirection};
use tokio::time::{Duration, Instant};

use crate::cli::SerialPort;

/// Per-robot feedback integrity tally, accumulated from the feedback stream.
#[derive(Default)]
struct RobotStat {
    /// Frames since the last 1 s report (→ measured feedback Hz).
    interval_count: u32,
    total_count: u64,
    last_recv: Option<Instant>,
    /// Worst inter-arrival gap seen (ms) — stall / loss indicator.
    max_gap_ms: u128,
    /// Worst `feedback_age_ms` seen — staleness indicator.
    max_age_ms: u32,
    online: bool,
    /// Whether the robot was ever reported offline (link flap).
    saw_offline: bool,
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
                        s.max_gap_ms = s.max_gap_ms.max((now - last).as_millis());
                    }
                    s.last_recv = Some(now);
                    s.interval_count += 1;
                    s.total_count += 1;
                    if let Some(age) = fb.feedback_age_ms {
                        s.max_age_ms = s.max_age_ms.max(age);
                    }
                    s.online = fb.online.unwrap_or(false);
                    if !s.online {
                        s.saw_offline = true;
                    }
                }
            }

            // Per-second report: link rates + per-robot feedback health.
            _ = report_tick.tick() => {
                let dt = last_report.elapsed().as_secs_f64();
                last_report = Instant::now();
                let bi = bs_handle.base_info();
                let (tx, loop_hz, conn) = bi.as_ref().map_or((0.0, 0.0, false), |b| {
                    (b.tx_hz.unwrap_or(0.0), b.loop_hz.unwrap_or(0.0), b.connected)
                });
                log::info!(
                    "--- t={:.0}s | link tx={:.0} loop={:.0} Hz connected={} ---",
                    start.elapsed().as_secs_f64(), tx, loop_hz, conn,
                );
                let mut robot_ids: Vec<u32> = stats.keys().copied().collect();
                robot_ids.sort_unstable();
                for id in robot_ids {
                    let s = stats.get_mut(&id).unwrap();
                    let hz = s.interval_count as f64 / dt;
                    log::info!(
                        "  robot {:>2}: {:>5.1} Hz | max_gap {:>4} ms | max_age {:>4} ms | online={} | total={}",
                        id, hz, s.max_gap_ms, s.max_age_ms, s.online, s.total_count,
                    );
                    s.interval_count = 0;
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
