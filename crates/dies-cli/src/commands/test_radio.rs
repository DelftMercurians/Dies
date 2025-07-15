use anyhow::Result;
use dies_basestation_client::{BasestationClientConfig, BasestationHandle};
use dies_core::{Angle, PlayerGlobalMoveCmd, PlayerId, RobotCmd, RotationDirection};
use tokio::time::{Duration, Instant};

use crate::cli::SerialPort;

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
    let port = port.select().await?;
    let bs_config =
        BasestationClientConfig::new(port, dies_basestation_client::BaseStationProtocol::V1);
    let mut bs_handle = BasestationHandle::spawn(bs_config)?;

    assert!(ids.len() > 0, "No IDs provided");

    let mut interval = tokio::time::interval(Duration::from_secs_f64(1.0 / 30.0));
    let start = Instant::now();

    loop {
        interval.tick().await;

        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= duration {
            break;
        }

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

    Ok(())
}
