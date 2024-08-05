use anyhow::Result;
use dies_basestation_client::{BasestationClientConfig, BasestationHandle};
use dies_core::{PlayerId, PlayerMoveCmd};
use tokio::time::{Duration, Instant};

use crate::cli::SerialPort;

pub async fn test_radio(
    port: SerialPort,
    ids: Vec<u32>,
    duration: f64,
    w: Option<f64>,
    sx: Option<f64>,
    sy: Option<f64>,
) -> Result<()> {
    let port = port.select().await?;
    let bs_config =
        BasestationClientConfig::new(port, dies_basestation_client::BaseStationProtocol::V1);
    let mut bs_handle = BasestationHandle::spawn(bs_config)?;

    let mut interval = tokio::time::interval(Duration::from_secs_f64(1.0 / 30.0));
    let start = Instant::now();
    loop {
        interval.tick().await;

        let elapsed = start.elapsed().as_secs_f64();
        if elapsed >= duration {
            break;
        }

        for id in ids.iter() {
            let mut cmd = PlayerMoveCmd::zero(PlayerId::new(*id));
            if let Some(w) = w {
                cmd.w = w;
            }
            if let Some(sx) = sx {
                cmd.sx = sx;
            }
            if let Some(sy) = sy {
                cmd.sy = sy;
            }

            println!("Sending {:?}", cmd);
            bs_handle.send_no_wait(dies_core::PlayerCmd::Move(cmd));
        }
    }

    Ok(())
}
