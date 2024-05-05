use anyhow::Result;
use dies_executor::{
    strategy::{AdHocStrategy, Strategy},
    Executor,
};
use tokio::sync::broadcast;

use super::utils::setup_vision_and_serial;

pub async fn run(args: crate::Args, stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let (vision, serial) = setup_vision_and_serial(&args).await?;

    let strategy = AdHocStrategy::new();

    let mut builder = Executor::builder();
    builder.with_strategy(Box::new(strategy) as Box<dyn Strategy>);
    builder.with_bs_client(serial.ok_or(anyhow::anyhow!("Serial client is required"))?);
    builder.with_ssl_client(vision);
    let executor = builder.build()?;

    executor.run_real_time(stop_rx).await
}
