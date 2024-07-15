use anyhow::Result;
use clap::ValueEnum;
use tokio::sync::broadcast;

use crate::{convert_logs::convert_log, tui_utils::CliArgs};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CliMode {
    Ui,
    ConvertLog,
}

impl CliMode {
    /// Run the selected mode.
    pub async fn run(args: CliArgs, stop_rx: broadcast::Receiver<()>) -> Result<()> {
        match args.mode {
            CliMode::Ui => dies_webui::start(args.into_ui().await?, stop_rx).await,
            CliMode::ConvertLog => convert_log(&args.log_input, &args.log_output)?,
        }
        Ok(())
    }
}
