use clap::ValueEnum;
use tokio::sync::broadcast;

use crate::tui_utils::CliArgs;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CliMode {
    Ui,
}

impl CliMode {
    /// Run the selected mode.
    pub async fn run(args: CliArgs, stop_rx: broadcast::Receiver<()>) {
        match args.mode {
            CliMode::Ui => dies_webui::start(args.into_ui().await, stop_rx).await,
        }
    }
}
