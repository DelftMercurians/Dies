use dies_protos::dies_log_line::{LogLevel, LogLine};
use log::{Log, Metadata, Record};
use protobuf::{EnumOrUnknown, Message};
use std::path::PathBuf;
use tokio::{io::AsyncWriteExt, sync::mpsc};

enum WorkerMsg {
    Log(Vec<u8>),
    Flush,
}

pub struct AsyncProtobufLogger {
    env_logger: env_logger::Logger,
    sender: mpsc::UnboundedSender<WorkerMsg>,
}

impl AsyncProtobufLogger {
    pub fn init(log_file_path: PathBuf) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        tokio::spawn(Self::run_worker(receiver, log_file_path));
        Self {
            env_logger: env_logger::Logger::from_default_env(),
            sender,
        }
    }

    async fn run_worker(mut receiver: mpsc::UnboundedReceiver<WorkerMsg>, log_file_path: PathBuf) {
        let log_file = tokio::fs::File::create(log_file_path)
            .await
            .expect("Failed to create log file");
        let mut log_file = tokio::io::BufWriter::new(log_file);
        while let Some(msg) = receiver.recv().await {
            match msg {
                WorkerMsg::Log(buf) => {
                    if let Err(e) = log_file.write_all(&buf).await {
                        eprintln!("Failed to write to log file: {}", e);
                    }
                }
                WorkerMsg::Flush => {
                    if let Err(e) = log_file.flush().await {
                        eprintln!("Failed to flush log file: {}", e);
                    }
                }
            }
            let _ = log_file.flush().await;
        }
    }
}

impl Drop for AsyncProtobufLogger {
    fn drop(&mut self) {
        self.flush();
    }
}

impl Log for AsyncProtobufLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.target().starts_with("dies")
    }

    fn log(&self, record: &Record) {
        // Log to env_logger
        self.env_logger.log(record);

        // Serialize log line
        let log_line = LogLine {
            level: Some(EnumOrUnknown::new(match record.level() {
                log::Level::Error => LogLevel::ERROR,
                log::Level::Warn => LogLevel::WARN,
                log::Level::Info => LogLevel::INFO,
                log::Level::Debug => LogLevel::DEBUG,
                log::Level::Trace => LogLevel::TRACE,
            })),
            target: Some(record.target().to_string()),
            message: Some(format!("{}", record.args())),
            source: Some(format!(
                "{}:{}",
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0)
            )),
            ..Default::default()
        };
        let mut buf = Vec::new();
        log_line.write_to_vec(&mut buf).unwrap();
        self.sender.send(WorkerMsg::Log(buf)).unwrap();
    }

    fn flush(&self) {
        self.env_logger.flush();
        self.sender.send(WorkerMsg::Flush).unwrap();
    }
}
