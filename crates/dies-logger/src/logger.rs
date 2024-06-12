use dies_protos::{
    dies_log_line::{LogLevel, LogLine},
    ssl_gc_referee_message::Referee,
    ssl_vision_wrapper::SSL_WrapperPacket,
};
use log::{Log, Metadata, Record};
use protobuf::{EnumOrUnknown, Message};
use std::{path::PathBuf, sync::OnceLock, thread};
use tokio::sync::mpsc;

use crate::log_codec::{LogFileWriter, LogMessage};

static PROTOBUF_LOGGER: OnceLock<AsyncProtobufLogger> = OnceLock::new();
const LOG_VERSION: u32 = 1;

enum WorkerMsg {
    Log(LogMessage),
    Flush,
}

pub fn log_vision(data: &SSL_WrapperPacket) {
    if let Some(logger) = PROTOBUF_LOGGER.get() {
        logger
            .sender
            .send(WorkerMsg::Log(LogMessage::Vision(data.clone())))
            .unwrap();
    }
}

pub fn log_referee(data: &Referee) {
    if let Some(logger) = PROTOBUF_LOGGER.get() {
        logger
            .sender
            .send(WorkerMsg::Log(LogMessage::Referee(data.clone())))
            .unwrap();
    }
}

pub fn log_bytes(bytes: &[u8]) {
    if let Some(logger) = PROTOBUF_LOGGER.get() {
        logger
            .sender
            .send(WorkerMsg::Log(LogMessage::Bytes(bytes.to_vec())))
            .unwrap();
    }
}

pub struct AsyncProtobufLogger {
    env_logger: env_logger::Logger,
    sender: mpsc::UnboundedSender<WorkerMsg>,
}

impl AsyncProtobufLogger {
    /// Initialize the logger with the given log file path and stdout logger.
    pub fn init_with_env_logger(log_file_path: PathBuf, env: env_logger::Logger) -> &'static Self {
        PROTOBUF_LOGGER.get_or_init(|| {
            let (sender, receiver) = mpsc::unbounded_channel();
            thread::spawn(|| Self::run_worker(receiver, log_file_path));
            Self {
                env_logger: env,
                sender,
            }
        })
    }

    /// Initialize the logger with the given log file path and the default environment for the stdout logger.
    pub fn init(log_file_path: PathBuf) -> &'static Self {
        Self::init_with_env_logger(log_file_path, env_logger::Logger::from_default_env())
    }

    fn run_worker(mut receiver: mpsc::UnboundedReceiver<WorkerMsg>, log_file_path: PathBuf) {
        let mut log_file = match LogFileWriter::open(log_file_path, LOG_VERSION) {
            Ok(file) => file,
            Err(e) => {
                eprintln!("Failed to open log file: {}", e);
                return;
            }
        };

        while let Some(msg) = receiver.blocking_recv() {
            match msg {
                WorkerMsg::Log(msg) => {
                    if let Err(e) = log_file.write_log_message(&msg) {
                        eprintln!("Failed to write to log file: {}", e);
                    }
                }
                WorkerMsg::Flush => {
                    if let Err(e) = log_file.flush() {
                        eprintln!("Failed to flush log file: {}", e);
                    }
                }
            }
            // TODO: This is a temporary workaround to ensure that the log file is flushed
            let _ = log_file.flush();
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
        self.sender
            .send(WorkerMsg::Log(LogMessage::DiesLog(log_line)))
            .unwrap();
    }

    fn flush(&self) {
        self.env_logger.flush();
        self.sender.send(WorkerMsg::Flush).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use log::Log;
    use tempfile::NamedTempFile;

    use crate::{LogFile, LogMessage};

    #[tokio::test]
    async fn one_message() {
        let temp = NamedTempFile::new().unwrap();
        if temp.path().exists() {
            std::fs::remove_file(temp.path()).unwrap();
        }

        let logger = super::AsyncProtobufLogger::init(temp.path().into());
        log::set_logger(logger).unwrap();
        log::set_max_level(log::LevelFilter::Trace);

        log::info!("testing!");

        // Wait for the log message to be written
        logger.flush();
        tokio::time::sleep(Duration::from_millis(10)).await;

        let logfile = LogFile::open(temp.path()).unwrap();
        assert!(logfile.messages().len() == 1);
        match &logfile.messages()[0] {
            LogMessage::DiesLog(line) => assert_eq!(line.message, Some("testing!".into())),
            _ => panic!("unexpected message type"),
        }
    }
}
