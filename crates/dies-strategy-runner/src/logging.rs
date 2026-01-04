//! Log forwarding for strategy processes.
//!
//! Captures log messages from strategy code and forwards them to the host
//! via IPC. Also sets up local logging for debugging.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;

use dies_strategy_protocol::{LogLevel, StrategyMessage};
use tracing::{debug, Level};
use tracing_subscriber::EnvFilter;

/// A log message captured from strategy code.
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
}

/// Global channel for log messages to be forwarded to host.
static LOG_SENDER: Mutex<Option<Sender<LogEntry>>> = Mutex::new(None);

/// Initialize the logging system.
///
/// Sets up:
/// 1. Local console logging (for debugging strategy processes)
/// 2. Log capture for forwarding to host
///
/// Returns a receiver for log entries that should be forwarded.
pub fn init_logging() -> Receiver<LogEntry> {
    let (tx, rx) = mpsc::channel();

    // Store sender globally for the log capture layer
    *LOG_SENDER.lock().unwrap() = Some(tx);

    // Set up tracing subscriber with console output
    let filter = EnvFilter::from_default_env().add_directive(Level::INFO.into());

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .finish();

    // Set as global default (ignore error if already set in tests)
    let _ = tracing::subscriber::set_global_default(subscriber);

    debug!("logging initialized");

    rx
}

/// Log a message that will be forwarded to the host.
///
/// This is called internally when log macros are used.
#[allow(dead_code)]
pub fn forward_log(level: LogLevel, message: String) {
    if let Some(sender) = LOG_SENDER.lock().unwrap().as_ref() {
        let entry = LogEntry { level, message };
        // Ignore send errors (receiver may have been dropped)
        let _ = sender.send(entry);
    }
}

/// Collect all pending log entries.
///
/// Called by the runner to collect logs for the current frame.
pub fn collect_logs(receiver: &Receiver<LogEntry>) -> Vec<LogEntry> {
    let mut logs = Vec::new();
    while let Ok(entry) = receiver.try_recv() {
        logs.push(entry);
    }
    logs
}

/// Convert log entries to strategy messages for forwarding.
pub fn logs_to_messages(logs: Vec<LogEntry>) -> Vec<StrategyMessage> {
    logs.into_iter()
        .map(|entry| StrategyMessage::Log {
            level: entry.level,
            message: entry.message,
        })
        .collect()
}

/// Convert tracing Level to our LogLevel.
#[allow(dead_code)]
pub fn level_to_log_level(level: Level) -> LogLevel {
    match level {
        Level::TRACE => LogLevel::Trace,
        Level::DEBUG => LogLevel::Debug,
        Level::INFO => LogLevel::Info,
        Level::WARN => LogLevel::Warn,
        Level::ERROR => LogLevel::Error,
    }
}

/// Macro to log with forwarding to host.
///
/// Usage:
/// ```ignore
/// strategy_log!(Info, "Player {} moved to position {:?}", id, pos);
/// ```
#[macro_export]
macro_rules! strategy_log {
    ($level:ident, $($arg:tt)*) => {
        {
            let msg = format!($($arg)*);
            tracing::$level!("{}", msg);
            $crate::logging::forward_log(
                dies_strategy_protocol::LogLevel::$level,
                msg,
            );
        }
    };
}

/// Log at info level with forwarding.
#[macro_export]
macro_rules! strategy_info {
    ($($arg:tt)*) => {
        $crate::strategy_log!(info, $($arg)*)
    };
}

/// Log at debug level with forwarding.
#[macro_export]
macro_rules! strategy_debug {
    ($($arg:tt)*) => {
        $crate::strategy_log!(debug, $($arg)*)
    };
}

/// Log at warn level with forwarding.
#[macro_export]
macro_rules! strategy_warn {
    ($($arg:tt)*) => {
        $crate::strategy_log!(warn, $($arg)*)
    };
}

/// Log at error level with forwarding.
#[macro_export]
macro_rules! strategy_error {
    ($($arg:tt)*) => {
        $crate::strategy_log!(error, $($arg)*)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_conversion() {
        assert_eq!(level_to_log_level(Level::TRACE), LogLevel::Trace);
        assert_eq!(level_to_log_level(Level::DEBUG), LogLevel::Debug);
        assert_eq!(level_to_log_level(Level::INFO), LogLevel::Info);
        assert_eq!(level_to_log_level(Level::WARN), LogLevel::Warn);
        assert_eq!(level_to_log_level(Level::ERROR), LogLevel::Error);
    }

    #[test]
    fn test_log_forwarding() {
        // Create a fresh channel for this test
        let (tx, rx) = mpsc::channel();
        *LOG_SENDER.lock().unwrap() = Some(tx);

        // Forward some logs
        forward_log(LogLevel::Info, "test message 1".to_string());
        forward_log(LogLevel::Warn, "test message 2".to_string());

        // Collect them
        let logs = collect_logs(&rx);
        assert_eq!(logs.len(), 2);
        assert_eq!(logs[0].level, LogLevel::Info);
        assert_eq!(logs[0].message, "test message 1");
        assert_eq!(logs[1].level, LogLevel::Warn);
        assert_eq!(logs[1].message, "test message 2");
    }

    #[test]
    fn test_logs_to_messages() {
        let logs = vec![
            LogEntry {
                level: LogLevel::Info,
                message: "hello".to_string(),
            },
            LogEntry {
                level: LogLevel::Error,
                message: "world".to_string(),
            },
        ];

        let messages = logs_to_messages(logs);
        assert_eq!(messages.len(), 2);

        match &messages[0] {
            StrategyMessage::Log { level, message } => {
                assert_eq!(*level, LogLevel::Info);
                assert_eq!(message, "hello");
            }
            _ => panic!("Expected Log message"),
        }
    }
}
