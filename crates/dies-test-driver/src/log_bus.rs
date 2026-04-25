use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use typeshare::typeshare;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[typeshare]
pub enum TestLogLevel {
    Info,
    Warn,
    Error,
    Record,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct TestLogEntry {
    pub level: TestLogLevel,
    pub tag: Option<String>,
    pub message: String,
    /// Optional structured payload (JSON-serialized). `log.record` uses this.
    pub value_json: Option<String>,
    /// Milliseconds since UNIX epoch at emission time.
    #[typeshare(serialized_as = "number")]
    pub ts_ms: i64,
}

#[derive(Clone)]
pub struct LogBus {
    tx: broadcast::Sender<TestLogEntry>,
}

impl std::fmt::Debug for LogBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogBus")
            .field("subscribers", &self.tx.receiver_count())
            .finish()
    }
}

impl LogBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn sender(&self) -> broadcast::Sender<TestLogEntry> {
        self.tx.clone()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<TestLogEntry> {
        self.tx.subscribe()
    }

    pub fn emit(&self, entry: TestLogEntry) {
        let _ = self.tx.send(entry);
    }
}

impl Default for LogBus {
    fn default() -> Self {
        Self::new(1024)
    }
}

pub(crate) fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
