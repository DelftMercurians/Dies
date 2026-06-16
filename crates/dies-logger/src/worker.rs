//! The columnar logger: a process-wide singleton backed by a worker thread that
//! owns the `LogWriter`. The public free functions push messages onto a
//! `std::sync::mpsc` channel; the worker drains it with `recv_timeout` so the
//! periodic flush fires even when the channel is idle.
//!
//! This is the new (Arrow/Parquet) write path. During migration it coexists with
//! the legacy `logger` module; only one is registered as the `log::Log` backend.

use std::path::PathBuf;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use dies_core::{DebugMap, ExecutorSettings, FieldGeometry, WorldData};
use log::{Log, Metadata, Record};

use crate::flatten::{self, FlatSettings};
use crate::frame::{EventRecord, FrameRecord, LogLineRecord, MarkerRecord, RawRecord};
use crate::meta::MetaJson;
use crate::writer::{LogWriter, FLUSH_INTERVAL};

static ARROW_LOGGER: OnceLock<ArrowLogger> = OnceLock::new();

enum WorkerMsg {
    StartLog { dir: PathBuf, meta: Box<MetaJson> },
    SetFieldGeom(Box<FieldGeometry>),
    CloseLog { ack: Option<Sender<()>> },
    Frame(Box<FrameRecord>),
    Settings {
        frame_id: u64,
        t: f64,
        flat: FlatSettings,
        baseline: bool,
    },
    Event(EventRecord),
    Marker(MarkerRecord),
    Raw(Box<RawRecord>),
    LogLine(LogLineRecord),
    Flush,
}

pub struct ArrowLogger {
    env_logger: env_logger::Logger,
    sender: Sender<WorkerMsg>,
    base_dir: PathBuf,
    start: Instant,
}

impl ArrowLogger {
    /// Initialize the columnar logger, spawning the worker thread. `base_dir` is
    /// the directory under which each session's log directory is created.
    /// Returns the singleton so the caller can register it via
    /// `log::set_logger`.
    pub fn init_with_env_logger(base_dir: PathBuf, env: env_logger::Logger) -> &'static Self {
        ARROW_LOGGER.get_or_init(|| {
            let (sender, receiver) = mpsc::channel();
            thread::spawn(move || run_worker(receiver));
            Self {
                env_logger: env,
                sender,
                base_dir,
                start: Instant::now(),
            }
        })
    }

    pub fn init(base_dir: PathBuf) -> &'static Self {
        Self::init_with_env_logger(base_dir, env_logger::Logger::from_default_env())
    }

    fn send(&self, msg: WorkerMsg) {
        let _ = self.sender.send(msg);
    }
}

/// Whether the columnar logger is the active backend (initialized). Callers use
/// this to skip building per-frame records when the legacy path is active.
pub fn is_active() -> bool {
    ARROW_LOGGER.get().is_some()
}

fn run_worker(receiver: mpsc::Receiver<WorkerMsg>) {
    let mut writer: Option<LogWriter> = None;
    loop {
        match receiver.recv_timeout(FLUSH_INTERVAL) {
            Ok(msg) => handle(&mut writer, msg),
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => break,
        }
        if let Some(w) = &mut writer {
            if let Err(e) = w.maybe_flush(Instant::now(), false) {
                eprintln!("dies-logger: flush failed: {e}");
            }
        }
    }
}

fn handle(writer: &mut Option<LogWriter>, msg: WorkerMsg) {
    match msg {
        WorkerMsg::StartLog { dir, meta } => {
            if let Some(w) = writer.take() {
                if let Err(e) = w.close(Instant::now()) {
                    eprintln!("dies-logger: closing previous log failed: {e}");
                }
            }
            match LogWriter::open(dir.clone(), *meta, Instant::now()) {
                Ok(w) => {
                    println!("Logging to {}", dir.display());
                    *writer = Some(w);
                }
                Err(e) => eprintln!("dies-logger: failed to open log dir: {e}"),
            }
        }
        WorkerMsg::SetFieldGeom(geom) => {
            if let Some(w) = writer {
                w.set_field_geom(*geom);
            }
        }
        WorkerMsg::CloseLog { ack } => {
            if let Some(w) = writer.take() {
                match w.close(Instant::now()) {
                    Ok(dir) => match crate::compact::finalize(&dir) {
                        Ok(zip) => println!("Compacted log to {}", zip.display()),
                        Err(e) => eprintln!("dies-logger: compaction failed: {e}"),
                    },
                    Err(e) => eprintln!("dies-logger: closing log failed: {e}"),
                }
            }
            if let Some(ack) = ack {
                let _ = ack.send(());
            }
        }
        WorkerMsg::Frame(rec) => {
            if let Some(w) = writer {
                w.push_frame(&rec);
            }
        }
        WorkerMsg::Settings {
            frame_id,
            t,
            flat,
            baseline,
        } => {
            if let Some(w) = writer {
                w.push_settings(frame_id, t, flat, baseline);
            }
        }
        WorkerMsg::Event(rec) => {
            if let Some(w) = writer {
                w.push_event(&rec);
            }
        }
        WorkerMsg::Marker(rec) => {
            if let Some(w) = writer {
                w.push_marker(&rec);
            }
        }
        WorkerMsg::Raw(rec) => {
            if let Some(w) = writer {
                w.push_raw(&rec);
            }
        }
        WorkerMsg::LogLine(rec) => {
            if let Some(w) = writer {
                w.push_log_line(&rec);
            }
        }
        WorkerMsg::Flush => {
            if let Some(w) = writer {
                let _ = w.maybe_flush(Instant::now(), true);
            }
        }
    }
}

// --- Public API. All no-op until the logger is initialized. ---

/// Start a new log in a session directory named `name` under the configured
/// base dir. Closes any currently open log.
pub fn log_start(name: &str, meta: MetaJson) {
    if let Some(l) = ARROW_LOGGER.get() {
        l.send(WorkerMsg::StartLog {
            dir: l.base_dir.join(name),
            meta: Box::new(meta),
        });
    }
}

/// Patch the static field geometry into `meta.json` (call once, on first frame).
pub fn log_set_field_geom(geom: &FieldGeometry) {
    if let Some(l) = ARROW_LOGGER.get() {
        l.send(WorkerMsg::SetFieldGeom(Box::new(geom.clone())));
    }
}

/// Flush and close the current log, triggering Parquet compaction + zip. Does
/// not wait for compaction to finish (use `log_close_blocking` on shutdown).
pub fn log_close() {
    if let Some(l) = ARROW_LOGGER.get() {
        l.send(WorkerMsg::CloseLog { ack: None });
    }
}

/// Close the log and block until compaction has finished (or `timeout` elapses).
/// Use this on shutdown so the process does not exit mid-compaction.
pub fn log_close_blocking(timeout: Duration) {
    if let Some(l) = ARROW_LOGGER.get() {
        let (tx, rx) = mpsc::channel();
        l.send(WorkerMsg::CloseLog { ack: Some(tx) });
        let _ = rx.recv_timeout(timeout);
    }
}

/// Log one world frame plus its debug snapshot. The columnar projection happens
/// on the caller thread (cheap; no `WorldData` clone).
pub fn log_frame(frame_id: u64, world: &WorldData, debug: &DebugMap) {
    if let Some(l) = ARROW_LOGGER.get() {
        let rec = FrameRecord::from_world(frame_id, world, debug);
        l.send(WorkerMsg::Frame(Box::new(rec)));
    }
}

/// Emit the full settings baseline (all keys) at `frame_id` (use 0 at start).
pub fn log_settings_baseline(frame_id: u64, settings: &ExecutorSettings) {
    if let Some(l) = ARROW_LOGGER.get() {
        l.send(WorkerMsg::Settings {
            frame_id,
            t: 0.0,
            flat: flatten::flatten(settings),
            baseline: true,
        });
    }
}

/// Emit only the settings keys that changed since the last snapshot.
pub fn log_settings_diff(frame_id: u64, t: f64, settings: &ExecutorSettings) {
    if let Some(l) = ARROW_LOGGER.get() {
        l.send(WorkerMsg::Settings {
            frame_id,
            t,
            flat: flatten::flatten(settings),
            baseline: false,
        });
    }
}

/// Log a discrete event with a JSON payload.
pub fn log_event(
    frame_id: u64,
    t: f64,
    event_type: impl Into<String>,
    payload_json: impl Into<String>,
) {
    if let Some(l) = ARROW_LOGGER.get() {
        l.send(WorkerMsg::Event(EventRecord {
            frame_id,
            t,
            event_type: event_type.into(),
            payload_json: payload_json.into(),
        }));
    }
}

/// Drop a user point-of-interest marker.
pub fn log_marker(frame_id: u64, t: f64, label: Option<String>) {
    if let Some(l) = ARROW_LOGGER.get() {
        l.send(WorkerMsg::Marker(MarkerRecord { frame_id, t, label }));
    }
}

/// Log raw received wire bytes (gated vision/GC stream).
pub fn log_raw(frame_id: u64, t: f64, kind: &str, bytes: &[u8]) {
    if let Some(l) = ARROW_LOGGER.get() {
        l.send(WorkerMsg::Raw(Box::new(RawRecord {
            frame_id,
            t,
            kind: kind.to_string(),
            bytes: bytes.to_vec(),
        })));
    }
}

impl Log for ArrowLogger {
    fn enabled(&self, metadata: &Metadata) -> bool {
        metadata.target().starts_with("dies")
    }

    fn log(&self, record: &Record) {
        self.env_logger.log(record);
        let rec = LogLineRecord {
            t: self.start.elapsed().as_secs_f64(),
            level: record.level().to_string(),
            target: record.target().to_string(),
            source: format!(
                "{}:{}",
                record.file().unwrap_or("unknown"),
                record.line().unwrap_or(0)
            ),
            message: format!("{}", record.args()),
        };
        self.send(WorkerMsg::LogLine(rec));
    }

    fn flush(&self) {
        self.env_logger.flush();
        self.send(WorkerMsg::Flush);
    }
}
