//! `LogWriter` — owns the per-table Arrow builders and IPC stream writers for
//! one open log directory. Lives entirely on the worker thread.
//!
//! Rows accumulate in the builders; a `RecordBatch` is flushed to each table's
//! `<table>.arrow` stream every `FLUSH_INTERVAL` or once a builder crosses
//! `FLUSH_ROWS`. On close, all builders are drained, the streams are finished,
//! and (Phase 3) the directory is compacted to Parquet + zipped.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use arrow::ipc::writer::StreamWriter;
use dies_core::FieldGeometry;

use crate::builders::*;
use crate::flatten::{self, FlatSettings};
use crate::frame::{EventRecord, FrameRecord, LogLineRecord, MarkerRecord, RawRecord};
use crate::meta::MetaJson;
use crate::schema;

pub const FLUSH_INTERVAL: Duration = Duration::from_millis(500);
pub const FLUSH_ROWS: usize = 8192;

type Stream = StreamWriter<BufWriter<File>>;

#[derive(Default)]
struct Builders {
    frames: FramesBuilder,
    ball: BallBuilder,
    players: PlayersBuilder,
    debug_values: DebugValuesBuilder,
    debug_shapes: DebugShapesBuilder,
    debug_tree: DebugTreeBuilder,
    settings_changes: SettingsChangesBuilder,
    events: EventsBuilder,
    markers: MarkersBuilder,
    logs: LogsBuilder,
    vision: VisionBuilder,
}

pub struct LogWriter {
    dir: PathBuf,
    meta: MetaJson,
    field_geom_written: bool,
    streams: HashMap<&'static str, Stream>,
    builders: Builders,
    settings_state: FlatSettings,
    last_flush: Instant,
    // Session stats, patched into meta.json on close.
    frame_count: u64,
    first_t: Option<f64>,
    last_t: f64,
}

/// Drain `builder` into the matching `streams` entry if it has rows.
macro_rules! flush_table {
    ($self:ident, $name:literal, $field:ident) => {
        if $self.builders.$field.len() > 0 {
            let batch = $self.builders.$field.finish()?;
            if let Some(stream) = $self.streams.get_mut($name) {
                stream.write(&batch)?;
            }
        }
    };
}

impl LogWriter {
    /// Create the log directory, write `meta.json`, and open one IPC stream per
    /// table.
    pub fn open(dir: PathBuf, meta: MetaJson, now: Instant) -> Result<Self> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating log dir {}", dir.display()))?;
        meta.write(&dir)?;

        let mut streams = HashMap::new();
        for &table in schema::TABLES {
            let path = dir.join(format!("{table}.arrow"));
            let file = File::create(&path)
                .with_context(|| format!("creating {}", path.display()))?;
            let s = schema::schema_for(table).expect("known table");
            let writer = StreamWriter::try_new(BufWriter::new(file), &s)?;
            streams.insert(table, writer);
        }

        Ok(Self {
            dir,
            meta,
            field_geom_written: false,
            streams,
            builders: Builders::default(),
            settings_state: FlatSettings::new(),
            last_flush: now,
            frame_count: 0,
            first_t: None,
            last_t: 0.0,
        })
    }

    pub fn push_frame(&mut self, rec: &FrameRecord) {
        self.frame_count += 1;
        self.first_t.get_or_insert(rec.t_received);
        self.last_t = rec.t_received;
        self.builders.frames.push(rec);
        if let Some(ball) = &rec.ball {
            self.builders.ball.push(rec.frame_id, ball);
        }
        for p in &rec.players {
            self.builders.players.push(rec.frame_id, p);
        }
        for d in &rec.debug_values {
            self.builders.debug_values.push(rec.frame_id, d);
        }
        for s in &rec.debug_shapes {
            self.builders.debug_shapes.push(rec.frame_id, s);
        }
        for t in &rec.debug_tree {
            self.builders.debug_tree.push(rec.frame_id, t);
        }
    }

    /// Apply a flattened settings snapshot. `baseline` emits every key; otherwise
    /// only keys that differ from the last snapshot are emitted.
    pub fn push_settings(&mut self, frame_id: u64, t: f64, flat: FlatSettings, baseline: bool) {
        let changes = if baseline {
            flat.iter().map(|(k, v)| (k.clone(), Some(v.clone()))).collect()
        } else {
            flatten::diff(&self.settings_state, &flat)
        };
        for (key, scalar) in changes {
            let (num, s) = match &scalar {
                Some(sc) => sc.to_columns(),
                None => (None, None),
            };
            self.builders
                .settings_changes
                .push(frame_id, t, &key, num, s.as_deref());
        }
        self.settings_state = flat;
    }

    pub fn push_event(&mut self, rec: &EventRecord) {
        self.builders.events.push(rec);
    }

    pub fn push_marker(&mut self, rec: &MarkerRecord) {
        self.builders.markers.push(rec);
    }

    pub fn push_log_line(&mut self, rec: &LogLineRecord) {
        self.builders.logs.push(rec);
    }

    pub fn push_raw(&mut self, rec: &RawRecord) {
        self.builders.vision.push(rec);
    }

    /// Patch the field geometry into `meta.json` once (geometry isn't known when
    /// logging starts).
    pub fn set_field_geom(&mut self, geom: FieldGeometry) {
        if !self.field_geom_written {
            self.meta.field_geom = Some(geom);
            if let Err(e) = self.meta.write(&self.dir) {
                log::warn!("failed to patch field geometry into meta.json: {e}");
            }
            self.field_geom_written = true;
        }
    }

    /// Flush builders to disk if the interval elapsed, a builder is large, or
    /// `force` is set.
    pub fn maybe_flush(&mut self, now: Instant, force: bool) -> Result<()> {
        let due = force
            || now.duration_since(self.last_flush) >= FLUSH_INTERVAL
            || self.largest_builder() >= FLUSH_ROWS;
        if !due {
            return Ok(());
        }
        flush_table!(self, "frames", frames);
        flush_table!(self, "ball", ball);
        flush_table!(self, "players", players);
        flush_table!(self, "debug_values", debug_values);
        flush_table!(self, "debug_shapes", debug_shapes);
        flush_table!(self, "debug_tree", debug_tree);
        flush_table!(self, "settings_changes", settings_changes);
        flush_table!(self, "events", events);
        flush_table!(self, "markers", markers);
        flush_table!(self, "logs", logs);
        flush_table!(self, "vision", vision);
        for stream in self.streams.values_mut() {
            stream.flush()?;
        }
        self.last_flush = now;
        Ok(())
    }

    fn largest_builder(&self) -> usize {
        [
            self.builders.frames.len(),
            self.builders.ball.len(),
            self.builders.players.len(),
            self.builders.debug_values.len(),
            self.builders.debug_shapes.len(),
            self.builders.debug_tree.len(),
            self.builders.settings_changes.len(),
            self.builders.events.len(),
            self.builders.markers.len(),
            self.builders.logs.len(),
            self.builders.vision.len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0)
    }

    /// Flush remaining rows and finish all IPC streams. Returns the log
    /// directory so the caller (Phase 3) can compact it.
    pub fn close(mut self, now: Instant) -> Result<PathBuf> {
        self.maybe_flush(now, true)?;
        for (_, mut stream) in self.streams.drain() {
            stream.finish()?;
            // into_inner flushes the BufWriter.
            stream.into_inner()?.into_inner()?;
        }
        // Patch final session stats into meta.json for cheap listing.
        self.meta.frame_count = Some(self.frame_count);
        self.meta.first_t = self.first_t;
        self.meta.last_t = Some(self.last_t);
        if let Err(e) = self.meta.write(&self.dir) {
            log::warn!("failed to write final stats to meta.json: {e}");
        }
        Ok(self.dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::{mock_world_data, DebugMap, ExecutorSettings};

    #[test]
    fn writes_a_valid_log_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("session");
        let meta = MetaJson::new(0.0, true, Some("concerto".into()), None, "yellow_on_positive".into());
        let mut w = LogWriter::open(dir.clone(), meta, Instant::now()).unwrap();

        let settings = ExecutorSettings::default();
        w.push_settings(0, 0.0, flatten::flatten(&settings), true);

        let world = mock_world_data();
        let debug = DebugMap::new();
        for fid in 0..3 {
            let rec = FrameRecord::from_world(fid, &world, &debug);
            w.push_frame(&rec);
        }
        let returned = w.close(Instant::now()).unwrap();
        assert_eq!(returned, dir);

        // meta.json + every table file exists
        assert!(dir.join("meta.json").exists());
        for &t in schema::TABLES {
            assert!(dir.join(format!("{t}.arrow")).exists(), "missing {t}.arrow");
        }
    }
}
