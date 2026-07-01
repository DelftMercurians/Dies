//! Read side of the columnar log: load a log directory (or a `.dieslog` zip) and
//! reconstruct per-frame `WorldData` + `DebugMap` for replay.
//!
//! The reader is **windowed and lazy**. Only a small eager *spine* is loaded up
//! front: the `frames` table (one row per frame: timestamps, game state, …) plus
//! the sparse `markers`/`events`. The heavy per-frame tables (`players`, `ball`,
//! `debug_*`) are left on disk; `reconstruct(frame_id)` faults in only the Parquet
//! row group covering that frame, caches it (byte-budget LRU), and decodes the
//! frame's rows directly from Arrow. This keeps memory bounded to a few hundred MB
//! regardless of log length (a dense 2h log is ~77M rows / ~40 GiB if loaded whole).
//!
//! Row groups are sorted by `frame_id` (frames are written monotonically), so a
//! frame maps to one — occasionally two, at a row-group boundary — covering row
//! groups, found via Parquet column statistics. `reconstruct` rebuilds the same
//! `WorldData` the live UI rendered, so replay feeds the identical render path.
//!
//! `scan()` offers a sequential whole-log pass for analysis/export that streams
//! row-group by row-group without touching the random-access LRU cache.
//!
//! Some fields are intentionally lossy (documented inline): the columnar schema
//! drops a few rarely-used `WorldData` details that the UI does not render.

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, ListArray, RecordBatch, StringArray,
    UInt32Array, UInt64Array,
};
use arrow::ipc::reader::StreamReader;
use dies_core::{
    Angle, BallData, DebugColor, DebugMap, DebugShape, DebugValue, GameState, Handicap, PlayerData,
    PlayerId, RawGameStateData, SideAssignment, SysStatus, TeamColor, Vector2, Vector3, WorldData,
};
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReaderBuilder,
};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::statistics::Statistics;

use crate::meta::MetaJson;

/// Default cache budget for resident heavy-table row groups. Sequential playback
/// needs only the current group per table; this leaves ample room for seeks.
const DEFAULT_CACHE_BUDGET: usize = 256 << 20; // 256 MiB

/// The scalar, one-row-per-frame spine. Held eagerly for the whole log.
#[derive(Clone)]
struct FrameSpine {
    frame_id: u64,
    t_received: f64,
    t_capture: f64,
    dt: f64,
    game_state: String,
    operating_team: String,
    side_assignment: String,
    ball_on_blue_side: Option<f64>,
    ball_on_yellow_side: Option<f64>,
}

/// A user point-of-interest marker.
#[derive(Clone, Debug)]
pub struct Marker {
    pub frame_id: u64,
    pub t: f64,
    pub label: Option<String>,
}

/// A discrete logged event (referee/control actions, autoref kicks, …).
#[derive(Clone, Debug)]
pub struct EventRow {
    pub frame_id: u64,
    pub t: f64,
    pub event_type: String,
    pub payload_json: String,
}

/// One row of the `player_feedback` table: the full basestation feedback for a
/// robot on a frame, captured verbatim as JSON.
#[derive(Clone, Debug)]
pub struct PlayerFeedbackRowR {
    pub frame_id: u64,
    pub team: String,
    pub player_id: u32,
    pub feedback_json: String,
}

/// Which heavy table a cache entry belongs to (cache-key discriminant).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum HeavyId {
    Players,
    Ball,
    DebugValues,
    DebugShapes,
    DebugTree,
}

/// One row group's (or eager batch's) frame_id coverage. `slot` indexes the
/// Parquet row group or the eager batch vec.
#[derive(Clone, Copy)]
struct Span {
    min_fid: u64,
    max_fid: u64,
    slot: usize,
}

/// A heavy per-frame table, either lazily read from Parquet or (for uncompacted
/// `.arrow` streams, missing statistics, or absent tables) fully materialized.
enum HeavyTable {
    Parquet {
        path: PathBuf,
        meta: ArrowReaderMetadata,
        index: Vec<Span>,
    },
    Eager {
        batches: Vec<Arc<RecordBatch>>,
        index: Vec<Span>,
    },
}

impl HeavyTable {
    fn open(dir: &Path, name: &str) -> Result<Self> {
        let pq = dir.join(format!("{name}.parquet"));
        if pq.exists() {
            let file = File::open(&pq).with_context(|| format!("opening {}", pq.display()))?;
            let meta = ArrowReaderMetadata::load(&file, ArrowReaderOptions::new())?;
            if let Some(index) = build_index(meta.metadata()) {
                return Ok(Self::Parquet {
                    path: pq,
                    meta,
                    index,
                });
            }
            // No column statistics — we can't window; fall back to eager.
            return Ok(Self::eager(read_parquet_all(&pq)?));
        }
        let ar = dir.join(format!("{name}.arrow"));
        if ar.exists() {
            // Uncompacted IPC stream (in-progress / compaction-failed log): eager.
            return Ok(Self::eager(read_arrow_all(&ar)?));
        }
        // Absent table: every frame has no rows here.
        Ok(Self::eager(Vec::new()))
    }

    /// Build an eager table from fully-read batches, indexing each non-empty batch
    /// by its first/last frame_id.
    fn eager(batches: Vec<RecordBatch>) -> Self {
        let mut arcs = Vec::new();
        let mut index = Vec::new();
        for b in batches {
            if b.num_rows() == 0 {
                continue;
            }
            let col = u64s(&b, "frame_id");
            let slot = arcs.len();
            index.push(Span {
                min_fid: col.value(0),
                max_fid: col.value(col.len() - 1),
                slot,
            });
            arcs.push(Arc::new(b));
        }
        index.sort_by_key(|s| s.min_fid);
        Self::Eager {
            batches: arcs,
            index,
        }
    }

    fn index(&self) -> &[Span] {
        match self {
            Self::Parquet { index, .. } => index,
            Self::Eager { index, .. } => index,
        }
    }
}

/// Byte-budget LRU of decoded heavy-table row groups, keyed by (table, slot).
struct RowGroupCache {
    map: HashMap<(HeavyId, usize), Arc<RecordBatch>>,
    order: VecDeque<(HeavyId, usize)>,
    bytes: usize,
    budget: usize,
}

impl RowGroupCache {
    fn new(budget: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: VecDeque::new(),
            bytes: 0,
            budget,
        }
    }

    fn get(&mut self, key: &(HeavyId, usize)) -> Option<Arc<RecordBatch>> {
        let batch = self.map.get(key)?.clone();
        self.touch(key);
        Some(batch)
    }

    fn touch(&mut self, key: &(HeavyId, usize)) {
        if let Some(p) = self.order.iter().position(|k| k == key) {
            self.order.remove(p);
        }
        self.order.push_back(*key);
    }

    fn insert(&mut self, key: (HeavyId, usize), batch: Arc<RecordBatch>) {
        if self.map.contains_key(&key) {
            self.touch(&key);
            return;
        }
        self.bytes += batch.get_array_memory_size();
        self.map.insert(key, batch);
        self.order.push_back(key);
        // Evict LRU entries until under budget, but always keep ≥1 (a single row
        // group larger than the budget stays resident).
        while self.bytes > self.budget && self.order.len() > 1 {
            if let Some(k) = self.order.pop_front() {
                if let Some(b) = self.map.remove(&k) {
                    self.bytes -= b.get_array_memory_size();
                }
            }
        }
    }
}

pub struct LogReader {
    pub meta: MetaJson,
    /// One row per frame, sorted by frame_id.
    spine: Vec<FrameSpine>,
    /// (t_received, frame_id) in frame order, for time-based seeking.
    time_index: Vec<(f64, u64)>,
    markers: Vec<Marker>,
    events: Vec<EventRow>,
    players: HeavyTable,
    ball: HeavyTable,
    debug_values: HeavyTable,
    debug_shapes: HeavyTable,
    debug_tree: HeavyTable,
    cache: Mutex<RowGroupCache>,
    /// Log directory (or extracted temp dir), retained for on-demand reads of
    /// dense tables not needed for `reconstruct` (e.g. `player_feedback`).
    dir: PathBuf,
    _extracted: Option<tempfile::TempDir>,
}

#[derive(Clone)]
struct ShapeRowR {
    key: String,
    shape_type: String,
    cx: Option<f64>,
    cy: Option<f64>,
    radius: Option<f64>,
    x1: Option<f64>,
    y1: Option<f64>,
    x2: Option<f64>,
    y2: Option<f64>,
    color: Option<String>,
    fill: Option<String>,
    stroke: Option<String>,
}

struct PlayerRowR {
    team: String,
    player_id: u32,
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    yaw: f64,
    raw_yaw: f64,
    angular_speed: f64,
    position_noise: f64,
    primary_status: Option<String>,
    kicker_cap_voltage: Option<f32>,
    kicker_temp: Option<f32>,
    pack_voltage_0: Option<f32>,
    pack_voltage_1: Option<f32>,
    breakbeam_ball_detected: bool,
    has_ball: bool,
    imu_status: Option<String>,
    kicker_status: Option<String>,
    handicaps: String,
}

impl LogReader {
    /// Open a log directory or a `.dieslog` zip (extracted to a temp dir). Loads
    /// the eager spine + sparse tables; heavy tables stay on disk for lazy reads.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let (dir, extracted) = if path.is_dir() {
            (path.to_path_buf(), None)
        } else {
            let tmp = extract_zip(path)?;
            let dir = tmp.path().to_path_buf();
            (dir, Some(tmp))
        };

        let meta = MetaJson::read(&dir).context("reading meta.json")?;

        let spine = load_spine(&dir)?;
        let mut time_index: Vec<(f64, u64)> =
            spine.iter().map(|s| (s.t_received, s.frame_id)).collect();
        time_index.sort_by_key(|x| x.1);

        Ok(Self {
            meta,
            spine,
            time_index,
            markers: load_markers(&dir)?,
            events: load_events(&dir)?,
            players: HeavyTable::open(&dir, "players")?,
            ball: HeavyTable::open(&dir, "ball")?,
            debug_values: HeavyTable::open(&dir, "debug_values")?,
            debug_shapes: HeavyTable::open(&dir, "debug_shapes")?,
            debug_tree: HeavyTable::open(&dir, "debug_tree")?,
            cache: Mutex::new(RowGroupCache::new(DEFAULT_CACHE_BUDGET)),
            dir,
            _extracted: extracted,
        })
    }

    pub fn frame_count(&self) -> usize {
        self.time_index.len()
    }

    /// (min, max) relative timestamp across frames.
    pub fn time_bounds(&self) -> (f64, f64) {
        match (self.time_index.first(), self.time_index.last()) {
            (Some(a), Some(b)) => (a.0, b.0),
            _ => (0.0, 0.0),
        }
    }

    pub fn frame_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.time_index.iter().map(|(_, id)| *id)
    }

    /// `(t_received, frame_id)` pairs in frame order — lets callers build a pacing
    /// index from the spine without reconstructing every frame.
    pub fn frame_times(&self) -> &[(f64, u64)] {
        &self.time_index
    }

    pub fn markers(&self) -> &[Marker] {
        &self.markers
    }

    pub fn events(&self) -> &[EventRow] {
        &self.events
    }

    /// Read the full `player_feedback` table on demand — every logged robot
    /// feedback frame as `(frame_id, team, player_id, feedback_json)`. Dense
    /// table, so this is not cached; read once. Empty for logs recorded before
    /// this table existed.
    pub fn player_feedback_rows(&self) -> Result<Vec<PlayerFeedbackRowR>> {
        let mut out = Vec::new();
        for b in read_table(&self.dir, "player_feedback")? {
            let frame_id = u64s(&b, "frame_id");
            let team = strs(&b, "team");
            let player_id = u32s(&b, "player_id");
            let json = strs(&b, "feedback_json");
            for i in 0..b.num_rows() {
                out.push(PlayerFeedbackRowR {
                    frame_id: frame_id.value(i),
                    team: team.value(i).to_string(),
                    player_id: player_id.value(i),
                    feedback_json: json.value(i).to_string(),
                });
            }
        }
        Ok(out)
    }

    /// The frame id active at time `t` (first frame whose time is >= `t`, else
    /// the last frame).
    pub fn frame_at_time(&self, t: f64) -> Option<u64> {
        if self.time_index.is_empty() {
            return None;
        }
        let idx = self
            .time_index
            .partition_point(|(ft, _)| *ft < t)
            .min(self.time_index.len() - 1);
        Some(self.time_index[idx].1)
    }

    /// Rebuild the `WorldData` and `DebugMap` for a frame (random access; faults
    /// in + caches the covering row groups).
    pub fn reconstruct(&self, frame_id: u64) -> Option<(WorldData, DebugMap)> {
        let fr = self.spine_for(frame_id)?.clone();
        let world = self.reconstruct_world(frame_id, &fr);
        let debug = self.reconstruct_debug(frame_id);
        Some((world, debug))
    }

    /// Sequentially reconstruct every frame in order, streaming each heavy table
    /// row-group by row-group. Bounded memory (≤ one row group per table), and does
    /// not touch the random-access LRU. For analysis/export, not the replay UI.
    pub fn scan<F: FnMut(u64, &WorldData, &DebugMap)>(&self, mut f: F) -> Result<()> {
        let mut c_players = ScanCursor::new(&self.players);
        let mut c_ball = ScanCursor::new(&self.ball);
        let mut c_values = ScanCursor::new(&self.debug_values);
        let mut c_shapes = ScanCursor::new(&self.debug_shapes);
        let mut c_tree = ScanCursor::new(&self.debug_tree);

        for fr in &self.spine {
            let fid = fr.frame_id;
            let mut blue = Vec::new();
            let mut yellow = Vec::new();
            c_players.advance_to(fid, &mut |b, lo, hi| {
                append_players(b, lo, hi, fr.t_received, &mut blue, &mut yellow)
            })?;
            let mut ball = None;
            c_ball.advance_to(fid, &mut |b, lo, _| {
                ball = Some(ball_at(b, lo, fr.t_capture))
            })?;
            let mut debug = DebugMap::new();
            c_values.advance_to(fid, &mut |b, lo, hi| {
                append_debug_values(b, lo, hi, &mut debug)
            })?;
            c_shapes.advance_to(fid, &mut |b, lo, hi| {
                append_debug_shapes(b, lo, hi, &mut debug)
            })?;
            c_tree.advance_to(fid, &mut |b, lo, hi| {
                append_debug_tree(b, lo, hi, &mut debug)
            })?;

            let world = build_world(fr, blue, yellow, ball, &self.meta);
            f(fid, &world, &debug);
        }
        Ok(())
    }

    fn spine_for(&self, frame_id: u64) -> Option<&FrameSpine> {
        let i = self
            .spine
            .binary_search_by_key(&frame_id, |s| s.frame_id)
            .ok()?;
        Some(&self.spine[i])
    }

    fn reconstruct_world(&self, frame_id: u64, fr: &FrameSpine) -> WorldData {
        let mut blue = Vec::new();
        let mut yellow = Vec::new();
        self.collect_rows(
            &self.players,
            HeavyId::Players,
            frame_id,
            &mut |b, lo, hi| append_players(b, lo, hi, fr.t_received, &mut blue, &mut yellow),
        );
        let mut ball = None;
        self.collect_rows(&self.ball, HeavyId::Ball, frame_id, &mut |b, lo, _| {
            ball = Some(ball_at(b, lo, fr.t_capture))
        });
        build_world(fr, blue, yellow, ball, &self.meta)
    }

    fn reconstruct_debug(&self, frame_id: u64) -> DebugMap {
        let mut map = DebugMap::new();
        self.collect_rows(
            &self.debug_values,
            HeavyId::DebugValues,
            frame_id,
            &mut |b, lo, hi| append_debug_values(b, lo, hi, &mut map),
        );
        self.collect_rows(
            &self.debug_shapes,
            HeavyId::DebugShapes,
            frame_id,
            &mut |b, lo, hi| append_debug_shapes(b, lo, hi, &mut map),
        );
        self.collect_rows(
            &self.debug_tree,
            HeavyId::DebugTree,
            frame_id,
            &mut |b, lo, hi| append_debug_tree(b, lo, hi, &mut map),
        );
        map
    }

    /// Find the (≤2) covering row groups for `fid`, load each, and invoke `f` with
    /// the contiguous row range matching `fid` within each. A frame's rows may
    /// straddle a row-group boundary, hence "covering run" rather than one group.
    fn collect_rows(
        &self,
        table: &HeavyTable,
        id: HeavyId,
        fid: u64,
        f: &mut dyn FnMut(&RecordBatch, usize, usize),
    ) {
        let index = table.index();
        // First span whose max_fid >= fid; iterate while min_fid <= fid.
        let mut k = index.partition_point(|s| s.max_fid < fid);
        while k < index.len() && index[k].min_fid <= fid {
            let batch = match table {
                HeavyTable::Eager { batches, .. } => Some(batches[index[k].slot].clone()),
                HeavyTable::Parquet { path, meta, .. } => {
                    self.load_cached(id, path, meta, index[k].slot)
                }
            };
            if let Some(b) = batch {
                let (lo, hi) = frame_range(&b, fid);
                if lo < hi {
                    f(&b, lo, hi);
                }
            }
            k += 1;
        }
    }

    /// Fetch a Parquet row group from the LRU, loading + inserting on a miss.
    fn load_cached(
        &self,
        id: HeavyId,
        path: &Path,
        meta: &ArrowReaderMetadata,
        slot: usize,
    ) -> Option<Arc<RecordBatch>> {
        let key = (id, slot);
        if let Some(b) = self.cache.lock().unwrap().get(&key) {
            return Some(b);
        }
        let batch = match read_row_group(path, meta, slot) {
            Ok(b) => Arc::new(b),
            Err(e) => {
                log::warn!(
                    "dies-logger: failed to read row group {slot} of {}: {e}",
                    path.display()
                );
                return None;
            }
        };
        self.cache.lock().unwrap().insert(key, batch.clone());
        Some(batch)
    }

    #[cfg(test)]
    fn cache_bytes(&self) -> usize {
        self.cache.lock().unwrap().bytes
    }

    #[cfg(test)]
    fn set_cache_budget(&self, budget: usize) {
        self.cache.lock().unwrap().budget = budget;
    }
}

/// A forward-only sequential reader over one heavy table's row groups, used by
/// `scan`. Holds at most one decoded row group at a time and never caches.
struct ScanCursor<'a> {
    table: &'a HeavyTable,
    span_idx: usize,
    pos: usize,
    cur: Option<Arc<RecordBatch>>,
}

impl<'a> ScanCursor<'a> {
    fn new(table: &'a HeavyTable) -> Self {
        Self {
            table,
            span_idx: 0,
            pos: 0,
            cur: None,
        }
    }

    fn load_next(&mut self) -> Result<bool> {
        let index = self.table.index();
        if self.span_idx >= index.len() {
            return Ok(false);
        }
        let slot = index[self.span_idx].slot;
        let batch = match self.table {
            HeavyTable::Eager { batches, .. } => batches[slot].clone(),
            HeavyTable::Parquet { path, meta, .. } => Arc::new(read_row_group(path, meta, slot)?),
        };
        self.span_idx += 1;
        self.pos = 0;
        self.cur = Some(batch);
        Ok(true)
    }

    /// Emit the contiguous row block(s) for `fid` (in order, possibly spanning a
    /// row-group boundary) via `f`. `fid` must be non-decreasing across calls.
    fn advance_to(
        &mut self,
        fid: u64,
        f: &mut dyn FnMut(&RecordBatch, usize, usize),
    ) -> Result<()> {
        loop {
            if self.cur.is_none() && !self.load_next()? {
                return Ok(());
            }
            let batch = self.cur.clone().unwrap();
            let col = u64s(&batch, "frame_id");
            let n = col.len();
            while self.pos < n && col.value(self.pos) < fid {
                self.pos += 1;
            }
            if self.pos >= n {
                self.cur = None; // exhausted this group; try the next
                continue;
            }
            if col.value(self.pos) != fid {
                return Ok(()); // no rows for this fid
            }
            let lo = self.pos;
            while self.pos < n && col.value(self.pos) == fid {
                self.pos += 1;
            }
            f(&batch, lo, self.pos);
            if self.pos >= n {
                // The frame may continue into the next row group (boundary split).
                self.cur = None;
                continue;
            }
            return Ok(());
        }
    }
}

// --- per-frame decoders (shared by reconstruct + scan) ------------------------

fn append_players(
    b: &RecordBatch,
    lo: usize,
    hi: usize,
    t: f64,
    blue: &mut Vec<PlayerData>,
    yellow: &mut Vec<PlayerData>,
) {
    let team = strs(b, "team");
    let player_id = u32s(b, "player_id");
    let x = f64s(b, "x");
    let y = f64s(b, "y");
    let vx = f64s(b, "vx");
    let vy = f64s(b, "vy");
    let yaw = f64s(b, "yaw");
    let raw_yaw = f64s(b, "raw_yaw");
    let angular_speed = f64s(b, "angular_speed");
    let position_noise = f64s(b, "position_noise");
    let primary_status = strs(b, "primary_status");
    let kicker_cap_voltage = f32s(b, "kicker_cap_voltage");
    let kicker_temp = f32s(b, "kicker_temp");
    let pack_voltage_0 = f32s(b, "pack_voltage_0");
    let pack_voltage_1 = f32s(b, "pack_voltage_1");
    let breakbeam = bools(b, "breakbeam_ball_detected");
    let has_ball = bools(b, "has_ball");
    let imu_status = strs(b, "imu_status");
    let kicker_status = strs(b, "kicker_status");
    let handicaps = strs(b, "handicaps");
    for i in lo..hi {
        let r = PlayerRowR {
            team: team.value(i).to_string(),
            player_id: player_id.value(i),
            x: x.value(i),
            y: y.value(i),
            vx: vx.value(i),
            vy: vy.value(i),
            yaw: yaw.value(i),
            raw_yaw: raw_yaw.value(i),
            angular_speed: angular_speed.value(i),
            position_noise: position_noise.value(i),
            primary_status: opt_str(primary_status, i),
            kicker_cap_voltage: opt_f32(kicker_cap_voltage, i),
            kicker_temp: opt_f32(kicker_temp, i),
            pack_voltage_0: opt_f32(pack_voltage_0, i),
            pack_voltage_1: opt_f32(pack_voltage_1, i),
            breakbeam_ball_detected: breakbeam.value(i),
            has_ball: has_ball.value(i),
            imu_status: opt_str(imu_status, i),
            kicker_status: opt_str(kicker_status, i),
            handicaps: handicaps.value(i).to_string(),
        };
        let pd = player_data_from_row(&r, t);
        if r.team == "yellow" {
            yellow.push(pd);
        } else {
            blue.push(pd);
        }
    }
}

fn ball_at(b: &RecordBatch, i: usize, t_capture: f64) -> BallData {
    BallData {
        timestamp: t_capture,
        position: Vector3::new(
            f64s(b, "x").value(i),
            f64s(b, "y").value(i),
            f64s(b, "z").value(i),
        ),
        // raw_position is not stored (lossy); replay does not render it.
        raw_position: Vec::new(),
        velocity: Vector3::new(
            f64s(b, "vx").value(i),
            f64s(b, "vy").value(i),
            f64s(b, "vz").value(i),
        ),
        detected: bools(b, "detected").value(i),
    }
}

fn append_debug_values(b: &RecordBatch, lo: usize, hi: usize, map: &mut DebugMap) {
    let key = strs(b, "key");
    let value = f64s(b, "value");
    let value_str = strs(b, "value_str");
    for i in lo..hi {
        if value.is_valid(i) {
            map.insert(key.value(i).to_string(), DebugValue::Number(value.value(i)));
        } else if value_str.is_valid(i) {
            map.insert(
                key.value(i).to_string(),
                DebugValue::String(value_str.value(i).to_string()),
            );
        }
    }
}

fn append_debug_shapes(b: &RecordBatch, lo: usize, hi: usize, map: &mut DebugMap) {
    let key = strs(b, "key");
    let shape_type = strs(b, "shape_type");
    let cx = f64s(b, "cx");
    let cy = f64s(b, "cy");
    let radius = f64s(b, "radius");
    let x1 = f64s(b, "x1");
    let y1 = f64s(b, "y1");
    let x2 = f64s(b, "x2");
    let y2 = f64s(b, "y2");
    let color = strs(b, "color");
    let fill = strs(b, "fill");
    let stroke = strs(b, "stroke");
    for i in lo..hi {
        let r = ShapeRowR {
            key: key.value(i).to_string(),
            shape_type: shape_type.value(i).to_string(),
            cx: opt_f64(cx, i),
            cy: opt_f64(cy, i),
            radius: opt_f64(radius, i),
            x1: opt_f64(x1, i),
            y1: opt_f64(y1, i),
            x2: opt_f64(x2, i),
            y2: opt_f64(y2, i),
            color: opt_str(color, i),
            fill: opt_str(fill, i),
            stroke: opt_str(stroke, i),
        };
        if let Some(shape) = shape_from_row(&r) {
            map.insert(r.key.clone(), DebugValue::Shape(shape));
        }
    }
}

fn append_debug_tree(b: &RecordBatch, lo: usize, hi: usize, map: &mut DebugMap) {
    let key = strs(b, "key");
    let name = strs(b, "name");
    let node_id = strs(b, "node_id");
    let children = b
        .column_by_name("children_ids")
        .and_then(|c| c.as_any().downcast_ref::<ListArray>());
    let is_active = bools(b, "is_active");
    let node_type = strs(b, "node_type");
    let internal_state = strs(b, "internal_state");
    let additional_info = strs(b, "additional_info");
    for i in lo..hi {
        let children_ids = children
            .map(|c| {
                let v = c.value(i);
                let sa = v.as_any().downcast_ref::<StringArray>();
                sa.map(|sa| (0..sa.len()).map(|j| sa.value(j).to_string()).collect())
                    .unwrap_or_default()
            })
            .unwrap_or_default();
        map.insert(
            key.value(i).to_string(),
            DebugValue::Shape(DebugShape::TreeNode {
                name: name.value(i).to_string(),
                id: node_id.value(i).to_string(),
                children_ids,
                is_active: is_active.value(i),
                node_type: node_type.value(i).to_string(),
                internal_state: opt_str(internal_state, i),
                additional_info: opt_str(additional_info, i),
            }),
        );
    }
}

/// Assemble the scalar half of `WorldData` from the spine + decoded entities.
fn build_world(
    fr: &FrameSpine,
    blue_team: Vec<PlayerData>,
    yellow_team: Vec<PlayerData>,
    ball: Option<BallData>,
    meta: &MetaJson,
) -> WorldData {
    let operating_team = team_color_from_str(&fr.operating_team);
    let game_state = RawGameStateData {
        game_state: game_state_from_str(&fr.game_state),
        operating_team,
        // The columnar `frames` table only carries game_state + operating team;
        // the rest is defaulted (lossy, not rendered in replay).
        freekick_kicker: None,
        blue_team_max_allowed_bots: 11,
        yellow_team_max_allowed_bots: 11,
        blue_team_yellow_cards: 0,
        yellow_team_yellow_cards: 0,
        blue_team_score: 0,
        yellow_team_score: 0,
        blue_team_keeper_id: None,
        yellow_team_keeper_id: None,
        stage: None,
        stage_time_left: None,
        action_time_remaining: None,
        next_command: None,
        predicted_next_game_state: None,
        predicted_operating_team: None,
        status_message: None,
        blue_team_name: None,
        yellow_team_name: None,
    };

    WorldData {
        t_received: fr.t_received,
        t_capture: fr.t_capture,
        dt: fr.dt,
        blue_team,
        yellow_team,
        ball,
        field_geom: meta.field_geom.clone(),
        game_state,
        side_assignment: side_assignment_from_str(&fr.side_assignment),
        ball_on_blue_side: fr.ball_on_blue_side.map(Duration::from_secs_f64),
        ball_on_yellow_side: fr.ball_on_yellow_side.map(Duration::from_secs_f64),
        // Reconstructed from events if needed; defaulted here (lossy).
        autoref_info: None,
        // Possession is recomputed live, not stored; replay defaults it.
        possession: Default::default(),
    }
}

fn player_data_from_row(r: &PlayerRowR, t: f64) -> PlayerData {
    let pack_voltages = match (r.pack_voltage_0, r.pack_voltage_1) {
        (Some(a), Some(b)) => Some([a, b]),
        _ => None,
    };
    let mut pd = PlayerData::new(PlayerId::new(r.player_id));
    pd.timestamp = t;
    pd.position = Vector2::new(r.x, r.y);
    // raw_position is not stored (lossy); use the filtered position.
    pd.raw_position = Vector2::new(r.x, r.y);
    pd.velocity = Vector2::new(r.vx, r.vy);
    pd.yaw = Angle::from_radians(r.yaw);
    pd.raw_yaw = Angle::from_radians(r.raw_yaw);
    pd.angular_speed = r.angular_speed;
    pd.position_noise = r.position_noise;
    pd.primary_status = r.primary_status.as_deref().and_then(sys_status_from_str);
    pd.kicker_cap_voltage = r.kicker_cap_voltage;
    pd.kicker_temp = r.kicker_temp;
    pd.pack_voltages = pack_voltages;
    pd.breakbeam_ball_detected = r.breakbeam_ball_detected;
    pd.has_ball = r.has_ball;
    pd.imu_status = r.imu_status.as_deref().and_then(sys_status_from_str);
    pd.kicker_status = r.kicker_status.as_deref().and_then(sys_status_from_str);
    pd.handicaps = parse_handicaps(&r.handicaps);
    pd
}

fn shape_from_row(r: &ShapeRowR) -> Option<DebugShape> {
    let color = |o: &Option<String>| o.as_deref().map(debug_color_from_str).unwrap_or_default();
    match r.shape_type.as_str() {
        "cross" => Some(DebugShape::Cross {
            center: Vector2::new(r.cx?, r.cy?),
            color: color(&r.color),
        }),
        "circle" => Some(DebugShape::Circle {
            center: Vector2::new(r.cx?, r.cy?),
            radius: r.radius.unwrap_or(0.0),
            fill: r.fill.as_deref().map(debug_color_from_str),
            stroke: r.stroke.as_deref().map(debug_color_from_str),
        }),
        "line" => Some(DebugShape::Line {
            start: Vector2::new(r.x1?, r.y1?),
            end: Vector2::new(r.x2?, r.y2?),
            color: color(&r.color),
        }),
        _ => None,
    }
}

// --- table loaders (eager spine + sparse) -------------------------------------

fn load_spine(dir: &Path) -> Result<Vec<FrameSpine>> {
    let mut spine = Vec::new();
    for b in read_table(dir, "frames")? {
        let frame_id = u64s(&b, "frame_id");
        let t_received = f64s(&b, "t_received");
        let t_capture = f64s(&b, "t_capture");
        let dt = f64s(&b, "dt");
        let game_state = strs(&b, "game_state");
        let operating_team = strs(&b, "operating_team");
        let side_assignment = strs(&b, "side_assignment");
        let bob = f64s(&b, "ball_on_blue_side");
        let boy = f64s(&b, "ball_on_yellow_side");
        for i in 0..b.num_rows() {
            spine.push(FrameSpine {
                frame_id: frame_id.value(i),
                t_received: t_received.value(i),
                t_capture: t_capture.value(i),
                dt: dt.value(i),
                game_state: game_state.value(i).to_string(),
                operating_team: operating_team.value(i).to_string(),
                side_assignment: side_assignment.value(i).to_string(),
                ball_on_blue_side: opt_f64(bob, i),
                ball_on_yellow_side: opt_f64(boy, i),
            });
        }
    }
    spine.sort_by_key(|s| s.frame_id);
    Ok(spine)
}

fn load_markers(dir: &Path) -> Result<Vec<Marker>> {
    let mut markers = Vec::new();
    for b in read_table(dir, "markers")? {
        let frame_id = u64s(&b, "frame_id");
        let t = f64s(&b, "t");
        let label = strs(&b, "label");
        for i in 0..b.num_rows() {
            markers.push(Marker {
                frame_id: frame_id.value(i),
                t: t.value(i),
                label: opt_str(label, i),
            });
        }
    }
    Ok(markers)
}

fn load_events(dir: &Path) -> Result<Vec<EventRow>> {
    let mut events = Vec::new();
    for b in read_table(dir, "events")? {
        let frame_id = u64s(&b, "frame_id");
        let t = f64s(&b, "t");
        let event_type = strs(&b, "event_type");
        let payload_json = strs(&b, "payload_json");
        for i in 0..b.num_rows() {
            events.push(EventRow {
                frame_id: frame_id.value(i),
                t: t.value(i),
                event_type: event_type.value(i).to_string(),
                payload_json: payload_json.value(i).to_string(),
            });
        }
    }
    Ok(events)
}

// --- enum reverse mappings (inverse of `frame.rs`) ----------------------------

fn game_state_from_str(s: &str) -> GameState {
    match s {
        "Halt" => GameState::Halt,
        "Timeout" => GameState::Timeout,
        "Stop" => GameState::Stop,
        "PrepareKickoff" => GameState::PrepareKickoff,
        "BallReplacement" => GameState::BallReplacement(Vector2::zeros()),
        "PreparePenalty" => GameState::PreparePenalty,
        "Kickoff" => GameState::Kickoff,
        "FreeKick" => GameState::FreeKick,
        "Penalty" => GameState::Penalty,
        "PenaltyRun" => GameState::PenaltyRun,
        "Run" => GameState::Run,
        _ => GameState::Unknown,
    }
}

fn team_color_from_str(s: &str) -> TeamColor {
    match s {
        "yellow" => TeamColor::Yellow,
        _ => TeamColor::Blue,
    }
}

fn side_assignment_from_str(s: &str) -> SideAssignment {
    match s {
        "blue_on_positive" => SideAssignment::BlueOnPositive,
        _ => SideAssignment::YellowOnPositive,
    }
}

fn debug_color_from_str(s: &str) -> DebugColor {
    match s {
        "green" => DebugColor::Green,
        "orange" => DebugColor::Orange,
        "purple" => DebugColor::Purple,
        "blue" => DebugColor::Blue,
        "gray" => DebugColor::Gray,
        _ => DebugColor::Red,
    }
}

fn sys_status_from_str(s: &str) -> Option<SysStatus> {
    Some(match s {
        "emergency" => SysStatus::Emergency,
        "ok" => SysStatus::Ok,
        "ready" => SysStatus::Ready,
        "stop" => SysStatus::Stop,
        "starting" => SysStatus::Starting,
        "overtemp" => SysStatus::Overtemp,
        "no_reply" => SysStatus::NoReply,
        "armed" => SysStatus::Armed,
        "disarmed" => SysStatus::Disarmed,
        "safe" => SysStatus::Safe,
        "not_installed" => SysStatus::NotInstalled,
        "standby" => SysStatus::Standby,
        _ => return None,
    })
}

fn parse_handicaps(s: &str) -> std::collections::HashSet<Handicap> {
    s.split(',')
        .filter_map(|h| match h.trim() {
            "no_kicker" => Some(Handicap::NoKicker),
            "no_dribbler" => Some(Handicap::NoDribbler),
            "no_breakbeam" => Some(Handicap::NoBreakbeam),
            _ => None,
        })
        .collect()
}

// --- parquet/arrow plumbing ---------------------------------------------------

/// Build a frame_id → row-group index from column-0 statistics. Returns `None` if
/// any row group lacks stats (caller falls back to an eager load).
fn build_index(meta: &ParquetMetaData) -> Option<Vec<Span>> {
    let mut spans = Vec::with_capacity(meta.num_row_groups());
    for i in 0..meta.num_row_groups() {
        let stats = meta.row_group(i).column(0).statistics()?;
        let (min, max) = match stats {
            // A UInt64 frame_id is stored as physical INT64 in Parquet.
            Statistics::Int64(v) => (*v.min_opt()? as u64, *v.max_opt()? as u64),
            _ => return None,
        };
        spans.push(Span {
            min_fid: min,
            max_fid: max,
            slot: i,
        });
    }
    spans.sort_by_key(|s| s.min_fid);
    Some(spans)
}

/// Read a single Parquet row group into one `RecordBatch`. `with_batch_size` set
/// to the row count guarantees a single batch (no `concat` needed).
fn read_row_group(path: &Path, meta: &ArrowReaderMetadata, slot: usize) -> Result<RecordBatch> {
    let file = File::open(path)?;
    let rg_rows = meta.metadata().row_group(slot).num_rows().max(0) as usize;
    let mut reader = ParquetRecordBatchReaderBuilder::new_with_metadata(file, meta.clone())
        .with_row_groups(vec![slot])
        .with_batch_size(rg_rows.max(1))
        .build()?;
    reader
        .next()
        .transpose()?
        .ok_or_else(|| anyhow!("empty row group {slot} in {}", path.display()))
}

fn read_parquet_all(path: &Path) -> Result<Vec<RecordBatch>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?.build()?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn read_arrow_all(path: &Path) -> Result<Vec<RecordBatch>> {
    let reader = StreamReader::try_new(BufReader::new(File::open(path)?), None)?;
    reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

/// Read an entire table (Parquet preferred, else uncompacted `.arrow`, else empty)
/// into memory — used for the eager spine + sparse tables.
fn read_table(dir: &Path, name: &str) -> Result<Vec<RecordBatch>> {
    let pq = dir.join(format!("{name}.parquet"));
    if pq.exists() {
        return read_parquet_all(&pq);
    }
    let ar = dir.join(format!("{name}.arrow"));
    if ar.exists() {
        return read_arrow_all(&ar);
    }
    Ok(Vec::new())
}

fn extract_zip(zip_path: &Path) -> Result<tempfile::TempDir> {
    let tmp = tempfile::tempdir()?;
    let mut archive = zip::ZipArchive::new(File::open(zip_path)?)?;
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        let Some(name) = entry.enclosed_name() else {
            continue;
        };
        let out = tmp.path().join(name);
        let mut f = File::create(&out)?;
        std::io::copy(&mut entry, &mut f)?;
    }
    Ok(tmp)
}

/// Find the contiguous `[lo, hi)` row range whose `frame_id == fid` in a batch
/// whose `frame_id` column is sorted ascending (non-null per schema).
fn frame_range(b: &RecordBatch, fid: u64) -> (usize, usize) {
    let col = u64s(b, "frame_id");
    let n = col.len();
    let lo = lower_bound(col, fid);
    let mut hi = lo;
    while hi < n && col.value(hi) == fid {
        hi += 1;
    }
    (lo, hi)
}

fn lower_bound(col: &UInt64Array, fid: u64) -> usize {
    let (mut a, mut b) = (0usize, col.len());
    while a < b {
        let m = a + (b - a) / 2;
        if col.value(m) < fid {
            a = m + 1;
        } else {
            b = m;
        }
    }
    a
}

// --- arrow column accessors ---------------------------------------------------

fn u64s<'a>(b: &'a RecordBatch, name: &str) -> &'a UInt64Array {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
        .unwrap_or_else(|| panic!("column {name} not u64"))
}
fn u32s<'a>(b: &'a RecordBatch, name: &str) -> &'a UInt32Array {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .unwrap_or_else(|| panic!("column {name} not u32"))
}
fn f64s<'a>(b: &'a RecordBatch, name: &str) -> &'a Float64Array {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())
        .unwrap_or_else(|| panic!("column {name} not f64"))
}
fn f32s<'a>(b: &'a RecordBatch, name: &str) -> &'a Float32Array {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        .unwrap_or_else(|| panic!("column {name} not f32"))
}
fn strs<'a>(b: &'a RecordBatch, name: &str) -> &'a StringArray {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .unwrap_or_else(|| panic!("column {name} not utf8"))
}
fn bools<'a>(b: &'a RecordBatch, name: &str) -> &'a BooleanArray {
    b.column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
        .unwrap_or_else(|| panic!("column {name} not bool"))
}

fn opt_f64(a: &Float64Array, i: usize) -> Option<f64> {
    a.is_valid(i).then(|| a.value(i))
}
fn opt_f32(a: &Float32Array, i: usize) -> Option<f32> {
    a.is_valid(i).then(|| a.value(i))
}
fn opt_str(a: &StringArray, i: usize) -> Option<String> {
    a.is_valid(i).then(|| a.value(i).to_string())
}

/// `LogReader` must stay `Send + Sync` — it is moved into a tokio replay task and
/// `reconstruct(&self)` mutates the cache through a `Mutex`.
fn _assert_send_sync() {
    fn f<T: Send + Sync>() {}
    f::<LogReader>();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameRecord;
    use crate::meta::MetaJson;
    use crate::writer::LogWriter;
    use dies_core::{mock_world_data, DebugColor, DebugMap, DebugShape, DebugValue, WorldData};
    use std::time::Instant;

    fn meta() -> MetaJson {
        MetaJson::new(0.0, true, None, None, "yellow_on_positive".into())
    }

    #[test]
    fn write_then_reconstruct_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("session");
        let mut w = LogWriter::open(dir.clone(), meta(), Instant::now()).unwrap();

        let mut world = mock_world_data();
        world.blue_team[0].has_ball = true;
        let mut debug = DebugMap::new();
        debug.insert(
            "team_blue.p0.target".into(),
            DebugValue::Shape(DebugShape::Cross {
                center: Vector2::new(11.0, 22.0),
                color: DebugColor::Green,
            }),
        );
        debug.insert("k".into(), DebugValue::Number(3.5));
        for fid in 0..4 {
            w.push_frame(&FrameRecord::from_world(fid, &world, &debug));
        }
        w.close(Instant::now()).unwrap();

        let reader = LogReader::open(&dir).unwrap();
        assert_eq!(reader.frame_count(), 4);

        let (rw, rd) = reader.reconstruct(0).unwrap();
        assert_eq!(rw.blue_team.len(), 1);
        assert_eq!(rw.yellow_team.len(), 1);
        assert_eq!(rw.blue_team[0].position, world.blue_team[0].position);
        assert!(rw.blue_team[0].has_ball, "has_ball should round-trip");
        assert!(!rw.yellow_team[0].has_ball);
        assert_eq!(rw.game_state.game_state, GameState::Run);
        assert_eq!(rd.len(), 2);
        match rd.get("k") {
            Some(DebugValue::Number(n)) => assert_eq!(*n, 3.5),
            _ => panic!("missing number debug value"),
        }
        match rd.get("team_blue.p0.target") {
            Some(DebugValue::Shape(DebugShape::Cross { center, .. })) => {
                assert_eq!(*center, Vector2::new(11.0, 22.0));
            }
            _ => panic!("missing cross shape"),
        }
    }

    #[test]
    fn player_feedback_roundtrips() {
        use crate::frame::{FeedbackRecord, PlayerFeedbackRow};
        use dies_core::{PlayerFeedbackMsg, PlayerId};

        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("session");
        let mut w = LogWriter::open(dir.clone(), meta(), Instant::now()).unwrap();

        let world = mock_world_data();
        let debug = DebugMap::new();
        let mut msg = PlayerFeedbackMsg::empty(PlayerId::new(3));
        msg.kicker_cap_voltage = Some(212.5);
        let json = serde_json::to_string(&msg).unwrap();
        for fid in 0..3 {
            w.push_frame(&FrameRecord::from_world(fid, &world, &debug));
            w.push_feedback(&FeedbackRecord {
                frame_id: fid,
                players: vec![PlayerFeedbackRow {
                    team: "yellow",
                    player_id: 3,
                    feedback_json: json.clone(),
                }],
            });
        }
        w.close(Instant::now()).unwrap();

        let reader = LogReader::open(&dir).unwrap();
        let rows = reader.player_feedback_rows().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].team, "yellow");
        assert_eq!(rows[0].player_id, 3);
        // The JSON preserves every field verbatim, including ones left at None.
        let v: serde_json::Value = serde_json::from_str(&rows[0].feedback_json).unwrap();
        assert_eq!(v["kicker_cap_voltage"], 212.5);
        assert!(
            v.get("motor_speeds").is_some(),
            "all fields present in JSON"
        );
    }

    #[test]
    fn reconstruct_from_zip() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("session");
        let mut w = LogWriter::open(dir.clone(), meta(), Instant::now()).unwrap();
        let world = mock_world_data();
        let debug = DebugMap::new();
        for fid in 0..3 {
            w.push_frame(&FrameRecord::from_world(fid, &world, &debug));
        }
        w.close(Instant::now()).unwrap();
        let zip = crate::compact::finalize(&dir).unwrap();

        let reader = LogReader::open(&zip).unwrap();
        assert_eq!(reader.frame_count(), 3);
        assert!(reader.reconstruct(1).is_some());
    }

    /// Write a dense log large enough to span multiple 64k-row row groups, compact
    /// it, and return the reader.
    fn dense_compacted(n: u64) -> (tempfile::TempDir, LogReader) {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("session");
        let mut w = LogWriter::open(dir.clone(), meta(), Instant::now()).unwrap();
        let base = mock_world_data();
        for fid in 0..n {
            let mut world = base.clone();
            // vary a little
            world.blue_team[0].position.x = fid as f64;
            let mut debug = DebugMap::new();
            for k in 0..100u32 {
                debug.insert(format!("v{k}"), DebugValue::Number((fid + k as u64) as f64));
            }
            debug.insert(
                "shape".into(),
                DebugValue::Shape(DebugShape::Cross {
                    center: Vector2::new(fid as f64, 0.0),
                    color: DebugColor::Green,
                }),
            );
            debug.insert(
                "tree".into(),
                DebugValue::Shape(DebugShape::TreeNode {
                    name: "n".into(),
                    id: format!("id{fid}"),
                    children_ids: vec!["a".into(), "b".into()],
                    is_active: true,
                    node_type: "Selector".into(),
                    internal_state: None,
                    additional_info: None,
                }),
            );
            w.push_frame(&FrameRecord::from_world(fid, &world, &debug));
        }
        w.close(Instant::now()).unwrap();
        crate::compact::finalize(&dir).unwrap();
        let reader = LogReader::open(&dir).unwrap();
        (tmp, reader)
    }

    #[test]
    fn rowgroup_index_spans_multiple_groups() {
        // 1000 frames × 100 debug_values = 100k rows > 64k → ≥2 row groups.
        let (_tmp, reader) = dense_compacted(1000);
        let spans = match &reader.debug_values {
            HeavyTable::Parquet { index, .. } => index.clone(),
            _ => panic!("debug_values should be lazy Parquet"),
        };
        assert!(
            spans.len() >= 2,
            "expected multiple row groups, got {}",
            spans.len()
        );
        // sorted, covering [0, 1000)
        for w in spans.windows(2) {
            assert!(w[0].min_fid <= w[1].min_fid);
        }
        assert_eq!(spans.first().unwrap().min_fid, 0);
        assert_eq!(spans.last().unwrap().max_fid, 999);
    }

    #[test]
    fn reconstruct_across_row_groups() {
        let (_tmp, reader) = dense_compacted(1000);
        // A frame near the boundary and one well into the 2nd group.
        for fid in [0u64, 819, 820, 999] {
            let (world, debug) = reader.reconstruct(fid).unwrap();
            assert_eq!(world.blue_team[0].position.x, fid as f64);
            assert_eq!(debug.len(), 102); // 100 values + shape + tree
            match debug.get("v0") {
                Some(DebugValue::Number(n)) => assert_eq!(*n, fid as f64),
                _ => panic!("missing v0 for frame {fid}"),
            }
        }
    }

    #[test]
    fn scan_matches_reconstruct() {
        let (_tmp, reader) = dense_compacted(1000);
        let mut scanned = Vec::new();
        reader
            .scan(|fid, world, debug| {
                scanned.push((
                    fid,
                    world.blue_team.len() + world.yellow_team.len(),
                    debug.len(),
                ));
            })
            .unwrap();
        assert_eq!(scanned.len(), 1000);
        // scan must not pollute the random-access cache
        assert_eq!(reader.cache_bytes(), 0);
        for (fid, players, dbg) in &scanned {
            let (w, d) = reader.reconstruct(*fid).unwrap();
            assert_eq!(*players, w.blue_team.len() + w.yellow_team.len());
            assert_eq!(*dbg, d.len());
        }
    }

    #[test]
    fn debug_tree_children_ids_roundtrip() {
        let (_tmp, reader) = dense_compacted(1000);
        let (_w, d) = reader.reconstruct(820).unwrap();
        match d.get("tree") {
            Some(DebugValue::Shape(DebugShape::TreeNode {
                children_ids, id, ..
            })) => {
                assert_eq!(children_ids, &vec!["a".to_string(), "b".to_string()]);
                assert_eq!(id, "id820");
            }
            _ => panic!("missing tree node"),
        }
    }

    #[test]
    fn ball_absent_frame() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("session");
        let mut w = LogWriter::open(dir.clone(), meta(), Instant::now()).unwrap();
        let mut world = mock_world_data();
        world.ball = None; // mock already has None, be explicit
        let debug = DebugMap::new();
        for fid in 0..3 {
            w.push_frame(&FrameRecord::from_world(fid, &world, &debug));
        }
        w.close(Instant::now()).unwrap();
        crate::compact::finalize(&dir).unwrap();
        let reader = LogReader::open(&dir).unwrap();
        let (rw, _) = reader.reconstruct(1).unwrap();
        assert!(rw.ball.is_none());
    }

    #[test]
    fn arrow_fallback_eager() {
        // Uncompacted: only .arrow files (no finalize). Heavy tables take the
        // eager path; reconstruction must still work.
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("session");
        let mut w = LogWriter::open(dir.clone(), meta(), Instant::now()).unwrap();
        let world = mock_world_data();
        let mut debug = DebugMap::new();
        debug.insert("k".into(), DebugValue::Number(7.0));
        for fid in 0..5 {
            w.push_frame(&FrameRecord::from_world(fid, &world, &debug));
        }
        w.close(Instant::now()).unwrap();
        // no finalize → .arrow files remain
        let reader = LogReader::open(&dir).unwrap();
        assert!(matches!(reader.players, HeavyTable::Eager { .. }));
        let (rw, rd) = reader.reconstruct(2).unwrap();
        assert_eq!(rw.blue_team.len(), 1);
        match rd.get("k") {
            Some(DebugValue::Number(n)) => assert_eq!(*n, 7.0),
            _ => panic!("missing k"),
        }
    }

    #[test]
    fn lru_eviction_bounds_memory() {
        let (_tmp, reader) = dense_compacted(1000);
        // Tiny budget: force eviction as we touch far-apart frames.
        reader.set_cache_budget(64 * 1024);
        for fid in [0u64, 999, 0, 500, 999, 200] {
            let _ = reader.reconstruct(fid).unwrap();
        }
        // bytes stays within budget except for a single oversized resident entry.
        let n = reader.cache.lock().unwrap().map.len();
        assert!(n >= 1);
        let bytes = reader.cache_bytes();
        // at most one entry may exceed the budget alone; with n==1 that's allowed.
        if n > 1 {
            assert!(
                bytes <= 64 * 1024,
                "cache bytes {bytes} over budget with {n} entries"
            );
        }
    }

    #[test]
    fn frame_times_aligned() {
        let (_tmp, reader) = dense_compacted(50);
        let times = reader.frame_times();
        assert_eq!(times.len(), reader.frame_count());
        for &(t, id) in times {
            let (w, _) = reader.reconstruct(id).unwrap();
            assert_eq!(w.t_received, t);
        }
    }

    // Silence unused-import warnings for WorldData when only used in closures.
    #[allow(dead_code)]
    fn _uses(_: WorldData) {}
}
