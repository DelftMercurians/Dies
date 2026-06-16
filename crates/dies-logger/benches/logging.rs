//! Macro-benchmarks for the columnar (Arrow/Parquet) logger.
//!
//! Two scenarios, mirroring how the logger is actually used in production (only
//! the public `worker` write API and `replay::LogReader` read API are exercised —
//! the same paths the executor and the replay UI use):
//!
//! 1. **record** — stream ~2h of *dense* logging (12 players + a fat per-frame
//!    `DebugMap` of shapes/values/tree nodes, plus a steady trickle of events and
//!    captured log lines) through the worker thread, then close + compact to
//!    Parquet + `.dieslog` zip. Reports producer-side throughput (×realtime),
//!    close/compaction cost, and on-disk artifact sizes.
//!
//! 2. **read** — open the produced log and reconstruct every frame's `WorldData`
//!    and `DebugMap` (the replay hot path), plus a batch of time-seeks. Reports
//!    load time, reconstruct throughput, and peak working-set hints.
//!
//! This is a custom-harness bench (`harness = false`): the workloads are minutes
//! long and many-GB, so Criterion's repeated-sampling model is the wrong fit. It
//! runs each section once and prints a report.
//!
//! ## Running
//!
//! ```bash
//! cargo bench -p dies-logger --bench logging                 # full 2h dense default
//! DIES_BENCH_SECS=300 cargo bench -p dies-logger --bench logging   # 5-min slice
//! DIES_BENCH_SECTION=record cargo bench -p dies-logger --bench logging
//! ```
//!
//! ## Knobs (env vars)
//!
//! | var | default | meaning |
//! |---|---|---|
//! | `DIES_BENCH_SECS`          | 7200 | simulated match duration (s) |
//! | `DIES_BENCH_FPS`           | 60   | frames logged per second |
//! | `DIES_BENCH_PLAYERS`       | 12   | robots on the field (6v6) |
//! | `DIES_BENCH_VALUES`        | 80   | debug values per frame (numbers + a few strings) |
//! | `DIES_BENCH_SHAPES`        | 60   | debug shapes per frame (cross/circle/line) |
//! | `DIES_BENCH_TREE`          | 25   | behaviour-tree nodes per frame |
//! | `DIES_BENCH_EVENTS_PER_SEC`| 5    | discrete events/s |
//! | `DIES_BENCH_LOGS_PER_SEC`  | 20   | captured `tracing`/`log` lines/s |
//! | `DIES_BENCH_SECTION`       | all  | `record` \| `read` \| `all` |
//! | `DIES_BENCH_DIR`           | tmp  | where to write the log (else a temp dir) |
//! | `DIES_BENCH_KEEP`          | 0    | keep the log dir after the run |

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use dies_core::{
    mock_world_data, BallData, DebugColor, DebugMap, DebugShape, DebugValue, ExecutorSettings,
    PlayerData, PlayerId, Vector2, Vector3, WorldData,
};
use dies_logger::{replay::LogReader, worker, MetaJson};

// --- parameters ---------------------------------------------------------------

#[derive(Clone, Copy)]
struct Params {
    secs: u64,
    fps: u64,
    players: usize,
    values: usize,
    shapes: usize,
    tree: usize,
    events_per_sec: u64,
    logs_per_sec: u64,
    /// Cap the producer's enqueue rate (frames/s) so the unbounded worker channel
    /// can't balloon when the producer vastly outruns the consumer. 0 = unbounded.
    max_fps: u64,
}

impl Params {
    fn from_env() -> Self {
        Self {
            secs: env_u64("DIES_BENCH_SECS", 7200),
            fps: env_u64("DIES_BENCH_FPS", 60),
            players: env_u64("DIES_BENCH_PLAYERS", 12) as usize,
            values: env_u64("DIES_BENCH_VALUES", 80) as usize,
            shapes: env_u64("DIES_BENCH_SHAPES", 60) as usize,
            tree: env_u64("DIES_BENCH_TREE", 25) as usize,
            events_per_sec: env_u64("DIES_BENCH_EVENTS_PER_SEC", 5),
            logs_per_sec: env_u64("DIES_BENCH_LOGS_PER_SEC", 20),
            // Default cap keeps the in-flight channel bounded on long runs while
            // still enqueueing ~133× realtime. Set 0 to measure the true max.
            max_fps: env_u64("DIES_BENCH_MAX_FPS", 8000),
        }
    }

    /// Bytes the read side must hold to load the whole log into RAM. The
    /// `LogReader` keeps every row of every table in `HashMap`s, so this grows
    /// linearly with the run — the dominant cost is the per-frame debug tables.
    /// ~600 B/row is measured empirically (String keys + boxed values).
    fn projected_read_bytes(&self) -> u64 {
        self.projected_rows() * 600
    }

    fn frames(&self) -> u64 {
        self.secs * self.fps
    }

    /// Total rows that land in each per-frame table over the whole run.
    fn projected_rows(&self) -> u64 {
        let f = self.frames();
        f * (1 + self.players as u64 + self.values as u64 + self.shapes as u64 + self.tree as u64)
            + self.secs * (self.events_per_sec + self.logs_per_sec)
    }
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

// --- synthetic dense data -----------------------------------------------------

/// Build a 6v6 world from the single-player mock by cloning + spreading players
/// across the field and attaching a ball.
fn build_world(players: usize) -> WorldData {
    let mut world = mock_world_data();
    let base = world.blue_team[0].clone();
    world.blue_team.clear();
    world.yellow_team.clear();
    let half = players / 2;
    for i in 0..players {
        let mut p: PlayerData = base.clone();
        p.id = PlayerId::new(i as u32);
        // Spread them out so positions aren't all identical.
        let row = (i % half) as f64;
        p.position = Vector2::new(
            -3000.0 + row * 800.0,
            if i < half { 1500.0 } else { -1500.0 },
        );
        if i < half {
            world.blue_team.push(p);
        } else {
            world.yellow_team.push(p);
        }
    }
    world.ball = Some(BallData {
        timestamp: 0.0,
        position: Vector3::new(0.0, 0.0, 0.0),
        raw_position: Vec::new(),
        velocity: Vector3::new(0.0, 0.0, 0.0),
        detected: true,
    });
    world
}

/// Nudge positions/velocities each frame so the columnar data is not trivially
/// constant (which would make Parquet compression unrealistically good).
fn mutate_world(world: &mut WorldData, frame_id: u64) {
    let t = frame_id as f64 * 0.016;
    let phase = (frame_id % 360) as f64 * std::f64::consts::PI / 180.0;
    for (i, p) in world
        .blue_team
        .iter_mut()
        .chain(world.yellow_team.iter_mut())
        .enumerate()
    {
        let d = i as f64;
        p.position.x = 2000.0 * (phase + d).sin();
        p.position.y = 1500.0 * (phase + d * 0.5).cos();
        p.velocity.x = 800.0 * (phase + d).cos();
        p.velocity.y = -800.0 * (phase + d * 0.5).sin();
        p.angular_speed = (phase + d).sin();
    }
    if let Some(ball) = &mut world.ball {
        ball.position.x = 3000.0 * (t).sin();
        ball.position.y = 2000.0 * (t * 0.7).cos();
        ball.velocity.x = 1500.0 * (t).cos();
    }
}

/// A fat per-frame debug map: `values` numbers/strings, `shapes` geometric
/// shapes, `tree` behaviour-tree nodes. Keys are stable across frames (realistic:
/// the same debug keys get re-emitted every tick) so dictionary encoding kicks in.
fn build_debug(params: &Params, frame_id: u64) -> DebugMap {
    let mut map = DebugMap::with_capacity(params.values + params.shapes + params.tree);
    let f = frame_id as f64;

    for i in 0..params.values {
        // ~1 in 8 is a string value; the rest are numbers.
        if i % 8 == 0 {
            map.insert(
                format!("p{}.skill.state", i % 12),
                DebugValue::String(format!("MoveTo#{}", frame_id % 4)),
            );
        } else {
            map.insert(
                format!("p{}.metric{i}", i % 12),
                DebugValue::Number((f * 0.01 + i as f64).sin() * 1000.0),
            );
        }
    }

    for i in 0..params.shapes {
        let c = match i % 6 {
            0 => DebugColor::Red,
            1 => DebugColor::Green,
            2 => DebugColor::Orange,
            3 => DebugColor::Purple,
            4 => DebugColor::Blue,
            _ => DebugColor::Gray,
        };
        let x = (f * 0.02 + i as f64).cos() * 3000.0;
        let y = (f * 0.02 + i as f64).sin() * 2000.0;
        let shape = match i % 3 {
            0 => DebugShape::Cross {
                center: Vector2::new(x, y),
                color: c,
            },
            1 => DebugShape::Circle {
                center: Vector2::new(x, y),
                radius: 90.0 + (i % 50) as f64,
                fill: Some(c),
                stroke: Some(DebugColor::Gray),
            },
            _ => DebugShape::Line {
                start: Vector2::new(x, y),
                end: Vector2::new(x + 500.0, y - 500.0),
                color: c,
            },
        };
        map.insert(format!("dbg.shape{i}"), DebugValue::Shape(shape));
    }

    for i in 0..params.tree {
        map.insert(
            format!("tree.node{i}"),
            DebugValue::Shape(DebugShape::TreeNode {
                name: format!("Node{i}"),
                id: format!("n{i}"),
                children_ids: vec![format!("n{}", i + 1), format!("n{}", i + 2)],
                is_active: frame_id.is_multiple_of(2),
                node_type: if i.is_multiple_of(2) {
                    "Selector".into()
                } else {
                    "Sequence".into()
                },
                internal_state: Some(format!("tick={}", frame_id % 100)),
                additional_info: None,
            }),
        );
    }

    map
}

// --- formatting helpers -------------------------------------------------------

fn human_bytes(n: u64) -> String {
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    format!("{v:.2} {}", U[i])
}

fn human_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

fn dir_parquet_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "parquet") {
                if let Ok(m) = p.metadata() {
                    total += m.len();
                }
            }
        }
    }
    total
}

fn file_size(p: &Path) -> u64 {
    p.metadata().map(|m| m.len()).unwrap_or(0)
}

fn rule(title: &str) {
    println!("\n========== {title} ==========");
}

// --- record benchmark ---------------------------------------------------------

fn bench_record(params: &Params, base_dir: &Path) -> PathBuf {
    rule("RECORD — stream + compact dense logging");
    let frames = params.frames();
    println!(
        "scenario: {}h {:02}m @ {} fps  ({} frames, {} players)",
        params.secs / 3600,
        (params.secs % 3600) / 60,
        params.fps,
        human_count(frames),
        params.players,
    );
    println!(
        "per frame: {} debug values, {} shapes, {} tree nodes  |  {} events/s, {} log lines/s",
        params.values, params.shapes, params.tree, params.events_per_sec, params.logs_per_sec,
    );
    println!(
        "projected total rows across tables: {}",
        human_count(params.projected_rows())
    );
    if params.max_fps > 0 {
        println!(
            "producer paced to {} fps ({}× realtime) to bound the worker channel; set DIES_BENCH_MAX_FPS=0 for true max",
            human_count(params.max_fps),
            params.max_fps / params.fps,
        );
    }

    // Spin the worker thread and register it so captured log lines exercise the
    // `logs` table (events use the dedicated `log_event` API).
    let logger = worker::ArrowLogger::init(base_dir.to_path_buf());
    let _ = log::set_logger(logger);
    log::set_max_level(log::LevelFilter::Debug);

    let meta = MetaJson::new(
        0.0,
        true,
        Some("concerto".into()),
        None,
        "blue_on_positive".into(),
    );
    worker::log_start("record", meta);
    worker::log_settings_baseline(0, &ExecutorSettings::default());

    let mut world = build_world(params.players);
    let dt = 1.0 / params.fps as f64;
    // event / log cadence in frames
    let event_every = (params.fps / params.events_per_sec.max(1)).max(1);
    let log_every = (params.fps / params.logs_per_sec.max(1)).max(1);

    let mut gen_time = Duration::ZERO; // synthetic data construction (not the logger)
    let mut produce_time = Duration::ZERO; // log_frame projection + channel send

    let wall = Instant::now();
    for frame_id in 0..frames {
        let t = frame_id as f64 * dt;

        let g0 = Instant::now();
        mutate_world(&mut world, frame_id);
        world.t_received = t;
        world.t_capture = t;
        world.dt = dt;
        let debug = build_debug(params, frame_id);
        gen_time += g0.elapsed();

        let p0 = Instant::now();
        worker::log_frame(frame_id, &world, &debug);
        produce_time += p0.elapsed();

        if frame_id % event_every == 0 {
            worker::log_event(
                frame_id,
                t,
                "autoref",
                format!(
                    "{{\"kind\":\"kick\",\"player\":{},\"frame\":{frame_id}}}",
                    frame_id % 12
                ),
            );
        }
        if frame_id % log_every == 0 {
            log::debug!(target: "dies_logger_bench", "frame {frame_id} t={t:.3} processed");
        }

        // Pace the producer (outside the timed region) so the unbounded channel
        // stays bounded on long runs. Sleep only when we're running ahead of cap.
        if params.max_fps > 0 {
            let target = Duration::from_secs_f64(frame_id as f64 / params.max_fps as f64);
            let elapsed = wall.elapsed();
            if target > elapsed {
                std::thread::sleep(target - elapsed);
            }
        }
    }
    let enqueue_wall = wall.elapsed();

    // Close + compact (Parquet + zip). Block until the worker is fully drained.
    let close = Instant::now();
    worker::log_close_blocking(Duration::from_secs(1800));
    let close_wall = close.elapsed();
    let total_wall = wall.elapsed();

    let log_dir = base_dir.join("record");
    let zip = base_dir.join("record.dieslog");
    let parquet = dir_parquet_size(&log_dir);
    let zip_sz = file_size(&zip);

    let realtime = params.secs as f64;
    let fps_produce = frames as f64 / produce_time.as_secs_f64();
    let fps_total = frames as f64 / total_wall.as_secs_f64();

    println!("\n-- timing --");
    println!(
        "  data synthesis (not logger): {:>8.2} s",
        gen_time.as_secs_f64()
    );
    println!(
        "  producer  log_frame() loop:  {:>8.2} s  ({:.0} fps, {:.1}× realtime, {:.1} µs/frame)",
        produce_time.as_secs_f64(),
        fps_produce,
        fps_produce / params.fps as f64,
        produce_time.as_secs_f64() * 1e6 / frames as f64,
    );
    println!(
        "  enqueue wall (gen+produce): {:>8.2} s",
        enqueue_wall.as_secs_f64()
    );
    println!(
        "  close + compaction (drain):  {:>8.2} s",
        close_wall.as_secs_f64()
    );
    println!(
        "  TOTAL wall:                  {:>8.2} s  ({:.1}× realtime — recording {:.0} s of match){}",
        total_wall.as_secs_f64(),
        realtime / total_wall.as_secs_f64(),
        realtime,
        if params.max_fps > 0 { "  [producer paced]" } else { "" },
    );
    println!("  → end-to-end effective rate: {:.0} fps", fps_total);

    println!("\n-- artifacts on disk --");
    println!("  Parquet (analysis):  {:>10}", human_bytes(parquet));
    println!("  .dieslog zip (share):{:>10}", human_bytes(zip_sz));
    if frames > 0 {
        println!(
            "  bytes/frame (parquet): {:.0} B",
            parquet as f64 / frames as f64
        );
        println!(
            "  storage rate: {} / hour of match",
            human_bytes((parquet as f64 / realtime * 3600.0) as u64)
        );
    }

    log_dir
}

// --- read benchmark -----------------------------------------------------------

fn bench_read(params: &Params, log_dir: &Path, base_dir: &Path) {
    rule("READ — open + reconstruct every frame");
    if !log_dir.exists() {
        println!(
            "  (no log at {} — run the record section first)",
            log_dir.display()
        );
        return;
    }

    // The LogReader is windowed + lazy: open loads only the eager spine; heavy
    // tables fault in per row group through a byte-budget LRU, so peak RSS stays
    // bounded regardless of log length. (The old eager reader needed ~the figure
    // below — kept here as a "what we no longer pay" reference.)
    println!(
        "  (eager-load equivalent, no longer paid: ~{})",
        human_bytes(params.projected_read_bytes())
    );

    // 1. Open from the Parquet directory (the replay UI's normal path).
    let t = Instant::now();
    let reader = match LogReader::open(log_dir) {
        Ok(r) => r,
        Err(e) => {
            println!("  open failed: {e}");
            return;
        }
    };
    let open_wall = t.elapsed();
    let n = reader.frame_count();
    let (t0, t1) = reader.time_bounds();
    println!(
        "  open (spine + lazy index):{:>8.2} s  ({} frames indexed)",
        open_wall.as_secs_f64(),
        human_count(n as u64)
    );
    println!("  time span: {:.1} s .. {:.1} s", t0, t1);

    // 2. Reconstruct every frame's WorldData + DebugMap (the replay hot path).
    let ids: Vec<u64> = reader.frame_ids().collect();
    let t = Instant::now();
    let mut checksum = 0usize;
    for &id in &ids {
        if let Some((world, debug)) = reader.reconstruct(id) {
            checksum += world.blue_team.len() + world.yellow_team.len() + debug.len();
        }
    }
    let recon_wall = t.elapsed();
    let fps_recon = ids.len() as f64 / recon_wall.as_secs_f64();
    println!(
        "  reconstruct all frames:  {:>8.2} s  ({:.0} fps, {:.1} µs/frame)  [checksum {}]",
        recon_wall.as_secs_f64(),
        fps_recon,
        recon_wall.as_secs_f64() * 1e6 / ids.len().max(1) as f64,
        checksum,
    );
    println!(
        "  → replaying all {} frames at this rate takes {:.2} s ({:.0}× realtime)",
        human_count(ids.len() as u64),
        ids.len() as f64 / fps_recon.max(1.0),
        fps_recon / params.fps as f64,
    );

    // 2b. Streamed scan over the whole log (analysis/export path; bounded memory,
    // no LRU). Should be at least as fast as random-access reconstruct.
    let t = Instant::now();
    let mut scan_sum = 0usize;
    let mut scanned = 0usize;
    if let Err(e) = reader.scan(|_fid, world, debug| {
        scan_sum += world.blue_team.len() + world.yellow_team.len() + debug.len();
        scanned += 1;
    }) {
        println!("  scan failed: {e}");
    } else {
        let scan_wall = t.elapsed();
        println!(
            "  scan() all frames:       {:>8.2} s  ({:.0} fps, {:.1} µs/frame)  [{} frames, checksum {}]",
            scan_wall.as_secs_f64(),
            scanned as f64 / scan_wall.as_secs_f64(),
            scan_wall.as_secs_f64() * 1e6 / scanned.max(1) as f64,
            human_count(scanned as u64),
            scan_sum,
        );
    }

    // 3. Time-seek throughput (scrubbing the replay timeline).
    const SEEKS: usize = 10_000;
    let span = (t1 - t0).max(1e-9);
    let t = Instant::now();
    let mut hit = 0usize;
    for k in 0..SEEKS {
        let frac = (k as f64) / SEEKS as f64;
        if reader.frame_at_time(t0 + frac * span).is_some() {
            hit += 1;
        }
    }
    let seek_wall = t.elapsed();
    println!(
        "  {} time-seeks:           {:>8.3} s  ({:.0} ns/seek, {hit} hits)",
        human_count(SEEKS as u64),
        seek_wall.as_secs_f64(),
        seek_wall.as_secs_f64() * 1e9 / SEEKS as f64,
    );

    // 4. Open from the shared `.dieslog` zip (measures extract-to-tmp overhead).
    let zip = base_dir.join("record.dieslog");
    if zip.exists() {
        let t = Instant::now();
        match LogReader::open(&zip) {
            Ok(r) => println!(
                "  open .dieslog zip:       {:>8.2} s  (extract + load, {} frames)",
                t.elapsed().as_secs_f64(),
                human_count(r.frame_count() as u64),
            ),
            Err(e) => println!("  zip open failed: {e}"),
        }
    }
}

// --- main ---------------------------------------------------------------------

fn main() {
    let params = Params::from_env();
    let section = std::env::var("DIES_BENCH_SECTION").unwrap_or_else(|_| "all".into());
    let keep = env_u64("DIES_BENCH_KEEP", 0) != 0;

    let (base_dir, _tmp) = match std::env::var("DIES_BENCH_DIR") {
        Ok(d) => {
            let p = PathBuf::from(d);
            let _ = std::fs::create_dir_all(&p);
            (p, None)
        }
        Err(_) => {
            let tmp = tempfile::tempdir().expect("tempdir");
            (tmp.path().to_path_buf(), Some(tmp))
        }
    };

    println!("dies-logger benchmark");
    println!(
        "log base dir: {}{}",
        base_dir.display(),
        if keep { " (kept)" } else { "" }
    );

    let run_record = section == "all" || section == "record";
    let run_read = section == "all" || section == "read";

    let log_dir = base_dir.join("record");
    if run_record {
        bench_record(&params, &base_dir);
    }
    if run_read {
        bench_read(&params, &log_dir, &base_dir);
    }

    if keep {
        // Re-leak the tempdir so it isn't deleted on drop.
        if let Some(tmp) = _tmp {
            let _ = tmp.keep();
        }
    }
    println!();
}
