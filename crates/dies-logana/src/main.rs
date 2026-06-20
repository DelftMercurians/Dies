//! Analysis tool for the OLD pre-Arrow `.log` format (header `SSL_LOG_FILE`,
//! big-endian framed messages; vision = protobuf `SSL_WrapperPacket`, dies data =
//! msgpack `DataLog` { World(WorldData) | Debug(DebugMap) }).
//!
//! All timestamps are the message `receiver_timestamp` (ns since log start),
//! the common AI-side clock shared by vision, commands and world state.
//!
//! Modes:
//!   fp <file...>          one fingerprint line per file (classify sim vs real)
//!   extract <file> <dir>  dump commands.csv / world.csv / vision.csv

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use anyhow::{bail, Result};
use dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
use dies_protos::Message;
use rmpv::Value;

const HEADER: &str = "SSL_LOG_FILE";

enum Msg {
    Vision(SSL_WrapperPacket),
    /// msgpack `DataLog`: single-key map {"World"|"Debug": ...}
    Data(Value),
    Referee,
    Other,
}

struct LogReader<R: Read> {
    inner: R,
}

impl<R: Read> LogReader<R> {
    fn new(mut inner: R) -> Result<Self> {
        let mut buf = [0u8; 4 + HEADER.len()];
        inner.read_exact(&mut buf)?;
        if &buf[0..12] != HEADER.as_bytes() {
            bail!("bad header");
        }
        let version = i32::from_be_bytes(buf[12..16].try_into()?);
        if version != 1 {
            bail!("unsupported version {version}");
        }
        Ok(Self { inner })
    }

    /// Returns (timestamp_seconds, msg) or None at EOF.
    fn next_msg(&mut self) -> Result<Option<(f64, Msg)>> {
        let mut hdr = [0u8; 16];
        match self.inner.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e.into()),
        }
        let ts_ns = i64::from_be_bytes(hdr[0..8].try_into()?);
        let mtype = i32::from_be_bytes(hdr[8..12].try_into()?);
        let size = i32::from_be_bytes(hdr[12..16].try_into()?) as usize;
        let mut payload = vec![0u8; size];
        self.inner.read_exact(&mut payload)?;
        let ts = ts_ns as f64 / 1e9;
        let msg = match mtype {
            2 | 4 => match SSL_WrapperPacket::parse_from_bytes(&payload) {
                Ok(p) => Msg::Vision(p),
                Err(_) => Msg::Other,
            },
            3 => Msg::Referee,
            _ => {
                // Blank: either a protobuf LogLine or msgpack DataLog. Try msgpack
                // and accept only if it looks like our single-key enum map.
                match rmpv::decode::read_value(&mut &payload[..]) {
                    Ok(v) if is_datalog(&v) => Msg::Data(v),
                    _ => Msg::Other,
                }
            }
        };
        Ok(Some((ts, msg)))
    }
}

fn is_datalog(v: &Value) -> bool {
    if let Value::Map(entries) = v {
        if entries.len() == 1 {
            if let Some((Value::String(k), _)) = entries.first() {
                let k = k.as_str().unwrap_or("");
                return k == "World" || k == "Debug";
            }
        }
    }
    false
}

// --- generic msgpack navigation helpers ---

fn map_get<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    if let Value::Map(entries) = v {
        for (k, val) in entries {
            if k.as_str() == Some(key) {
                return Some(val);
            }
        }
    }
    None
}

fn as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::F64(f) => Some(*f),
        Value::F32(f) => Some(*f as f64),
        Value::Integer(i) => i.as_f64(),
        _ => None,
    }
}

/// PlayerData position/velocity are nalgebra Vector2 → serialized as a 2-element
/// array (serde serializes Vector2 as a seq of its coords).
fn vec2(v: &Value) -> Option<(f64, f64)> {
    if let Value::Array(a) = v {
        if a.len() == 2 {
            return Some((as_f64(&a[0])?, as_f64(&a[1])?));
        }
    }
    None
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("");
    match mode {
        "fp" => {
            print_fp_header();
            for f in &args[2..] {
                if let Err(e) = fingerprint(Path::new(f)) {
                    eprintln!("{f}: ERROR {e}");
                }
            }
        }
        "extract" => {
            let file = &args[2];
            let dir = &args[3];
            extract(Path::new(file), Path::new(dir))?;
        }
        "probe" => {
            for f in &args[2..] {
                if let Err(e) = probe(Path::new(f)) {
                    eprintln!("{f}: ERROR {e}");
                }
            }
        }
        _ => {
            eprintln!("usage: dies-logana fp|probe <file...> | extract <file> <outdir>");
        }
    }
    Ok(())
}

struct RobotTrack {
    last: Option<(f64, f64, f64)>, // (t, x, y) within one camera stream
}

fn fingerprint(path: &Path) -> Result<()> {
    let f = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut r = LogReader::new(f)?;

    let mut first_ts = f64::NAN;
    let mut last_ts = 0.0;
    let mut n_vision = 0u64;
    let mut n_ref = 0u64;
    let mut n_world = 0u64;
    let mut cams: HashSet<u32> = HashSet::new();
    let mut max_blue = 0usize;
    let mut max_yellow = 0usize;
    let mut cmd_team: HashSet<String> = HashSet::new();
    let mut n_cmd = 0u64;

    // Noise floor: RMS of frame-to-frame delta for near-stationary robots,
    // keyed per (color,id,camera) so multi-camera handoff doesn't pollute it.
    let mut tracks: HashMap<(u8, u32, u32), RobotTrack> = HashMap::new();
    let mut noise_sumsq = 0.0;
    let mut noise_n = 0u64;

    while let Some((ts, msg)) = r.next_msg()? {
        if first_ts.is_nan() {
            first_ts = ts;
        }
        last_ts = ts;
        match msg {
            Msg::Vision(p) => {
                if p.detection.is_some() {
                    n_vision += 1;
                    let det = p.detection.get_or_default();
                    let cam = det.camera_id();
                    cams.insert(cam);
                    let tcap = det.t_capture();
                    max_blue = max_blue.max(det.robots_blue.len());
                    max_yellow = max_yellow.max(det.robots_yellow.len());
                    for (color, robots) in
                        [(0u8, &det.robots_blue), (1u8, &det.robots_yellow)]
                    {
                        for rb in robots {
                            let id = rb.robot_id();
                            let (x, y) = (rb.x() as f64, rb.y() as f64);
                            let tr = tracks
                                .entry((color, id, cam))
                                .or_insert(RobotTrack { last: None });
                            if let Some((pt, px, py)) = tr.last {
                                let dt = tcap - pt;
                                if dt > 0.0 && dt < 0.1 {
                                    let dx = x - px;
                                    let dy = y - py;
                                    let speed = (dx * dx + dy * dy).sqrt() / dt;
                                    // near-stationary → delta ≈ pure vision noise
                                    if speed < 50.0 {
                                        noise_sumsq += dx * dx + dy * dy;
                                        noise_n += 1;
                                    }
                                }
                            }
                            tr.last = Some((tcap, x, y));
                        }
                    }
                }
            }
            Msg::Referee => n_ref += 1,
            Msg::Data(v) => {
                if map_get(&v, "World").is_some() {
                    n_world += 1;
                } else if let Some(dbg) = map_get(&v, "Debug") {
                    if let Value::Map(entries) = dbg {
                        for (k, _) in entries {
                            if let Some(ks) = k.as_str() {
                                if ks.ends_with(".target_vel") {
                                    n_cmd += 1;
                                    if let Some(team) = ks.split('.').next() {
                                        cmd_team.insert(team.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Msg::Other => {}
        }
    }

    let dur = (last_ts - first_ts).max(1e-9);
    let vrate = n_vision as f64 / dur;
    let noise_rms = if noise_n > 0 {
        (noise_sumsq / noise_n as f64).sqrt()
    } else {
        f64::NAN
    };
    let mut camv: Vec<u32> = cams.into_iter().collect();
    camv.sort_unstable();
    let team = {
        let mut t: Vec<String> = cmd_team.into_iter().collect();
        t.sort();
        t.join("|")
    };
    let class = classify(camv.len(), vrate, noise_rms, n_cmd);

    println!(
        "{},{:.1},{},{},{:.1},{},{},{},{:?},{:.2},{},{}",
        path.file_name().unwrap().to_string_lossy(),
        dur,
        n_vision,
        n_ref,
        vrate,
        max_blue,
        max_yellow,
        n_world,
        camv,
        noise_rms,
        if team.is_empty() { "-".into() } else { team },
        class,
    );
    Ok(())
}

fn classify(n_cams: usize, _vrate: f64, noise_rms: f64, n_cmd: u64) -> &'static str {
    // Heuristic: real SSL has multiple cameras and non-trivial vision noise;
    // sim typically has 1 synthetic camera and near-zero stationary noise.
    let noisy = noise_rms.is_finite() && noise_rms > 0.3;
    if n_cmd == 0 {
        return "no-cmd";
    }
    if n_cams >= 2 || noisy {
        "REAL?"
    } else {
        "SIM?"
    }
}

fn print_fp_header() {
    println!("file,dur_s,n_vision,n_ref,vis_hz,max_blue,max_yellow,n_world,cams,noise_rms_mm,cmd_team,class");
}

fn extract(path: &Path, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let f = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut r = LogReader::new(f)?;

    let mut cmd_w = BufWriter::new(File::create(dir.join("commands.csv"))?);
    let mut world_w = BufWriter::new(File::create(dir.join("world.csv"))?);
    let mut vis_w = BufWriter::new(File::create(dir.join("vision.csv"))?);
    writeln!(cmd_w, "t,team,id,cmd_vx,cmd_vy")?;
    writeln!(
        world_w,
        "t,id,filt_x,filt_y,filt_vx,filt_vy,raw_x,raw_y,yaw"
    )?;
    writeln!(vis_w, "t,color,cam,id,x,y,orient")?;

    let mut first_ts = f64::NAN;
    while let Some((ts, msg)) = r.next_msg()? {
        if first_ts.is_nan() {
            first_ts = ts;
        }
        let t = ts - first_ts;
        match msg {
            Msg::Vision(p) => {
                if let Some(det) = p.detection.as_ref() {
                    let cam = det.camera_id();
                    for (color, robots) in [("blue", &det.robots_blue), ("yellow", &det.robots_yellow)] {
                        for rb in robots {
                            writeln!(
                                vis_w,
                                "{:.4},{},{},{},{:.1},{:.1},{:.4}",
                                t,
                                color,
                                cam,
                                rb.robot_id(),
                                rb.x(),
                                rb.y(),
                                rb.orientation()
                            )?;
                        }
                    }
                }
            }
            Msg::Data(v) => {
                if let Some(dbg) = map_get(&v, "Debug") {
                    if let Value::Map(entries) = dbg {
                        for (k, val) in entries {
                            let Some(ks) = k.as_str() else { continue };
                            if !ks.ends_with(".target_vel") {
                                continue;
                            }
                            // key: team_<color>.p<id>.target_vel ; value DebugValue::String
                            let parts: Vec<&str> = ks.split('.').collect();
                            if parts.len() < 3 {
                                continue;
                            }
                            let team = parts[0].strip_prefix("team_").unwrap_or(parts[0]);
                            let id = parts[1].strip_prefix('p').unwrap_or(parts[1]);
                            if let Some(s) = debugvalue_string(val) {
                                let mut it = s.split_whitespace();
                                if let (Some(a), Some(b)) = (it.next(), it.next()) {
                                    if let (Ok(vx), Ok(vy)) = (a.parse::<f64>(), b.parse::<f64>()) {
                                        writeln!(cmd_w, "{:.4},{},{},{:.2},{:.2}", t, team, id, vx, vy)?;
                                    }
                                }
                            }
                        }
                    }
                } else if let Some(world) = map_get(&v, "World") {
                    for team_key in ["blue_team", "yellow_team"] {
                        if let Some(Value::Array(players)) = map_get(world, team_key) {
                            for pl in players {
                                let id = map_get(pl, "id").and_then(as_f64).unwrap_or(-1.0);
                                let pos = map_get(pl, "position").and_then(vec2);
                                let vel = map_get(pl, "velocity").and_then(vec2);
                                let raw = map_get(pl, "raw_position").and_then(vec2);
                                let yaw = map_get(pl, "yaw").and_then(as_f64).unwrap_or(0.0);
                                if let (Some((x, y)), Some((vx, vy))) = (pos, vel) {
                                    let (rx, ry) = raw.unwrap_or((f64::NAN, f64::NAN));
                                    writeln!(
                                        world_w,
                                        "{:.4},{},{:.1},{:.1},{:.2},{:.2},{:.1},{:.1},{:.4}",
                                        t, id as i64, x, y, vx, vy, rx, ry, yaw
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    cmd_w.flush()?;
    world_w.flush()?;
    vis_w.flush()?;
    eprintln!("wrote commands.csv / world.csv / vision.csv to {}", dir.display());
    Ok(())
}

/// DebugValue is `#[serde(tag="type", content="data")]` → map {type:.., data:..}.
fn debugvalue_string(v: &Value) -> Option<String> {
    if map_get(v, "type").and_then(|t| t.as_str()) == Some("String") {
        return map_get(v, "data").and_then(|d| d.as_str().map(|s| s.to_string()));
    }
    // Fallback: plain string
    v.as_str().map(|s| s.to_string())
}

// --- probe mode: categorise message types and sample type-0 (Blank) content ---

fn probe(path: &Path) -> Result<()> {
    use dies_protos::dies_log_line::LogLine;
    let mut f = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut h = [0u8; 16];
    f.read_exact(&mut h)?;
    let mut by_type: HashMap<i32, u64> = HashMap::new();
    let mut blank_logline = 0u64;
    let mut blank_msgpack_world = 0u64;
    let mut blank_msgpack_debug = 0u64;
    let mut blank_msgpack_other = 0u64;
    let mut blank_unparsed = 0u64;
    let mut logline_samples: Vec<String> = Vec::new();
    let mut logline_targets: HashMap<String, u64> = HashMap::new();
    let mut msgpack_keys: HashMap<String, u64> = HashMap::new();
    loop {
        let mut hdr = [0u8; 16];
        match f.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let mtype = i32::from_be_bytes(hdr[8..12].try_into()?);
        let size = i32::from_be_bytes(hdr[12..16].try_into()?) as usize;
        let mut payload = vec![0u8; size];
        f.read_exact(&mut payload)?;
        *by_type.entry(mtype).or_default() += 1;
        if mtype == 2 || mtype == 4 {
            if let Ok(p) = SSL_WrapperPacket::parse_from_bytes(&payload) {
                let top = p.special_fields.unknown_fields().iter().count();
                let det = p.detection.as_ref();
                let din = det
                    .map(|d| d.special_fields.unknown_fields().iter().count())
                    .unwrap_or(0);
                let rob = det
                    .map(|d| {
                        d.robots_blue
                            .iter()
                            .chain(d.robots_yellow.iter())
                            .map(|r| r.special_fields.unknown_fields().iter().count())
                            .sum::<usize>()
                    })
                    .unwrap_or(0);
                if top + din + rob > 0 {
                    *msgpack_keys
                        .entry(format!("VISION_UNKNOWN_FIELDS top={top} det={din} robot={rob}"))
                        .or_default() += 1;
                }
            }
            continue;
        }
        if mtype == 0 || mtype == 1 {
            if let Ok(ll) = LogLine::parse_from_bytes(&payload) {
                if !ll.message().is_empty() || !ll.target().is_empty() {
                    blank_logline += 1;
                    *logline_targets.entry(ll.target().to_string()).or_default() += 1;
                    if logline_samples.len() < 50 {
                        logline_samples.push(format!(
                            "[{}] {}",
                            ll.target(),
                            ll.message().chars().take(180).collect::<String>()
                        ));
                    }
                    continue;
                }
            }
            match rmpv::decode::read_value(&mut &payload[..]) {
                Ok(v) => {
                    if let Value::Map(entries) = &v {
                        for (k, _) in entries {
                            if let Some(ks) = k.as_str() {
                                *msgpack_keys.entry(ks.to_string()).or_default() += 1;
                            }
                        }
                    }
                    if map_get(&v, "World").is_some() {
                        blank_msgpack_world += 1;
                    } else if map_get(&v, "Debug").is_some() {
                        blank_msgpack_debug += 1;
                    } else {
                        blank_msgpack_other += 1;
                    }
                }
                Err(_) => blank_unparsed += 1,
            }
        }
    }
    println!("== {} ==", path.file_name().unwrap().to_string_lossy());
    let mut tv: Vec<_> = by_type.iter().collect();
    tv.sort();
    for (t, c) in tv {
        println!("  msg_type {t}: {c}");
    }
    println!(
        "  type0/1: logline={blank_logline} world={blank_msgpack_world} \
         debug={blank_msgpack_debug} mp_other={blank_msgpack_other} unparsed={blank_unparsed}"
    );
    let mut tg: Vec<_> = logline_targets.iter().collect();
    tg.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    println!("  logline targets (top): {:?}", &tg[..tg.len().min(15)]);
    if !msgpack_keys.is_empty() {
        let mut mk: Vec<_> = msgpack_keys.iter().collect();
        mk.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
        println!("  msgpack top-level keys: {:?}", &mk[..mk.len().min(20)]);
    }
    println!("  --- logline samples ---");
    for s in &logline_samples {
        println!("    {s}");
    }
    Ok(())
}
