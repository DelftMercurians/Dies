//! Online per-robot open-loop delay estimator (command → first visible motion
//! on vision). One [`DelayDetector`] per robot, fed one frame at a time with the
//! robot's raw (unfiltered) vision position and the velocity command sent that
//! frame, plus world time.
//!
//! It isolates clean "go from standstill" events (command idle and robot
//! physically still for a pre-window, then command jumps past a go threshold and
//! the robot accelerates away) and estimates the delay by extrapolating the
//! `sqrt(displacement)`-vs-time line back to its x-intercept — the same method
//! validated offline, which removes the accelerate-from-rest travel bias and is
//! independent of the detection threshold (~46 ms on real hardware).
//!
//! All timing is driven off world time (seconds), never wall-clock, so it stays
//! deterministic under faster-than-real-time sim.

use std::collections::VecDeque;

use dies_core::{OpenLoopDelayStats, Vector2};

// --- tunables (mm, mm/s, s) -- identical to the validated offline detector ---
const IDLE: f64 = 20.0; // |cmd| at/below this == "no command"
const GO: f64 = 350.0; // |cmd| must exceed this for a real go-event
const PRE: f64 = 0.30; // required standstill window before onset
const STILL_MM: f64 = 8.0; // max raw displacement allowed during the pre-window
const FIT_LO: f64 = 5.0; // displacement band used for the sqrt-linear fit
const FIT_HI: f64 = 60.0;
const MAX_DELAY: f64 = 0.60; // abort if no motion within this after the command
const GO_WINDOW: f64 = 0.25; // command must reach GO within this after onset
const MIN_FIT_PTS: usize = 4;
const MIN_ALIGN: f64 = 0.3; // motion must align with commanded direction
const MIN_R2: f64 = 0.90; // sqrt-linear fit quality gate

const WINDOW_S: f64 = 600.0; // rolling window (10 min), anchored to latest event
const MAX_EVENTS: usize = 1024; // memory cap
const RING_KEEP: f64 = PRE + 0.1; // pre-window ring-buffer horizon

struct Sample {
    t: f64,
    pos: Vector2,
    cmd: f64,
}

enum State {
    Idle,
    /// A command onset was seen; collecting the acceleration segment.
    Collecting {
        t_cmd: f64,
        anchor: Vector2,
        cmd_dir: Vector2,
        seg: Vec<(f64, f64)>, // (t, displacement)
        armed_go: bool,
    },
}

#[derive(Default)]
pub struct DelayDetector {
    ring: VecDeque<Sample>,
    state: State,
    events: VecDeque<(f64, f64)>, // (event world-time, delay seconds)
}

impl Default for State {
    fn default() -> Self {
        State::Idle
    }
}

impl DelayDetector {
    /// Feed one frame. `pos` is the raw/unfiltered vision position (mm, absolute
    /// frame) and `cmd_vel` the commanded velocity that frame (mm/s, same frame
    /// as `pos`). `t` is world time in seconds.
    pub fn push(&mut self, t: f64, pos: Vector2, cmd_vel: Vector2) {
        let cmd = cmd_vel.norm();
        let prev_cmd = self.ring.back().map(|s| s.cmd);
        self.ring.push_back(Sample { t, pos, cmd });
        while self.ring.front().map_or(false, |s| s.t < t - RING_KEEP) {
            self.ring.pop_front();
        }

        match &mut self.state {
            State::Idle => {
                // command onset: previous frame idle, this frame not
                if prev_cmd.map_or(false, |c| c <= IDLE) && cmd > IDLE {
                    if let Some(anchor) = self.prewindow(t) {
                        self.state = State::Collecting {
                            t_cmd: t,
                            anchor,
                            cmd_dir: cmd_vel,
                            seg: Vec::new(),
                            armed_go: false,
                        };
                    }
                }
            }
            State::Collecting {
                t_cmd,
                anchor,
                cmd_dir,
                seg,
                armed_go,
            } => {
                let t_cmd = *t_cmd;
                if t - t_cmd <= GO_WINDOW && cmd >= GO {
                    *armed_go = true;
                }
                let d = (pos - *anchor).norm();
                if (FIT_LO..=FIT_HI).contains(&d) {
                    seg.push((t, d));
                }
                let done = d > FIT_HI && *armed_go && seg.len() >= MIN_FIT_PTS;
                if done {
                    let ev = finish(t_cmd, *anchor, *cmd_dir, seg, pos);
                    self.state = State::Idle;
                    if let Some(delay) = ev {
                        self.record(t_cmd, delay);
                    }
                } else if t - t_cmd > MAX_DELAY
                    || (t - t_cmd > GO_WINDOW && !*armed_go)
                {
                    self.state = State::Idle; // stalled / command never went
                }
            }
        }
    }

    /// Pre-window anchor (median position) over the standstill window strictly
    /// before the onset frame `t`. `None` if the window is too short, the
    /// command wasn't idle throughout, or the robot wasn't physically still.
    fn prewindow(&self, t: f64) -> Option<Vector2> {
        let pw: Vec<&Sample> = self
            .ring
            .iter()
            .filter(|s| s.t >= t - PRE && s.t < t)
            .collect();
        if pw.len() < 8 || pw[0].t > t - PRE + 0.05 {
            return None;
        }
        if pw.iter().any(|s| s.cmd > IDLE) {
            return None;
        }
        let mut xs: Vec<f64> = pw.iter().map(|s| s.pos.x).collect();
        let mut ys: Vec<f64> = pw.iter().map(|s| s.pos.y).collect();
        let anchor = Vector2::new(median(&mut xs), median(&mut ys));
        if pw.iter().any(|s| (s.pos - anchor).norm() > STILL_MM) {
            return None;
        }
        Some(anchor)
    }

    fn record(&mut self, t_event: f64, delay: f64) {
        self.events.push_back((t_event, delay));
        // Window anchored to the most recent event: nothing ages out while the
        // robot is idle (no new events), so the last estimate persists.
        let latest = self.events.back().map(|e| e.0).unwrap_or(t_event);
        while self.events.front().map_or(false, |e| e.0 < latest - WINDOW_S) {
            self.events.pop_front();
        }
        while self.events.len() > MAX_EVENTS {
            self.events.pop_front();
        }
    }

    /// Current rolling stats, or `None` if no event has been measured yet.
    pub fn stats(&self, now: f64) -> Option<OpenLoopDelayStats> {
        let latest = self.events.back()?.0;
        let mut delays: Vec<f64> = self
            .events
            .iter()
            .filter(|e| e.0 >= latest - WINDOW_S)
            .map(|e| e.1 * 1e3)
            .collect();
        if delays.is_empty() {
            return None;
        }
        let max_ms = delays.iter().cloned().fold(f64::MIN, f64::max);
        let median_ms = median(&mut delays);
        Some(OpenLoopDelayStats {
            median_ms,
            max_ms,
            age_s: (now - latest).max(0.0),
            sample_count: delays.len() as u32,
        })
    }
}

fn median(vals: &mut [f64]) -> f64 {
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = vals.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        0.5 * (vals[n / 2 - 1] + vals[n / 2])
    }
}

/// Extrapolate the acceleration segment back to zero displacement and return the
/// command→motion delay (s), or `None` if it fails a quality/alignment gate.
fn finish(
    t_cmd: f64,
    anchor: Vector2,
    cmd_dir: Vector2,
    seg: &[(f64, f64)],
    pos: Vector2,
) -> Option<f64> {
    // Linear fit of sqrt(displacement) vs time; x-intercept = motion start.
    let n = seg.len() as f64;
    let sx: f64 = seg.iter().map(|s| s.0).sum();
    let sy: f64 = seg.iter().map(|s| s.1.sqrt()).sum();
    let sxx: f64 = seg.iter().map(|s| s.0 * s.0).sum();
    let sxy: f64 = seg.iter().map(|s| s.0 * s.1.sqrt()).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-9 {
        return None;
    }
    let a = (n * sxy - sx * sy) / denom; // slope
    let b = (sy - a * sx) / n; // intercept
    if a <= 0.0 {
        return None;
    }
    // R^2 of the fit.
    let mean_y = sy / n;
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    for s in seg {
        let yy = s.1.sqrt();
        let pred = a * s.0 + b;
        ss_res += (yy - pred).powi(2);
        ss_tot += (yy - mean_y).powi(2);
    }
    let r2 = 1.0 - ss_res / (ss_tot + 1e-9);
    if r2 < MIN_R2 {
        return None;
    }
    let t_start = -b / a;
    let delay = t_start - t_cmd;
    if !(0.0..0.4).contains(&delay) {
        return None;
    }
    // Direction sanity: net motion must align with the commanded direction.
    let mv = pos - anchor;
    let denom = mv.norm() * cmd_dir.norm();
    if denom < 1e-9 {
        return None;
    }
    if mv.dot(&cmd_dir) / denom < MIN_ALIGN {
        return None;
    }
    Some(delay)
}
