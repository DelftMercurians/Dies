//! Replay player: drives a recorded log back through the same channels the live
//! UI consumes. Reconstructed `WorldUpdate`s go to the shared `update_tx` watch
//! and debug shapes are re-published via the global `debug_record`, so the
//! frontend renders replay with the exact same code path as live.

use std::path::Path;

use anyhow::Result;
use dies_core::{debug_clear, debug_record, WorldUpdate};
use dies_logger::replay::LogReader;
use tokio::sync::{mpsc, watch};

use crate::{ReplayMarker, ReplayState};

enum ReplayCmd {
    Play,
    Pause,
    Seek(f64),
    SetSpeed(f64),
    /// Step by N frames (negative = backward). Pauses playback.
    Step(i64),
    /// Step by a time delta in seconds (negative = backward). Pauses playback.
    StepTime(f64),
}

pub struct ReplayController {
    ctrl_tx: mpsc::UnboundedSender<ReplayCmd>,
    task: tokio::task::JoinHandle<()>,
}

impl ReplayController {
    /// Load a log (directory or `.dieslog` zip) and start the (paused) player.
    pub fn load(
        path: impl AsRef<Path>,
        update_tx: watch::Sender<Option<WorldUpdate>>,
        replay_tx: watch::Sender<ReplayState>,
    ) -> Result<Self> {
        let reader = LogReader::open(path)?;
        let (ctrl_tx, ctrl_rx) = mpsc::unbounded_channel();
        let task = tokio::spawn(playback_loop(reader, ctrl_rx, update_tx, replay_tx));
        Ok(Self { ctrl_tx, task })
    }

    pub fn play(&self) {
        let _ = self.ctrl_tx.send(ReplayCmd::Play);
    }
    pub fn pause(&self) {
        let _ = self.ctrl_tx.send(ReplayCmd::Pause);
    }
    pub fn seek(&self, t: f64) {
        let _ = self.ctrl_tx.send(ReplayCmd::Seek(t));
    }
    pub fn set_speed(&self, speed: f64) {
        let _ = self.ctrl_tx.send(ReplayCmd::SetSpeed(speed));
    }
    pub fn step(&self, delta: i64) {
        let _ = self.ctrl_tx.send(ReplayCmd::Step(delta));
    }
    pub fn step_time(&self, dt: f64) {
        let _ = self.ctrl_tx.send(ReplayCmd::StepTime(dt));
    }

    /// Stop the player and clear the replay overlay.
    pub fn stop(self, replay_tx: &watch::Sender<ReplayState>) {
        self.task.abort();
        debug_clear();
        let _ = replay_tx.send(ReplayState::default());
    }
}

async fn playback_loop(
    reader: LogReader,
    mut ctrl_rx: mpsc::UnboundedReceiver<ReplayCmd>,
    update_tx: watch::Sender<Option<WorldUpdate>>,
    replay_tx: watch::Sender<ReplayState>,
) {
    // Frame ids + per-frame timestamps for pacing come straight from the spine —
    // no need to reconstruct every frame (which would fault in the whole log).
    let (times, frames): (Vec<f64>, Vec<u64>) = reader.frame_times().iter().copied().unzip();
    if frames.is_empty() {
        return;
    }
    let (t_min, t_max) = reader.time_bounds();
    // Nominal per-frame timestep: median of consecutive frame-time deltas. Robust
    // to the occasional large gap (stoppages), unlike a plain mean.
    let nominal_dt = {
        let mut deltas: Vec<f64> = times
            .windows(2)
            .map(|w| w[1] - w[0])
            .filter(|d| *d > 0.0)
            .collect();
        if deltas.is_empty() {
            0.0
        } else {
            deltas.sort_by(|a, b| a.partial_cmp(b).unwrap());
            deltas[deltas.len() / 2]
        }
    };
    let markers: Vec<ReplayMarker> = reader
        .markers()
        .iter()
        .map(|m| ReplayMarker {
            frame_id: m.frame_id,
            t: m.t,
            label: m.label.clone(),
        })
        .collect();

    let mut idx = 0usize;
    let mut playing = false;
    let mut speed = 1.0_f64;

    let emit = |idx: usize| {
        if let Some((world, debug)) = reader.reconstruct(frames[idx]) {
            let _ = update_tx.send(Some(WorldUpdate {
                world_data: world,
                frame_id: frames[idx],
                announcements: Vec::new(),
            }));
            // Re-publish this frame's debug overlay through the global subscriber.
            debug_clear();
            for (key, value) in debug.iter() {
                debug_record(key.clone(), value.clone());
            }
        }
    };
    let publish_state = |idx: usize, playing: bool, speed: f64| {
        let _ = replay_tx.send(ReplayState {
            loaded: true,
            playing,
            speed,
            t_min,
            t_max,
            current_t: times[idx],
            current_frame_id: frames[idx],
            frame_count: frames.len() as u64,
            dt: nominal_dt,
            markers: markers.clone(),
        });
    };

    emit(idx);
    publish_state(idx, playing, speed);

    loop {
        if playing && idx + 1 < frames.len() {
            let dt = ((times[idx + 1] - times[idx]).max(0.0) / speed.max(1e-3)).min(1.0);
            tokio::select! {
                cmd = ctrl_rx.recv() => match cmd {
                    Some(c) => apply_cmd(c, &reader, &frames, &times, &mut idx, &mut playing, &mut speed, &emit, &publish_state),
                    None => break,
                },
                _ = tokio::time::sleep(std::time::Duration::from_secs_f64(dt)) => {
                    idx += 1;
                    emit(idx);
                    if idx + 1 >= frames.len() { playing = false; }
                    publish_state(idx, playing, speed);
                }
            }
        } else {
            // Paused or at the end: block until a command arrives.
            match ctrl_rx.recv().await {
                Some(c) => apply_cmd(
                    c,
                    &reader,
                    &frames,
                    &times,
                    &mut idx,
                    &mut playing,
                    &mut speed,
                    &emit,
                    &publish_state,
                ),
                None => break,
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_cmd(
    cmd: ReplayCmd,
    reader: &LogReader,
    frames: &[u64],
    times: &[f64],
    idx: &mut usize,
    playing: &mut bool,
    speed: &mut f64,
    emit: &impl Fn(usize),
    publish_state: &impl Fn(usize, bool, f64),
) {
    match cmd {
        ReplayCmd::Play => *playing = true,
        ReplayCmd::Pause => *playing = false,
        ReplayCmd::SetSpeed(s) => *speed = s.clamp(0.05, 16.0),
        ReplayCmd::Step(delta) => {
            *playing = false;
            let last = frames.len().saturating_sub(1) as i64;
            *idx = (*idx as i64 + delta).clamp(0, last) as usize;
            emit(*idx);
        }
        ReplayCmd::StepTime(dt) => {
            *playing = false;
            let last = frames.len().saturating_sub(1);
            let target = times[*idx] + dt;
            let mut next = if dt >= 0.0 {
                // First frame at/after the target time.
                times.partition_point(|&ft| ft < target).min(last)
            } else {
                // Last frame at/before the target time.
                times.partition_point(|&ft| ft <= target).saturating_sub(1)
            };
            // Always move at least one frame in the requested direction, so a
            // jump smaller than a single frame's dt still advances.
            if dt > 0.0 && next <= *idx {
                next = (*idx + 1).min(last);
            } else if dt < 0.0 && next >= *idx {
                next = idx.saturating_sub(1);
            }
            *idx = next;
            emit(*idx);
        }
        ReplayCmd::Seek(t) => {
            if let Some(fid) = reader.frame_at_time(t) {
                if let Some(found) = frames.iter().position(|&f| f == fid) {
                    *idx = found;
                    emit(*idx);
                }
            } else {
                // Fall back to nearest by timestamp.
                *idx = times
                    .partition_point(|&ft| ft < t)
                    .min(frames.len().saturating_sub(1));
                emit(*idx);
            }
        }
    }
    publish_state(*idx, *playing, *speed);
}
