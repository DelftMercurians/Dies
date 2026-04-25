//! Excitation profile generators for sysid.
//!
//! Profiles are *Rust-side generators*: JS passes a spec, Rust produces the
//! command sample by sample at command rate. This keeps Boa out of the hot loop.

use nalgebra::Vector2;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExcitationAxis {
    Forward,
    Strafe,
    Yaw,
}

impl ExcitationAxis {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "forward" | "fwd" | "x" => Some(Self::Forward),
            "strafe" | "y" => Some(Self::Strafe),
            "yaw" | "w" => Some(Self::Yaw),
            _ => None,
        }
    }
}

/// Per-axis excitation profile. Emits `Vector2` body-frame velocity commands
/// (and optional angular rate). Time `t` is seconds since profile start.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExcitationProfile {
    /// Linear chirp on a single axis.
    Chirp {
        axis: ExcitationAxis,
        f0: f64,
        f1: f64,
        amp: f64,
        duration: f64,
    },
    /// Square step: `magnitude` on the axis for `hold_sec`, zero for remainder.
    Step {
        axis: ExcitationAxis,
        magnitude: f64,
        hold_sec: f64,
        duration: f64,
    },
    /// Pseudo-random binary sequence at approx bandwidth.
    Prbs {
        axis: ExcitationAxis,
        amp: f64,
        bandwidth_hz: f64,
        duration: f64,
        seed: u64,
    },
    /// Linear ramp from `start` to `end` on a single axis.
    Ramp {
        axis: ExcitationAxis,
        start: f64,
        end: f64,
        duration: f64,
    },
    /// Zero command — useful to record a baseline during capture.
    Zero { duration: f64 },
}

/// Value of the excitation at time `t`. The `vel` field is body-frame velocity
/// command (mm/s). `angular` is optional angular rate (rad/s) — non-zero only
/// for yaw-axis profiles.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExcitationSample {
    pub vel: Vector2<f64>,
    pub angular: f64,
}

impl ExcitationProfile {
    pub fn duration(&self) -> f64 {
        match self {
            Self::Chirp { duration, .. }
            | Self::Step { duration, .. }
            | Self::Prbs { duration, .. }
            | Self::Ramp { duration, .. }
            | Self::Zero { duration } => *duration,
        }
    }

    /// Sample the profile at time `t` (seconds since profile start).
    pub fn sample(&self, t: f64) -> ExcitationSample {
        match *self {
            Self::Chirp {
                axis,
                f0,
                f1,
                amp,
                duration,
            } => {
                // Instantaneous phase for linear chirp:
                //   phi(t) = 2π (f0 t + (f1 - f0) t² / (2 T))
                let t = t.clamp(0.0, duration);
                let k = if duration > 0.0 {
                    (f1 - f0) / duration
                } else {
                    0.0
                };
                let phase = 2.0 * std::f64::consts::PI * (f0 * t + 0.5 * k * t * t);
                axis_to_sample(axis, amp * phase.sin())
            }
            Self::Step {
                axis,
                magnitude,
                hold_sec,
                ..
            } => {
                let val = if t >= 0.0 && t < hold_sec {
                    magnitude
                } else {
                    0.0
                };
                axis_to_sample(axis, val)
            }
            Self::Prbs {
                axis,
                amp,
                bandwidth_hz,
                seed,
                ..
            } => {
                let bit_period = if bandwidth_hz > 0.0 {
                    1.0 / bandwidth_hz
                } else {
                    1.0
                };
                let bit_idx = (t / bit_period).floor() as i64;
                let val = if lfsr_bit(seed, bit_idx) { amp } else { -amp };
                axis_to_sample(axis, val)
            }
            Self::Ramp {
                axis,
                start,
                end,
                duration,
            } => {
                let a = if duration > 0.0 {
                    (t / duration).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                axis_to_sample(axis, start + (end - start) * a)
            }
            Self::Zero { .. } => ExcitationSample::default(),
        }
    }
}

fn axis_to_sample(axis: ExcitationAxis, val: f64) -> ExcitationSample {
    match axis {
        ExcitationAxis::Forward => ExcitationSample {
            vel: Vector2::new(val, 0.0),
            angular: 0.0,
        },
        ExcitationAxis::Strafe => ExcitationSample {
            vel: Vector2::new(0.0, val),
            angular: 0.0,
        },
        ExcitationAxis::Yaw => ExcitationSample {
            vel: Vector2::zeros(),
            angular: val,
        },
    }
}

/// Galois LFSR advanced `bit_idx` times from `seed`, emit the low bit.
fn lfsr_bit(seed: u64, bit_idx: i64) -> bool {
    let seed = if seed == 0 { 0xACE1u64 } else { seed };
    let mut state = seed;
    let steps = bit_idx.unsigned_abs();
    for _ in 0..steps {
        let bit = state & 1;
        state >>= 1;
        if bit == 1 {
            state ^= 0xD800_0000_0000_0000u64;
        }
    }
    (state & 1) == 1
}
