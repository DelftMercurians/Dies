//! Sample buffer for sysid capture.
//!
//! Stores per-tick `(t, cmd, heading, state)` rows and converts to the
//! `dies_mpc::types::Sample` format used by the offline LM fit.

use dies_mpc::types::{RobotState, Sample, Vec2};

#[derive(Debug, Clone)]
pub struct CapturedSample {
    pub t: f64,
    pub cmd_x: f64,
    pub cmd_y: f64,
    pub heading: f64,
    pub pos_x: f64,
    pub pos_y: f64,
    pub vel_x: f64,
    pub vel_y: f64,
}

impl CapturedSample {
    pub fn to_mpc(&self) -> Sample {
        Sample {
            t: self.t,
            cmd: Vec2::new(self.cmd_x, self.cmd_y),
            heading: self.heading,
            state: RobotState {
                pos: Vec2::new(self.pos_x, self.pos_y),
                vel: Vec2::new(self.vel_x, self.vel_y),
            },
        }
    }
}

#[derive(Default, Debug)]
pub struct CaptureBuffer {
    pub samples: Vec<CapturedSample>,
}

impl CaptureBuffer {
    pub fn push(&mut self, s: CapturedSample) {
        self.samples.push(s);
    }

    pub fn into_mpc(self) -> Vec<Sample> {
        self.samples.iter().map(|s| s.to_mpc()).collect()
    }
}
