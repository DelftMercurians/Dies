//! Sample buffer for sysid capture. Stores per-tick `(t, cmd, heading, state)`
//! rows; the offline LM fit runs in a Python notebook against the CSV dump.

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

#[derive(Default, Debug)]
pub struct CaptureBuffer {
    pub samples: Vec<CapturedSample>,
}

impl CaptureBuffer {
    pub fn push(&mut self, s: CapturedSample) {
        self.samples.push(s);
    }
}

/// Free-form recording started with `r.startRecording` and closed with
/// `r.stopRecording`. Samples the player's current slot at `rate_hz` regardless
/// of which operation is driving the robot, so the same buffer can capture a
/// sequence of `moveTo` / `setLocalVelocity` / `excite` / etc. calls.
#[derive(Debug)]
pub struct Recording {
    pub rate_hz: f64,
    pub last_sample_s: f64,
    pub buffer: CaptureBuffer,
}

impl Recording {
    pub fn new(rate_hz: f64, now: f64) -> Self {
        let interval = if rate_hz > 0.0 { 1.0 / rate_hz } else { 0.05 };
        Self {
            rate_hz,
            // Pre-arm so the first tick after start emits a sample immediately.
            last_sample_s: now - interval,
            buffer: CaptureBuffer::default(),
        }
    }
}

pub fn samples_to_csv(samples: &[CapturedSample]) -> String {
    let mut out = String::from("t,cmd_x,cmd_y,heading,pos_x,pos_y,vel_x,vel_y\n");
    for s in samples {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            s.t, s.cmd_x, s.cmd_y, s.heading, s.pos_x, s.pos_y, s.vel_x, s.vel_y,
        ));
    }
    out
}
