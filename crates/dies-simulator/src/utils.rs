const TRESHOLD: f64 = 1e-6;

/// A struct that triggers an event periodically at a given interval.
pub struct IntervalTrigger {
    interval: f64,
    triggered: bool,
}

impl IntervalTrigger {
    /// Creates a new `Interval` with the given interval.
    pub fn new(interval: f64) -> Self {
        Self {
            interval,
            triggered: false,
        }
    }

    /// Sets the interval at which the event should be triggered.
    pub fn set_interval(&mut self, interval: f64) {
        self.interval = interval;
    }

    /// Returns true if the event should be triggered at the given time.
    pub fn trigger(&mut self, time: f64) -> bool {
        if (time / self.interval).fract() < TRESHOLD {
            if !self.triggered {
                self.triggered = true;
                true
            } else {
                false
            }
        } else {
            self.triggered = false;
            false
        }
    }
}
