/// A struct that triggers an event periodically at a given interval.
pub struct IntervalTrigger {
    interval: f64,
    next_trigger: f64,
}

impl IntervalTrigger {
    /// Creates a new `Interval` with the given interval.
    pub fn new(interval: f64) -> Self {
        Self {
            interval,
            next_trigger: 0.0,
        }
    }

    /// Sets the interval at which the event should be triggered.
    #[allow(dead_code)]
    pub fn set_interval(&mut self, interval: f64) {
        self.interval = interval;
    }

    /// Returns true if the event should be triggered at the given time.
    pub fn trigger(&mut self, time: f64) -> bool {
        if time >= self.next_trigger {
            self.next_trigger += self.interval;
            true
        } else {
            false
        }
    }
}
