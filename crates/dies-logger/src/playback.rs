use std::path::Path;

use anyhow::Result;

use crate::{LogFile, TimestampedMessage};

pub struct LogPlayback {
    /// The messages in the log file. Guaranteed to be non-empty.
    messages: Vec<TimestampedMessage>,
    /// The index of the current message. Guaranteed to be in bounds.
    current_index: usize,
    max_time: f64,
}

impl LogPlayback {
    /// Read a log file and return a LogPlayback object
    pub fn read_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let messages = LogFile::open(path)?.take_messages();
        if messages.is_empty() {
            return Err(anyhow::anyhow!("No messages in log file"));
        }
        Ok(Self {
            max_time: messages.last().map(|msg| msg.timestamp).unwrap_or(0.0),
            current_index: 0,
            messages,
        })
    }

    /// Get the current message
    pub fn current(&self) -> &TimestampedMessage {
        &self.messages[self.current_index]
    }

    /// Get the next message, if there is one
    pub fn next(&mut self) -> Option<&TimestampedMessage> {
        if self.current_index < self.messages.len() {
            let msg = &self.messages[self.current_index];
            self.current_index += 1;
            Some(msg)
        } else {
            None
        }
    }

    /// Get the previous message, if there is one
    pub fn previous(&mut self) -> Option<&TimestampedMessage> {
        if self.current_index > 0 {
            self.current_index -= 1;
            Some(&self.messages[self.current_index])
        } else {
            None
        }
    }

    /// Reset the playback to the beginning
    pub fn reset(&mut self) {
        self.current_index = 0;
    }

    /// Jump to a specific time in the log file
    pub fn jump_to(&mut self, time: f64) {
        self.current_index = self
            .messages
            .iter()
            .position(|msg| msg.timestamp >= time)
            .unwrap_or(self.messages.len() - 1);
    }

    /// Get the maximum time in the log file
    pub fn max_time(&self) -> f64 {
        self.max_time
    }
}
