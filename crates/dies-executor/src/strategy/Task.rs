use dies_core::{Angle, PlayerData};
use crate::{PlayerControlInput, KickerControlInput};
use nalgebra::Vector2;
use tokio::{time::Instant};
#[derive(Clone, Debug, PartialEq, Copy)]
pub enum Status3 {
    NoGoal,
    Ongoing,
    Accomplished,
}

pub enum Status4 {
    NoGoal,
    Ongoing1,
    Ongoing2,
    Accomplished,
}

pub struct Task3Phase {
    status: Status3,
}

impl Task3Phase {
    pub fn new() -> Self {
        Self {
            status: Status3::NoGoal,
        }
    }

    pub fn relocate(&mut self, player_data: &PlayerData, goal: Vector2<f64>, yaw: Angle) -> PlayerControlInput {
        let new_status = match self.status {
            Status3::NoGoal => {
                Status3::Ongoing
            },
            Status3::Ongoing => {
                if (player_data.position - goal).norm() < 1e-6 && (player_data.yaw - yaw).abs() < 1e-6{
                    Status3::Accomplished
                } else {
                    Status3::Ongoing
                }
            },
            Status3::Accomplished => {
                Status3::Accomplished
            }
        };
        self.status = new_status;
        PlayerControlInput::new().with_position(goal).with_yaw(yaw).clone()
    }
    
    pub fn is_accomplished(&self) -> bool {
        match self.status {
            Status3::Accomplished => true,
            _ => false,
        }
    }
}

pub struct Task4Phase {
    status: Status4,
    timer: Instant,
    
}

impl Task4Phase {
    pub fn new() -> Self {
        Self {
            status: Status4::NoGoal,
            timer: Instant::now(),
        }
    }

    pub fn kick(&mut self) -> PlayerControlInput {
        let mut res = PlayerControlInput::new();
        let new_status = match self.status {
            Status4::NoGoal => {
                self.timer = Instant::now();
                res.with_kicker(KickerControlInput::Arm);
                Status4::Ongoing1
            },
            Status4::Ongoing1 => {
                if self.timer.elapsed().as_secs() > 10 {
                    res.with_kicker(KickerControlInput::Kick);
                    Status4::Ongoing2
                } else {
                    res.with_kicker(KickerControlInput::Arm);
                    Status4::Ongoing1
                }
            },
            Status4::Ongoing2 => {
                res.with_kicker(KickerControlInput::Disarm);
                Status4::Accomplished
            },
            Status4::Accomplished => {
                Status4::Accomplished
            }
        };
        self.status = new_status;
        res
    }
    
    pub fn is_accomplished(&self) -> bool {
        match self.status {
            Status4::Accomplished => true,
            _ => false,
        }
    }
}
