use dies_core::{Angle, PlayerData};
use nalgebra::Vector2;
use tokio::time::Instant;

use crate::{KickerControlInput, PlayerControlInput};
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

    pub fn relocate(
        &mut self,
        player_data: &PlayerData,
        goal: Vector2<f64>,
        yaw: Angle,
        dribbling: f64,
        carefullness: f64, // specify how accurately do we have to move
                           // (0 - inaccurate, 1 - careful)
    ) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();
        let new_status = match self.status {
            Status3::NoGoal => {
                input
                    .with_position(goal)
                    .with_yaw(yaw)
                    .with_dribbling(dribbling)
                    .with_care(carefullness);
                Status3::Ongoing
            }
            Status3::Ongoing => {
                input
                    .with_position(goal)
                    .with_yaw(yaw)
                    .with_dribbling(dribbling)
                    .with_care(carefullness);
                if (player_data.position - goal).norm() < 50.0 // TODO: magic number, fix
                    && (player_data.yaw - yaw).abs() < 0.1
                {
                    Status3::Accomplished
                } else {
                    Status3::Ongoing
                }
            }
            Status3::Accomplished => {
                input
                    .with_position(goal)
                    .with_yaw(yaw)
                    .with_care(carefullness);
                Status3::Accomplished
            }
        };
        self.status = new_status;

        input
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

    pub fn kick(&mut self, base_control: PlayerControlInput) -> PlayerControlInput {
        let mut res = base_control;
        let new_status = match self.status {
            Status4::NoGoal => {
                self.timer = Instant::now();
                res.with_kicker(KickerControlInput::Arm);
                Status4::Ongoing1
            }
            Status4::Ongoing1 => {
                if self.timer.elapsed().as_secs() > 5 {
                    // need short delay, otherwise we fail
                    // penalty-style rule about 10s
                    res.with_kicker(KickerControlInput::Kick);
                    Status4::Ongoing2
                } else {
                    res.with_kicker(KickerControlInput::Arm);
                    Status4::Ongoing1
                }
            }
            Status4::Ongoing2 => {
                res.with_kicker(KickerControlInput::Disarm);
                Status4::Accomplished
            }
            Status4::Accomplished => Status4::Accomplished,
        };
        self.status = new_status;
        res
    }
}
