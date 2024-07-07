use dies_core::{Angle, DebugColor, PlayerData};
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

        let mut input = PlayerControlInput::new();
        //log::info!("playerid: {:?}, Relocating to {:?}, current position: {:?}, self status: {:?}", player_data.id, goal, player_data.position,self.status);
        dies_core::debug_cross("p0.cross", goal, DebugColor::Red);
        if player_data.id.as_u32() == 0 {
            dies_core::debug_string("p0.goal", format!("{}, {}", goal.x, goal.y));
            dies_core::debug_string("p0.position", format!("{}, {}", player_data.position.x, player_data.position.y));
            dies_core::debug_string("p0.status", format!("{:?}", self.status));
        } else {
            dies_core::debug_string("p1.goal", format!("{}, {}", goal.x, goal.y));
            dies_core::debug_string("p1.position", format!("{}, {}", player_data.position.x, player_data.position.y));
            dies_core::debug_string("p1.status", format!("{:?}", self.status));
        }
        let new_status = match self.status {
            Status3::NoGoal => {
                input.with_position(goal).with_yaw(yaw);
                Status3::Ongoing
            },
            Status3::Ongoing => {
                input.with_position(goal).with_yaw(yaw);
                if (player_data.position - goal).norm() < 50.0 && (player_data.yaw - yaw).abs() < 0.1{
                    Status3::Accomplished
                } else {
                    Status3::Ongoing
                }
            },
            Status3::Accomplished => {
                input.with_position(goal).with_yaw(yaw);
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
