use dies_core::PlayerOverrideCommand;

use crate::{control::Velocity, KickerControlInput, PlayerControlInput};

pub struct PlayerOverrideState {
    frame_counter: u32,
    current_command: PlayerOverrideCommand,
}

impl PlayerOverrideState {
    const VELOCITY_TIMEOUT: u32 = 5;

    pub fn new() -> Self {
        Self {
            frame_counter: 0,
            current_command: PlayerOverrideCommand::Stop,
        }
    }

    pub fn set_cmd(&mut self, cmd: PlayerOverrideCommand) {
        self.current_command = cmd;
        self.frame_counter = 0;
    }

    pub fn advance(&mut self) -> PlayerControlInput {
        let input = match self.current_command {
            PlayerOverrideCommand::Stop => PlayerControlInput::new(),
            PlayerOverrideCommand::MoveTo {
                position,
                yaw,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                position: Some(position),
                yaw: Some(yaw),
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::LocalVelocity {
                velocity,
                angular_velocity,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                velocity: Velocity::local(velocity),
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::GlobalVelocity {
                velocity,
                angular_velocity,
                dribble_speed,
                arm_kick,
            } => PlayerControlInput {
                velocity: Velocity::global(velocity),
                dribbling_speed: dribble_speed,
                kicker: if arm_kick {
                    KickerControlInput::Arm
                } else {
                    KickerControlInput::default()
                },
                ..Default::default()
            },
            PlayerOverrideCommand::Kick { speed } => PlayerControlInput {
                kicker: KickerControlInput::Kick,
                kick_speed: Some(speed),
                ..Default::default()
            },
            PlayerOverrideCommand::DischargeKicker => PlayerControlInput {
                kicker: KickerControlInput::Disarm,
                ..Default::default()
            },
            PlayerOverrideCommand::SetFanSpeed { speed } => PlayerControlInput {
                fan_speed: Some(speed),
                ..Default::default()
            },
        };

        // Advance the frame counter
        self.frame_counter += 1;
        match self.current_command {
            PlayerOverrideCommand::LocalVelocity { .. }
            | PlayerOverrideCommand::GlobalVelocity { .. } => {
                if self.frame_counter > Self::VELOCITY_TIMEOUT {
                    self.current_command = PlayerOverrideCommand::Stop;
                    self.frame_counter = 0;
                }
            }
            PlayerOverrideCommand::Kick { .. } => {
                self.current_command = PlayerOverrideCommand::Stop
            }
            _ => {}
        };

        input
    }
}
