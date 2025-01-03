use dies_core::{RobotCmd, RobotFeedback, RobotMainboardCmd};

pub(crate) fn convert_cmd(cmd: RobotCmd) -> glue::Radio_Command {
    match cmd {
        RobotCmd::Move(cmd) => glue::Radio_Command {
            speed: convert_pose(cmd),
            dribbler_speed: cmd.dribble_speed as f32,
            robot_command: convert_mainboard_cmd(cmd.mainboard_cmd),
            // TODO: Add kick time and fan speed
            kick_time: 5_000.0,
            fan_speed: 0.0,
            _pad: [0, 0, 0],
        },
        RobotCmd::SetHeadingReference { .. } => todo!(),
    }
}

pub(crate) fn convert_feedback(feedback: glue::Radio_Feedback) -> RobotFeedback {
    RobotFeedback {
        primary_status: Some(SysStatus::from(feedback.primary_status)),
        kicker_status: Some(SysStatus::from(feedback.kicker_status)),
        imu_status: Some(SysStatus::from(feedback.imu_status)),
        fan_status: Some(SysStatus::from(feedback.fan_status)),
        kicker_cap_voltage: Some(feedback.kicker_cap_voltage),
        kicker_temp: Some(feedback.kicker_temp),
        motor_statuses: Some([
            SysStatus::from(feedback.motor_statuses[0]),
            SysStatus::from(feedback.motor_statuses[1]),
            SysStatus::from(feedback.motor_statuses[2]),
            SysStatus::from(feedback.motor_statuses[3]),
            SysStatus::from(feedback.motor_statuses[4]),
        ]),
        motor_speeds: Some(feedback.motor_speeds),
        motor_temps: Some(feedback.motor_temps),
        breakbeam_ball_detected: Some(feedback.breakbeam_ball_detected),
        breakbeam_sensor_ok: Some(feedback.breakbeam_sensor_ok),
        pack_voltages: Some(feedback.pack_voltages),
    }
}

fn convert_mainboard_cmd(cmd: RobotMainboardCmd) -> glue::Radio_RobotCommand {
    match cmd {
        RobotMainboardCmd::None => glue::Radio_RobotCommand::NONE,
        RobotMainboardCmd::Arm => glue::Radio_RobotCommand::ARM,
        RobotMainboardCmd::Disarm => glue::Radio_RobotCommand::DISARM,
        RobotMainboardCmd::Discharge => glue::Radio_RobotCommand::DISCHARGE,
        RobotMainboardCmd::Kick => glue::Radio_RobotCommand::KICK,
        RobotMainboardCmd::Chip => glue::Radio_RobotCommand::CHIP,
        RobotMainboardCmd::PowerBoardOff => glue::Radio_RobotCommand::POWER_BOARD_OFF,
        RobotMainboardCmd::Reboot => glue::Radio_RobotCommand::REBOOT,
        RobotMainboardCmd::Beep => glue::Radio_RobotCommand::BEEP,
        RobotMainboardCmd::Coast => glue::Radio_RobotCommand::COAST,
        RobotMainboardCmd::HeadingControl => glue::Radio_RobotCommand::HEADING_CONTROL,
        RobotMainboardCmd::YawRateControl => glue::Radio_RobotCommand::YAW_RATE_CONTROL,
    }
}

fn convert_pose(cmd: MoveCmd) -> glue::HG_Pose {
    glue::HG_Pose {
        x: cmd.sx as f32,
        y: cmd.sy as f32,
        z: cmd.w as f32,
    }
}
