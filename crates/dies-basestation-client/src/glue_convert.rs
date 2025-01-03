use dies_core::{RobotCmd, RobotFeedback, RobotMainboardCmd, RobotMoveCmd, SysStatus};

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

pub(crate) fn convert_sys_status(status: glue::HG_Status) -> SysStatus {
    match status {
        glue::HG_Status::EMERGENCY => SysStatus::Emergency,
        glue::HG_Status::OK => SysStatus::Ok,
        glue::HG_Status::READY => SysStatus::Ready,
        glue::HG_Status::STOP => SysStatus::Stop,
        glue::HG_Status::STARTING => SysStatus::Starting,
        glue::HG_Status::OVERTEMP => SysStatus::Overtemp,
        glue::HG_Status::NO_REPLY => SysStatus::NoReply,
        glue::HG_Status::ARMED => SysStatus::Armed,
        glue::HG_Status::DISARMED => SysStatus::Disarmed,
        glue::HG_Status::SAFE => SysStatus::Safe,
        glue::HG_Status::NOT_INSTALLED => SysStatus::NotInstalled,
        glue::HG_Status::STANDBY => SysStatus::Standby,
    }
}

pub(crate) fn convert_sys_status_opt(status: Option<glue::HG_Status>) -> Option<SysStatus> {
    status.map(|status| convert_sys_status(status))
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

fn convert_pose(cmd: RobotMoveCmd) -> glue::HG_Pose {
    glue::HG_Pose {
        x: cmd.sx as f32,
        y: cmd.sy as f32,
        z: cmd.w as f32,
    }
}
