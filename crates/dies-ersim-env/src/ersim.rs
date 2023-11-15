use anyhow::Result;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};

use dies_core::Env;
use dies_protos::{
    ssl_simulation_robot_control::{
        MoveLocalVelocity, RobotCommand, RobotControl, RobotMoveCommand,
    },
    Message,
};

use crate::{docker_wrapper::DockerWrapper, RecvTransport};

#[derive(Debug, Clone)]
pub struct ErSimConfig {
    pub vision_host: String,
    pub vision_port: u16,
    pub gc_host: String,
    pub gc_port: u16,
    pub bridge_port: u16,
    pub sim_control_remote_host: String,
    pub sim_control_remote_port: u16,
}

impl Default for ErSimConfig {
    fn default() -> Self {
        Self {
            vision_host: String::from("224.5.23.2"),
            vision_port: 10020,
            gc_host: String::from("224.5.23.1"),
            gc_port: 11003,
            bridge_port: 10050,
            sim_control_remote_host: String::from("127.0.0.1"),
            sim_control_remote_port: 10301,
        }
    }
}

pub struct ErSimEnv {
    #[allow(dead_code)]
    docker: DockerWrapper,
    rx: RecvTransport,
    sim_control_socket: UdpSocket,
    sim_control_remote_addr: SocketAddr,
}

impl ErSimEnv {
    pub fn new(config: ErSimConfig) -> Result<Self> {
        let sim_control_socket = UdpSocket::bind("127.0.0.1:0")?;
        log::debug!(
            "Bound sim control socket to {:?}",
            sim_control_socket.local_addr()
        );

        Ok(Self {
            docker: DockerWrapper::new("dies-ersim-env".into())?,
            rx: RecvTransport::new(&config)?,
            sim_control_socket,
            sim_control_remote_addr: SocketAddr::new(
                config.sim_control_remote_host.parse::<Ipv4Addr>()?.into(),
                config.sim_control_remote_port,
            ),
        })
    }
}

impl Env for ErSimEnv {
    fn recv(&mut self) -> Vec<dies_core::EnvEvent> {
        self.rx.recv()
    }

    fn send_player(&self, msg: dies_core::PlayerCmd) -> Result<()> {
        let mut move_local_vel = MoveLocalVelocity::new();
        move_local_vel.set_left(msg.sx);
        move_local_vel.set_forward(msg.sy);
        move_local_vel.set_angular(msg.w);

        let mut robot_move_cmd = RobotMoveCommand::new();
        robot_move_cmd.set_local_velocity(move_local_vel);

        let mut robot_cmd = RobotCommand::new();
        robot_cmd.set_id(msg.id);
        robot_cmd.move_command = Some(robot_move_cmd).into();

        let mut robot_control = RobotControl::new();
        robot_control.robot_commands.push(robot_cmd);
        let buf = match robot_control.write_to_bytes() {
            Ok(buf) => buf,
            Err(err) => {
                log::error!("Failed to serialize robot control message: {}", err);
                return Err(err.into());
            }
        };

        match self
            .sim_control_socket
            .send_to(&buf, self.sim_control_remote_addr)
        {
            Ok(_) => Ok(()),
            Err(err) => {
                log::error!("Failed to send robot control message: {}", err);
                Err(err.into())
            }
        }
    }
}
