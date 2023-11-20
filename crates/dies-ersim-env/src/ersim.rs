use anyhow::Result;
use std::{
    net::{IpAddr, SocketAddr, UdpSocket},
    str::FromStr,
    sync::Arc,
};

use dies_core::{EnvEvent, EnvReceiver, EnvSender};
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
pub struct ErSimEnvSender {
    #[allow(dead_code)]
    docker: Arc<DockerWrapper>,
    sim_control_socket: UdpSocket,
    sim_control_remote_addr: SocketAddr,
}

impl EnvSender for ErSimEnvSender {
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

pub struct ErSimEnvReceiver {
    #[allow(dead_code)]
    docker: Arc<DockerWrapper>,
    rx: RecvTransport,
}

impl EnvReceiver for ErSimEnvReceiver {
    fn recv(&mut self) -> Result<EnvEvent> {
        self.rx.recv()
    }
}

pub fn create_ersim_env(config: ErSimConfig) -> Result<(Box<dyn EnvSender>, Box<dyn EnvReceiver>)> {
    let docker = Arc::new(DockerWrapper::new("dies-ersim-env".into())?);

    let sim_control_socket = UdpSocket::bind("127.0.0.1:0")?;
    log::debug!(
        "Bound sim control socket to {:?}",
        sim_control_socket.local_addr()
    );
    let sender = Box::new(ErSimEnvSender {
        docker: Arc::clone(&docker),
        sim_control_socket,
        sim_control_remote_addr: SocketAddr::new(
            IpAddr::from_str(&config.sim_control_remote_host)?,
            config.sim_control_remote_port,
        ),
    });

    let rx = RecvTransport::new(&config)?;
    let receiver = Box::new(ErSimEnvReceiver { docker, rx });

    Ok((sender, receiver))
}
