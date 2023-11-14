use anyhow::Result;
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};

use dies_core::Env;
use dies_protos::{
    ssl_simulation_robot_control::{
        MoveLocalVelocity, RobotCommand, RobotControl, RobotMoveCommand,
    },
    Message,
};

use crate::docker_wrapper::DockerWrapper;

const BUF_SIZE: usize = 2 * 1024;

#[derive(Debug, Clone)]
pub struct ErSimConfig {
    pub bridge_port: u16,
    pub sim_control_remote_host: String,
    pub sim_control_remote_port: u16,
}

impl Default for ErSimConfig {
    fn default() -> Self {
        Self {
            bridge_port: 10050,
            sim_control_remote_host: String::from("127.0.0.1"),
            sim_control_remote_port: 10301,
        }
    }
}

pub struct ErSimEnv {
    #[allow(dead_code)]
    docker: DockerWrapper,
    buf: Vec<u8>,
    rx_socket: UdpSocket,
    sim_control_socket: UdpSocket,
    sim_control_remote_addr: SocketAddr,
}

impl ErSimEnv {
    pub fn new(config: ErSimConfig) -> Result<Self> {
        let ErSimConfig {
            bridge_port,
            sim_control_remote_host,
            sim_control_remote_port,
            ..
        } = config;

        let rx_socket = UdpSocket::bind(format!("127.0.0.1:{}", bridge_port))?;
        log::debug!("Bound bridge rx socket to {:?}", rx_socket.local_addr());
        let sim_control_socket = UdpSocket::bind("127.0.0.1:0")?;
        log::debug!(
            "Bound sim control socket to {:?}",
            sim_control_socket.local_addr()
        );

        Ok(Self {
            docker: DockerWrapper::new("dies-ersim-env".into())?,
            buf: vec![0; BUF_SIZE],
            rx_socket,
            sim_control_socket,
            sim_control_remote_addr: SocketAddr::new(
                sim_control_remote_host.parse::<Ipv4Addr>()?.into(),
                sim_control_remote_port,
            ),
        })
    }
}

impl Env for ErSimEnv {
    fn recv(&mut self) -> Vec<dies_core::EnvEvent> {
        let data = match self.rx_socket.recv(&mut self.buf) {
          Ok(data) => data,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::WouldBlock {
                    return vec![];
                } else {
                    log::error!("Failed to receive data: {}", err);
                    return vec![];
                }
            }  
        };

        // Check first byte for packet type
        match self.buf[0] {
            0 => {
                // Vision message
                let msg = match dies_protos::ssl_vision_wrapper::SSL_WrapperPacket::parse_from_bytes(&self.buf[1..data]) {
                    Ok(msg) => msg,
                    Err(err) => {
                        log::error!("Failed to parse vision message: {}", err);
                        return vec![];
                    }
                };
                vec![dies_core::EnvEvent::VisionMsg(msg)]
            },
            1 => {
                // GC message
                let msg = match dies_protos::ssl_gc_referee_message::Referee::parse_from_bytes(&self.buf[1..data]) {
                    Ok(msg) => msg,
                    Err(err) => {
                        log::error!("Failed to parse GC message: {}", err);
                        return vec![];
                    }
                };
                vec![dies_core::EnvEvent::GcRefereeMsg(msg)]
            },
            _ => {
                log::error!("Received message with unknown type: {}", self.buf[0]);
                vec![]
            },
        }
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
