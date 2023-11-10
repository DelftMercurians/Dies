use anyhow::Result;
use polling::{Events, Poller};
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};

use dies_core::Env;
use dies_protos::{
    ssl_simulation_robot_control::{
        MoveLocalVelocity, RobotCommand, RobotControl, RobotMoveCommand,
    },
    Message,
};

use socket2::{Domain, Protocol, SockAddr, Socket, Type};

use crate::docker_wrapper::DockerWrapper;

const BUF_SIZE: usize = 2 * 1024;
const VISION_SOCK_KEY: usize = 1;
const GC_SOCK_KEY: usize = 2;

#[derive(Debug, Clone)]
pub struct ErSimConfig {
    pub vision_host: String,
    pub vision_port: u16,
    pub gc_host: String,
    pub gc_port: u16,
    pub sim_control_local_host: String,
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
            sim_control_local_host: String::from("127.0.0.1"),
            sim_control_remote_host: String::from("127.0.0.1"),
            sim_control_remote_port: 10301,
        }
    }
}

pub struct ErSimEnv {
    #[allow(dead_code)]
    docker: DockerWrapper,
    poller: Poller,
    events: Events,
    buf: Vec<u8>,
    vision_socket: UdpSocket,
    gc_socket: UdpSocket,
    sim_control_socket: UdpSocket,
    sim_control_remote_addr: SocketAddr,
}

impl ErSimEnv {
    pub fn new(config: ErSimConfig) -> Result<Self> {
        let ErSimConfig {
            vision_host,
            vision_port,
            gc_host,
            gc_port,
            sim_control_local_host,
            sim_control_remote_host,
            sim_control_remote_port,
            ..
        } = config;
        let vision_host = vision_host.parse::<Ipv4Addr>()?;

        let docker = DockerWrapper::new("dies-ersim-env".into())?;

        let vision_sock = {
            let vision_sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
            vision_sock.set_reuse_address(true)?;

            let bind_addr = SockAddr::from(SocketAddr::new(vision_host.into(), vision_port));
            vision_sock.bind(&bind_addr)?;

            let host = "127.0.0.1".parse::<Ipv4Addr>()?;
            vision_sock.set_multicast_if_v4(&host)?;
            vision_sock.join_multicast_v4(&vision_host, &host)?;

            log::debug!(
                "Bound to simulator vision at {}:{}",
                vision_host,
                vision_port
            );

            vision_sock
        };

        let gc_sock = {
            let gc_sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
            gc_sock.set_reuse_address(true)?;

            let bind_addr = SockAddr::from(SocketAddr::new(
                gc_host.parse::<Ipv4Addr>()?.into(),
                gc_port,
            ));
            gc_sock.bind(&bind_addr)?;

            let host = "127.0.0.1".parse::<Ipv4Addr>()?;
            gc_sock.set_multicast_if_v4(&host)?;
            gc_sock.join_multicast_v4(&gc_host.parse::<Ipv4Addr>()?, &host)?;

            log::debug!("Bound to simulator GC at {}:{}", gc_host, gc_port);

            gc_sock
        };

        let sim_control_sock = {
            let host = sim_control_local_host.parse::<Ipv4Addr>()?;
            let sim_control_sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;

            let bind_addr = SockAddr::from(SocketAddr::new(host.into(), 0));
            sim_control_sock.bind(&bind_addr)?;

            log::debug!(
                "Created simulator control socket at {}:{}",
                host,
                sim_control_sock
                    .local_addr()?
                    .as_socket_ipv4()
                    .unwrap()
                    .port()
            );

            sim_control_sock
        };

        let vision_sock = UdpSocket::from(vision_sock);
        vision_sock.set_nonblocking(true)?;

        let gc_sock = UdpSocket::from(gc_sock);
        gc_sock.set_nonblocking(true)?;

        let poller = Poller::new()?;
        // Safety: We just created these sockets, so they are valid. We have to make
        //         sure to remove them from the poller before they are dropped.
        //         See the Drop impl for ErSimEnv.
        unsafe {
            poller.add(&vision_sock, polling::Event::readable(VISION_SOCK_KEY))?;
            poller.add(&gc_sock, polling::Event::readable(GC_SOCK_KEY))?;
        }

        Ok(Self {
            docker,
            poller,
            events: Events::new(),
            buf: vec![0; BUF_SIZE],
            vision_socket: vision_sock,
            gc_socket: gc_sock,
            sim_control_socket: sim_control_sock.into(),
            sim_control_remote_addr: SocketAddr::new(
                sim_control_remote_host.parse::<Ipv4Addr>()?.into(),
                sim_control_remote_port,
            ),
        })
    }
}

impl Drop for ErSimEnv {
    fn drop(&mut self) {
        self.poller.delete(&self.vision_socket).unwrap();
        self.poller.delete(&self.gc_socket).unwrap();
    }
}

impl Env for ErSimEnv {
    fn recv(&mut self) -> Vec<dies_core::EnvEvent> {
        if self.events.is_empty() {
            match self.poller.wait(&mut self.events, None) {
                Ok(_) => {}
                Err(err) => {
                    log::error!("Failed to poll sockets: {}", err);
                    return vec![];
                }
            }
        }

        let result: Vec<_> = self
            .events
            .iter()
            .filter_map(|ev| {
                if ev.key == VISION_SOCK_KEY {
                    let amt = match self.vision_socket.recv(&mut self.buf) {
                        Ok(amt) => amt,
                        Err(err) => {
                            if err.kind() == std::io::ErrorKind::WouldBlock {
                                return None;
                            } else {
                                log::error!("Failed to receive vision message: {}", err);
                                return None;
                            }
                        }
                    };
                    log::debug!("Received {} bytes from vision", amt);
                    match dies_core::VisionMsg::parse_from_bytes(&self.buf[..amt]) {
                        Ok(msg) => return Some(dies_core::EnvEvent::VisionMsg(msg)),
                        Err(err) => {
                            log::error!("Failed to parse vision message: {}", err);
                            return None;
                        }
                    };
                }
                if ev.key == GC_SOCK_KEY {
                    let amt = match self.gc_socket.recv(&mut self.buf) {
                        Ok(amt) => amt,
                        Err(err) => {
                            if err.kind() == std::io::ErrorKind::WouldBlock {
                                return None;
                            } else {
                                log::error!("Failed to receive GC message: {}", err);
                                return None;
                            }
                        }
                    };
                    log::debug!("Received {} bytes from GC", amt);
                    match dies_core::GcRefereeMsg::parse_from_bytes(&self.buf[..amt]) {
                        Ok(msg) => return Some(dies_core::EnvEvent::GcRefereeMsg(msg)),
                        Err(err) => {
                            log::error!("Failed to parse GC message: {}", err);
                            return None;
                        }
                    };
                }
                None
            })
            .collect();

        self.events.clear();

        return result;
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
