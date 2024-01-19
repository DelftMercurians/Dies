use anyhow::{Context, Result};
use dies_protos::Message;
use serialport::SerialPort;
use std::{io::Read, net::TcpStream, sync::Mutex, time::Duration};

use dies_core::{EnvConfig, EnvEvent, EnvReceiver, EnvSender};

const BUF_SIZE: usize = 2 * 1024;

/// Configuration for the environment.
#[derive(Debug, Clone)]
pub struct RobotTestConfig {
    pub port_name: String,
    pub vision_host: String,
    pub vision_port: u16,
}

/// Sender half of the environment.
pub struct RobotTestEnvSender {
    port: Mutex<Box<dyn SerialPort>>,
}

/// Receiver half of the environment.
pub struct RobotTestEnvReceiver {
    buf: [u8; BUF_SIZE],
    socket: TcpStream,
}

impl EnvSender for RobotTestEnvSender {
    fn send_player(&self, msg: dies_core::PlayerCmd) -> Result<()> {
        println!("{}", cmd);
        self.port
            .lock()
            .unwrap()
            .write_all(cmd.as_bytes())
            .context("Failed to write to port")?;
        Ok(())
    }
}

impl EnvReceiver for RobotTestEnvReceiver {
    fn recv(&mut self) -> Result<EnvEvent> {
        let data = self.socket.read(&mut self.buf)?;
        let mut msg = dies_protos::ssl_vision_wrapper::SSL_WrapperPacket::parse_from_bytes(
            &self.buf[..data],
        )?;

        if let Some(detection) = msg.detection.as_mut() {
            detection.robots_blue.retain(|robot| {
                robot.confidence.is_some_and(|c| c > 0.8)
                    && robot.robot_id.is_some_and(|id| id == 12)
            });
        }

        Ok(EnvEvent::VisionMsg(msg))
        // Err(anyhow::anyhow!("Not implemented"))
    }
}

impl EnvConfig for RobotTestConfig {
    fn build(self) -> Result<(Box<dyn EnvSender>, Box<dyn EnvReceiver>)> {
        let port = serialport::new(self.port_name.clone(), 115200)
            .timeout(Duration::from_millis(10))
            .open()
            .context("Failed to open port")?;
        let sender = Box::new(RobotTestEnvSender {
            port: Mutex::new(port),
        });

        let socket = TcpStream::connect(format!("{}:{}", self.vision_host, self.vision_port))
            .context("Failed to connect to vision")?;
        let receiver = Box::new(RobotTestEnvReceiver {
            socket,
            buf: [0; BUF_SIZE],
        });

        Ok((sender, receiver))
    }
}
