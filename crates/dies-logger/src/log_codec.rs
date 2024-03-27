use anyhow::{bail, Result};
use std::{io::Read, path::Path};
use tokio::{
    fs::File,
    io::{AsyncWrite, AsyncWriteExt},
};

use dies_protos::{
    dies_log_line::LogLine, ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket,
};
use protobuf::Message;

enum LogMessage {
    DiesLog(LogLine),
    Vision(SSL_WrapperPacket),
    Referee(Referee),
}

#[derive(Debug)]
enum LogFileMessageType {
    Blank = 0,
    Unknown = 1,
    SSLVision2010 = 2,
    SSLRefbox2013 = 3,
    SSLVision2014 = 4,
}

impl LogFileMessageType {
    fn from_i32(value: i32) -> LogFileMessageType {
        match value {
            0 => LogFileMessageType::Blank,
            1 => LogFileMessageType::Unknown,
            2 => LogFileMessageType::SSLVision2010,
            3 => LogFileMessageType::SSLRefbox2013,
            4 => LogFileMessageType::SSLVision2014,
            _ => LogFileMessageType::Unknown,
        }
    }
}

pub struct LogFileWriter {
    file: File,
    buf: Vec<u8>,
}

impl LogFileWriter {
    /// Open a new log file for writing and write the header.
    ///
    /// # Errors
    ///
    /// Returns an error if the file already exists or if an I/O error occurs.
    pub async fn open(path: impl AsRef<Path>) -> Result<Self> {
        if path.as_ref().exists() {
            bail!("Log file already exists: {:?}", path.as_ref());
        }

        let file = File::create(path).await?;
        let mut writer = LogFileWriter {
            file,
            buf: Vec::new(),
        };
        writer.write_header().await?;
        Ok(writer)
    }

    async fn write_header(&mut self) -> Result<()> {
        let header = "SSL_LOG_FILE".as_bytes();
        let version = 1i32.to_be_bytes();
        self.file.write_all(header).await?;
        self.file.write_all(&version).await?;
        Ok(())
    }

    /// Write a Dies log line to the log file.
    pub async fn write_log_line(&mut self, log_line: &LogLine) -> Result<()> {
        self.buf.clear();
        log_line.write_to_vec(&mut self.buf)?;
        self.write_message(LogFileMessageType::Blank).await
    }

    async fn write_message(&mut self, message_type: LogFileMessageType) -> Result<()> {
        let receiver_timestamp = 0i64.to_be_bytes();
        let message_type = message_type as i32;
        let message_size = self.buf.len() as i32;
        self.file.write_all(&receiver_timestamp).await?;
        self.file.write_all(&message_type.to_be_bytes()).await?;
        self.file.write_all(&message_size.to_be_bytes()).await?;
        self.file.write_all(&self.buf).await?;
        Ok(())
    }
}

struct LogFile {
    version: i32,
    messages: Vec<LogMessage>,
}

impl LogFile {
    fn read(mut source: impl Read) -> Result<Self> {
        // Read header + version
        let mut buf = [0u8; 12 + 4];
        source.read_exact(&mut buf)?;
        let file_type = String::from_utf8(buf[0..12].to_vec())?;
        if file_type != "SSL_LOG_FILE" {
            bail!("Invalid log file type: {}", file_type);
        }
        let version = i32::from_be_bytes(buf[12..16].try_into()?);
        if version != 1 {
            bail!("Unsupported log file version: {}", version);
        }

        // Read messages
        let mut messages = Vec::new();
        'msg_loop: loop {
            // Read message header
            let mut buf = [0u8; 16];
            match source.read_exact(&mut buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let _receiver_timestamp = i64::from_be_bytes(buf[0..8].try_into()?);
            let message_type =
                LogFileMessageType::from_i32(i32::from_be_bytes(buf[8..12].try_into()?));
            let message_size = i32::from_be_bytes(buf[12..16].try_into()?);

            // Read message
            let mut message_buf = vec![0u8; message_size as usize];
            match source.read_exact(&mut message_buf) {
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }

            let message = match message_type {
                LogFileMessageType::SSLVision2014 | LogFileMessageType::SSLVision2010 => {
                    let packet = match SSL_WrapperPacket::parse_from_bytes(&message_buf) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("Failed to parse SSL_WrapperPacket: {}", e);
                            continue;
                        }
                    };
                    LogMessage::Vision(packet)
                }
                LogFileMessageType::SSLRefbox2013 => {
                    let referee = match Referee::parse_from_bytes(&message_buf) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("Failed to parse Referee: {}", e);
                            continue;
                        }
                    };
                    LogMessage::Referee(referee)
                }
                _ => 'msg: {
                    // Try to parse as LogLine, then Vision, then Referee
                    if let Ok(log_line) = LogLine::parse_from_bytes(&message_buf) {
                        break 'msg LogMessage::DiesLog(log_line);
                    }
                    if let Ok(packet) = SSL_WrapperPacket::parse_from_bytes(&message_buf) {
                        break 'msg LogMessage::Vision(packet);
                    }
                    if let Ok(referee) = Referee::parse_from_bytes(&message_buf) {
                        break 'msg LogMessage::Referee(referee);
                    }
                    eprintln!("Failed to parse message");
                    continue 'msg_loop;
                }
            };

            messages.push(message);
        }

        Ok(LogFile { version, messages })
    }
}
