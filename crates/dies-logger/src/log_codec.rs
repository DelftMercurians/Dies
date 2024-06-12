use anyhow::{bail, Result};
use std::{io::Read, path::Path};
use tokio::{
    fs::File,
    io::{AsyncWriteExt, BufWriter},
};

use dies_protos::{
    dies_log_line::LogLine, ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket,
};
use protobuf::Message;

// *GENERAL NOTE*: All log files use big-endian byte order for all integers.

const LOG_FILE_HEADER: &str = "SSL_LOG_FILE";

#[derive(Debug, Clone, PartialEq)]
pub enum LogMessage {
    DiesLog(LogLine),
    Vision(SSL_WrapperPacket),
    Referee(Referee),
    Bytes(Vec<u8>),
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
    writer: BufWriter<File>,
    buf: Vec<u8>,
}

impl LogFileWriter {
    /// Open a new log file for writing and write the header.
    ///
    /// # Errors
    ///
    /// Returns an error if the file already exists or if an I/O error occurs.
    pub async fn open(path: impl AsRef<Path>, version: u32) -> Result<Self> {
        if path.as_ref().exists() {
            bail!("Log file already exists: {:?}", path.as_ref());
        }

        let file = File::create(path).await?;
        let mut writer = LogFileWriter {
            writer: BufWriter::new(file),
            buf: Vec::new(),
        };
        writer.write_header(version).await?;
        Ok(writer)
    }

    async fn write_header(&mut self, version: u32) -> Result<()> {
        let header = LOG_FILE_HEADER.as_bytes();
        self.writer.write_all(header).await?;
        self.writer
            .write_all(&(version as i32).to_be_bytes())
            .await?;
        self.writer.flush().await?;
        Ok(())
    }

    pub async fn write_log_message(&mut self, message: &LogMessage) -> Result<()> {
        match message {
            LogMessage::DiesLog(log_line) => self.write_log_line(&log_line).await,
            LogMessage::Vision(vision) => self.write_vision(&vision).await,
            LogMessage::Referee(referee) => self.write_referee(&referee).await,
            LogMessage::Bytes(bytes) => self.write_bytes(&bytes).await,
        }
    }

    pub async fn write_vision(&mut self, vision: &SSL_WrapperPacket) -> Result<()> {
        self.buf.clear();
        vision.write_to_vec(&mut self.buf)?;
        self.write_message(LogFileMessageType::SSLVision2014).await
    }

    /// Write a referee message to the log file.
    pub async fn write_referee(&mut self, referee: &Referee) -> Result<()> {
        self.buf.clear();
        referee.write_to_vec(&mut self.buf)?;
        self.write_message(LogFileMessageType::SSLRefbox2013).await
    }

    /// Write a Dies log line to the log file.
    pub async fn write_log_line(&mut self, log_line: &LogLine) -> Result<()> {
        self.buf.clear();
        log_line.write_to_vec(&mut self.buf)?;
        self.write_message(LogFileMessageType::Blank).await
    }

    /// Write a raw byte buffer to the log file.
    pub async fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.buf.clear();
        self.buf.extend_from_slice(bytes);
        self.write_message(LogFileMessageType::Blank).await
    }

    /// Flush the log file.
    pub async fn flush(&mut self) -> Result<()> {
        self.writer.flush().await?;
        Ok(())
    }

    async fn write_message(&mut self, message_type: LogFileMessageType) -> Result<()> {
        let receiver_timestamp = 0i64.to_be_bytes();
        let message_type = message_type as i32;
        let message_size = self.buf.len() as i32;
        self.writer.write_all(&receiver_timestamp).await?;
        self.writer.write_all(&message_type.to_be_bytes()).await?;
        self.writer.write_all(&message_size.to_be_bytes()).await?;
        self.writer.write_all(&self.buf).await?;
        Ok(())
    }
}

pub struct LogFile {
    version: u32,
    messages: Vec<LogMessage>,
}

impl LogFile {
    pub fn read(mut source: impl Read) -> Result<Self> {
        // Read header + version
        let mut buf = [0u8; 4 + LOG_FILE_HEADER.len()];
        source.read_exact(&mut buf)?;
        let file_type = String::from_utf8(buf[0..12].to_vec())?;
        if file_type != LOG_FILE_HEADER {
            bail!("Invalid log file type: {}", file_type);
        }
        let version = i32::from_be_bytes(buf[12..16].try_into()?) as u32;
        if version != 1 {
            bail!("Unsupported log file version: {}", version);
        }

        // Read messages
        let mut messages = Vec::new();
        'msg_loop: loop {
            // Read message header
            let mut buf = [0u8; 8 + 4 + 4];
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
            source.read_exact(&mut message_buf)?;

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
                LogFileMessageType::Blank => {
                    if let Ok(log_line) = LogLine::parse_from_bytes(&message_buf) {
                        LogMessage::DiesLog(log_line)
                    } else {
                        LogMessage::Bytes(message_buf.clone())
                    }
                }
                LogFileMessageType::Unknown => 'msg: {
                    // Try to parse as Vision, then Referee
                    if let Ok(packet) = SSL_WrapperPacket::parse_from_bytes(&message_buf) {
                        break 'msg LogMessage::Vision(packet);
                    }
                    if let Ok(referee) = Referee::parse_from_bytes(&message_buf) {
                        break 'msg LogMessage::Referee(referee);
                    }
                    continue 'msg_loop;
                }
            };

            messages.push(message);
        }

        Ok(LogFile { version, messages })
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        println!(
            "{} file len: {:?}",
            path.as_ref().display(),
            std::fs::read(path.as_ref())?.len()
        );
        Self::read(std::fs::File::open(path)?)
    }

    pub fn messages(&self) -> &[LogMessage] {
        &self.messages
    }

    pub fn version(&self) -> u32 {
        self.version
    }
}

#[cfg(test)]
mod tests {
    use dies_protos::dies_log_line::LogLevel;
    use flate2::read::GzDecoder;
    use tempfile::NamedTempFile;

    use super::*;

    const TEST_FILE: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/test.log.gz"));

    #[test]
    fn test_read_standard_file() {
        let test_data = GzDecoder::new(TEST_FILE)
            .bytes()
            .collect::<Result<Vec<u8>, _>>()
            .unwrap();
        let log_file = LogFile::read(test_data.as_slice()).unwrap();

        assert_eq!(log_file.version, 1);
        assert_eq!(log_file.messages.len(), 409519);

        // Check that all messages are either vision or referee
        for message in &log_file.messages {
            match message {
                LogMessage::Vision(_) | LogMessage::Referee(_) => {}
                LogMessage::Bytes(_) => panic!("Unexpected message type: Bytes"),
                LogMessage::DiesLog(_) => panic!("Unexpected message type: DiesLog"),
            }
        }
    }

    #[tokio::test]
    async fn test_read_standard_file_then_write() {
        let temp = NamedTempFile::new().unwrap();
        if temp.path().exists() {
            std::fs::remove_file(temp.path()).unwrap();
        }

        let test_data = GzDecoder::new(TEST_FILE)
            .bytes()
            .collect::<Result<Vec<u8>, _>>()
            .unwrap();
        let log_file = LogFile::read(test_data.as_slice()).unwrap();
        let mut writer = LogFileWriter::open(temp.path(), log_file.version())
            .await
            .unwrap();

        for message in &log_file.messages {
            writer.write_log_message(message).await.unwrap();
        }

        // Close writer
        writer.flush().await.unwrap();
        drop(writer);

        let written_log = LogFile::open(temp.path()).unwrap();
        assert_eq!(written_log.version, log_file.version);
        assert_eq!(written_log.messages(), log_file.messages());
    }

    #[tokio::test]
    async fn test_roundtrip() {
        let temp = NamedTempFile::new().unwrap();
        if temp.path().exists() {
            std::fs::remove_file(temp.path()).unwrap();
        }

        let version = 1;
        let mut writer = LogFileWriter::open(temp.path(), version).await.unwrap();

        let messages = vec![
            LogMessage::DiesLog(LogLine {
                level: Some(LogLevel::DEBUG.into()),
                target: Some("test".to_string()),
                message: Some("test".to_string()),
                source: Some("unknown:0".to_string()),
                ..Default::default()
            }),
            LogMessage::Vision(SSL_WrapperPacket::default()),
            LogMessage::DiesLog(LogLine {
                level: Some(LogLevel::DEBUG.into()),
                target: Some("test".to_string()),
                message: Some("test".to_string()),
                source: Some("unknown:0".to_string()),
                ..Default::default()
            }),
        ];
        for message in &messages {
            writer.write_log_message(message).await.unwrap();
        }

        // Close writer
        writer.flush().await.unwrap();
        drop(writer);

        // Check that the written file is the same as the original
        let log_file = LogFile::open(temp.path()).unwrap();
        assert_eq!(log_file.version, version);
        assert_eq!(log_file.messages(), messages);
    }
}
