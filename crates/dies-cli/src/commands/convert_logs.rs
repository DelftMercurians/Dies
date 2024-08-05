use std::{
    io::{BufWriter, Write},
    path::Path,
};

use anyhow::Context;
use dies_logger::LogFile;

fn msg_to_json(msg: &dies_logger::TimestampedMessage) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "timestamp".to_string(),
        serde_json::Value::from(msg.timestamp),
    );
    match &msg.message {
        dies_logger::LogMessage::DiesData(data) => {
            map.insert("data".to_string(), serde_json::to_value(data).unwrap());
        }
        dies_logger::LogMessage::DiesLog(log) => {
            map.insert(
                "log".to_string(),
                log.message.clone().unwrap_or_default().into(),
            );
        }
        dies_logger::LogMessage::Vision(msg) => {
            map.insert(
                "vision".to_string(),
                protobuf_json_mapping::print_to_string(msg)
                    .unwrap_or_default()
                    .into(),
            );
        }
        dies_logger::LogMessage::Referee(msg) => {
            map.insert(
                "referee".to_string(),
                protobuf_json_mapping::print_to_string(msg)
                    .unwrap_or_default()
                    .into(),
            );
        }
        dies_logger::LogMessage::Bytes(_) => {}
    }
    serde_json::Value::Object(map)
}

pub fn convert_log(input: &Path, output: &Path) -> anyhow::Result<()> {
    println!("Converting log file {} to JSON", input.display());
    if !input.exists() || !input.is_file() {
        return Err(anyhow::anyhow!("Input file does not exist"));
    }

    let logs = LogFile::open(input)?;
    let out_file = std::fs::File::create(output)?;
    let mut writer = BufWriter::new(out_file);

    writer.write_all(b"[")?;

    let mut first = true;
    let mut count = 0;

    for msg in logs.messages() {
        if !first {
            writer.write_all(b",")?;
        } else {
            first = false;
        }

        let json = msg_to_json(msg);
        serde_json::to_writer(&mut writer, &json).context("Failed to serialize message")?;

        count += 1;
    }

    writer.write_all(b"]")?;
    writer.flush()?;

    println!("Converted {} messages", count);
    println!("Output written to {}", output.display());
    Ok(())
}
