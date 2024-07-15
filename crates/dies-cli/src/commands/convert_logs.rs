use std::path::Path;

use dies_logger::LogFile;

fn msg_to_json(msg: &dies_logger::TimestampedMessage) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert(
        "timestamp".to_string(),
        serde_json::Value::from(msg.timestamp),
    );
    match &msg.message {
        dies_logger::LogMessage::DiesData(data) => {
            map.insert(
                "data".to_string(),
                serde_json::to_value(&data).unwrap(),
            );
        }
        dies_logger::LogMessage::DiesLog(_) => {}
        dies_logger::LogMessage::Vision(_) => {}
        dies_logger::LogMessage::Referee(_) => {}
        dies_logger::LogMessage::Bytes(_) => {}
    }
    serde_json::Value::Object(map)
}

pub fn convert_log(input: &Path, output: &Path) -> anyhow::Result<()> {
    println!("Converting log file {} to JSON", input.display());

    if !input.exists() || !input.is_file() {
        return Err(anyhow::anyhow!("Input file does not exist"));
    }
    if output.exists() {
        return Err(anyhow::anyhow!("Output file already exists"));
    }

    let logs = LogFile::open(input)?;

    let out = std::fs::File::create(output)?;
    let messages = logs.messages()
        .iter()
        .map(|msg| msg_to_json(msg))
        .collect::<Vec<_>>();

    serde_json::to_writer_pretty(out, &messages)?;

    println!("Converted {} messages", messages.len());
    println!("Output written to {}", output.display());

    Ok(())
}
