use dummy_strat::{StrategyInput, StrategyOutput};
use interprocess::local_socket::{
    tokio::{prelude::*, Stream},
    GenericFilePath, GenericNamespaced,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt, BufReader},
    try_join,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ToDo why would we want generic namespaced if generic file path is always supported?
    let name = if GenericNamespaced::is_supported() {
        "example.sock".to_ns_name::<GenericNamespaced>()?
    } else {
        "/tmp/example.sock".to_fs_name::<GenericFilePath>()?
    };

    // Await this here since we can't do a whole lot without a connection.
    let conn = Stream::connect(name).await?;

    // This consumes our connection and splits it into two halves, so that we can concurrently use
    // both.
    let (recver, mut sender) = conn.split();
    let mut recver = BufReader::new(recver);

    // Allocate a buffer with a fixed size of 128 bytes
    let mut buffer = vec![0u8; 1024]; // 128-byte buffer

    // Create a StrategyOutput instance to send
    let strategy_output = StrategyOutput {
        message: String::from("Hello from client!"),
    };

    // Serialize StrategyOutput to JSON
    let serialized_output =
        serde_json::to_vec(&strategy_output).expect("Failed to serialize output");

    // Describe the send operation as sending the serialized StrategyOutput
    let send = sender.write_all(&serialized_output);

    // Describe the receive operation as receiving the serialized StrategyInput
    let recv = recver.read(&mut buffer);

    // Concurrently perform both operations.
    try_join!(send, recv)?;

    // Close the connection a bit earlier than you'd think we would. Nice practice!
    drop((recver, sender));

    // Deserialize the received data into StrategyInput
    let strategy_input: StrategyInput =
        serde_json::from_slice(&buffer).expect("Failed to deserialize input");

    // Produce output after receiving the input from server
    println!("Server sent: {}", strategy_input.message);
    Ok(())
}
