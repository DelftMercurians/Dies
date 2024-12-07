use ipc_channel::ipc::IpcSender;
use test_ipc::StrategyInput;

mod config;

fn main() {
    let tx: IpcSender<StrategyInput> = IpcSender::connect(config::get_server_name()).unwrap();
    
    let data = StrategyInput {
        message: "Hello from client".to_string(),
    };

    tx.send(data).unwrap();
}