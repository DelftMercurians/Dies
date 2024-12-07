use ipc_channel::ipc::IpcOneShotServer;
use test_ipc::StrategyInput;

mod config;

fn main() {
    let (server, server_name) = IpcOneShotServer::new().unwrap();

    config::set_server_name(server_name);

    let (_, data): (_, StrategyInput) = server.accept().unwrap();
    println!("Received {:?}", data);
}