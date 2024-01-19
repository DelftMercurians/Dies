mod executor;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use dies_core::{EnvConfig, EnvEvent, PlayerCmd, RuntimeConfig};
use dies_ersim_env::ErSimConfig;
use dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
use dies_python_rt::PyRuntimeConfig;
use dies_robot_test_env::RobotTestConfig;

use crate::executor::run;

fn list_ports() -> Vec<String> {
    let ports = serialport::available_ports().expect("No ports found!");
    ports.into_iter().map(|p| p.port_name).collect()
}

fn main() {
    env_logger::init();

    // Print ports
    println!("Available ports:");
    for port in list_ports() {
        println!("  {}", port);
    }

    let env = RobotTestConfig {
        port_name: "/dev/ttyACM0".into(),
        vision_host: String::from("localhost"),
        vision_port: 6078,
    }
    .build()
    .expect("Failed to create ersim env");
    let rt = PyRuntimeConfig {
        workspace: std::env::current_dir().unwrap(),
        package: "dies_test_strat".into(),
        module: "__main__".into(),
        sync: false,
    }
    .build()
    .expect("Failed to create python runtime");

    let should_stop = Arc::new(AtomicBool::new(false));

    ctrlc::set_handler({
        let should_stop = should_stop.clone();
        move || {
            println!("Stopping...");
            should_stop.store(true, Ordering::Relaxed);
        }
    })
    .expect("Failed to set ctrl-c handler");

    run(env, rt, should_stop).expect("Failed to run executor");
}
