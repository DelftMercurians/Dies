use anyhow::Result;
use dies_ssl_client::SslVisionClientConfig;
use dies_world::WorldConfig;
use tokio::{sync::oneshot, time};

use dies_python_rt::PyRuntimeConfig;
use dies_serial_client::list_serial_ports;
use tokio_util::sync::CancellationToken;

mod executor;

use crate::executor::{run, ExecutorConfig};

// This is the main function of the program. It's asynchronous, which means it can perform tasks
// without blocking the execution of the rest of the program. The `#[tokio::main]` attribute is
// used to enable the async runtime provided by the Tokio crate.
#[tokio::main]
async fn main() -> Result<()> {
    // This initializes a logger with a warning level filter. This means that only log messages
    // with a severity of warning or higher will be displayed.
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    // This calls a function that lists the available serial ports. The `?` operator is used for
    // error propagation. If the function returns an error, the main function will immediately
    // return that error and the rest of the code will not be executed.
    let ports = list_serial_ports()?;

    // This block of code checks if there are any available serial ports.
    let port = if !ports.is_empty() {
        // If there's only one port, it prints a message and connects to that port.
        if ports.len() == 1 {
            println!("Connecting to serial port {}", ports[0]);
            Some(ports[0].clone())
        } else {
            // If there are multiple ports, it prints all available ports and prompts the user to
            // choose one.
            println!("Available ports:");
            for (idx, port) in ports.iter().enumerate() {
                println!("{}: {}", idx, port);
            }

            // This loop keeps asking for user input until a valid port number is entered.
            loop {
                println!("Enter port number:");
                let mut input = String::new();
                // This line reads a line from the standard input and stores it in the `input`
                // string.
                std::io::stdin().read_line(&mut input)?;
                // This line trims the input string and tries to parse it into a `usize` (an
                // unsigned integer type).
                let port_idx = input.trim().parse::<usize>()?;
                // If the entered number is less than the length of the list, the program breaks
                // the loop and connects to the chosen port.
                if port_idx < ports.len() {
                    break Some(ports[port_idx].clone());
                } else {
                    // If the entered number is not valid (either it's not a number or it's out of
                    // range), the program prints an error message and prompts the user again.
                    println!("Invalid port number");
                }
            }
        }
    } else {
        // If there are no available serial ports, the program prints a message and won't be
        // connecting to a basestation.
        println!("No serial ports available, not connecting to basestation");
        None
    };

// This block of code is creating a configuration for an Executor. The ExecutorConfig struct
// presumably controls the behavior of some kind of task executor in the program.

let config = ExecutorConfig {
    // This field determines whether a web user interface should be used. It's currently set to
    // false, so no web UI will be used.
    webui: false,

    // This field is a HashMap that maps robot IDs to some other integer. In this case, the robot
    // with ID 14 is mapped to 2, and the robot with ID 5 is mapped to 3.
    robot_ids: std::collections::HashMap::from([(14, 2), (5, 3)]),

    // This field is a configuration for a Python runtime. It specifies the workspace directory,
    // the Python package to use, the module to run, and whether the Python code should be run
    // synchronously.
    py_config: PyRuntimeConfig {
        workspace: std::env::current_dir().unwrap(),
        package: "dies_test_strat".into(),
        module: "__main__".into(),
        sync: true,
    },

    // This field is a configuration for the world. It specifies whether the team is blue and the
    // initial x-coordinate of the opponent's goal.
    world_config: WorldConfig {
        is_blue: true,
        initial_opp_goal_x: 1.0,
    },

    // This field is an optional configuration for an SSL vision client. If it's Some, the client
    // will connect to a server at localhost on port 6078 using TCP. If it's None, no vision client
    // will be used.
    vision_config: Some(SslVisionClientConfig {
        host: "localhost".to_string(),
        port: 6078,
        socket_type: dies_ssl_client::SocketType::Tcp,
    }),

    // This field is an optional configuration for a serial client. If a port was selected earlier
    // in the program, the client will connect to that port with the default settings. If no port
    // was selected, no serial client will be used.
    serial_config: match port {
        Some(port) => Some(dies_serial_client::SerialClientConfig {
            port_name: port.clone(),
            ..Default::default()
        }),
        None => None,
    },
};

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    run(config, cancel_clone.clone())
        .await
        .expect("Failed to run executor");

    // let ctrc = match tokio::signal::ctrl_c().await {
    //     Ok(fut) => fut,
    //     Err(err) => {
    //         eprintln!("Unable to listen for shutdown signal: {}", err);
    //         // Send stop command
    //         cancel.cancel();
    //         handle.await?;
    //         return Ok(());
    //     }
    // };
    // tokio::select! {
    //     // _ = tokio::signal::ctrl_c() => {}
    //     _ = &mut handle => {}
    // };

    println!("Shutting down");

    // Send stop command
    // cancel.cancel();

    // if !handle.is_finished() {
    //     handle.abort();
    //     handle.await.ok();
    // }

    Ok(())
}
