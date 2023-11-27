mod executor;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use dies_core::{EnvConfig, RuntimeConfig};
use dies_python_rt::PyRuntimeConfig;
use executor::run;

use dies_ersim_env::ErSimConfig;

fn main() {
    env_logger::init();

    let env = ErSimConfig::default()
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
