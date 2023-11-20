mod executor;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use dies_python_rt::{create_py_runtime, PyRuntimeConfig};
use executor::run;

use dies_ersim_env::{create_ersim_env, ErSimConfig};

fn main() {
    let env = create_ersim_env(ErSimConfig::default()).expect("Failed to create ersim env");
    let rt = create_py_runtime(PyRuntimeConfig {
        workspace: std::env::current_dir().unwrap(),
        package: "dies".into(),
        module: "dies".into(),
        sync: true,
    })
    .expect("Failed to create python runtime");

    let should_stop = Arc::new(AtomicBool::new(false));

    ctrlc::set_handler({
        let should_stop = should_stop.clone();
        move || {
            should_stop.store(true, Ordering::Relaxed);
        }
    })
    .expect("Failed to set ctrl-c handler");

    run(env, rt, should_stop).expect("Failed to run executor");
}
