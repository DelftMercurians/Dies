// test_script.rhai

fn hello_world() {
    print("Hello from test_script.rhai!");
    debug("This is a debug message from Rhai.");
}

// Call it for initial testing if TeamController tries to run a function.
// hello_world(); // We usually call functions from Rust side via engine.call_fn() 