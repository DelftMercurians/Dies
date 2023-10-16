use anyhow::Result;
use dies::{PyRunner, StratCmd, StratMsg};
use env_logger::{Builder, Env};
use workspace_utils::get_workspace_root;

fn main() -> Result<()> {
    // Setup logging to stderr
    Builder::from_env(Env::default().default_filter_or("info")).init();

    let workspace = get_workspace_root();
    println!("Workspace: {}", workspace.display());
    let mut py_runner = PyRunner::new(workspace, "dies_py", "__main__")?;

    let msg = py_runner.recv()?;
    match msg {
        StratCmd::Debug { message } => println!("Debug: {}", message),
    }

    py_runner.send(&StratMsg::Hello {
        message: "Hello from Rust!".to_owned(),
    })?;

    Ok(())
}
