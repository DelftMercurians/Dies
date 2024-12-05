use tokio::process::Command;

use dummy_strat::start_server;

#[tokio::test]
async fn test_spawn_with_join() -> Result<(), Box<dyn std::error::Error>> {
    // I tried to use tokio::spawn, but it requires + Send
    println!("Starting server and main process concurrently");
    let server1 = async { start_server().await };

    let server2 = async {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        run_client().await
    };

    // Run both concurrently and wait for them to finish
    let (res1, res2) = tokio::join!(server1, server2);

    res1?;
    res2?;

    Ok(())
}

async fn run_client() -> Result<(), Box<dyn std::error::Error>> {
    let mut child = Command::new("cargo")
        .arg("run")
        .arg("--package")
        .arg("dummy-strat")
        .spawn()?;

    let output = child.wait().await?;

    if !output.success() {
        return Err("Main process failed".into());
    }

    Ok(())
}
