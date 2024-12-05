use interprocess::local_socket::{
    tokio::{prelude::*, Stream}, GenericNamespaced, ListenerOptions,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader}, process::Command, try_join
}; 
use std::io;

#[tokio::test]
async fn test_spawn_with_join() -> Result<(), Box<dyn std::error::Error>> {
    // I tried to use tokio::spawn, but it requires + Send
    println!("Starting server and main process concurrently");
    let server1 = async {
        start_server().await
    };
    
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

async fn start_server() -> Result<(), Box<dyn std::error::Error>> {

	// Describe the things we do when we've got a connection ready.
	async fn handle_conn(conn: Stream) -> io::Result<()> {
		let mut recver = BufReader::new(&conn);
		let mut sender = &conn;

		// Allocate a sizeable buffer for receiving. This size should be big enough and easy to
		// find for the allocator.
		let mut buffer = String::with_capacity(128);

		// Describe the send operation as sending our whole message.
		let send = sender.write_all(b"Hello from server!\n");
		// Describe the receive operation as receiving a line into our big buffer.
		let recv = recver.read_line(&mut buffer);

		// Run both operations concurrently.
		try_join!(recv, send)?;

		// Produce our output!
		println!("Client answered: {}", buffer.trim());
		Ok(())
	}

	// Pick a name.
	let printname = "example.sock";
	let name = printname.to_ns_name::<GenericNamespaced>()?;

	// Configure our listener...
	let opts = ListenerOptions::new().name(name);

	// ...and create it.
	let listener = match opts.create_tokio() {
		Err(e) if e.kind() == io::ErrorKind::AddrInUse => {
			// When a program that uses a file-type socket name terminates its socket server
			// without deleting the file, a "corpse socket" remains, which can neither be
			// connected to nor reused by a new listener. Normally, Interprocess takes care of
			// this on affected platforms by deleting the socket file when the listener is
			// dropped. (This is vulnerable to all sorts of races and thus can be disabled.)
			//
			// There are multiple ways this error can be handled, if it occurs, but when the
			// listener only comes from Interprocess, it can be assumed that its previous instance
			// either has crashed or simply hasn't exited yet. In this example, we leave cleanup
			// up to the user, but in a real application, you usually don't want to do that.
			eprintln!(
				"
Error: could not start server because the socket file is occupied. Please check if {printname}
is in use by another process and try again."
			);
			return Err(e.into());
		}
		x => x?,
	};

	// The syncronization between the server and client, if any is used, goes here.
	eprintln!("Server running at {printname}");

	// Set up our loop boilerplate that processes our incoming connections.
	loop {
		// Sort out situations when establishing an incoming connection caused an error.
		let conn = match listener.accept().await {
			Ok(c) => c,
			Err(e) => {
				eprintln!("There was an error with an incoming connection: {e}");
				continue;
			}
		};

		// Spawn new parallel asynchronous tasks onto the Tokio runtime and hand the connection
		// over to them so that multiple clients could be processed simultaneously in a
		// lightweight fashion.
		tokio::spawn(async move {
			// The outer match processes errors that happen when we're connecting to something.
			// The inner if-let processes errors that happen during the connection.
			if let Err(e) = handle_conn(conn).await {
				eprintln!("Error while handling connection: {e}");
			}
		});
	}
} 



/*
types should be serializable
trait serialize
Make the server generic <StrategyInput, StrategyOutput>
comit + rebase to main
*/