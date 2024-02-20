use dies_core::workspace_utils;
use git2::Repository;
use sha2::{Digest, Sha256};
use ssh2::Session;
use std::{
    collections::HashMap,
    fs,
    io::{self, prelude::*, BufReader},
    net::TcpStream,
    path::Path,
};
use walkdir::WalkDir;

pub(crate) fn run_on_server() {
    let repo_path = workspace_utils::get_workspace_root();
    let remote_host = "100.119.147.55";
    let remote_user = "mercury";
    let remote_path = "/home/mercury/Code/dies/";
    let repo = Repository::open(repo_path).expect("Failed to open repository");

    // Establish an SSH session
    println!("Connecting to {}", remote_host);
    let tcp = TcpStream::connect(format!("{}:22", remote_host))
        .expect("Failed to connect to remote host");
    let mut session = Session::new().unwrap();
    session.set_tcp_stream(tcp);
    session.handshake().expect("Failed to handshake");
    session
        .userauth_agent(remote_user)
        .expect("Failed to authenticate");

    let mut channel = session.channel_session().expect("Failed to open channel");
    channel
        .exec("mkdir -p ~/Code/dies")
        .expect("Failed to create directory");
    channel.wait_eof().expect("Failed to wait for EOF");
    channel
        .wait_close()
        .expect("Failed to wait for channel to close");

    // Get remote file hashes
    println!("Getting remote file hashes...");
    let mut channel = session.channel_session().expect("Failed to open channel");
    channel
        .exec(&format!(
            "cd {} && find . -type d \\( -path ./target -o -path ./.venv \\) -prune -o -type f -exec sha256sum {{}} +",
            remote_path,
        ))
        .expect("Failed to execute command");
    let mut remote_file_hashes = HashMap::new();
    {
        let reader = BufReader::new(channel);
        for line in reader.lines() {
            let line = line.unwrap();
            let mut parts = line.split_whitespace();
            if let (Some(hash), Some(path)) = (parts.next(), parts.next()) {
                let path = path.strip_prefix("./").unwrap_or(path);
                remote_file_hashes.insert(path.to_string(), hash.to_string());
            }
        }
    };

    println!("Copying files...");

    // List files and copy them respecting .gitignore
    for entry in WalkDir::new(repo_path) {
        let entry = entry.unwrap();
        let path = entry.path();
        if path.is_dir() || repo.status_should_ignore(path).unwrap() {
            continue;
        }

        let relative_path = path.strip_prefix(repo_path).unwrap().to_str().unwrap();
        let remote_file_path = format!("{}/{}", remote_path, relative_path.replace("\\", "/"));

        // Hash the file to check if it has changed
        let local_hash = hash_file(path).unwrap();

        // Check if the file has changed
        if let Some(remote_hash) = remote_file_hashes.get(relative_path) {
            if local_hash == *remote_hash {
                continue;
            }
        }
        println!("Copying {}", relative_path);

        let mut remote_file = session
            .scp_send(
                Path::new(&remote_file_path),
                0o644,
                entry.metadata().unwrap().len(),
                None,
            )
            .unwrap();

        let mut local_file = fs::File::open(path).unwrap();
        let mut contents = Vec::new();
        local_file.read_to_end(&mut contents).unwrap();
        remote_file.write_all(&contents).unwrap();
    }

    println!("Compiling and running...");

    let mut channel = session.channel_session().unwrap();
    channel.request_pty("xterm", None, None).unwrap();
    channel
        .exec("cd ~/Code/dies && /home/mercury/.cargo/bin/cargo run")
        .unwrap();

    // Create buffered reader
    let stdout = io::BufReader::new(channel.stream(0));
    for line in stdout.lines() {
        match line {
            Ok(line) => println!("{}", line),
            Err(e) => eprintln!("stdout error: {}", e),
        }
    }

    // Ensure the channel is closed after command execution
    channel.send_eof().expect("Failed to send EOF");
    channel.wait_eof().expect("Failed to wait for EOF");
    channel
        .wait_close()
        .expect("Failed to wait for channel to close");

    // Wait for SSH channel to close.
    channel.wait_close().unwrap();
    let exit_status = channel.exit_status().unwrap();
    println!("Exit status: {}", exit_status);
}

fn hash_file<P: AsRef<Path>>(path: P) -> Result<String, std::io::Error> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    hasher.update(&buffer);
    Ok(hex::encode(hasher.finalize()))
}
