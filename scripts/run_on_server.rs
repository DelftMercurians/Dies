//! ```cargo
//! [dependencies]
//! git2 = "0.18.2"
//! sha2 = "0.10.8"
//! walkdir = "2.4.0"
//! hex = "0.4.3"
//! ```

use git2::Repository;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet}, fs, io::{self, prelude::*}, path::Path, process::{exit, Command}
};
use walkdir::WalkDir;

fn main() {
    let repo_path = std::env::current_dir().expect("Failed to get current directory");
    let remote_host = "merucryvision";
    let remote_user = "mercury";
    let remote_path = "/home/mercury/Code/dies/";
    let repo = Repository::open(repo_path.clone()).expect("Failed to open repository");

    // Create remote directory
    println!("Creating remote directory...");
    ssh_command(remote_host, remote_user, "mkdir -p ~/Code/dies");

    // Create remote directories
    println!("Creating remote directories...");
    create_dirs(&repo, &repo_path, remote_host, remote_user, remote_path);

    // Get remote file hashes
    println!("Getting remote file hashes...");
    let remote_file_hashes = get_remote_file_hashes(remote_host, remote_user, remote_path);

    println!("Copying files...");
    copy_files(&repo, &repo_path, remote_host, remote_user, remote_path, &remote_file_hashes);

    println!("Removing unmatched remote files...");
    remove_unmatched_remote_files(&repo, &repo_path, remote_host, remote_user, remote_path, &remote_file_hashes);

    // Compile and run
    println!("Compiling and running...");
    ssh_command_with_pty(remote_host, remote_user, "cd ~/Code/dies && /home/mercury/.cargo/bin/cargo run -- --vision-socket-type udp --vision-host 224.5.23.2 --vision-port 10006");
}

fn ssh_command(remote_host: &str, remote_user: &str, command: &str) {
    let status = Command::new("ssh")
        .arg(format!("{}@{}", remote_user, remote_host))
        .arg(command)
        .status()
        .expect("failed to execute command");

    assert!(status.success());
}

fn ssh_command_with_pty(remote_host: &str, remote_user: &str, command: &str) -> ! {
    let output = Command::new("ssh")
        .arg("-t")
        .arg(format!("{}@{}", remote_user, remote_host))
        .arg(command)
        .stdin(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .output()
        .expect("failed to execute command");

    match output.status.code() {
        Some(0) => exit(0),
        Some(code) => {
            eprintln!("Command failed with exit code {}", code);
            exit(code);
        }
        None => {
            eprintln!("Command was terminated by a signal");
            exit(1);
        }
    }
}

fn get_remote_file_hashes(remote_host: &str, remote_user: &str, remote_path: &str) -> HashMap<String, String> {
    let output = Command::new("ssh")
        .arg(format!("{}@{}", remote_user, remote_host))
        .arg(format!(
            "cd {} && find . -type d \\( -path ./target -o -path ./.venv \\) -prune -o -type f -exec sha256sum {{}} +",
            remote_path
        ))
        .output()
        .expect("Failed to execute command");

    let output_str = String::from_utf8_lossy(&output.stdout);
    output_str.lines().fold(HashMap::new(), |mut acc, line| {
        let mut parts = line.split_whitespace();
        if let (Some(hash), Some(path)) = (parts.next(), parts.next()) {
            let path = path.strip_prefix("./").unwrap_or(path);
            acc.insert(path.to_string(), hash.to_string());
        }
        acc
    })
}

fn create_dirs(repo: &Repository, repo_path: &Path, remote_host: &str, remote_user: &str, remote_path: &str) {
    let dirs = WalkDir::new(repo_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .filter(|e| !repo.status_should_ignore(e.path()).unwrap())
        .map(|entry| {
            let path = entry.path();
            let relative_path = path.strip_prefix(repo_path).unwrap();
            let relative_path = relative_path
                .iter()
                .map(|s| s.to_string_lossy().to_string())
                .collect::<Vec<_>>()
                .join("/");
            let remote_dir_path = if !remote_path.ends_with('/') && !relative_path.starts_with('/')
            {
                format!("{}/{}", remote_path, relative_path)
            } else {
                format!("{}{}", remote_path, relative_path)
            };
            remote_dir_path
        })
        .collect::<Vec<_>>()
        .join(" ");

    let mkdir_command = format!("mkdir -p {}", dirs);

    let status = Command::new("ssh")
        .arg(format!("{}@{}", remote_user, remote_host))
        .arg(mkdir_command)
        .status()
        .expect("Failed to execute ssh command");

    if !status.success() {
        panic!("Failed to create remote directories");
    }
}

fn copy_files(repo: &Repository, repo_path: &Path, remote_host: &str, remote_user: &str, remote_path: &str, remote_file_hashes: &HashMap<String, String>) {
    for entry in WalkDir::new(repo_path) {
        let entry = entry.unwrap();
        let path = entry.path();
        if repo.status_should_ignore(path).unwrap() || !path.is_file() {
            continue;
        }

        let relative_path = path.strip_prefix(repo_path).unwrap();
        // Convert to unix style path
        let relative_path = relative_path
            .iter()
            .map(|s| s.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join("/");
        let remote_file_path = if !remote_path.ends_with('/') && !relative_path.starts_with('/') {
            format!("{}/{}", remote_path, relative_path)
        } else {
            format!("{}{}", remote_path, relative_path)
        };

        // Hash the file to check if it has changed
        let local_hash = hash_file(path).unwrap();

        // Check if the file has changed
        if let Some(remote_hash) = remote_file_hashes.get(&relative_path) {
            if local_hash == *remote_hash {
                continue;
            }
        }

        println!("Copying {}", relative_path);
        let scp_res = Command::new("scp")
            .arg(path.to_str().unwrap())
            .arg(format!("{}@{}:{}", remote_user, remote_host, remote_file_path))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .expect("Failed to copy file");

        if !scp_res.status.success() {
            panic!("Failed to copy file: {}", String::from_utf8_lossy(&scp_res.stderr));
        }
    }
}

fn remove_unmatched_remote_files(repo: &Repository, repo_path: &Path, remote_host: &str, remote_user: &str, remote_path: &str, remote_file_hashes: &HashMap<String, String>) {
    // Collect all local files as relative paths
    let local_files = WalkDir::new(repo_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().is_file() && !repo.status_should_ignore(e.path()).unwrap())
        .map(|entry| {
            let path = entry.path().strip_prefix(repo_path).unwrap();
            path.to_str().unwrap().replace("\\", "/") // Convert to unix style path
        })
        .collect::<HashSet<String>>();

    // Determine which remote files are not present locally
    let files_to_remove = remote_file_hashes.keys()
        .filter(|remote_file| !local_files.contains(*remote_file))
        .cloned()
        .collect::<Vec<String>>();

    if files_to_remove.is_empty() {
        return;
    }

    // Print the files to be removed
    println!("Removing the following files on the remote:");
    for file in &files_to_remove {
        println!("{}", file);
    }

    // Construct a single ssh command to remove all unmatched files
    let remove_commands = files_to_remove.iter()
        .map(|file| format!("rm -f '{}/{}'", remote_path, file))
        .collect::<Vec<_>>()
        .join(" && ");

    // Execute the command via ssh
    let status = Command::new("ssh")
        .arg(format!("{}@{}", remote_user, remote_host))
        .arg(remove_commands)
        .status()
        .expect("Failed to execute ssh command for removing files");

    if !status.success() {
        eprintln!("Failed to remove unmatched remote files");
    }
}

fn hash_file<P: AsRef<Path>>(path: P) -> Result<String, io::Error> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    hasher.update(&buffer);
    Ok(hex::encode(hasher.finalize()))
}
