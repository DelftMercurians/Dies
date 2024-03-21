use git2::Repository;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs,
    io::{self, prelude::*},
    path::Path,
    process::{exit, Command},
};
use walkdir::WalkDir;

// pb 1. line 204: deletes files on the remote, but no dirs => should delete dirs too (if no coresponding dir in local remove it from remote)
// pb 2. It's slow
// - minimize nr of ssh commands
// combine creates dirs + computing hashes
// later - upload all files with a single command
// parallelize
// while computes hashes on the remote, compute them locally as well
// when testing change remote_path to Code/test

fn main() {
    let repo_path = std::env::current_dir().expect("Failed to get current directory");
    let remote_host = "merucryvision";
    let remote_user = "mercury";
    let remote_path = "/home/mercury/Code/dies/";
    let repo = Repository::open(repo_path.clone()).expect("Failed to open repository");

    // Create remote directories
    println!("Creating remote directories...");
    create_dirs(&repo, &repo_path, remote_host, remote_user, remote_path);

    // Get remote file hashes
    println!("Getting remote file hashes and dirs...");
    // let remote_file_hashes = get_remote_file_hashes(remote_host, remote_user, remote_path);
    let (remote_file_hashes, remote_dirs) =
        get_remote_file_hashes_and_dirs(remote_host, remote_user, remote_path);

    println!("Copying files...");
    copy_files(
        &repo,
        &repo_path,
        remote_host,
        remote_user,
        remote_path,
        &remote_file_hashes,
    );

    println!("Removing unmatched remote files...");
    remove_unmatched_remote_files_and_dirs(
        &repo_path,
        remote_host,
        remote_user,
        remote_path,
        &remote_file_hashes,
        &remote_dirs,
    );

    // check for empty folders on remote not present locally and delete them
    // get all empty folders on remote

    // Compile and run
    println!("Compiling and running...");
    ssh_command_with_pty(remote_host, remote_user, "cd ~/Code/dies && /home/mercury/.cargo/bin/cargo run -- --vision udp --vision-addr 224.5.23.2:10006");
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

fn get_remote_file_hashes_and_dirs(
    remote_host: &str,
    remote_user: &str,
    remote_path: &str,
) -> (HashMap<String, String>, HashSet<String>) {
    let output = Command::new("ssh")
        .arg(format!("{}@{}", remote_user, remote_host))
        .arg(format!(
            "cd {} && find . -type d \\( -path ./target -o -path ./.venv \\) -prune -o \\( -type f -exec sha256sum {{}} + \\) -o \\( -type d -print \\)",
            remote_path
        ))
        .output()
        .expect("Failed to execute command");

    let output_str = String::from_utf8_lossy(&output.stdout);

    let mut file_hashes = HashMap::new();
    let mut dirs = HashSet::new();

    for line in output_str.lines() {
        let mut parts = line.split_whitespace();
        let pair = (parts.next(), parts.next());
        if let (Some(hash), Some(path)) = pair {
            let path = path.strip_prefix("./").unwrap_or(path);
            file_hashes.insert(path.to_string(), hash.to_string());
        } else if let (Some(path), _) = pair {
            dirs.insert(path.strip_prefix("./").unwrap_or(path).to_string());
        }
    }

    (file_hashes, dirs)
}

fn create_dirs(
    repo: &Repository,
    repo_path: &Path,
    remote_host: &str,
    remote_user: &str,
    remote_path: &str,
) {
    println!("Getting dirs");
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

    println!("Creating dirs: {}", dirs);

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

fn copy_files(
    repo: &Repository,
    repo_path: &Path,
    remote_host: &str,
    remote_user: &str,
    remote_path: &str,
    remote_file_hashes: &HashMap<String, String>,
) {
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
            .arg(format!(
                "{}@{}:{}",
                remote_user, remote_host, remote_file_path
            ))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .expect("Failed to copy file");

        if !scp_res.status.success() {
            panic!(
                "Failed to copy file: {}",
                String::from_utf8_lossy(&scp_res.stderr)
            );
        }
    }
}

fn remove_unmatched_remote_files_and_dirs(
    repo_path: &Path,
    remote_host: &str,
    remote_user: &str,
    remote_path: &str,
    remote_file_hashes: &HashMap<String, String>,
    remote_dirs: &HashSet<String>,
) {
    // Collect all local files as relative paths
    let local_files = WalkDir::new(repo_path)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.path().is_file())
        .map(|entry| {
            let path = entry.path().strip_prefix(repo_path).unwrap();
            path.to_str().unwrap().replace("\\", "/") // Convert to unix style path
        })
        .collect::<HashSet<String>>();

    // Determine which remote files are not present locally
    let files_to_remove = remote_file_hashes
        .keys()
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

    // collect all remote dirs that are not present locally
    let dirs_to_remove = remote_dirs
        .iter()
        .filter(|remote_dir| !local_files.contains(*remote_dir))
        .cloned()
        .collect::<Vec<String>>();

    println!(
        "Removing the following dirs on the remote: {}",
        dirs_to_remove.join(", ")
    );

    // Construct a single ssh command to remove all unmatched files and dirs
    let mut remove_commands = files_to_remove
        .iter()
        .map(|file| format!("rm -f '{}/{}'", remote_path, file))
        .collect::<Vec<_>>()
        .join(" && ");

    let remove_dirs_commands = dirs_to_remove
        .iter()
        .map(|dir| format!("rm -rf '{}/{}'", remote_path, dir))
        .collect::<Vec<_>>()
        .join(" && ");

    if (!remove_commands.is_empty()) && (!remove_dirs_commands.is_empty()) {
        remove_commands = format!("{} && {}", remove_commands, remove_dirs_commands);
    } else if remove_dirs_commands.is_empty() {
        remove_commands = remove_commands;
    } else {
        remove_commands = remove_dirs_commands;
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delete_empty_folder() {
        // create folder tests on this repository
        let repo_path = std::env::current_dir()
            .expect("Failed to get current directory")
            .join("..") // Move up one directory
            .join("..") // Move up another directory
            .canonicalize() // Resolve the path to its absolute form, removing any '..' components.
            .expect("Failed to resolve path");

        let test_path = repo_path.join("test");
        if !test_path.exists() {
            let _ = fs::create_dir(&test_path);
        }

        // add test.txt file to test folder with some content
        let test_file_path = test_path.join("test.txt");
        fs::write(&test_file_path, "test content").expect("Failed to write file");

        // run run-on-server to create the same folder on remote
        let output = Command::new("powershell")
            .arg(format!(
                "cd {} ; cargo make run-on-server",
                repo_path.to_str().unwrap()
            ))
            .output()
            .expect("Failed to execute command");

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("Command output: {}", stdout);
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Error executing command: {}", stderr);
        }

        // check if test folder was created on remote
        let output = Command::new("powershell")
            .arg(format!(
                "cd {} ; ssh mercury@merucryvision 'ls ~/Code/test-dies'",
                repo_path.to_str().unwrap()
            ))
            .output()
            .expect("Failed to execute command");

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("Command output: {}", stdout);
            assert!(stdout.contains("test"));
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Error executing command: {}", stderr);
        }

        // delete test.txt file from local
        fs::remove_file(test_file_path).expect("Failed to remove file");

        // run run-on-server to delete the file from remote
        let output = Command::new("powershell")
            .arg(format!(
                "cd {} ; cargo make run-on-server",
                repo_path.to_str().unwrap()
            ))
            .output()
            .expect("Failed to execute command");

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("Command output: {}", stdout);
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Error executing command: {}", stderr);
        }

        // check if test folder was deleted from remote
        let output = Command::new("powershell")
            .arg(format!(
                "cd {} ; ssh mercury@merucryvision 'ls ~/Code/test-dies'",
                repo_path.to_str().unwrap()
            ))
            .output()
            .expect("Failed to execute command");

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("Command output: {}", stdout);
            assert!(!stdout.contains("test"));
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Error executing command: {}", stderr);
        }
    }
}
