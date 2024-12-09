#!/bin/bash

# Trap ctrl-c and call ctrl_c()
trap ctrl_c INT


TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG_FILE_PATH="/home/mercury/.local/share/dies/dies-$TIMESTAMP.log"
REMOTE_DIR="Code/dies/$(whoami)/"

function pull_logs() {
    echo "** Pulling logs (from $LOG_FILE_PATH)..."
    mkdir -p ./logs
    scp mercury@mercuryvision-old:$LOG_FILE_PATH ./logs/
}

has_pressed_ctrl_c=false
function ctrl_c() {
    if $has_pressed_ctrl_c; then
        echo "** Forced exit"
        exit 1
    fi
    has_pressed_ctrl_c=true
    
    echo "** Trapped CTRL-C"
    ssh mercury@mercuryvision-old "killall -s INT dies-cli && echo 'Sent SIGINT'"
    sleep 0.5
    pull_logs
    ssh mercury@mercuryvision-old "killall dies-cli"
    exit 1
}

# If ./crates/dies-webui/static is empty, or if --web-build is passed, build the webui
if [ ! "$(ls -A ./crates/dies-webui/static)" ] || [ "$1" == "--web-build" ]; then
    shift 1
    echo "Building webui..."
    cargo make webui
fi

ssh mercury@mercuryvision-old "mkdir -p $REMOTE_DIR"

# Sync files to the remote server
rsync -avz --delete --exclude-from='.gitignore' --exclude-from='.rsyncignore' --exclude .git  . mercury@mercuryvision-old:$REMOTE_DIR

# Run the program on the remote server
ssh -L 5555:localhost:5555 mercury@mercuryvision-old "cd $REMOTE_DIR && /home/mercury/.cargo/bin/cargo run -- --log-file $LOG_FILE_PATH $@"

# Pull logs
pull_logs

echo "** Done"
