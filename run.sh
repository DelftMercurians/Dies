#!/bin/bash

# Trap ctrl-c and call ctrl_c()
trap ctrl_c INT


TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG_FILE_PATH="/home/mercury/.local/share/dies/dies-$TIMESTAMP.log"

function pull_logs() {
    echo "** Pulling logs (from $LOG_FILE_PATH)..."
    mkdir -p ./logs
    scp mercury@mercuryvision:$LOG_FILE_PATH ./logs/
}

has_pressed_ctrl_c=false
function ctrl_c() {
    if $has_pressed_ctrl_c; then
        echo "** Forced exit"
        exit 1
    fi
    has_pressed_ctrl_c=true
    
    echo "** Trapped CTRL-C"
    ssh mercury@mercuryvision "killall dies-cli"
    pull_logs
    exit 1
}

# If ./crates/dies-webui/static is empty, or if --web-build is passed, build the webui
if [ ! "$(ls -A ./crates/dies-webui/static)" ] || [ "$1" == "--web-build" ]; then
    shift 1
    echo "Building webui..."
    cd ./webui
    npm run build
    cd ../
fi

# Sync files to the remote server
rsync -avz --delete --exclude-from='.gitignore' --exclude-from='.rsyncignore' --exclude .git  . mercury@mercuryvision:Code/dies/

# Run the program on the remote server
ssh -L 5555:localhost:5555 mercury@mercuryvision "cd Code/dies && /home/mercury/.cargo/bin/cargo run -- --log-file $LOG_FILE_PATH $@"

# Pull logs
pull_logs

echo "** Done"
