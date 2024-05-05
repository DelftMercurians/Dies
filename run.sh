#!/bin/bash

# Trap ctrl-c and call ctrl_c()
trap ctrl_c INT

function ctrl_c() {
    echo "** Trapped CTRL-C"
    ssh mercury@merucryvision "killall dies-cli"
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

rsync -avz --delete --exclude-from='.gitignore' --exclude .git  . mercury@merucryvision:Code/dies/
ssh -L 5555:localhost:5555 mercury@merucryvision "cd Code/dies && /home/mercury/.cargo/bin/cargo run -- $@"