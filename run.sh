#!/bin/bash

# Trap ctrl-c and call ctrl_c()
trap ctrl_c INT

function ctrl_c() {
    echo "** Trapped CTRL-C"
    ssh mercury@merucryvision "killall dies-cli"
    exit 1
}

rsync -avz --delete --exclude-from='.gitignore' --exclude .git  . mercury@merucryvision:Code/dies/
ssh -L 5555:localhost:5555 mercury@merucryvision "cd Code/dies && /home/mercury/.cargo/bin/cargo run -- $@"