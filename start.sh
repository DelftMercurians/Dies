#!/bin/bash

# trap ctrl-c and call ctrl_c()
trap ctrl_c INT

function ctrl_c() {
    echo "** Trapped CTRL-C"
    exit 0
}

# auto restart if the process is dead
while true
do
    cargo run -- --start-scenario play --interface enp4s0
    echo "Server crashed, restarting..."
done