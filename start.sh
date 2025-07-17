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
    cargo run -- --serial-port='/dev/tty.usbmodem4983466939361' --interface en7 --controlled-teams=yellow --auto-start
    echo "Server crashed, restarting..."
done