#!/bin/bash

# auto restart if the process is dead
while true
do
    cargo run -- --start-scenario play
    echo "Server crashed, restarting..."
done