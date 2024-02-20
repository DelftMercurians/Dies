from math import sqrt, cos, sin, pi
import time
from glob import glob
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd, Term, World
import matplotlib.pyplot as plt


print("Starting test-strat")

if __name__ == "__main__":
    bridge = Bridge()

    start_time = None
    start_pos = None
    to_save = []
    rid = 5
    while True:
        msg = bridge.recv()
        if isinstance(msg, Term):
            break
        if not msg:
            continue
        player = next((p for p in msg.own_players if p.id == rid), None)
        if not player:
            # print("Player not found")
            continue
        pos = np.array([player.position[0], player.position[1]])

        if start_time is None:
            start_time = time.time()
            start_pos = pos
        else:
            dist = np.linalg.norm(pos - start_pos)
            elapsed = time.time() - start_time
            to_save.append((elapsed, dist))
            if dist > 2:
                print("Distance travelled:", dist)
                print("Time taken:", elapsed)
                break

        bridge.send(PlayerCmd(rid, 10, 0))
        time.sleep(10e-4)
    np.save("test-strat.npy", to_save)
    print("Exiting test-strat")
