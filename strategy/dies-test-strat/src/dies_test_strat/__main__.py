import numpy as np
import time
from dies_py import init, next as next_world, world, send
from dies_py.messages import PlayerCmd, PlayerPosCmd


print("Starting test-strat")

if __name__ == "__main__":
    init()
    print("Initialized")

    start_time = time.time()
    start_pos = None
    while next_world():
        w = world()
        player = next((p for p in w.own_players if p.id == 5), None)
        if player:
            send(PlayerCmd(5, 100, 0))
            pos = np.array([player.position[0], player.position[1]])
            if start_pos is None:
                start_pos = pos
            if np.linalg.norm(start_pos - pos) > 50:
                dt = time.time() - start_time
                print(f"dist: {np.linalg.norm(start_pos - pos)}, time: {dt}")
                break
    print("Stopping test-strat")
