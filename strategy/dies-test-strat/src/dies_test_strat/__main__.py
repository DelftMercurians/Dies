from math import sqrt, cos, sin, pi
import time
from glob import glob
from dies_py.messages import PlayerCmd
import numpy as np
from dies_py import init, next_world, send, should_stop, world


print("Starting test-strat")

if __name__ == "__main__":
    init()
    print("Initialized")

    print("Sleeping for 2 seconds...")
    time.sleep(2)
    print("Staring!")
    last_time = time.time()
    is_first = True
    # to_save = []
    try:
        while next_world():
            send(PlayerCmd(5, 0, 0.5))
            if is_first:
                w = world()
                print(w.own_players[0])
                is_first = False
            # p = w.own_players[0]
            # to_save.append((p.timestamp, p.position[0], p.position[1], p.velocity[0], p.velocity[1]))
            if time.time() - last_time > 4:
                break
    except KeyboardInterrupt:
        pass
    # find last file named log##.npy
    # files = glob("log*.npy")
    # if len(files) > 0:
    #     last_file = sorted(files)[-1]
    #     last_num = int(last_file[3:-4])
    # else:
    #     last_num = 0
    # fn = f"log{last_num+1:02d}.npy"
    # np.save(fn, np.array(to_save))
    # print(f"Saving to {fn}")
    print("Stopping test-strat")
