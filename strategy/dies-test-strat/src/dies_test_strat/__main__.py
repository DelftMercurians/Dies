from math import sqrt, cos, sin, pi
import time
from glob import glob
from dies_py.messages import PlayerCmd
import numpy as np
from dies_py import init, next_world, send, should_stop, world
from mpc-rust-gen import make_solver, mpc_control
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
        prev_time = time.time()
        

        N = 100
        x_init = []
        x_ref = [-750, -750]
        obstacles = []
        u_constraints = [-3.0, 3.0]
        ts = 0.02
        prev_u = [1.0] * (2*N)
        solver = make_solver(N, x_init, x_ref, obstacles, u_constraints, ts)

        rid = 14
        while next_world():
            # send(PlayerCmd(5, 0, 0.5))
            # if is_first:
            #     w = world()
            #     print(w.own_players[0])
            #     is_first = False
            # # p = w.own_players[0]
            # # to_save.append((p.timestamp, p.position[0], p.position[1], p.velocity[0], p.velocity[1]))
            # if time.time() - last_time > 4:
            #     break

            w = world()
            x_init = w.own_players[0].position
            x_ref = [-750, -750] # in mm
            
            # if the distnace between the player and the target is less than 50mm, then stop
            if sqrt((x_init[0] - x_ref[0])**2 + (x_init[1] - x_ref[1])**2) < 50:
                send(PlayerCmd(rid, 0, 0))
                break

            u_mpc, prev_u = mpc_control(solver, x_init, x_ref, obstacles, prev_u)
            send(PlayerCmd(rid, u_mpc[0], u_mpc[1]))
            
            print(f"Position: {x_init}| Delay: {time.time() - prev_time}")
            prev_time = time.time()


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
