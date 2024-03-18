from math import sqrt, cos, sin, pi
import time
from glob import glob
from dies_py.messages import PlayerCmd
import numpy as np
from dies_py import init, next_world, send, should_stop, world

#from mpc_rust_gen.mpc import make_solver, mpc_control
from dies_test_strat.mpc import make_solver, mpc_control 


# import mpc
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
        

        N = 40
        x_init = []
        x_ref = [-750, -750]
        obstacles = []
        u_constraints = [-0.5, 0.5]
        ts = 0.02
        prev_u = [1.0] * (2*N)
        solver = make_solver(N, [], u_constraints, obstacles, obstacles, ts)

        def global_to_local_vel(velx, vely, theta):
            theta = theta + pi / 4
            new_x = -vely * sin(-theta) + velx * cos(-theta)
            new_y = vely * cos(-theta) + velx * sin(-theta)
            return new_x, new_y

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
            if len(w.own_players) != 0: 
                x_init = w.own_players[0].position 
                x_init = [x_init[0] / 1000, x_init[1] / 1000]
                x_ref = [-750 / 1000, -750 / 1000] 
                
                # if the distnace between the player and the target is less than 50mm, then stop
                if sqrt((x_init[0] - x_ref[0])**2 + (x_init[1] - x_ref[1])**2) < 0.2:
                    send(PlayerCmd(rid, 0, 0))
                    break
                print(f"X_init: {x_init} | x_ref: {x_ref} | obstacles: {obstacles} | prev_u: {prev_u} | ts: {ts}")
                u_mpc, prev_u = mpc_control(solver, x_init, x_ref, obstacles, prev_u)
                u_x, u_y = global_to_local_vel(u_mpc[0], u_mpc[1], w.own_players[0].orientation)
                send(PlayerCmd(rid, u_x, u_y))

                print(f"Position: {x_init}| Delay: {time.time() - prev_time}")
                prev_time = time.time()
            else:
                print("No player found")


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
