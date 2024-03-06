from math import sqrt, cos, sin, pi
import time
from glob import glob
import numpy as np
from dies_py import init, next, send, should_stop, world


print("Starting test-strat")

if __name__ == "__main__":
    init()
    print("Initialized")

    last_time = time.time()
    while next():
        w = world()
        dt = time.time() - last_time
        last_time = time.time()
        # print(f"[{dt:.0f}] World: {w}")
    print("Stopping test-strat")
