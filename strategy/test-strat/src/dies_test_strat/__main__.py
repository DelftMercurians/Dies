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

    while True:
        msg = bridge.recv()
        if isinstance(msg, Term):
            break
        elif isinstance(msg, World):
            print("Received world")
            print(msg)
            time.sleep(1)
    bridge.close()
    print("Exiting test-strat")
