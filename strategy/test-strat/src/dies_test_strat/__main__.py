from dies_py import Bridge


print("Starting test-strat")

if __name__ == "__main__":
    bridge = Bridge()
    last_pos = (0, 0, 0)
    while True:
        msg = bridge.recv()
        if msg:
            new_pos = msg.ball.position
            if new_pos != last_pos:
                print(f"Ball moved to {new_pos}")
                last_pos = new_pos
