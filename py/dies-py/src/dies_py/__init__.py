# import json
# import sys

# def recv():
#     """Receive a message through stdin and decode it."""
#     line = sys.stdin.readline()
#     return json.loads(line)

# def send(msg):
#     """Encode a message and send it through stdout."""
#     encoded = json.dumps(msg)
#     print(encoded)
#     sys.stdout.flush()