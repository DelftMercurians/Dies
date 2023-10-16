import sys
import zmq
import msgspec

from .messages import msg_dec, socket_info_dec, Term, Debug, Hello

if __name__ == '__main__':
    # Read socket info from stdin
    socket_info = socket_info_dec.decode(sys.stdin.readline().strip())
    print("Received socket info", socket_info)
    
    ctx = zmq.Context()
    
    # Connect to the strategy process
    msg_sock = ctx.socket(zmq.SUB)
    msg_sock.connect(f'tcp://localhost:{socket_info.pub_port}')
    msg_sock.setsockopt(zmq.SUBSCRIBE, b'')
    cmd_sock = ctx.socket(zmq.PUSH)
    cmd_sock.connect(f'tcp://localhost:{socket_info.pull_port}')
    
    # Send a test command
    cmd_sock.send(msgspec.json.encode(Debug(message='Hello, world!')))
    
    print("Listening for messages")
    while True:
        # Receive a message from the strategy process
        msg = msg_sock.recv()
        print("Received message", msg)
        msg = msg_dec.decode(msg)
        if isinstance(msg, Term):
            break
        elif isinstance(msg, Hello):
            print("Received hello message", msg)