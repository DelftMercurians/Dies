
from .bridge import Bridge, read_socket_info
from .messages import Term, Debug, Hello

if __name__ == '__main__':
    # Read socket info from stdin
    socket_info = read_socket_info()
    print("Received socket info", socket_info)

    bridge = Bridge(socket_info)
    
    # Send a test command
    bridge.send(Debug(message='Hello, world!'))
    
    print("Listening for messages")
    while True:
        msg = bridge.recv()
        print("Received message", msg)
        if isinstance(msg, Term):
            break
        elif isinstance(msg, Hello):
            print("Received hello message", msg)
    
    print("Exiting")