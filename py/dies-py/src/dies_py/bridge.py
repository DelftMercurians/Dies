import signal
import os
import socket
import sys
from threading import Event, Thread
import msgspec

from .messages import Ping, Term, World, msg_dec, Cmd, Msg

_sock = None
_world_state = None
_world_update_ev = Event()
_stop = False


def init():
    """Initialize the bridge. This must be called before any other function."""
    global _sock
    if _sock is not None:
        raise RuntimeError("Bridge already initialized")
    host = os.environ.get("DIES_IPC_HOST", "localhost")
    cmd_port = int(os.environ.get("DIES_IPC_PORT", 0))
    if cmd_port == 0:
        raise RuntimeError("DIES_IPC_PORT not set")

    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _sock.settimeout(1)
    _sock.connect((host, cmd_port))
    # Send a ping
    send(Ping())

    # Start the receive loop
    recv_thread = Thread(target=_recv_loop, daemon=True)
    recv_thread.start()

    # Set signal handler for SIGINT
    signal.signal(signal.SIGINT, _sig_handler)


def next_world() -> bool:
    """Receive the next update from Dies and return `True` if the program should
    continue running, or `False` if it should stop.

    This function will block until a new update is available or the program is
    stopping.
    """
    global _world_update_ev, _stop, _sock
    if _sock is None:
        raise RuntimeError("Bridge not initialized, call dies_py.init() first")
    _world_update_ev.wait()
    _world_update_ev.clear()
    return not _stop


def world() -> World:
    """Return the current world state."""
    global _world_state
    if _world_state is None:
        raise RuntimeError("No world state available. Call dies_py.next() first")
    return _world_state


def send(message: Cmd):
    """Send a message to the dies server."""
    global _sock
    if _sock is None:
        raise RuntimeError("Bridge not initialized, call dies_py.init() first")
    _sock.send(msgspec.json.encode(message))


def should_stop():
    """Check if the bridge should stop."""
    global _stop
    return _stop


def _recv_loop():
    global _world_state, _world_update_ev, _stop, _sock
    buf = bytearray(4096)
    while not _stop:
        n = _sock.recv_into(buf)
        msg: Msg = msg_dec.decode(buf[:n])
        if isinstance(msg, Term):
            _world_update_ev.set()
            _stop = True
            break
        elif isinstance(msg, World):
            _world_state = msg
            _world_update_ev.set()


def _sig_handler(signum, frame):
    global _stop
    _stop = True
