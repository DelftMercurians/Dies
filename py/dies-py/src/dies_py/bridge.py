import sys
import zmq
import msgspec

from .messages import msg_dec, socket_info_dec, Cmd, Msg, SocketInfo


def read_socket_info() -> SocketInfo:
    return socket_info_dec.decode(sys.stdin.readline().strip())


class Bridge:
    def __init__(self, socket_info: SocketInfo):
        self.ctx = zmq.Context()
        self.msg_sock = self.ctx.socket(zmq.SUB)
        self.msg_sock.connect(f'tcp://localhost:{socket_info.pub_port}')
        self.msg_sock.setsockopt(zmq.SUBSCRIBE, b'')
        self.cmd_sock = self.ctx.socket(zmq.PUSH)
        self.cmd_sock.connect(f'tcp://localhost:{socket_info.pull_port}')

    def send(self, message: Cmd):
        self.cmd_sock.send(msgspec.json.encode(message))

    def recv(self) -> Msg:
        msg = self.msg_sock.recv()
        return msg_dec.decode(msg)