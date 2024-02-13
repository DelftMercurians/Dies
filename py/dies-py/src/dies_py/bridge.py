from io import StringIO
import os
import socket
import msgspec

from .messages import msg_dec, Cmd, Msg

__BRIDGE__ = None


class Bridge:
    """Singleton class for communicating with the dies server."""

    def __new__(cls):
        global __BRIDGE__
        if __BRIDGE__ is None:
            __BRIDGE__ = object.__new__(cls)
        return __BRIDGE__

    def __init__(self):
        host = os.environ.get("DIES_IPC_HOST", "localhost")
        cmd_port = int(os.environ.get("DIES_IPC_PORT", 0))
        if cmd_port == 0:
            raise RuntimeError("DIES_IPC_PORT not set")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, cmd_port))
        self.sock.settimeout(1)

        self.buffer = StringIO()
        # self.msg_queue = queue.Queue()
        # self.listen_thread = threading.Thread(
        #     target=self._listen_for_messages, daemon=True
        # )
        # self.listen_thread.start()

    def send(self, message: Cmd):
        """Send a message to the dies server."""
        self.sock.send(msgspec.json.encode(message) + b"\n")

    def recv(self) -> Msg | None:
        """Receive a message from the dies server.

        Blocks until a message is available."""
        msg_line = None

        while True:
            try:
                data = self.sock.recv(1024).decode("utf-8")
                if data:
                    self.buffer.write(data)
                    self.buffer.seek(0)
                    line = self.buffer.readline()
                    while line:
                        if line.endswith("\n"):
                            msg_line = line
                            # Read the next line
                            line = self.buffer.readline()
                        else:
                            # Incomplete line, move the cursor back to its beginning
                            self.buffer.seek(self.buffer.tell() - len(line))
                            break
                        line = self.buffer.readline()

                    # Clear the self.buffer and write any incomplete line back to it
                    remainder = self.buffer.read()
                    self.buffer.seek(0)
                    self.buffer.truncate(0)
                    self.buffer.write(remainder)

                    if msg_line is not None:
                        try:
                            decoded_msg = msg_dec.decode(msg_line.rstrip("\n"))
                            return decoded_msg
                        except msgspec.DecodeError:
                            continue
                else:
                    # No data, break the loop
                    break
            except socket.timeout:
                continue
