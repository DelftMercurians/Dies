from io import StringIO
import os
import socket
import msgspec
import threading
import queue

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

        self.msg_queue = queue.Queue()

        self.listen_thread = threading.Thread(
            target=self._listen_for_messages, daemon=True
        )
        self.listen_thread.start()

    def send(self, message: Cmd):
        """Send a message to the dies server."""
        self.sock.send(msgspec.json.encode(message))

    def recv(self) -> Msg | None:
        """Receive a message from the dies server.

        Returns None if no message is available."""
        try:
            return self.msg_queue.get_nowait()
        except queue.Empty:
            return None

    def _listen_for_messages(self):
        buffer = StringIO()
        while True:
            try:
                data = self.sock.recv(1024).decode("utf-8")
                if data:
                    buffer.write(data)
                    buffer.seek(0)
                    line = buffer.readline()
                    while line:
                        if line.endswith("\n"):
                            decoded_msg = msg_dec.decode(line.rstrip("\n"))
                            self.msg_queue.put(decoded_msg)

                            # Read the next line
                            line = buffer.readline()
                        else:
                            # Incomplete line, move the cursor back to its beginning
                            buffer.seek(buffer.tell() - len(line))
                            break

                    # Clear the buffer and write any incomplete line back to it
                    remainder = buffer.read()
                    buffer.seek(0)
                    buffer.truncate(0)
                    buffer.write(remainder)
                else:
                    # No data, break the loop
                    break
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error in listen_for_messages: {e}")
                break
