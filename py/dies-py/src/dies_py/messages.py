import msgspec
from msgspec import Struct
from typing import Union

# Socket info
class SocketInfo(Struct):
    pub_port: int
    pull_port: int
    
socket_info_dec = msgspec.json.Decoder(SocketInfo)

# StratMsg - A message to the strategy process

class Term(Struct, tag=True):
    pass

class Hello(Struct, tag=True):
    message: str


Msg = Union[Term, Hello]
msg_dec = msgspec.json.Decoder(Msg)


# StratCmd - A message from the strategy process

class Debug(Struct, tag=True):
    message: str

Cmd = Debug