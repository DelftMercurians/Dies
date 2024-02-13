import msgspec
from msgspec import Struct
from typing import Union

Vector2f = tuple[float, float]

Vector3f = tuple[float, float, float]

# RuntimeMsg - A message to the strategy process


class BallData(Struct):
    timestamp: float
    position: Vector3f
    velocity: Vector3f


class PlayerData(Struct):
    timestamp: float
    id: int
    position: Vector2f
    velocity: Vector2f
    orientation: float
    angular_speed: float


class FieldCircularArc(Struct):
    name: str
    center: Vector2f
    radius: float
    a1: float
    a2: float
    thickness: float


class FieldLineSegment(Struct):
    name: str
    p1: Vector2f
    p2: Vector2f
    thickness: float


class FieldGeometry(Struct):
    field_length: int
    field_width: int
    goal_width: int
    goal_depth: int
    boundary_width: int
    line_segments: list[FieldLineSegment]
    circular_arcs: list[FieldCircularArc]


class World(Struct, tag=True):
    own_players: list[PlayerData]
    opp_players: list[PlayerData]
    ball: BallData | None
    field_geom: FieldGeometry | None


class Term(Struct, tag=True):
    pass


Msg = Union[Term, World]
msg_dec = msgspec.json.Decoder(Msg)


# RuntimeEvent - A message from the strategy process


class PlayerCmd(Struct, tag=True):
    id: int
    sx: float
    sy: float
    w: float = 0.0


class Debug(Struct, tag=True):
    message: str


Cmd = Union[Debug, PlayerCmd]
