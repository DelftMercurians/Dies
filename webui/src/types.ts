export type XY = [number, number];

export type XYZ = [number, number, number];

export interface PlayerCmd {
  id: number;
  /**
   * The player's x (left-right, with `+` left) velocity \[mm/s]
   */
  sx: number;
  /**
   * The player's y (forward-backward, with `+` forward) velocity \[mm/s]
   */
  sy: number;
  /**
   * The player's angular velocity (with `+` counter-clockwise, `-` clockwise) \[rad/s]
   */
  w: number;
  /**
   * The player's dribble speed
   */
  dribble_speed: number;

  arm: boolean;
  disarm: boolean;
  kick: boolean;
}

export interface BallData {
  timestamp: number;
  position: XYZ;
  velocity: XYZ;
}

export interface PlayerData {
  timestamp: number;
  id: number;
  raw_position: XY;
  position: XY;
  velocity: XY;
  orientation: number;
  angular_speed: number;
}

export interface FieldLineSegment {
  name: string;
  p1: XY;
  p2: XY;
  thickness: number;
}

export interface FieldCircularArc {
  name: string;
  center: XY;
  radius: number;
  a1: number;
  a2: number;
  thickness: number;
}

export interface FieldGeometry {
  /** The length of the field in mm. */
  field_length: number;
  /** The width of the field in mm. */
  field_width: number;
  goal_width: number;
  goal_depth: number;
  boundary_width: number;

  line_segments: FieldLineSegment[];
  circular_arcs: FieldCircularArc[];
}

export interface World {
  own_players: PlayerData[];
  opp_players: PlayerData[];
  ball: BallData | null;
  field_geom: FieldGeometry;
}
