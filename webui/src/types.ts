export type XY = [number, number];

export type XYZ = [number, number, number];

export interface BallData {
  timestamp: number;
  position: XYZ;
  velocity: XYZ;
}

export interface PlayerData {
  timestamp: number;
  id: number;
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
