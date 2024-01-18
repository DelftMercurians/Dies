/**
 * handlers.ts
 *
 * Contains handlers for mock http requests.
 */
// ---------------- IMPORTS ----------------
import { http, HttpResponse } from "msw";
import {
  World,
  FieldGeometry,
  BallData,
  PlayerData,
  FieldLineSegment,
  FieldCircularArc,
  XYZ,
  XY,
} from "../types";

// Generate the ball data.
const ball: BallData = {
  timestamp: 0,
  position: [0, 0, 0],
  velocity: [0, 0, 0],
};

// Generate field segments.
const line_segments: FieldLineSegment[] = [
  { name: "top", p1: [-4, 1], p2: [4, 1], thickness: 1 },
  { name: "right", p1: [4, 1], p2: [4, -1], thickness: 1 },
  { name: "bottom", p1: [4, -1], p2: [-4, -1], thickness: 1 },
  { name: "left", p1: [-4, -1], p2: [-4, 1], thickness: 1 },
];

// Generate circular arcs
const circular_arcs: FieldCircularArc[] = [
  { name: "center", center: [0, 0], radius: 1, a1: 1, a2: 1, thickness: 1 },
];

// Generate the field geometry. (all values are in mm)
const field_geom: FieldGeometry = {
  field_length: 10,
  field_width: 4,
  goal_width: 2,
  goal_depth: 1 / 2,
  boundary_width: 1 / 5,
  line_segments,
  circular_arcs,
};

// Generate the own players data
const own_players: PlayerData[] = [
  {
    id: 0,
    angular_speed: 0,
    orientation: 0,
    velocity: [0, 0],
    position: [-4, -1],
    timestamp: 0,
  },
];

const opp_players: PlayerData[] = [
  {
    id: 0,
    angular_speed: 0,
    orientation: 0,
    velocity: [0, 0],
    position: [4, 1],
    timestamp: 0,
  },
];

const state: World = {
  own_players,
  opp_players,
  ball,
  field_geom,
};

const stateResolver = (): HttpResponse => {
  return HttpResponse.json(state);
};

export const handlers = [http.get("/api/state", stateResolver)];
