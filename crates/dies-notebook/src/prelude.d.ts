// Type declarations of the notebook API (prelude.js), loaded into the editor
// next to the generated `bindings.ts` (world/feedback types).
//
// Units: mm, mm/s, radians, seconds. Coordinates are absolute field coordinates
// (the same frame as the field view and the logs).
//
// NOTE: `ControlInput` and `SkillCommand` mirror Rust types that typeshare can't
// generate (externally tagged enums). Keep them in sync with
// `dies-executor/src/control/player_input.rs` and
// `dies-strategy-protocol/src/skill.rs` until they are generated.

import type {
  PlayerData,
  PlayerFeedbackMsg,
  WorldData,
  RoleType,
  TeamColor,
  Vector2,
  Angle,
} from "./bindings";

declare global {
  type Vec2 = Vector2;

  type SkillStatus = "Idle" | "Running" | "Succeeded" | "Failed";

  /** Per-robot executor info in a snapshot. */
  interface RobotInfo {
    team: TeamColor;
    id: number;
    /** Status of the skill executed last frame. */
    skill: SkillStatus | null;
    /** Velocity setpoint the controller produced last frame (mm/s, global). */
    target_vel: Vec2 | null;
    pass: unknown | null;
    /** Latest raw robot telemetry. */
    feedback: PlayerFeedbackMsg | null;
  }

  /** One executor frame. */
  interface Frame {
    frame: number;
    /** Seconds. */
    t: number;
    dt: number;
    /** Controlled team colors. */
    teams: TeamColor[];
    world: WorldData;
    robots: RobotInfo[];
  }

  // ------------------------------------------------------------ commands

  type Velocity = { Global: Vec2 } | { Local: Vec2 };
  type KickerControlInput = "Idle" | "Arm" | "Kick" | "ReflexKick" | "Disarm";
  type MotionBounds =
    | { Arc: { center: Vec2; min_radius: number; max_radius: number; half_angle: number } }
    | { RoundedRect: { center: Vec2; half_extents: Vec2; min_offset: number; max_offset: number } };

  /** Player controller input. Every field is optional (defaults apply); unknown
   * fields are rejected. */
  interface ControlInput {
    /** Target position (closed-loop position control). */
    position?: Vec2 | null;
    /** Velocity, added to the position controller output. */
    velocity?: Velocity;
    /** Target heading (radians). */
    yaw?: Angle | null;
    angular_velocity?: number | null;
    /** Dribbler speed, 0..1. */
    dribbling_speed?: number;
    dribbler_brake?: boolean;
    kicker?: KickerControlInput;
    magnet?: boolean;
    care?: number;
    aggressiveness?: number;
    brake_gain?: number | null;
    acceleration_limit?: number | null;
    speed_limit?: number | null;
    angular_acceleration_limit?: number | null;
    angular_speed_limit?: number | null;
    avoid_ball?: boolean;
    avoid_ball_care?: number;
    /** Field-wall containment (default true). */
    avoid_wall?: boolean;
    wall_care?: number;
    hold_ground?: boolean;
    fan_speed?: number | null;
    kick_speed?: number | null;
    /** ORCA avoidance of other robots (default false). */
    avoid_robots?: boolean;
    avoid_defense_area?: boolean;
    /** Global path planner (default true). false = drive straight at the target. */
    use_planner?: boolean;
    bounds?: MotionBounds | null;
    role_type?: RoleType;
  }

  type BallAction =
    | { Shoot: { target: Vec2 } }
    | { Strike: { target: Vec2; acquire_first: boolean } }
    | { Carry: { to: Vec2; heading: Angle } }
    | { Hold: { heading: Angle } };
  type AcquirePosition = "Fastest" | "Default" | { Heading: Angle };

  /** A skill command, exactly as strategies issue it. */
  type SkillCommand =
    | { GoToPos: { position: Vec2; heading: Angle | null; hold_ground: boolean; avoid_robots: boolean } }
    | { GoToBounded: { position: Vec2; heading: Angle | null; bounds: MotionBounds } }
    | { Dribble: { target_pos: Vec2; target_heading: Angle } }
    | { Shoot: { target: Vec2 } }
    | { ReflexReceive: { from_pos: Vec2; intercept_pos: Vec2; target: Vec2; capture_limit: number } }
    | {
        Pass: {
          partner: number;
          role: "Passer" | "Receiver";
          target_hint: Vec2 | null;
          forward_to: Vec2 | null;
          seq: number;
        };
      }
    | { Snatch: { release_hint: Vec2 | null } }
    | { HandleBall: { action: BallAction; acquire: AcquirePosition; magnet: boolean } }
    | "Stop";

  // ------------------------------------------------------------ core

  interface Robot {
    readonly id: number;
    readonly team: TeamColor;
    /** This frame's tracked state, or undefined if the robot isn't seen. */
    readonly state: PlayerData | undefined;
    /** Latest raw telemetry, or undefined. */
    readonly feedback: PlayerFeedbackMsg | undefined;
    /** Status of the skill executed last frame. */
    readonly status: SkillStatus | null;
    /** Hold a controller input until changed or released. */
    set(input: ControlInput): void;
    /** Hold a skill command; resolves when it ends. */
    skill(cmd: SkillCommand): Promise<SkillStatus | "Cancelled">;
    /** Hand the robot back to the strategy (or idle). */
    release(): void;
    /** Debug values this frame, logged as team_<Color>.p<id>.<name>; [x, y] → _x/_y. */
    log(values: Record<string, number | string | boolean | Vec2>): void;
  }

  /** Resolves at the next executor frame. The only timing primitive. */
  function frame(): Promise<Frame>;
  /** Latest snapshot. */
  const now: Frame;
  /** A robot handle. `robot()` is the robot picked in the editor, or the only
   * one on the field. `team` may be omitted when one team is controlled. */
  function robot(id?: number, team?: "blue" | "yellow" | TeamColor): Robot;
  /** Write a row to the log's events table. */
  function emit(type: string, payload?: unknown): void;
  /** Cell output: a Plot → chart, an array of objects → table, else text/JSON. */
  function show(...items: unknown[]): void;

  // ------------------------------------------------------------ stdlib

  function wait(seconds: number): Promise<Frame>;
  /** Wait until `pred` holds (checked once per frame); throws after `timeout` s. */
  function until(pred: (f: Frame) => boolean, timeout?: number): Promise<Frame>;

  interface Trial<R = unknown> {
    readonly id: number;
    readonly name: string;
    readonly params: Record<string, unknown>;
    readonly status: "ok" | "failed" | "aborted";
    readonly error?: string;
    readonly result: R;
    /** Seconds. */
    readonly duration: number;
    /** One flat row per frame: state (position_x, ...), feedback (fb_*), controller
     * info (skill, target_vel_x, ...), logged values, and the params. */
    rows(robot: Robot | number, team?: string): Record<string, any>[];
    col(robot: Robot | number, name: string, team?: string): any[];
    last(robot: Robot | number, name: string, team?: string): any;
    summary(): Record<string, unknown>;
  }
  /** Run `body` as a recorded, logged trial. Failures don't throw (status
   * "failed", robots commanded in it are released); aborts propagate. */
  function trial<R>(name: string, params: Record<string, unknown>, body: () => Promise<R>): Promise<Trial<R>>;
  function trial<R>(name: string, body: () => Promise<R>): Promise<Trial<R>>;

  /** Observable Plot (https://observablehq.com/plot). Options take field names,
   * not functions. Wrap marks in show(). */
  const Plot: any;

  const v2: {
    add(a: Vec2, b: Vec2): Vec2;
    sub(a: Vec2, b: Vec2): Vec2;
    mul(a: Vec2, s: number): Vec2;
    dot(a: Vec2, b: Vec2): number;
    /** Positive when b is left of a. */
    cross(a: Vec2, b: Vec2): number;
    norm(a: Vec2): number;
    dist(a: Vec2, b: Vec2): number;
    unit(a: Vec2): Vec2;
    rot(a: Vec2, angle: number): Vec2;
    angle(a: Vec2): number;
    fromAngle(angle: number, len?: number): Vec2;
    lerp(a: Vec2, b: Vec2, s: number): Vec2;
  };
  /** Degrees → radians. */
  function deg(d: number): number;
  /** Radians → degrees. */
  function toDeg(r: number): number;
  function clamp(x: number, lo: number, hi: number): number;
  function wrapAngle(a: number): number;
  const stats: {
    mean(xs: number[]): number;
    rms(xs: number[]): number;
    std(xs: number[]): number;
    maxAbs(xs: number[]): number;
    linfit(xs: number[], ys: number[]): { slope: number; intercept: number; r: number; n: number };
  };
}

export {};
