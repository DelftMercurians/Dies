import {
  DebugColor,
  DebugMap,
  DebugShape,
  FieldMask,
  MarkerKind,
  PlayerData,
  Vector2,
  Vector3,
  WorldData,
  TeamColor,
} from "../bindings";

/** A manual MoveTo target to draw: a line from the robot to the target. */
export interface ManualTargetMarker {
  from: Vector2 | null;
  to: Vector2;
}

const ROBOT_RADIUS = 0.08 * 1000;
const BALL_RADIUS = 0.043 * 1000;
export const DEFAULT_FIELD_SIZE = [10400, 7400] as [number, number];
export const CANVAS_PADDING = 20;

const FIELD_GREEN = "#15803d";
const FIELD_LINE = "#ffffff";
const WALL_COLOR = "#111827";
const BLUE_ROBOT_FILTERED = "#2563eb";
const BLUE_ROBOT_RAW = "#7c3aed";
const YELLOW_ROBOT_FILTERED = "#facc15";
const YELLOW_ROBOT_RAW = "#f9731688";
const BALL_FILTERED = "#fb923c";
const BALL_RAW = "#eab308";
const MANUAL_OUTLINE = "#dc2626";
const SELECTED_OUTLINE = "#ffffff";
const MASK_OUTLINE = "#22d3ee";
const MASK_DRAFT = "#ffffff";
const MASK_SHADE = "#0008";
const TARGET_COLOR = "#dc2626";

// Below this robot→target distance (mm) a tether is pure noise — the target
// already sits on its robot — so it's only drawn when the robot is selected.
const TETHER_MIN_DIST = ROBOT_RADIUS * 2;

const DEBUG_COLORS: Record<DebugColor, string> = {
  green: "#14b8a688",
  red: "#dc262688",
  orange: "#f9731688",
  purple: "#9333ea88",
  blue: "#0000aa88",
  gray: "#66666622",
};

export type PositionDisplayMode = "raw" | "filtered" | "both";

export class FieldRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private world: WorldData | null = null;
  private debugShapes: DebugShape[] = [];
  private debugStrings: Record<string, string> = {};
  private fieldSize: [number, number] = DEFAULT_FIELD_SIZE;
  private positionDisplayMode: PositionDisplayMode = "filtered";
  private fieldMask: FieldMask | null = null;
  /** In-progress mask drag rectangle, in field mm: [x1, y1, x2, y2]. */
  private maskDraft: [number, number, number, number] | null = null;
  private manualTargets: ManualTargetMarker[] = [];
  /** In-progress slingshot kick: ball origin + current pull-back point (mm). */
  private simKickDraft: { ball: Vector2; pull: Vector2 } | null = null;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  setFieldMask(mask: FieldMask | null) {
    this.fieldMask = mask;
  }

  setMaskDraft(rect: [number, number, number, number] | null) {
    this.maskDraft = rect;
  }

  setManualTargets(targets: ManualTargetMarker[]) {
    this.manualTargets = targets;
  }

  setSimKickDraft(draft: { ball: Vector2; pull: Vector2 } | null) {
    this.simKickDraft = draft;
  }

  setWorldData(world: WorldData | null) {
    this.world = world;
    if (world && world.field_geom) {
      this.fieldSize = [
        world.field_geom.field_length + world.field_geom.boundary_width * 2,
        world.field_geom.field_width + world.field_geom.boundary_width * 2,
      ];
    }
  }

  setDebugData(data: DebugMap) {
    this.debugShapes = Object.values(data)
      .filter(({ type }) => type === "Shape")
      .map(({ data }) => data as DebugShape);

    // Extract string debug values
    this.debugStrings = {};
    Object.entries(data).forEach(([key, value]) => {
      if (value.type === "String") {
        this.debugStrings[key] = value.data as string;
      }
    });
  }

  private getPlayerRole(playerId: number, teamColor: TeamColor): string | null {
    const teamColorStr = teamColor === TeamColor.Blue ? "Blue" : "Yellow";
    const roleKey = `team_${teamColorStr}.p${playerId}.role`;
    return this.debugStrings[roleKey] || null;
  }

  setPositionDisplayMode(mode: PositionDisplayMode) {
    this.positionDisplayMode = mode;
  }

  clear() {
    this.ctx.fillStyle = FIELD_GREEN;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  render(
    selectedPlayerId: number | null,
    primaryTeam: TeamColor,
    manualControl: number[],
    manualBallPlacementPosition: Vector2 | null
  ) {
    this.clear();
    if (!this.world) return;
    const { blue_team, yellow_team, ball } = this.world;

    this.drawFieldLines();
    this.drawWalls();
    this.drawFieldMask();

    // Tethers are drawn first so they sit *under* the robots/ball/glyphs and
    // never occlude them.
    this.drawTethers(primaryTeam, selectedPlayerId);

    blue_team.forEach((player) =>
      this.drawPlayer(
        player,
        "blue",
        player.id === selectedPlayerId && primaryTeam === TeamColor.Blue,
        manualControl.includes(player.id) && primaryTeam === TeamColor.Blue,
        this.positionDisplayMode,
        TeamColor.Blue
      )
    );
    yellow_team.forEach((player) =>
      this.drawPlayer(
        player,
        "yellow",
        player.id === selectedPlayerId && primaryTeam === TeamColor.Yellow,
        manualControl.includes(player.id) && primaryTeam === TeamColor.Yellow,
        this.positionDisplayMode,
        TeamColor.Yellow
      )
    );

    if (ball) {
      if (
        this.positionDisplayMode === "both" ||
        this.positionDisplayMode === "filtered"
      ) {
        this.drawBall(ball.position, "filtered");
      }
      if (
        this.positionDisplayMode === "both" ||
        this.positionDisplayMode === "raw"
      ) {
        ball.raw_position.forEach((pos) => this.drawBall(pos, "raw"));
      }
    }

    if (manualBallPlacementPosition) {
      this.ctx.fillStyle = "#00000088";
      this.ctx.beginPath();
      this.ctx.arc(
        this.fieldToCanvas(manualBallPlacementPosition)[0],
        this.fieldToCanvas(manualBallPlacementPosition)[1],
        this.convertLength(BALL_RADIUS),
        0,
        2 * Math.PI
      );
      this.ctx.fill();
    }

    this.debugShapes.forEach((shape) => {
      // When a player is selected, fade other players' markers so the focused
      // robot's targets stand out.
      let alpha = 1;
      if (
        shape.type === "Marker" &&
        shape.data.owner != null &&
        selectedPlayerId !== null
      ) {
        alpha = selectedPlayerId === shape.data.owner ? 1 : 0.25;
      }
      this.ctx.save();
      this.ctx.globalAlpha = alpha;
      this.drawDebugShape(shape);
      this.ctx.restore();
    });

    this.manualTargets.forEach((t) => this.drawManualTarget(t));
    this.drawMaskDraft();
    this.drawSimKickDraft();
  }

  /** Slingshot aim: dashed pull-back line + solid launch arrow at the ball. */
  private drawSimKickDraft() {
    const k = this.simKickDraft;
    if (!k) return;
    const [bx, by] = this.fieldToCanvas(k.ball);
    const [px, py] = this.fieldToCanvas(k.pull);
    // Launch is opposite the pull, same magnitude (in canvas px).
    const lx = bx - (px - bx);
    const ly = by - (py - by);
    this.ctx.save();
    // Pull-back band (where the cursor is).
    this.ctx.strokeStyle = "#ffffff88";
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash([5, 4]);
    this.ctx.beginPath();
    this.ctx.moveTo(bx, by);
    this.ctx.lineTo(px, py);
    this.ctx.stroke();
    this.ctx.setLineDash([]);
    // Launch arrow.
    this.ctx.strokeStyle = "#fb923c";
    this.ctx.fillStyle = "#fb923c";
    this.ctx.lineWidth = 2.5;
    this.ctx.beginPath();
    this.ctx.moveTo(bx, by);
    this.ctx.lineTo(lx, ly);
    this.ctx.stroke();
    const ang = Math.atan2(ly - by, lx - bx);
    const head = 9;
    this.ctx.beginPath();
    this.ctx.moveTo(lx, ly);
    this.ctx.lineTo(
      lx - head * Math.cos(ang - Math.PI / 6),
      ly - head * Math.sin(ang - Math.PI / 6)
    );
    this.ctx.lineTo(
      lx - head * Math.cos(ang + Math.PI / 6),
      ly - head * Math.sin(ang + Math.PI / 6)
    );
    this.ctx.closePath();
    this.ctx.fill();
    this.ctx.restore();
  }

  /** Shade the region excluded by the field mask + outline the kept region. */
  private drawFieldMask() {
    const m = this.fieldMask;
    if (!m) return;
    const hx = this.fieldSize[0] / 2;
    const hy = this.fieldSize[1] / 2;
    const xMin = m.x_min * hx;
    const xMax = m.x_max * hx;
    const yMin = m.y_min * hy;
    const yMax = m.y_max * hy;
    // Skip drawing entirely when the mask is (effectively) the whole field.
    const full =
      m.x_min <= -0.999 &&
      m.x_max >= 0.999 &&
      m.y_min <= -0.999 &&
      m.y_max >= 0.999;
    if (full) return;

    // Shade everything outside [xMin,xMax]x[yMin,yMax] using four bands.
    this.ctx.fillStyle = MASK_SHADE;
    const band = (x1: number, y1: number, x2: number, y2: number) => {
      const [cx1, cy1] = this.fieldToCanvas([x1, y1]);
      const [cx2, cy2] = this.fieldToCanvas([x2, y2]);
      this.ctx.fillRect(
        Math.min(cx1, cx2),
        Math.min(cy1, cy2),
        Math.abs(cx2 - cx1),
        Math.abs(cy2 - cy1)
      );
    };
    band(-hx, hy, xMin, -hy); // left
    band(xMax, hy, hx, -hy); // right
    band(xMin, hy, xMax, yMax); // top (between mask sides)
    band(xMin, yMin, xMax, -hy); // bottom

    // Outline the kept region.
    const [rx1, ry1] = this.fieldToCanvas([xMin, yMax]);
    const [rx2, ry2] = this.fieldToCanvas([xMax, yMin]);
    this.ctx.save();
    this.ctx.strokeStyle = MASK_OUTLINE;
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash([6, 4]);
    this.ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);
    this.ctx.restore();
  }

  /** Draw the live drag rectangle while editing the mask. */
  private drawMaskDraft() {
    const d = this.maskDraft;
    if (!d) return;
    const [x1, y1] = this.fieldToCanvas([d[0], d[1]]);
    const [x2, y2] = this.fieldToCanvas([d[2], d[3]]);
    this.ctx.save();
    this.ctx.fillStyle = "#22d3ee22";
    this.ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    this.ctx.strokeStyle = MASK_DRAFT;
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash([4, 3]);
    this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    this.ctx.restore();
  }

  private drawManualTarget({ from, to }: ManualTargetMarker) {
    const [tx, ty] = this.fieldToCanvas(to);
    this.ctx.save();
    if (from) {
      const [fx, fy] = this.fieldToCanvas(from);
      this.ctx.strokeStyle = TARGET_COLOR + "aa";
      this.ctx.lineWidth = 1.5;
      this.ctx.setLineDash([5, 4]);
      this.ctx.beginPath();
      this.ctx.moveTo(fx, fy);
      this.ctx.lineTo(tx, ty);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }
    // Ring + cross marker at the target.
    this.ctx.strokeStyle = TARGET_COLOR;
    this.ctx.lineWidth = 2;
    this.ctx.beginPath();
    this.ctx.arc(tx, ty, 7, 0, 2 * Math.PI);
    this.ctx.stroke();
    this.ctx.beginPath();
    this.ctx.moveTo(tx - 10, ty);
    this.ctx.lineTo(tx + 10, ty);
    this.ctx.moveTo(tx, ty - 10);
    this.ctx.lineTo(tx, ty + 10);
    this.ctx.stroke();
    this.ctx.restore();
  }

  public fieldToCanvas(coords: Vector2 | Vector3): Vector2 {
    const [x, y] = coords;
    const [fieldW, fieldH] = this.fieldSize;
    const screenFieldW = this.canvas.width - CANVAS_PADDING * 2;
    const screenFieldH = this.canvas.height - CANVAS_PADDING * 2;

    return [
      (x + fieldW / 2) * (screenFieldW / fieldW) + CANVAS_PADDING,
      (-y + fieldH / 2) * (screenFieldH / fieldH) + CANVAS_PADDING,
    ];
  }

  public canvasToField(coords: Vector2): Vector2 {
    const [x, y] = coords;
    const [fieldW, fieldH] = this.fieldSize;
    const screenFieldW = this.canvas.width - CANVAS_PADDING * 2;
    const screenFieldH = this.canvas.height - CANVAS_PADDING * 2;

    return [
      (x - CANVAS_PADDING) * (fieldW / screenFieldW) - fieldW / 2,
      -(y - CANVAS_PADDING) * (fieldH / screenFieldH) + fieldH / 2,
    ];
  }

  private convertLength(length: number): number {
    const [fieldW] = this.fieldSize;
    const screenFieldW = this.canvas.width - CANVAS_PADDING * 2;
    return Math.ceil(length * (screenFieldW / fieldW));
  }

  private drawFieldLines() {
    if (!this.world?.field_geom?.line_segments) return;

    this.ctx.lineWidth = 1;
    this.world.field_geom.line_segments.forEach(({ p1, p2 }) => {
      const [x1, y1] = this.fieldToCanvas(p1);
      const [x2, y2] = this.fieldToCanvas(p2);
      this.ctx.strokeStyle = FIELD_LINE;
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.stroke();
    });
  }

  private drawWalls() {
    const geom = this.world?.field_geom;
    if (!geom) return;

    const {
      field_length,
      field_width,
      goal_width: gw,
      goal_depth: gd,
      boundary_width,
    } = geom;
    const fl = field_length + boundary_width * 2;
    const fw = field_width + boundary_width * 2;

    // Use a visible thickness independent of geometry thickness for clarity
    this.ctx.strokeStyle = WALL_COLOR;
    this.ctx.lineCap = "butt";
    this.ctx.lineJoin = "miter";
    this.ctx.lineWidth = Math.max(2, this.convertLength(10));

    const drawSeg = (p1: Vector2, p2: Vector2) => {
      const [x1, y1] = this.fieldToCanvas(p1);
      const [x2, y2] = this.fieldToCanvas(p2);
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.stroke();
    };

    // Top and bottom walls along touch lines
    drawSeg([-fl / 2, fw / 2], [fl / 2, fw / 2]);
    drawSeg([-fl / 2, -fw / 2], [fl / 2, -fw / 2]);

    // Goal-line walls on left and right
    // Left goal line (x = -fl/2): draw above and below mouth
    drawSeg([-fl / 2, fw / 2], [-fl / 2, -fw / 2]);
    // Right goal line (x = fl/2)
    drawSeg([fl / 2, fw / 2], [fl / 2, -fw / 2]);

    // Goal side walls (from goal line to back wall)
    drawSeg([-field_length / 2, gw / 2], [-(field_length / 2 + gd), gw / 2]);
    drawSeg([-field_length / 2, -gw / 2], [-(field_length / 2 + gd), -gw / 2]);
    drawSeg([field_length / 2, gw / 2], [field_length / 2 + gd, gw / 2]);
    drawSeg([field_length / 2, -gw / 2], [field_length / 2 + gd, -gw / 2]);

    // Goal back walls
    drawSeg(
      [-(field_length / 2 + gd), -gw / 2],
      [-(field_length / 2 + gd), gw / 2]
    );
    drawSeg([field_length / 2 + gd, -gw / 2], [field_length / 2 + gd, gw / 2]);
  }

  private drawPlayer(
    data: PlayerData,
    team: "blue" | "yellow",
    selected: boolean,
    manualControl: boolean,
    positionDisplayMode: PositionDisplayMode,
    teamColor: TeamColor,
    opacity = 1
  ) {
    if (positionDisplayMode === "both") {
      this.drawPlayer(
        data,
        team,
        selected,
        manualControl,
        "filtered",
        teamColor
      );
      this.drawPlayer(data, team, false, false, "raw", teamColor, 0.7);
      return;
    }

    const hexColor =
      team === "blue"
        ? positionDisplayMode === "raw"
          ? BLUE_ROBOT_RAW
          : BLUE_ROBOT_FILTERED
        : positionDisplayMode === "raw"
        ? YELLOW_ROBOT_RAW
        : YELLOW_ROBOT_FILTERED;

    const serverPos =
      positionDisplayMode === "raw" ? data.raw_position : data.position;

    const [x, y] = this.fieldToCanvas(serverPos);
    const robotCanvasRadius = this.convertLength(ROBOT_RADIUS);
    // add opaciy
    this.ctx.fillStyle =
      hexColor +
      Math.floor(opacity * 255)
        .toString(16)
        .padStart(2, "0");
    this.ctx.beginPath();
    this.ctx.arc(x, y, robotCanvasRadius, 0, 2 * Math.PI);
    this.ctx.fill();

    const angle = positionDisplayMode === "raw" ? -data.raw_yaw : -data.yaw;
    this.ctx.strokeStyle = "white";
    this.ctx.lineWidth = 2;
    this.ctx.save();
    this.ctx.translate(x, y);
    this.ctx.rotate(angle);
    this.ctx.beginPath();
    this.ctx.moveTo(0, 0);
    this.ctx.lineTo(robotCanvasRadius, 0);
    this.ctx.closePath();
    this.ctx.restore();
    this.ctx.stroke();

    if (selected) {
      this.ctx.strokeStyle = SELECTED_OUTLINE;
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.arc(x, y, robotCanvasRadius + 2, 0, 2 * Math.PI);
      this.ctx.stroke();
    }

    if (manualControl) {
      this.ctx.strokeStyle = MANUAL_OUTLINE;
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.arc(x, y, robotCanvasRadius - 2, 0, 2 * Math.PI);
      this.ctx.stroke();
    }

    // Draw role name and player id if opacity is full (not for the "both" mode
    // secondary rendering).
    if (opacity === 1) {
      const role = this.getPlayerRole(data.id, teamColor);
      if (role) {
        this.ctx.fillStyle = "white";
        this.ctx.font = "12px Arial";
        this.ctx.textAlign = "center";
        this.ctx.fillText(role, x, y - robotCanvasRadius - 5);
      }

      // Player id, centered on the body. A dark stroke keeps it legible over
      // either team's fill color.
      const idText = String(data.id);
      this.ctx.save();
      this.ctx.font = `bold ${Math.max(11, robotCanvasRadius)}px Arial`;
      this.ctx.textAlign = "center";
      this.ctx.textBaseline = "middle";
      this.ctx.lineWidth = 3;
      this.ctx.strokeStyle = "rgba(0,0,0,0.85)";
      this.ctx.strokeText(idText, x, y);
      this.ctx.fillStyle = "white";
      this.ctx.fillText(idText, x, y);
      this.ctx.restore();
    }
  }

  private drawBall(
    position: Vector2 | Vector3,
    positionDisplayMode: PositionDisplayMode
  ) {
    const [x, y] = this.fieldToCanvas(position);
    const ballCanvasRadius = this.convertLength(BALL_RADIUS);
    this.ctx.fillStyle =
      positionDisplayMode === "filtered" ? BALL_FILTERED : BALL_RAW;
    this.ctx.beginPath();
    this.ctx.arc(x, y, ballCanvasRadius, 0, 2 * Math.PI);
    this.ctx.fill();
  }

  /**
   * Draw a thin tether from each owned target marker back to its robot, so
   * ownership is unambiguous in a crowd. Kept calm by: only target markers (one
   * per robot), low opacity underneath everything, a distance gate that skips
   * targets sitting on their robot, and promotion of the selected robot's tether
   * while the rest recede.
   */
  private drawTethers(primaryTeam: TeamColor, selectedPlayerId: number | null) {
    if (!this.world) return;
    const team =
      primaryTeam === TeamColor.Blue
        ? this.world.blue_team
        : this.world.yellow_team;
    const someoneSelected = selectedPlayerId !== null;

    for (const shape of this.debugShapes) {
      if (shape.type !== "Marker") continue;
      const m = shape.data;
      if (m.kind !== MarkerKind.Target || m.owner == null) continue;
      const player = team.find((p) => p.id === m.owner);
      if (!player) continue;

      const isSelected = selectedPlayerId === m.owner;
      const dist = Math.hypot(
        player.position[0] - m.center[0],
        player.position[1] - m.center[1]
      );
      if (!isSelected && dist < TETHER_MIN_DIST) continue;

      const [px, py] = this.fieldToCanvas(player.position);
      const [tx, ty] = this.fieldToCanvas(m.center);
      const alpha = isSelected ? 0.9 : someoneSelected ? 0.08 : 0.28;

      this.ctx.save();
      this.ctx.globalAlpha = alpha;
      this.ctx.strokeStyle = DEBUG_COLORS[m.color].slice(0, 7);
      this.ctx.lineWidth = isSelected ? 2 : 1;
      this.ctx.beginPath();
      this.ctx.moveTo(px, py);
      this.ctx.lineTo(tx, ty);
      this.ctx.stroke();
      this.ctx.restore();
    }
  }

  private drawDebugShape(shape: DebugShape) {
    if (shape.type === "Line") {
      const { start, end, color } = shape.data;
      this.ctx.strokeStyle = DEBUG_COLORS[color];
      this.ctx.lineWidth = 3;
      const [x1, y1] = this.fieldToCanvas(start);
      const [x2, y2] = this.fieldToCanvas(end);
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.stroke();
    } else if (shape.type === "Circle") {
      const { center, radius, fill, stroke } = shape.data;
      const [x, y] = this.fieldToCanvas(center);
      const canvasRadius = this.convertLength(radius);
      this.ctx.beginPath();
      this.ctx.arc(x, y, canvasRadius, 0, 2 * Math.PI);
      if (fill) {
        this.ctx.fillStyle = DEBUG_COLORS[fill];
        this.ctx.fill();
      }
      if (stroke) {
        this.ctx.lineWidth = 3;
        this.ctx.strokeStyle = DEBUG_COLORS[stroke];
        this.ctx.stroke();
      }
    } else if (shape.type === "Cross") {
      const { center, color } = shape.data;
      const [x, y] = this.fieldToCanvas(center);
      this.ctx.strokeStyle = DEBUG_COLORS[color];
      this.ctx.lineWidth = 3;
      this.ctx.beginPath();
      this.ctx.moveTo(x - 10, y - 10);
      this.ctx.lineTo(x + 10, y + 10);
      this.ctx.moveTo(x + 10, y - 10);
      this.ctx.lineTo(x - 10, y + 10);
      this.ctx.stroke();
    } else if (shape.type === "Marker") {
      this.drawMarker(shape.data);
    }
  }

  /**
   * Draw a semantic marker glyph. Shape encodes *kind* (target / waypoint /
   * kick target); color is left to carry purpose/state; ownership is shown by
   * the tether (see `drawTethers`). Glyphs are screen-space sized so they stay
   * legible at any zoom.
   */
  private drawMarker(m: Extract<DebugShape, { type: "Marker" }>["data"]) {
    const [x, y] = this.fieldToCanvas(m.center);
    const c = DEBUG_COLORS[m.color].slice(0, 7);
    this.ctx.strokeStyle = c;
    this.ctx.fillStyle = c;
    this.ctx.lineWidth = 2;

    if (m.kind === MarkerKind.Target) {
      // Crosshair inside a ring — the "go here" destination.
      const r = 9;
      this.ctx.beginPath();
      this.ctx.moveTo(x - r, y);
      this.ctx.lineTo(x + r, y);
      this.ctx.moveTo(x, y - r);
      this.ctx.lineTo(x, y + r);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.arc(x, y, r, 0, 2 * Math.PI);
      this.ctx.stroke();
    } else if (m.kind === MarkerKind.Waypoint) {
      // Small hollow dot — clearly subordinate to a target.
      this.ctx.beginPath();
      this.ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
      this.ctx.stroke();
    } else if (m.kind === MarkerKind.KickTarget) {
      // Concentric reticle with a center dot — a shot aim point.
      this.ctx.beginPath();
      this.ctx.arc(x, y, 9, 0, 2 * Math.PI);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.arc(x, y, 4, 0, 2 * Math.PI);
      this.ctx.stroke();
      this.ctx.beginPath();
      this.ctx.arc(x, y, 1.5, 0, 2 * Math.PI);
      this.ctx.fill();
    }
  }

  getPlayerAt(x: number, y: number): [TeamColor, number] | null {
    if (!this.world) return null;

    const { blue_team, yellow_team } = this.world;
    for (const player of blue_team) {
      const [playerX, playerY] =
        this.positionDisplayMode === "raw"
          ? player.raw_position
          : player.position;
      const robotCanvasRadius = ROBOT_RADIUS + 100;
      if (
        Math.sqrt((x - playerX) ** 2 + (y - playerY) ** 2) <= robotCanvasRadius
      ) {
        return [TeamColor.Blue, player.id];
      }
    }
    for (const player of yellow_team) {
      const [playerX, playerY] =
        this.positionDisplayMode === "raw"
          ? player.raw_position
          : player.position;
      const robotCanvasRadius = ROBOT_RADIUS + 100;
      if (
        Math.sqrt((x - playerX) ** 2 + (y - playerY) ** 2) <= robotCanvasRadius
      ) {
        return [TeamColor.Yellow, player.id];
      }
    }

    return null;
  }
}
