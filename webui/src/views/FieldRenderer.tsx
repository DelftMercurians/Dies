import {
  DebugColor,
  DebugMap,
  DebugShape,
  PlayerData,
  Vector2,
  Vector3,
  TeamData,
  WorldData,
  TeamColor,
} from "../bindings";

const ROBOT_RADIUS = 0.08 * 1000;
const BALL_RADIUS = 0.043 * 1000;
export const DEFAULT_FIELD_SIZE = [10400, 7400] as [number, number];
export const CANVAS_PADDING = 20;

const FIELD_GREEN = "#15803d";
const FIELD_LINE = "#ffffff";
const BLUE_ROBOT_FILTERED = "#2563eb";
const BLUE_ROBOT_RAW = "#7c3aed";
const YELLOW_ROBOT_FILTERED = "#facc15";
const YELLOW_ROBOT_RAW = "#f9731688";
const BALL_FILTERED = "#fb923c";
const BALL_RAW = "#eab308";
const MANUAL_OUTLINE = "#dc2626";
const SELECTED_OUTLINE = "#ffffff";

const DEBUG_COLORS: Record<DebugColor, string> = {
  green: "#14b8a688",
  red: "#dc262688",
  orange: "#f9731688",
  purple: "#9333ea88",
  blue: "#0000aa88",
  gray: "#66666688"
};

export type PositionDisplayMode = "raw" | "filtered" | "both";

export class FieldRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private world: WorldData | null = null;
  private debugShapes: DebugShape[] = [];
  private fieldSize: [number, number] = DEFAULT_FIELD_SIZE;
  private positionDisplayMode: PositionDisplayMode = "filtered";

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  setWorldData(world: WorldData | null) {
    this.world = world;
    if (world) {
      this.fieldSize = [
        world.field_geom?.field_length ?? DEFAULT_FIELD_SIZE[0],
        world.field_geom?.field_width ?? DEFAULT_FIELD_SIZE[1],
      ];
    }
  }

  setDebugData(data: DebugMap) {
    this.debugShapes = Object.values(data)
      .filter(({ type }) => type === "Shape")
      .map(({ data }) => data as DebugShape);
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
    manualControl: number[]
  ) {
    this.clear();
    if (!this.world) return;
    const { blue_team, yellow_team, ball } = this.world;

    this.drawFieldLines();

    blue_team.forEach((player) =>
      this.drawPlayer(
        player,
        "blue",
        player.id === selectedPlayerId && primaryTeam === TeamColor.Blue,
        manualControl.includes(player.id) && primaryTeam === TeamColor.Blue,
        this.positionDisplayMode
      )
    );
    yellow_team.forEach((player) =>
      this.drawPlayer(
        player,
        "yellow",
        player.id === selectedPlayerId && primaryTeam === TeamColor.Yellow,
        manualControl.includes(player.id) && primaryTeam === TeamColor.Yellow,
        this.positionDisplayMode
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

    this.debugShapes.forEach((shape) => this.drawDebugShape(shape));
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

  private drawPlayer(
    data: PlayerData,
    team: "blue" | "yellow",
    selected: boolean,
    manualControl: boolean,
    positionDisplayMode: PositionDisplayMode,
    opacity = 1
  ) {
    if (positionDisplayMode === "both") {
      this.drawPlayer(data, team, selected, manualControl, "filtered");
      this.drawPlayer(data, team, false, false, "raw", 0.7);
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
