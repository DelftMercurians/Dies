import { PlayerData, Vector2, Vector3, WorldData } from "../bindings";

const ROBOT_RADIUS = 0.14 * 1000;
const BALL_RADIUS = 0.043 * 1000;
export const DEFAULT_FIELD_SIZE = [10400, 7400] as [number, number];
export const CANVAS_PADDING = 20;

const FIELD_GREEN = "#15803d";
const FIELD_LINE = "#ffffff";
const BLUE_ROBOT = "#2563eb";
const YELLOW_ROBOT = "#facc15";
const BALL = "#fb923c";
const MANUAL_OUTLINE = "#dc2626";
const SELECTED_OUTLINE = "#ffffff";

export type PositionDisplayMode = "raw" | "filtered" | "both";

export class FieldRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private worldData: WorldData | null = null;
  private fieldSize: [number, number] = DEFAULT_FIELD_SIZE;
  private positionDisplayMode: PositionDisplayMode = "filtered";

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d")!;
  }

  setWorldData(worldData: WorldData | null) {
    this.worldData = worldData;
    if (worldData) {
      this.fieldSize = [
        worldData.field_geom?.field_length ?? DEFAULT_FIELD_SIZE[0],
        worldData.field_geom?.field_width ?? DEFAULT_FIELD_SIZE[1],
      ];
    }
  }

  setPositionDisplayMode(mode: PositionDisplayMode) {
    this.positionDisplayMode = mode;
  }

  clear() {
    this.ctx.fillStyle = FIELD_GREEN;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  render(selectedPlayerId: number | null, manualControl: number[]) {
    this.clear();
    if (!this.worldData) return;
    const { own_players, opp_players, ball } = this.worldData;

    this.drawFieldLines();

    own_players.forEach((player) =>
      this.drawPlayer(
        player,
        BLUE_ROBOT,
        player.id === selectedPlayerId,
        manualControl.includes(player.id),
        this.positionDisplayMode
      )
    );
    opp_players.forEach((player) =>
      this.drawPlayer(
        player,
        YELLOW_ROBOT,
        false,
        false,
        this.positionDisplayMode
      )
    );

    if (ball) {
      this.drawBall(ball.position);
    }
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
    if (!this.worldData?.field_geom?.line_segments) return;

    this.worldData.field_geom.line_segments.forEach(({ p1, p2 }) => {
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
    hexColor: string,
    selected: boolean,
    manualControl: boolean,
    positionDisplayMode: PositionDisplayMode,
    opacity = 1
  ) {
    if (positionDisplayMode === "both") {
      this.drawPlayer(data, hexColor, false, false, "raw");
      this.drawPlayer(data, hexColor, selected, manualControl, "filtered", 0.7);
      return;
    }

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

    const angle = -data.yaw;
    this.ctx.strokeStyle = "white";
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

  private drawBall(position: Vector2 | Vector3) {
    const [x, y] = this.fieldToCanvas(position);
    const ballCanvasRadius = this.convertLength(BALL_RADIUS);
    this.ctx.fillStyle = BALL;
    this.ctx.beginPath();
    this.ctx.arc(x, y, ballCanvasRadius, 0, 2 * Math.PI);
    this.ctx.fill();
  }

  getPlayerAt(x: number, y: number): number | null {
    if (!this.worldData) return null;

    const { own_players, ball } = this.worldData;
    for (const player of own_players) {
      const [playerX, playerY] =
        this.positionDisplayMode === "raw"
          ? player.raw_position
          : player.position;
      const robotCanvasRadius = ROBOT_RADIUS + 100;
      if (
        Math.sqrt((x - playerX) ** 2 + (y - playerY) ** 2) <= robotCanvasRadius
      ) {
        return player.id;
      }
    }

    return null;
  }
}
