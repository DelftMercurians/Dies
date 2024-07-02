import React, { FC, useEffect, useRef, useCallback } from "react";
import { WorldStatus, useWorldState } from "../api";
import { Vector2, Vector3, WorldData } from "../bindings";
import { useResizeObserver } from "@/lib/useResizeObserver";

const ROBOT_RADIUS = 0.14 * 1000;
const BALL_RADIUS = 0.043 * 1000;
const DEFAULT_FIELD_SIZE = [10400, 7400] as [number, number];
const CANVAS_PADDING = 20;
const CONT_PADDING_PX = 8;

const FIELD_GREEN = "#15803d";

class FieldRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private worldData: WorldData | null = null;
  private fieldSize: [number, number] = DEFAULT_FIELD_SIZE;
  private selectedPlayerId: number | null = null;

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

  setSelectedPlayerId(id: number | null) {
    this.selectedPlayerId = id;
  }

  clear() {
    this.ctx.fillStyle = FIELD_GREEN;
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  render() {
    this.clear();
    if (!this.worldData) return;

    const { own_players, opp_players, ball } = this.worldData;
    const screenFieldW = this.canvas.width - CANVAS_PADDING * 2;
    const screenFieldH = this.canvas.height - CANVAS_PADDING * 2;

    this.drawFieldLines();
    own_players.forEach(({ id, raw_position, yaw }) =>
      this.drawPlayer(raw_position, yaw, "blue", id === this.selectedPlayerId)
    );
    opp_players.forEach(({ raw_position, yaw }) =>
      this.drawPlayer(raw_position, yaw, "yellow", false)
    );

    if (ball) {
      this.drawBall(ball.position);
    }
  }

  private convertCoords(coords: Vector2 | Vector3): Vector2 {
    const [x, y] = coords;
    const [fieldW, fieldH] = this.fieldSize;
    const screenFieldW = this.canvas.width - CANVAS_PADDING * 2;
    const screenFieldH = this.canvas.height - CANVAS_PADDING * 2;

    return [
      (x + fieldW / 2) * (screenFieldW / fieldW) + CANVAS_PADDING,
      (-y + fieldH / 2) * (screenFieldH / fieldH) + CANVAS_PADDING,
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
      const [x1, y1] = this.convertCoords(p1);
      const [x2, y2] = this.convertCoords(p2);
      this.ctx.strokeStyle = "white";
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.stroke();
    });
  }

  private drawPlayer(
    serverPos: Vector2,
    yaw: number,
    color: string,
    selected: boolean
  ) {
    const [x, y] = this.convertCoords(serverPos);
    const robotCanvasRadius = this.convertLength(ROBOT_RADIUS);
    this.ctx.fillStyle = color;
    this.ctx.beginPath();
    this.ctx.arc(x, y, robotCanvasRadius, 0, 2 * Math.PI);
    this.ctx.fill();

    const angle = -yaw;
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
      this.ctx.strokeStyle = "white";
      this.ctx.lineWidth = 2;
      this.ctx.beginPath();
      this.ctx.arc(x, y, robotCanvasRadius + 2, 0, 2 * Math.PI);
      this.ctx.stroke();
    }
  }

  private drawBall(position: Vector2 | Vector3) {
    const [x, y] = this.convertCoords(position);
    const ballCanvasRadius = this.convertLength(BALL_RADIUS);
    this.ctx.fillStyle = "red";
    this.ctx.beginPath();
    this.ctx.arc(x, y, ballCanvasRadius, 0, 2 * Math.PI);
    this.ctx.fill();
  }

  getClickedObject(x: number, y: number): { type: string; id?: number } | null {
    if (!this.worldData) return null;

    const { own_players, opp_players, ball } = this.worldData;
    const allPlayers = [...own_players, ...opp_players];

    for (const player of allPlayers) {
      const [playerX, playerY] = this.convertCoords(player.raw_position);
      const robotCanvasRadius = this.convertLength(ROBOT_RADIUS) + 10;
      if (
        Math.sqrt((x - playerX) ** 2 + (y - playerY) ** 2) <= robotCanvasRadius
      ) {
        return { type: "player", id: player.id };
      }
    }

    if (ball) {
      const [ballX, ballY] = this.convertCoords(ball.position);
      const ballCanvasRadius = this.convertLength(BALL_RADIUS);
      if (Math.sqrt((x - ballX) ** 2 + (y - ballY) ** 2) <= ballCanvasRadius) {
        return { type: "ball" };
      }
    }

    return null;
  }
}

const Field: FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<FieldRenderer | null>(null);
  const world = useWorldState();
  const worldData = world.status === "connected" ? world.data : null;
  const contRef = useRef(null);

  const { width: contWidth = 0, height: contHeight = 0 } = useResizeObserver({
    ref: contRef,
  });

  // Calculate canvas dimensions
  const fieldSize = [
    (worldData?.field_geom?.field_length ?? DEFAULT_FIELD_SIZE[0]) +
      2 * CANVAS_PADDING,
    (worldData?.field_geom?.field_width ?? DEFAULT_FIELD_SIZE[1]) +
      2 * CANVAS_PADDING,
  ];
  const availableWidth = contWidth - 2 * CONT_PADDING_PX;
  const availableHeight = contHeight - 2 * CONT_PADDING_PX;
  const fieldAspectRatio = fieldSize[0] / fieldSize[1];
  const containerAspectRatio = availableWidth / availableHeight;

  let canvasWidth: number;
  let canvasHeight: number;
  if (containerAspectRatio > fieldAspectRatio) {
    // Container is wider than the field aspect ratio
    canvasHeight = availableHeight;
    canvasWidth = canvasHeight * fieldAspectRatio;
  } else {
    // Container is taller than the field aspect ratio
    canvasWidth = availableWidth;
    canvasHeight = canvasWidth / fieldAspectRatio;
  }
  canvasWidth = Math.floor(canvasWidth);
  canvasHeight = Math.floor(canvasHeight);

  useEffect(() => {
    if (!canvasRef.current) return;

    if (!rendererRef.current) {
      rendererRef.current = new FieldRenderer(canvasRef.current);
    }

    rendererRef.current.setWorldData(worldData);
    rendererRef.current.render();
  }, [worldData, canvasWidth, canvasHeight]);

  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!canvasRef.current || !rendererRef.current) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const clickedObject = rendererRef.current.getClickedObject(x, y);

      if (clickedObject) {
        if (clickedObject.type === "player") {
          console.log(`Clicked on player with ID: ${clickedObject.id}`);
          rendererRef.current.setSelectedPlayerId(clickedObject.id ?? null);
        } else if (clickedObject.type === "ball") {
          console.log("Clicked on the ball");
          rendererRef.current.setSelectedPlayerId(null);
        }
        rendererRef.current.render();
      } else {
        console.log("Clicked on empty space");
        rendererRef.current.setSelectedPlayerId(null);
        rendererRef.current.render();
      }
    },
    []
  );

  return (
    <div
      ref={contRef}
      className="w-full h-full flex items-center justify-center overflow-hidden"
      style={{ padding: CONT_PADDING_PX }}
    >
      <canvas
        ref={canvasRef}
        className="border-8 border-green-950 rounded"
        width={canvasWidth}
        height={canvasHeight}
        onClick={handleCanvasClick}
      />
    </div>
  );
};

export default Field;
