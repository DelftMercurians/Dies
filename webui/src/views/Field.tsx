import React, { FC, useEffect, useRef } from "react";
import { WorldStatus, useWorldState } from "../api";
import { Vector2, Vector3, WorldData } from "../bindings";
import { useResizeObserver } from "@/lib/useResizeObserver";

const ROBOT_RADIUS = 0.14 * 1000;
const BALL_RADIUS = 0.043 * 1000;
const FIELD_ASPECT_RATIO = [52, 37];
const CANVAS_PADDING = 20;
const CONT_PADDING_PX = 8;

const FIELD_GREEN = "#15803d";

const Field: FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const world = useWorldState();
  const worldData = world.status === "connected" ? world.data : null;
  const contRef = useRef(null);
  const { width: contWidth = 0, height: contHeight = 0 } = useResizeObserver({
    ref: contRef,
  });

  const availableHeight = contHeight - 2 * CONT_PADDING_PX;
  const availableWidth = contWidth - 2 * CONT_PADDING_PX;
  const canvasWidth = Math.min(
    availableWidth,
    availableHeight * (FIELD_ASPECT_RATIO[0] / FIELD_ASPECT_RATIO[1])
  );
  const canvasHeight =
    canvasWidth * (FIELD_ASPECT_RATIO[1] / FIELD_ASPECT_RATIO[0]);

  // Render the field when the world state changes
  useEffect(() => {
    if (!canvasRef.current) return;

    clearCanvas(canvasRef.current);
    if (worldData) {
      render(canvasRef.current, worldData, null);
    }
  }, [worldData, canvasWidth]);

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
      />
    </div>
  );
};

const clearCanvas = (canvas: HTMLCanvasElement) => {
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = FIELD_GREEN;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillRect(0, 0, canvas.width, canvas.height);
};

/**
 * Render the field on the canvas
 */
const render = (
  canvas: HTMLCanvasElement,
  worldState: WorldData,
  selectedPlayerId: number | null
) => {
  const ctx = canvas.getContext("2d")!;

  const { own_players, opp_players, ball } = worldState;
  const fieldW = worldState.field_geom?.field_length ?? 0;
  const fieldH = worldState.field_geom?.field_width ?? 0;
  const width = canvas.width - CANVAS_PADDING * 2;
  const height = canvas.height - CANVAS_PADDING * 2;

  const convertCoords = (coords: Vector2 | Vector3): Vector2 => {
    const [x, y] = coords;

    return [
      (x + fieldW / 2) * (width / fieldW) + CANVAS_PADDING,
      (-y + fieldH / 2) * (height / fieldH) + CANVAS_PADDING,
    ];
  };

  const convertLength = (length: number): number => {
    return Math.ceil(length * (width / fieldW));
  };

  const drawPlayer = (
    serverPos: Vector2,
    yaw: number,
    color: string,
    selected: boolean
  ) => {
    const [x, y] = convertCoords(serverPos);
    const robotCanvasRadius = convertLength(ROBOT_RADIUS);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, robotCanvasRadius, 0, 2 * Math.PI);
    ctx.fill();

    const angle = -yaw;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(robotCanvasRadius, 0);
    ctx.closePath();
    ctx.restore();
    ctx.stroke();

    if (selected) {
      ctx.strokeStyle = "white";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, robotCanvasRadius + 2, 0, 2 * Math.PI);
      ctx.stroke();
    }
  };

  // Draw the field lines
  worldState.field_geom?.line_segments?.forEach?.(({ p1, p2 }) => {
    const [x1, y1] = convertCoords(p1);
    const [x2, y2] = convertCoords(p2);
    ctx.strokeStyle = "white";
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  });

  // Draw the player robots
  own_players.forEach(({ id, raw_position, yaw }) =>
    drawPlayer(raw_position, yaw, "blue", id === selectedPlayerId)
  );
  opp_players.forEach(({ raw_position, yaw }) =>
    drawPlayer(raw_position, yaw, "yellow", false)
  );

  // Draw the ball
  if (ball) {
    const ballPos = convertCoords(ball.position);
    const ballCanvasRadius = convertLength(BALL_RADIUS);
    ctx.fillStyle = "red";
    ctx.beginPath();
    ctx.arc(ballPos[0], ballPos[1], ballCanvasRadius, 0, 2 * Math.PI);
    ctx.fill();
  }
};

export default Field;
