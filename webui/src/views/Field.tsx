import React, { FC, useEffect, useRef } from "react";
import { WorldStatus, useWorldState } from "../api";
import { Vector2, Vector3, WorldData } from "../bindings";

const ROBOT_RADIUS = 0.14 * 1000;
const BALL_RADIUS = 0.043 * 1000;
const PADDING = 20;

const Field: FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const world = useWorldState();

  // Render the field when the world state changes
  useEffect(() => {
    if (!canvasRef.current || world.status !== "connected") {
      return;
    }

    render(canvasRef.current, world.data, null);
  }, [world]);

  return <canvas ref={canvasRef} width={1100} height={900} />;
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
  const width = canvas.width - PADDING * 2;
  const height = canvas.height - PADDING * 2;

  const convertCoords = (coords: Vector2 | Vector3): Vector2 => {
    const [x, y] = coords;

    return [
      (x + fieldW / 2) * (width / fieldW) + PADDING,
      (-y + fieldH / 2) * (height / fieldH) + PADDING,
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

  // Clear the canvas
  ctx.fillStyle = "#00aa00";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillRect(0, 0, canvas.width, canvas.height);

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
