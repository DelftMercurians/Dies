import React, { useCallback, useEffect, useRef, useState } from "react";
import { type PlayerCmd, type World, type XY, type XYZ, SELECT_SCENARIO_CMD, DIRECT_PLAYER_CMD , START_CMD, STOP_CMD } from "./types";
import { useWebSocket } from "./client";

const ROBOT_RADIUS = 0.14 * 1000;
const BALL_RADIUS = 0.043 * 1000;
const PADDING = 20;

const scenarios = ["Empty", "SinglePlayerWithoutBall", "SinglePlayer", "TwoPlayers"] as const;
export interface SymScenario {
  type: typeof scenarios[number];
}

const App: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedPlayerId, setSelectedPlayerId] = useState<number | null>(null);
  const worldStateRef = useRef<World | null>(null);
  const [crossX, setCrossX] = useState(0); // State for the X position of the cross
  const [crossY, setCrossY] = useState(0); // State for the Y position of the cross
  const fieldW = worldStateRef.current?.field_geom?.field_length! ?? 0;
  const fieldH = worldStateRef.current?.field_geom?.field_width! ?? 0;
  const canvasW = canvasRef.current?.width! - PADDING * 2;
  const canvasH = canvasRef.current?.height! - PADDING * 2;
  const [selectedScenario, setSelectedScenario] = useState<SymScenario>({ type: scenarios[0] });

  const convertCoords = (coords: XY | XYZ): XY => {
    const [x, y] = coords;

    return [
      (x + fieldW / 2) * (canvasW / fieldW) + PADDING,
      (-y + fieldH / 2) * (canvasH / fieldH) + PADDING,
    ];
  };

  const onUpdate = useCallback((world: World) => {
    worldStateRef.current = world;
  }, []);
  const { sendCommand } = useWebSocket({ onUpdate });

  // Keyboard input
  useEffect(() => {
    const pressedKeys = new Set<string>();
    const keydownHandler = (ev: KeyboardEvent) => {
      pressedKeys.add(ev.key);
    };
    const keyupHandler = (ev: KeyboardEvent) => {
      pressedKeys.delete(ev.key);
    };

    const cmdInterval = setInterval(() => {
      const worldState = worldStateRef.current;
      if (pressedKeys.size === 0 || !worldState) return;
      const player = worldState.own_players.find(
        (player) => player.id === selectedPlayerId
      );
      if (!player) {
        console.error("Player not found");
        return;
      }

      const cmd = createCmd(player.id, pressedKeys);
      sendCommand({ type: DIRECT_PLAYER_CMD, cmd });
    }, 1 / 10);

    window.addEventListener("keydown", keydownHandler);
    window.addEventListener("keyup", keyupHandler);
    return () => {
      clearInterval(cmdInterval);
      window.removeEventListener("keydown", keydownHandler);
      window.removeEventListener("keyup", keyupHandler);
    };
  }, [selectedPlayerId]);

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let frame: number | null = null;
    const render = () => {
      const worldState = worldStateRef.current;
      if (!worldState) {
        frame = requestAnimationFrame(render);
        return;
      }

      const { own_players, opp_players, ball } = worldState;
      const width = canvas.width - PADDING * 2;
      const height = canvas.height - PADDING * 2;

      const convertLength = (length: number): number => {
        return Math.ceil(length * (width / fieldW));
      };

      ctx.fillStyle = "#00aa00";
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      worldState.field_geom?.line_segments?.forEach?.(({ p1, p2 }) => {
        const [x1, y1] = convertCoords(p1);
        const [x2, y2] = convertCoords(p2);
        ctx.strokeStyle = "white";
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      });

      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.strokeRect(PADDING, PADDING, canvasW, canvasH);

      const drawPlayer = (
        serverPos: XY,
        orientation: number,
        color: string,
        selected: boolean
      ) => {
        const [x, y] = convertCoords(serverPos);
        const robotCanvasRadius = convertLength(ROBOT_RADIUS);
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, robotCanvasRadius, 0, 2 * Math.PI);
        ctx.fill();

        const angle = -orientation;
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
      own_players.forEach(({ id, raw_position, orientation }) =>
        drawPlayer(raw_position, orientation, "blue", id === selectedPlayerId)
      );
      opp_players.forEach(({ raw_position, orientation }) =>
        drawPlayer(raw_position, orientation, "yellow", false)
      );

      if (ball) {
        const ballPos = convertCoords(ball.position);
        const ballCanvasRadius = convertLength(BALL_RADIUS);
        ctx.fillStyle = "red";
        ctx.beginPath();
        ctx.arc(ballPos[0], ballPos[1], ballCanvasRadius, 0, 2 * Math.PI);
        ctx.fill();
      }

      if (worldState.own_players.length > 0) {
        let selectedPlayer = worldState.own_players[0];
        if (selectedPlayerId === null) {
          setSelectedPlayerId(worldState.own_players[0].id);
        } else {
          const _selectedPlayer = worldState.own_players.find(
            (player) => player.id === selectedPlayerId
          );
          if (_selectedPlayer) {
            selectedPlayer = _selectedPlayer;
          } else {
            setSelectedPlayerId(selectedPlayer.id);
          }
        }
      }

      drawCross(ctx, crossX, crossY);

      frame = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (frame !== null) {
        cancelAnimationFrame(frame);
      }
    };
  }, [selectedPlayerId, crossX, crossY]);

  function handleXChange(e: React.ChangeEvent<HTMLInputElement>): void {
    const newX = parseInt(e.target.value);
    setCrossX(newX);
  }

  function handleYChange(e: React.ChangeEvent<HTMLInputElement>): void {
    const newY = parseInt(e.target.value);
    setCrossY(newY);
  }

  function drawCross(ctx: CanvasRenderingContext2D, x: number, y: number) {
    const [crossCanvasX, crossCanvasY] = convertCoords([x, y]);

    const crossSize = 10; // Length of each arm of the cross
    ctx.strokeStyle = "red";
    ctx.beginPath();
    ctx.moveTo(crossCanvasX - crossSize, crossCanvasY); // Horizontal line
    ctx.lineTo(crossCanvasX + crossSize, crossCanvasY);
    ctx.moveTo(crossCanvasX, crossCanvasY - crossSize); // Vertical line
    ctx.lineTo(crossCanvasX, crossCanvasY + crossSize);
    ctx.stroke();
  }

  const handleStartSimulation = () => {
    sendCommand({ type: START_CMD });
  };

  const handleStopSimulation = () => {
    sendCommand({ type: STOP_CMD });
  };

  const handleScenarioChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newType = event.target.value as SymScenario['type'];
    const scenario: SymScenario = { type: newType };
    setSelectedScenario(scenario);
    sendCommand({ type: SELECT_SCENARIO_CMD, scenario });
  };

  return (
    <main className="cont">
      <div className="sidebar">
        <label>
          Select Scenario:
          <select id="scenario-select" value={selectedScenario.type} onChange={handleScenarioChange}>
            {scenarios.map((scenario) => (
              <option key={scenario} value={scenario}>
                {scenario}
              </option>
            ))}
          </select>
        </label>
        <label>
          Simulation:
          <button onClick={handleStartSimulation}>Start</button>
          <button onClick={handleStopSimulation}>Stop</button>
        </label>
        <label>
          X-Axis:
          <input type="range" min={-fieldW / 2} max={fieldW / 2} value={crossX} onChange={handleXChange} />
          {crossX}
        </label>
        <br />
        <label>
          Y-Axis:
          <input type="range" min={-fieldH / 2} max={fieldH / 2} value={crossY} onChange={handleYChange} />
          {crossY}
        </label>
      </div>

      <canvas ref={canvasRef} width={1100} height={900} className="canvas" />

      <div className="sidebar">
        <h3>Controls</h3>
        <ul>
          <li>Use <strong>W,A,S,D</strong> to move the robot</li>
          <li>Use <strong>Q,E</strong> to rotate the robot</li>
          <li>Hold <strong>Space</strong> to use the dribbler</li>
          <li>Press <strong>V</strong> to kick (not implemented yet, should also allow charging the kick + showing this)</li>
        </ul>
      </div>
    </main>
  );
};

export default App;

function createCmd(id: number, pressedKeys: Set<String>): PlayerCmd {
  const cmd: PlayerCmd = {
    id,
    sx: 0,
    sy: 0,
    w: 0,
    dribble_speed: 0,
    arm: false,
    disarm: false,
    kick: false,
  };

  if (pressedKeys.has("w")) {
    cmd.sy = 3;
  }
  if (pressedKeys.has("s")) {
    cmd.sy = -3;
  }
  if (pressedKeys.has("a")) {
    cmd.sx = 3;
  }
  if (pressedKeys.has("d")) {
    cmd.sx = -3;
  }
  if (pressedKeys.has("q")) {
    cmd.w = 3;
  }
  if (pressedKeys.has("e")) {
    cmd.w = -3;
  }
  if (pressedKeys.has(" ")) {
    cmd.dribble_speed = 200;
  }
  if (pressedKeys.has("v")) {
    cmd.kick = true;
  } else {
    cmd.kick = false;
  }

  return cmd;
}
