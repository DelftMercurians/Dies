<script lang="ts">
  import { onMount } from "svelte";
  // import Chart from "chart.js/auto";
  import { Canvas, Layer, t, type Render } from "svelte-canvas";
  import type { PlayerCmd, World, XY, XYZ } from "./types";
  import { connectWs } from "./client";

  /**
   * The radius of the robots and ball, in mm
   */
  const ROBOT_RADIUS = 0.14 * 1000;
  /**
   * The radius of the ball, in mm
   */
  const BALL_RADIUS = 0.043 * 1000;

  const PADDING = 20;

  const { sendCommand, worldState } = connectWs();

  let selectedPlayerId: number | null = null;
  onMount(() => {
    let pressedKeys = new Set<string>();
    const keydownHandler = (ev: KeyboardEvent) => {
      pressedKeys.add(ev.key);
    };
    const keyupHandler = (ev: KeyboardEvent) => {
      pressedKeys.delete(ev.key);
    };

    let interval = setInterval(() => {
      if (pressedKeys.size === 0) return;
      const player = $worldState?.own_players.find(
        (player) => player.id === selectedPlayerId
      );
      if (!player) {
        console.error("Player not found");
        return;
      }

      const cmd: PlayerCmd = {
        id: player.id,
        sx: 0,
        sy: 0,
        w: 0,
        dribble_speed: 0,
        arm: false,
        disarm: false,
        kick: false,
      };

      if (pressedKeys.has("w")) {
        cmd.sx = 1;
      }
      if (pressedKeys.has("s")) {
        cmd.sx = -1;
      }
      if (pressedKeys.has("a")) {
        cmd.sy = 1;
      }
      if (pressedKeys.has("d")) {
        cmd.sy = -1;
      }
      if (pressedKeys.has("q")) {
        cmd.w = 1;
      }
      if (pressedKeys.has("e")) {
        cmd.w = -1;
      }
      if (pressedKeys.has(" ")) {
        cmd.dribble_speed = 200;
      }
      if (cmd.sx !== 0 || cmd.sy !== 0 || cmd.w !== 0) {
        sendCommand({ type: "directPlayerCmd", cmd });
      }
    }, 1 / 10);

    window.addEventListener("keydown", keydownHandler);
    window.addEventListener("keyup", keyupHandler);
    return () => {
      clearInterval(interval);
      window.removeEventListener("keydown", keydownHandler);
      window.removeEventListener("keyup", keyupHandler);
    };
  });

  let firstTs: number | null = null;
  let render: Render;
  $: render = ({ context: ctx, width: canvasWidth, height: canvasHeight }) => {
    if (!$worldState) return;

    // Add some padding to the canvas
    const width = canvasWidth - PADDING * 2;
    const height = canvasHeight - PADDING * 2;

    const { own_players, opp_players, ball } = $worldState;
    const fieldH = $worldState.field_geom?.field_width ?? 0;
    const fieldW = $worldState.field_geom?.field_length ?? 0;

    /**
     * Convert from server length to canvas length.
     */
    const convertLength = (length: number): number => {
      return Math.ceil(length * (width / fieldW));
    };

    /**
     * Convert from server coordinates to canvas coordinates.
     *
     * The server's coordinate system has its origin at the center of the field,
     */
    const convertCoords = (coords: XY | XYZ): XY => {
      const [x, y] = coords;

      return [
        (x + fieldW / 2) * (width / fieldW) + PADDING,
        (-y + fieldH / 2) * (height / fieldH) + PADDING,
      ];
    };

    // Draw field
    ctx.fillStyle = "#00aa00";
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw field lines
    $worldState.field_geom?.line_segments?.forEach?.(({ p1, p2 }) => {
      const [x1, y1] = convertCoords(p1);
      const [x2, y2] = convertCoords(p2);
      ctx.strokeStyle = "white";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });

    // Draw players
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

      // Draw arrow for orientation
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

      // Draw outline around selected player
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

    // Draw ball
    if (ball) {
      const ballPos = convertCoords(ball.position);
      const ballCanvasRadius = convertLength(BALL_RADIUS);
      ctx.fillStyle = "red";
      ctx.beginPath();
      ctx.arc(ballPos[0], ballPos[1], ballCanvasRadius, 0, 2 * Math.PI);
      ctx.fill();
    }

    if ($worldState?.own_players.length > 0) {
      let selectedPlayer = $worldState.own_players[0];
      if (selectedPlayerId === null) {
        selectedPlayerId = $worldState.own_players[0].id;
      } else {
        const _selectedPlayer = $worldState.own_players.find(
          (player) => player.id === selectedPlayerId
        );
        if (_selectedPlayer) {
          selectedPlayer = _selectedPlayer;
        } else {
          selectedPlayerId = selectedPlayer.id;
        }
      }

      const velocityMagnitude = Math.sqrt(
        selectedPlayer.velocity[0] ** 2 + selectedPlayer.velocity[1] ** 2
      );
      if (firstTs === null) {
        firstTs = selectedPlayer.timestamp;
      }
      const ts = selectedPlayer.timestamp - firstTs;

      // const labels = velocityChart.data.labels!;
      // const data = velocityChart.data.datasets[0].data;
      // if (data.length > 100) {
      //   labels.shift();
      //   data.shift();
      // }
      // labels.push(ts);
      // data.push(velocityMagnitude);
      // velocityChart.update();
    }
    if ($worldState?.own_players.length > 0) {
      let selectedPlayer = $worldState.own_players[0];
      if (selectedPlayerId === null) {
        selectedPlayerId = $worldState.own_players[0].id;
      } else {
        const _selectedPlayer = $worldState.own_players.find(
          (player) => player.id === selectedPlayerId
        );
        if (_selectedPlayer) {
          selectedPlayer = _selectedPlayer;
        } else {
          selectedPlayerId = selectedPlayer.id;
        }
      }

      const velocityMagnitude = Math.sqrt(
        selectedPlayer.velocity[0] ** 2 + selectedPlayer.velocity[1] ** 2
      );
      if (firstTs === null) {
        firstTs = selectedPlayer.timestamp;
      }
      const ts = selectedPlayer.timestamp - firstTs;

      // const labels = velocityChart.data.labels!;
      // const data = velocityChart.data.datasets[0].data;
      // if (data.length > 100) {
      //   labels.shift();
      //   data.shift();
      // }
      // labels.push(ts);
      // data.push(velocityMagnitude);
      // velocityChart.update();
    }
  };
</script>

<main class="cont">
  <div class="sidebar">
    <canvas id="velocityChart" width="400" height="400"></canvas>
  </div>
  <div class="sidebar">
    <canvas id="velocityChart" width="400" height="400"></canvas>
  </div>

  <Canvas width={840} height={600} class="canvas">
    <Layer {render} />
  </Canvas>

  <div class="sidebar"></div>
</main>

<style>
  .cont {
    width: 100vw;
    height: 100vh;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #191918;
  }

  .sidebar {
    height: 100%;
    flex: 1;
    background-color: #1c1c1c;
  }

  /* .canvas-container {
    flex: 7;
    display: flex;
    justify-content: center;
    align-items: center;
  } */

  :global(.canvas) {
    width: 50vw !important;
    height: auto !important;
  }
</style>
