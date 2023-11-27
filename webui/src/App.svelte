<script lang="ts">
  import { onMount } from "svelte";
  import { Canvas, Layer, t, type Render } from "svelte-canvas";
  import type { World, XY, XYZ } from "./types";

  /**
   * The radius of the robots and ball, in mm
   */
  const ROBOT_RADIUS = 0.14 * 1000;
  /**
   * The radius of the ball, in mm
   */
  const BALL_RADIUS = 0.043 * 1000;

  const PADDING = 20;

  let state: World | null = null;
  onMount(() => {
    const interval = setInterval(async () => {
      const res = await fetch("/api/state");
      state = await res.json();
    }, 100);

    return () => clearInterval(interval);
  });

  let render: Render;
  $: render = ({ context: ctx, width: canvasWidth, height: canvasHeight }) => {
    if (!state) return;

    // Add some padding to the canvas
    const width = canvasWidth - PADDING * 2;
    const height = canvasHeight - PADDING * 2;

    const { own_players, opp_players, ball } = state;
    const fieldH = state.field_geom.field_width;
    const fieldW = state.field_geom.field_length;

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
    state.field_geom.line_segments.forEach(({ p1, p2 }) => {
      const [x1, y1] = convertCoords(p1);
      const [x2, y2] = convertCoords(p2);
      ctx.strokeStyle = "white";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });

    // Draw ball
    const ballPos = convertCoords(ball.position);
    const ballCanvasRadius = convertLength(BALL_RADIUS);
    ctx.fillStyle = "red";
    ctx.beginPath();
    ctx.arc(ballPos[0], ballPos[1], ballCanvasRadius, 0, 2 * Math.PI);
    ctx.fill();

    // Draw players
    const drawPlayer = (serverPos: XY, color: string) => {
      const [x, y] = convertCoords(serverPos);
      const robotCanvasRadius = convertLength(ROBOT_RADIUS);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, robotCanvasRadius, 0, 2 * Math.PI);
      ctx.fill();
    };
    own_players.forEach(({ position }) => drawPlayer(position, "blue"));
    opp_players.forEach(({ position }) => drawPlayer(position, "yellow"));

    // // Render vector field on top
    // if (state.vector_field) {
    //   const img = new Image();
    //   img.src = state.vector_field;
    //   ctx.drawImage(img, PADDING, PADDING, width, height);
    // }
  };
</script>

<main class="cont">
  <div class="sidebar"></div>

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
