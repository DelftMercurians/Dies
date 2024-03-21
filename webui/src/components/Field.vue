<script setup lang="ts">
// ---------------- IMPORTS ----------------
// core
import { ref, onMounted } from "vue";

// store
import { storeToRefs } from "pinia";
import { useFieldStore } from "../store/field";

// typesa
import type { World, XY, XYZ } from "../types";

// components
import Bot from "./Bot.vue";

// ---------------- CODE ----------------
// Store setup
const fieldStore = useFieldStore();
const { PADDING, ROBOT_RADIUS, BALL_RADIUS } = storeToRefs(fieldStore);

// Canvas variables
const canvas = ref<HTMLCanvasElement | null>(null);
let canvasWidth = ref<number>(0);
let canvasHeight = ref<number>(0);

let state: World | null = null;

/**
 * Fits the canvas to its full width/height
 * @param canvas - the canvas where our filed is rendered
 */
function fitToContainer(canvas: HTMLCanvasElement) {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
}

/**
 * Renders the field
 */
function render(ctx: CanvasRenderingContext2D) {
  if (canvas.value && state) {
    // Adjusting the canvas to the right dimenstions
    fitToContainer(canvas.value);

    canvasWidth.value = canvas.value.width;
    canvasHeight.value = canvas.value.height;

    // Add some padding to the canvas
    const width = canvasWidth.value - PADDING.value * 2;
    const height = canvasHeight.value - PADDING.value * 2;

    const { own_players, opp_players, ball } = state;
    const fieldH = state.field_geom.field_width;
    const fieldW = state.field_geom.field_length;

    // console.log(canvasWidth.value, canvasHeight.value);
    console.log(state.field_geom);

    // ---------------- FILL FIELD WITH COLOR ----------------
    ctx.fillStyle = "#3bad59";
    ctx.clearRect(0, 0, canvasWidth.value, canvasHeight.value);
    ctx.fillRect(0, 0, canvasWidth.value, canvasHeight.value);

    // ---------------- DRAW LINES ----------------
    state.field_geom.line_segments.forEach(({ p1, p2 }: { p1: XY; p2: XY }) => {
      const [x1, y1]: [number, number] = convertCoords(
        p1,
        width,
        fieldW,
        height,
        fieldH
      );
      const [x2, y2]: [number, number] = convertCoords(
        p2,
        width,
        fieldW,
        height,
        fieldH
      );
      ctx.strokeStyle = "white";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });

    // ---------------- DRAW BALL ----------------
    if (ball) {
      const ballPos = convertCoords(
        ball.position,
        width,
        fieldW,
        height,
        fieldH
      );
      const ballCanvasRadius = convertLength(BALL_RADIUS.value, width, fieldW);
      ctx.fillStyle = "red";
      ctx.beginPath();
      ctx.arc(ballPos[0], ballPos[1], ballCanvasRadius, 0, 2 * Math.PI);
      ctx.fill();
    }

    // ---------------- DRAW PLAYERS ----------------
    const drawPlayer = (serverPos: XY, color: string) => {
      const [x, y]: [number, number] = convertCoords(
        serverPos,
        width,
        fieldW,
        height,
        fieldH
      );

      const robotCanvasRadius = convertLength(
        ROBOT_RADIUS.value,
        width,
        fieldW
      );
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, robotCanvasRadius, 0, 2 * Math.PI);
      ctx.fill();
    };

    own_players.forEach(({ position }: { position: XY }) => {
      drawPlayer(position, "blue");
    });
    opp_players.forEach(({ position }: { position: XY }) => {
      //BEFORECOMMIT: Get position back to normal
      position[0] = 6000;
      drawPlayer(position, "yellow");
    });
  } else {
    console.log("state is not working");
  }
}

/**
 * Convert from server length to canvas length.
 */
const convertLength = (
  length: number,
  width: number,
  fieldW: number
): number => {
  return Math.ceil(length * (width / fieldW));
};

/**
 * Convert from server coordinates to canvas coordinates.
 *
 * The server's coordinate system has its origin at the center of the field,
 */
const convertCoords = (
  coords: XY | XYZ,
  width: number,
  fieldW: number,
  height: number,
  fieldH: number
): XY => {
  const [x, y] = coords;

  return [
    (x + fieldW / 2) * (width / fieldW) + PADDING.value,
    (-y + fieldH / 2) * (height / fieldH) + PADDING.value,
  ];
};

// ---------------- Lifecycle hooks ----------------
onMounted(async () => {
  const context = ref(canvas.value?.getContext("2d"));
  if (context.value) {
    // Retrieve state
    const res = await fetch("/api/state");
    state = await res.json();

    render(context.value);
  }
});
</script>

<template>
  <div class="field-container">
    <canvas ref="canvas" class="football-field"> </canvas>
    <Bot
      v-if="state?.own_players"
      :height="50"
      :width="50"
      color="red"
      :offset="[0, 0]"
    />
  </div>
</template>

<style scoped>
.football-field {
  position: relative;
  width: 100%;
  aspect-ratio: 5/2;
}

.field-container {
  width: 100%;
  aspect-ratio: 5/2;
}
</style>
