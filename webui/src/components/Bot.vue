<script setup lang="ts">
import { XY } from "@/types";

interface Props {
  width: number;
  height: number;
  color: string;
  offset: XY;
}

const props = defineProps<Props>();

const showDetails = (): void => {
  console.log("Test");
};
</script>

<template>
  <div
    class="bot"
    @click="showDetails"
    :style="{
      height: props.height + 'px',
      width: props.width + 'px',
      backgroundColor: props.color,
      top: props.offset[0] + 'px',
      left: props.offset[1] + 'px',
    }"
  ></div>
</template>

<style scoped>
.bot {
  position: absolute;
  border-radius: 50%;
}
</style>
