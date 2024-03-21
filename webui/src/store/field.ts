import { ref, computed } from "vue";
import { defineStore } from "pinia";

export const useFieldStore = defineStore("field", () => {
  /**
   * The radius of the robots and ball, in mm
   */
  const ROBOT_RADIUS = ref(0.14 * 1000);
  /**
   * The radius of the ball, in mm
   */
  const BALL_RADIUS = ref(0.043 * 1000);

  const PADDING = ref(20);

  return { PADDING, ROBOT_RADIUS, BALL_RADIUS };
});
