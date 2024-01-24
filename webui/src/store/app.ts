// Utilities
import { defineStore } from "pinia";

export const useAppStore = defineStore("app", {
  state: () => ({
    PADDING: 20,
    ROBOT_RADIUS: 0.14,
    BALL_RADIUS: 0.043,
  }),
});
