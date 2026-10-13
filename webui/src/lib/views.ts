/**
 * Built-in views: task-specific Dockview layouts (match, motion testing, ...).
 *
 * A view is a named layout preset. The user's edits to a view are persisted
 * under the view's id (see DockviewWrapper); "Reset view" restores the preset.
 * Stored relative sizes are scaled to the real container by Dockview's
 * `fromJSON`. Bump LAYOUT_SCHEMA_VERSION in DockviewWrapper whenever a preset
 * changes shape so stale persisted copies are dropped.
 */

export interface ViewPreset {
  id: string;
  label: string;
  description: string;
  layout: object;
}

const panel = (id: string, title: string) => ({ id, contentComponent: id, title });

// FIELD fills the left; the right-hand column is split into a top tab group
// (INSPECTOR / SETTINGS) and a bottom tab group (CONSOLE / DEBUG / GAME CTRL).
const DEFAULT_LAYOUT = {
  grid: {
    root: {
      type: "branch",
      data: [
        {
          type: "leaf",
          data: { views: ["field"], activeView: "field", id: "1", hideHeader: true },
          size: 1247,
        },
        {
          type: "branch",
          data: [
            {
              type: "leaf",
              data: {
                views: ["player-inspector", "settings"],
                activeView: "player-inspector",
                id: "4",
              },
              size: 655,
            },
            {
              type: "leaf",
              data: {
                views: ["console", "debug-layers", "game-controller"],
                activeView: "console",
                id: "3",
              },
              size: 338,
            },
          ],
          size: 414.5,
        },
      ],
      size: 993,
    },
    width: 1661.5,
    height: 993,
    orientation: "HORIZONTAL",
  },
  panels: {
    field: panel("field", "FIELD"),
    "player-inspector": panel("player-inspector", "INSPECTOR"),
    settings: panel("settings", "SETTINGS"),
    console: panel("console", "CONSOLE"),
    "debug-layers": panel("debug-layers", "DEBUG"),
    "game-controller": panel("game-controller", "GAME CTRL"),
  },
  activeGroup: "4",
};

// Motion testing: the notebook gets as much room as the field. The field stays
// visible for watching the robot (and right-click → copy a position); the
// inspector / game controller sit underneath the notebook.
const MOTION_LAYOUT = {
  grid: {
    root: {
      type: "branch",
      data: [
        {
          type: "leaf",
          data: { views: ["field"], activeView: "field", id: "1", hideHeader: true },
          size: 830,
        },
        {
          type: "branch",
          data: [
            {
              type: "leaf",
              data: { views: ["notebook"], activeView: "notebook", id: "2" },
              size: 700,
            },
            {
              type: "leaf",
              data: {
                views: ["player-inspector", "game-controller", "console"],
                activeView: "player-inspector",
                id: "3",
              },
              size: 293,
            },
          ],
          size: 831.5,
        },
      ],
      size: 993,
    },
    width: 1661.5,
    height: 993,
    orientation: "HORIZONTAL",
  },
  panels: {
    field: panel("field", "FIELD"),
    notebook: panel("notebook", "NOTEBOOKS"),
    "player-inspector": panel("player-inspector", "INSPECTOR"),
    "game-controller": panel("game-controller", "GAME CTRL"),
    console: panel("console", "CONSOLE"),
  },
  activeGroup: "2",
};

export const DEFAULT_VIEW_ID = "default";

export const VIEWS: ViewPreset[] = [
  {
    id: "default",
    label: "Default",
    description: "Field, inspector, settings and game controller",
    layout: DEFAULT_LAYOUT,
  },
  {
    id: "motion",
    label: "Motion",
    description: "Motion testing: field + scripting notebook",
    layout: MOTION_LAYOUT,
  },
];

export const viewPreset = (id: string): ViewPreset | undefined =>
  VIEWS.find((v) => v.id === id);

export const isBuiltinView = (id: string): boolean => VIEWS.some((v) => v.id === id);
