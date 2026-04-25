/**
 * Panel Components Index
 *
 * Export all panel components for use with Dockview.
 * Each panel wraps an existing view component.
 */

import FieldPanel from "./FieldPanel";
import GameControllerPanel from "./GameControllerPanel";
import TeamOverviewPanel from "./TeamOverviewPanel";
import PlayerInspectorPanel from "./PlayerInspectorPanel";
import BasestationPanel from "./BasestationPanel";
import SettingsPanel from "./SettingsPanel";
import DebugLayerPanel from "./DebugLayerPanel";
import ScenarioPanel from "./ScenarioPanel";

// Re-export panel components
export {
  FieldPanel,
  GameControllerPanel,
  TeamOverviewPanel,
  PlayerInspectorPanel,
  BasestationPanel,
  SettingsPanel,
  DebugLayerPanel,
  ScenarioPanel,
};

// Panel IDs used for layout configuration
export const PANEL_IDS = {
  FIELD: "field",
  GAME_CONTROLLER: "game-controller",
  TEAM_OVERVIEW: "team-overview",
  PLAYER_INSPECTOR: "player-inspector",
  BASESTATION: "basestation",
  SETTINGS: "settings",
  DEBUG_LAYERS: "debug-layers",
  SCENARIO: "scenario",
} as const;

export type PanelId = (typeof PANEL_IDS)[keyof typeof PANEL_IDS];

// Human-readable tab titles for each panel
export const PANEL_TITLES: Record<PanelId, string> = {
  [PANEL_IDS.FIELD]: "FIELD",
  [PANEL_IDS.GAME_CONTROLLER]: "GAME CTRL",
  [PANEL_IDS.TEAM_OVERVIEW]: "TEAM",
  [PANEL_IDS.PLAYER_INSPECTOR]: "PLAYER",
  [PANEL_IDS.BASESTATION]: "BASESTATION",
  [PANEL_IDS.SETTINGS]: "SETTINGS",
  [PANEL_IDS.DEBUG_LAYERS]: "LAYERS",
  [PANEL_IDS.SCENARIO]: "SCENARIO",
};

// All panels in a stable order for menus
export const ALL_PANELS: { id: PanelId; title: string }[] = (
  Object.values(PANEL_IDS) as PanelId[]
).map((id) => ({ id, title: PANEL_TITLES[id] }));

// Panel components map for DockviewReact
export const panelComponents = {
  [PANEL_IDS.FIELD]: FieldPanel,
  [PANEL_IDS.GAME_CONTROLLER]: GameControllerPanel,
  [PANEL_IDS.TEAM_OVERVIEW]: TeamOverviewPanel,
  [PANEL_IDS.PLAYER_INSPECTOR]: PlayerInspectorPanel,
  [PANEL_IDS.BASESTATION]: BasestationPanel,
  [PANEL_IDS.SETTINGS]: SettingsPanel,
  [PANEL_IDS.DEBUG_LAYERS]: DebugLayerPanel,
  [PANEL_IDS.SCENARIO]: ScenarioPanel,
};
