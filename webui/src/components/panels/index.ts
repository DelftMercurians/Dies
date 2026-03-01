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

// Re-export panel components
export {
  FieldPanel,
  GameControllerPanel,
  TeamOverviewPanel,
  PlayerInspectorPanel,
  BasestationPanel,
  SettingsPanel,
  DebugLayerPanel,
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
} as const;

// Panel components map for DockviewReact
export const panelComponents = {
  [PANEL_IDS.FIELD]: FieldPanel,
  [PANEL_IDS.GAME_CONTROLLER]: GameControllerPanel,
  [PANEL_IDS.TEAM_OVERVIEW]: TeamOverviewPanel,
  [PANEL_IDS.PLAYER_INSPECTOR]: PlayerInspectorPanel,
  [PANEL_IDS.BASESTATION]: BasestationPanel,
  [PANEL_IDS.SETTINGS]: SettingsPanel,
  [PANEL_IDS.DEBUG_LAYERS]: DebugLayerPanel,
};
