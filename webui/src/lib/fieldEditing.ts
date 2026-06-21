import { atom } from "jotai";
import { TeamColor, Vector2 } from "../bindings";

/**
 * Whether the field-mask drag editor is active. Toggled from the Field Mask
 * settings tab; consumed by the Field canvas to capture a drag rectangle.
 */
export const maskEditModeAtom = atom<boolean>(false);

/**
 * Last manual MoveTo target per player, tracked client-side (the executor does
 * not echo override targets back). Keyed by `${team}:${playerId}`. Ephemeral —
 * pruned when a player leaves manual control.
 */
export const manualTargetsAtom = atom<Record<string, Vector2>>({});

export const manualTargetKey = (team: TeamColor, playerId: number): string =>
  `${team}:${playerId}`;
