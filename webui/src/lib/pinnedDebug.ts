import { atomWithStorage } from "jotai/utils";
import {
  DebugMap,
  TeamColor,
} from "../bindings";
import { formatDebugString, formatNumber } from "./utils";

/**
 * Debug sub-keys (relative to `team_{color}.p{id}.`) pinned by the user.
 * Pinned values are shown for every robot in the team overview, persisted
 * across sessions.
 */
export const pinnedDebugKeysAtom = atomWithStorage<string[]>(
  "dies-pinned-debug-keys",
  []
);

/** Look up + format a player's value for a pinned sub-key. Null if absent/shape. */
export const formatPlayerDebugValue = (
  debugData: DebugMap | null,
  playerId: number,
  team: TeamColor,
  subkey: string
): string | null => {
  if (!debugData) return null;
  const teamStr = team === TeamColor.Blue ? "Blue" : "Yellow";
  const v = debugData[`team_${teamStr}.p${playerId}.${subkey}`];
  if (!v) return null;
  if (v.type === "Number") return formatNumber(v.data);
  if (v.type === "String") return formatDebugString(v.data);
  return null;
};
