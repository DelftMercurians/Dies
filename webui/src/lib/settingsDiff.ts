import { ExecutorSettings } from "@/bindings";

/** A single field that differs between two settings configs. */
export type SettingsDiffEntry = {
  /** Nested settings group key, or "general" for top-level scalar fields. */
  section: string;
  field: string;
  current: unknown;
  baseline: unknown;
};

const GENERAL = "general";

/**
 * Sections that are per-run operational state, not persisted tuning — they're
 * set from CLI flags + toolbar controls each session and carried on their own
 * command path, so they don't belong in the settings baseline/revert diff (and
 * aren't written to the settings JSON). Keep them out of the diff entirely.
 */
const IGNORED_SECTIONS = new Set<string>(["team_configuration"]);

const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === "object" && !Array.isArray(v);

const eq = (a: unknown, b: unknown): boolean =>
  JSON.stringify(a) === JSON.stringify(b);

/**
 * Deep-diff two settings configs, grouped by section. Nested groups (e.g.
 * `controller_settings`) diff per-field; top-level scalars (e.g.
 * `goal_area_avoidance`) are collected under the synthetic "general" section.
 */
export function diffSettings(
  current: ExecutorSettings,
  baseline: ExecutorSettings,
): SettingsDiffEntry[] {
  const out: SettingsDiffEntry[] = [];
  const cur = current as unknown as Record<string, unknown>;
  const base = baseline as unknown as Record<string, unknown>;

  for (const key of Object.keys(cur)) {
    if (IGNORED_SECTIONS.has(key)) continue;
    const cv = cur[key];
    const bv = base[key];
    if (isPlainObject(cv)) {
      const bobj = isPlainObject(bv) ? bv : {};
      for (const f of Object.keys(cv)) {
        if (!eq(cv[f], bobj[f])) {
          out.push({ section: key, field: f, current: cv[f], baseline: bobj[f] });
        }
      }
    } else if (!eq(cv, bv)) {
      out.push({ section: GENERAL, field: key, current: cv, baseline: bv });
    }
  }
  return out;
}

/** Return a copy of `current` with one field restored from `baseline`. */
export function revertField(
  current: ExecutorSettings,
  baseline: ExecutorSettings,
  section: string,
  field: string,
): ExecutorSettings {
  const next = structuredClone(current) as unknown as Record<string, any>;
  const base = baseline as unknown as Record<string, any>;
  if (section === GENERAL) {
    next[field] = base[field];
  } else {
    next[section] = { ...next[section], [field]: base[section][field] };
  }
  return next as ExecutorSettings;
}

/** Return a copy of `current` with a whole section restored from `baseline`. */
export function revertSection(
  current: ExecutorSettings,
  baseline: ExecutorSettings,
  section: string,
): ExecutorSettings {
  const next = structuredClone(current) as unknown as Record<string, any>;
  const base = baseline as unknown as Record<string, any>;
  if (section === GENERAL) {
    // Restore every top-level scalar field.
    for (const k of Object.keys(next)) {
      if (!isPlainObject(next[k])) next[k] = base[k];
    }
  } else {
    next[section] = structuredClone(base[section]);
  }
  return next as ExecutorSettings;
}

const SECTION_LABELS: Record<string, string> = {
  controller_settings: "Controller",
  tracker_settings: "Tracker",
  avoidance: "Avoidance",
  skill_tunables: "Skill tunables",
  team_configuration: "Team",
  blue_team_settings: "Blue team",
  yellow_team_settings: "Yellow team",
  general: "General",
};

export const sectionLabel = (section: string): string =>
  SECTION_LABELS[section] ?? section;

export const fieldLabel = (field: string): string =>
  field.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());

/** Compact, human-readable rendering of a settings value for the diff list. */
export function formatValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "on" : "off";
  if (typeof v === "number") {
    // Trim float noise without losing small values.
    return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(4)));
  }
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}
