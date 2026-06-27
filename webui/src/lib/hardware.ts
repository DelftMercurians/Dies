import { PlayerFeedbackMsg, SkillState, SysStatus } from "@/bindings";

/**
 * Hardware health helpers, ported from the standalone basestation viewer
 * (`Scripts_RemoteControl/src/tabs.rs`). Centralizes all thresholds and the
 * status -> severity mapping so the player readout and the team overview color
 * things consistently.
 *
 * Severity ordering (worst wins): "alert" > "warn" > "info" > "ok" > "idle".
 */
export type Severity = "idle" | "ok" | "info" | "warn" | "alert";

const SEVERITY_RANK: Record<Severity, number> = {
  idle: 0,
  ok: 1,
  info: 2,
  warn: 3,
  alert: 4,
};

export const worstSeverity = (...severities: Severity[]): Severity =>
  severities.reduce<Severity>(
    (worst, s) => (SEVERITY_RANK[s] > SEVERITY_RANK[worst] ? s : worst),
    "idle",
  );

/** Tailwind text color for a severity. "ok"/"idle" are intentionally quiet. */
export const severityTextClass = (s: Severity): string =>
  ({
    idle: "text-text-muted",
    ok: "text-text-dim",
    info: "text-accent-blue",
    warn: "text-accent-amber",
    alert: "text-accent-red",
  })[s];

/** Tailwind background color for a severity dot. "ok"/"idle" are quiet. */
export const severityDotClass = (s: Severity): string =>
  ({
    idle: "bg-border-muted",
    ok: "bg-text-muted",
    info: "bg-accent-blue",
    warn: "bg-accent-amber",
    alert: "bg-accent-red",
  })[s];

/** Map a firmware `SysStatus` to a severity. */
export const sysStatusSeverity = (status?: SysStatus): Severity => {
  switch (status) {
    case SysStatus.Ok:
    case SysStatus.Safe:
    case SysStatus.Standby:
      return "ok";
    case SysStatus.Emergency:
    case SysStatus.Overtemp:
    case SysStatus.NoReply:
      return "alert";
    case SysStatus.Armed:
    case SysStatus.Stop:
      return "warn";
    case SysStatus.Starting:
    case SysStatus.Disarmed:
    case SysStatus.Ready:
      return "info";
    case SysStatus.NotInstalled:
    case undefined:
    default:
      return "idle";
  }
};

// --- Numeric thresholds (from the viewer) ---

/** Battery pack voltage (V): >11 ok, 10.2-11 warn, <10.2 alert. */
export const batterySeverity = (v?: number): Severity => {
  if (v === undefined) return "idle";
  if (v < 10.2) return "alert";
  if (v < 11.0) return "warn";
  return "ok";
};

/** Motor temperature (°C): <70 ok, 70-80 warn, >80 alert. */
export const motorTempSeverity = (t?: number): Severity => {
  if (t === undefined) return "idle";
  if (t > 80) return "alert";
  if (t >= 70) return "warn";
  return "ok";
};

/** Kicker capacitor voltage (V): >250 alert (overcharged), 20-250 ok, <20 idle. */
export const capVoltageSeverity = (v?: number): Severity => {
  if (v === undefined) return "idle";
  if (v > 250) return "alert";
  if (v >= 20) return "ok";
  return "idle";
};

export interface PlayerHealth {
  /** Worst-of severity across all subsystems. */
  severity: Severity;
  /** Whether any feedback at all was received. */
  online: boolean;
  /** Short human-readable issue strings for the worst subsystems. */
  issues: string[];
}

/**
 * Summarize the overall health of a robot from its basestation feedback.
 * Returns a single worst-of severity plus a list of issue strings for a
 * tooltip. Used by the team overview's glanceable dot.
 */
export const playerHealth = (fb?: PlayerFeedbackMsg): PlayerHealth => {
  if (!fb) return { severity: "idle", online: false, issues: [] };

  const issues: string[] = [];
  const note = (sev: Severity, label: string): Severity => {
    if (sev === "warn" || sev === "alert") issues.push(label);
    return sev;
  };

  const severities: Severity[] = [];
  severities.push(note(sysStatusSeverity(fb.primary_status), "primary status"));
  severities.push(note(sysStatusSeverity(fb.kicker_status), "kicker"));
  severities.push(note(sysStatusSeverity(fb.imu_status), "imu"));

  fb.motor_statuses?.forEach((s, i) => {
    severities.push(
      note(sysStatusSeverity(s), i === 4 ? "dribbler" : `motor ${i}`),
    );
  });
  fb.motor_temps?.forEach((t, i) => {
    severities.push(
      note(motorTempSeverity(t), i === 4 ? "dribbler temp" : `motor ${i} temp`),
    );
  });
  fb.pack_voltages?.forEach((v, i) => {
    severities.push(note(batterySeverity(v), `battery ${i === 0 ? "L" : "R"}`));
  });

  return { severity: worstSeverity(...severities), online: true, issues };
};

/** Tailwind text color for a skill execution state. */
export const skillStateTextClass = (s: SkillState): string =>
  ({
    [SkillState.Idle]: "text-text-muted",
    [SkillState.Running]: "text-accent-blue",
    [SkillState.Succeeded]: "text-accent-green",
    [SkillState.Failed]: "text-accent-red",
  })[s];

/** Zero-padded pattern image path for a robot id + team. */
export const patternSrc = (id: number, team: "blue" | "yellow"): string =>
  `/patterns/team_${team}/${String(id).padStart(2, "0")}.png`;
