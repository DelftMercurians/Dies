import { PlayerFeedbackMsg, SysStatus } from "@/bindings";
import { atomWithStorage } from "jotai/utils";
import { useEffect, useRef } from "react";
import { batterySeverity, motorTempSeverity } from "./hardware";

/**
 * Audible hardware alarms for the team inspector.
 *
 * Three distinct, synthesized (Web Audio, no asset files) sounds, by priority:
 *  - `fault`      — robot faulted or lost comms (HIGH): urgent fast klaxon.
 *  - `lowBattery` — a pack voltage below the alert threshold (HIGH): descending two-tone.
 *  - `overtemp`   — overtemp status or a motor over its ceiling (LOWER): soft slow double-beep.
 *
 * A persistent condition re-alerts on a per-kind cooldown rather than every
 * frame. Alarms are armed/muted by the user (off until armed — browsers also
 * require a user gesture before audio can play, so arming doubles as that gesture).
 */
export type AlarmKind = "fault" | "lowBattery" | "overtemp";

/** Whether audible alarms are armed. Off by default; persisted across reloads. */
export const alarmsArmedAtom = atomWithStorage<boolean>("dies.alarmsArmed", false);

/** Re-alert interval per kind, in ms. High-priority conditions repeat sooner. */
const COOLDOWN_MS: Record<AlarmKind, number> = {
  fault: 3000,
  lowBattery: 5000,
  overtemp: 8000,
};

/** Higher number = higher priority (plays first when several are due at once). */
const PRIORITY: Record<AlarmKind, number> = {
  fault: 3,
  lowBattery: 2,
  overtemp: 1,
};

/** Statuses that constitute a fault / loss of comms. */
const FAULT_STATUSES: ReadonlySet<SysStatus> = new Set([
  SysStatus.Emergency,
  SysStatus.NoReply,
  SysStatus.Stop,
]);

/**
 * Which alarm conditions a single robot's feedback is currently in. Robots with
 * no feedback at all are intentionally ignored (commonly powered off on purpose
 * during setup) — only a live `NoReply` counts as disconnected.
 */
export const playerAlarms = (fb: PlayerFeedbackMsg): AlarmKind[] => {
  const kinds: AlarmKind[] = [];

  const statuses = [
    fb.primary_status,
    fb.kicker_status,
    fb.imu_status,
    ...(fb.motor_statuses ?? []),
  ];
  if (statuses.some((s) => s !== undefined && FAULT_STATUSES.has(s))) {
    kinds.push("fault");
  }

  if (fb.pack_voltages?.some((v) => batterySeverity(v) === "alert")) {
    kinds.push("lowBattery");
  }

  const overtempStatus = statuses.some((s) => s === SysStatus.Overtemp);
  const overtempMotor = fb.motor_temps?.some(
    (t) => motorTempSeverity(t) === "alert",
  );
  if (overtempStatus || overtempMotor) {
    kinds.push("overtemp");
  }

  return kinds;
};

/**
 * Singleton Web Audio engine. Lazily creates an `AudioContext` (only valid
 * after a user gesture) and plays a distinct tone pattern per alarm kind, with
 * per-kind cooldown bookkeeping so persistent conditions don't machine-gun.
 */
class AlarmEngine {
  private ctx: AudioContext | null = null;
  private lastPlayed: Record<AlarmKind, number> = {
    fault: 0,
    lowBattery: 0,
    overtemp: 0,
  };

  private ensureCtx(): AudioContext | null {
    if (!this.ctx) {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext })
          .webkitAudioContext;
      if (!Ctor) return null;
      this.ctx = new Ctor();
    }
    if (this.ctx.state === "suspended") void this.ctx.resume();
    return this.ctx;
  }

  /** One [freq(Hz), start(s), dur(s)] beep using the given waveform. */
  private beep(
    ctx: AudioContext,
    type: OscillatorType,
    freq: number,
    at: number,
    dur: number,
    gain: number,
  ) {
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type;
    osc.frequency.value = freq;
    // Short attack/release to avoid clicks.
    g.gain.setValueAtTime(0, at);
    g.gain.linearRampToValueAtTime(gain, at + 0.01);
    g.gain.setValueAtTime(gain, at + dur - 0.02);
    g.gain.linearRampToValueAtTime(0, at + dur);
    osc.connect(g).connect(ctx.destination);
    osc.start(at);
    osc.stop(at + dur);
  }

  private playPattern(kind: AlarmKind) {
    const ctx = this.ensureCtx();
    if (!ctx) return;
    const t = ctx.currentTime;
    switch (kind) {
      case "fault":
        // Urgent klaxon: fast alternating high tones, square wave.
        for (let i = 0; i < 3; i++) {
          this.beep(ctx, "square", 880, t + i * 0.18, 0.09, 0.18);
          this.beep(ctx, "square", 660, t + i * 0.18 + 0.09, 0.09, 0.18);
        }
        break;
      case "lowBattery":
        // Descending two-tone, sine.
        this.beep(ctx, "sine", 600, t, 0.18, 0.22);
        this.beep(ctx, "sine", 300, t + 0.2, 0.28, 0.22);
        break;
      case "overtemp":
        // Soft slow double-beep, triangle.
        this.beep(ctx, "triangle", 500, t, 0.14, 0.13);
        this.beep(ctx, "triangle", 500, t + 0.32, 0.14, 0.13);
        break;
    }
  }

  /**
   * Given the set of currently-active alarm kinds, play any whose cooldown has
   * elapsed (highest priority first). Call once per frame.
   */
  update(active: Set<AlarmKind>) {
    const now = Date.now();
    const due = [...active]
      .filter((k) => now - this.lastPlayed[k] >= COOLDOWN_MS[k])
      .sort((a, b) => PRIORITY[b] - PRIORITY[a]);
    for (const kind of due) {
      this.playPattern(kind);
      this.lastPlayed[kind] = now;
    }
  }

  /** Forget cooldowns so re-arming alarms fires immediately on the next active condition. */
  reset() {
    this.lastPlayed = { fault: 0, lowBattery: 0, overtemp: 0 };
  }
}

const engine = new AlarmEngine();

/**
 * Drive the alarm engine from the current own-team feedback. When `armed`,
 * evaluates the active alarm kinds across all robots every time `feedback`
 * changes and lets the engine play due sounds.
 */
export const useHardwareAlarms = (
  feedback: PlayerFeedbackMsg[],
  armed: boolean,
) => {
  const wasArmed = useRef(false);

  useEffect(() => {
    if (!armed) {
      wasArmed.current = false;
      return;
    }
    if (!wasArmed.current) {
      engine.reset();
      wasArmed.current = true;
    }
    const active = new Set<AlarmKind>();
    for (const fb of feedback) {
      for (const kind of playerAlarms(fb)) active.add(kind);
    }
    engine.update(active);
  }, [feedback, armed]);
};
