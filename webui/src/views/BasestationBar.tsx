import { FC } from "react";
import { useBasestationInfo } from "@/api";
import { PlayerFeedbackMsg } from "@/bindings";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Severity, severityDotClass, severityTextClass } from "@/lib/hardware";

/**
 * Compact basestation/link telemetry strip shown at the very top of the
 * Inspector, above the strategy plan and the player list. Surfaces the RF link
 * health (connection, channel, firmware, per-radio status), the client's own
 * loop metrics (tx / rx / loop rate) and a per-robot feedback summary
 * (rate, age, firmware) — all of which only exist on a live basestation, so the
 * bar hides itself entirely in simulation (no `base_info`).
 */

const LOOP_TARGET_HZ = 50;

/** Severity for the client IO-loop rate relative to its 50 Hz target. */
const loopSeverity = (hz?: number): Severity => {
  if (hz === undefined) return "idle";
  if (hz >= 0.9 * LOOP_TARGET_HZ) return "ok";
  if (hz >= 0.5 * LOOP_TARGET_HZ) return "warn";
  return "alert";
};

/** Severity for a single robot from its feedback freshness. */
const robotSeverity = (fb: PlayerFeedbackMsg): Severity => {
  if (!fb.online) return "idle";
  const age = fb.feedback_age_ms;
  if (age === undefined) return "ok";
  if (age <= 120) return "ok";
  if (age <= 300) return "warn";
  return "alert";
};

const fmtVersion = (
  v?: { major: number; minor: number; patch: number } | undefined
): string => (v ? `${v.major}.${v.minor}.${v.patch}` : "?");

const Metric: FC<{
  label: string;
  value?: number;
  sev?: Severity;
  title: string;
}> = ({ label, value, sev = "idle", title }) => (
  <SimpleTooltip title={title}>
    <span className="flex items-center gap-1">
      <span className="text-text-muted">{label}</span>
      <span className={cn("font-mono", severityTextClass(sev))}>
        {value !== undefined ? Math.round(value) : "—"}
      </span>
    </span>
  </SimpleTooltip>
);

const BasestationBar: FC<{ className?: string }> = ({ className }) => {
  const { data } = useBasestationInfo();
  const base = data?.base_info;

  // No basestation at all (e.g. simulation): nothing meaningful to show.
  if (!base) return null;

  const robots = data
    ? [
        ...data.blue_team.map((fb) => ({ team: "blue" as const, fb })),
        ...data.yellow_team.map((fb) => ({ team: "yellow" as const, fb })),
        ...data.unknown_team.map((fb) => ({ team: null, fb })),
      ].sort((a, b) => a.fb.id - b.fb.id)
    : [];

  const onlineCount = robots.filter((r) => r.fb.online).length;
  const onlineSev: Severity = !base.connected
    ? "idle"
    : robots.length === 0
    ? "idle"
    : onlineCount === robots.length
    ? "ok"
    : onlineCount === 0
    ? "alert"
    : "warn";

  // Firmware across online robots: a single value if uniform, else "mixed".
  const versions = new Set(
    robots
      .filter((r) => r.fb.online && r.fb.firmware_version)
      .map((r) => fmtVersion(r.fb.firmware_version))
  );
  const fwSummary =
    versions.size === 0 ? null : versions.size === 1 ? [...versions][0] : "mixed";

  const linkSev: Severity = !base.connected
    ? "alert"
    : !base.protocol_ok
    ? "warn"
    : "ok";
  const linkTitle = !base.connected
    ? "Basestation disconnected"
    : `${base.version || "?"} · protocol ${base.protocol_version || "?"}${
        base.protocol_ok ? "" : " (mismatch!)"
      }`;

  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-x-3 gap-y-1 px-2 py-1.5 text-[10px] text-text-dim border-b border-border-subtle bg-bg-surface",
        className
      )}
    >
      {/* RF link */}
      <span className="flex items-center gap-1.5">
        <SimpleTooltip title={linkTitle}>
          <span
            className={cn("w-2 h-2 rounded-full shrink-0", severityDotClass(linkSev))}
          />
        </SimpleTooltip>
        <span className="uppercase tracking-wider text-text-muted">base</span>
        {base.connected ? (
          <>
            <span className="font-mono text-text-std">{base.channel_mhz}MHz</span>
            {base.version && (
              <span className="font-mono text-text-dim">v{base.version}</span>
            )}
            {base.num_radios > 0 && (
              <SimpleTooltip
                title={`Radios online: ${base.radios_online
                  .map((on, i) => `${i}:${on ? "up" : "down"}`)
                  .join("  ")}`}
              >
                <span className="flex items-center gap-0.5">
                  {base.radios_online.map((on, i) => (
                    <span
                      key={i}
                      className={cn(
                        "w-1.5 h-1.5 rounded-full",
                        severityDotClass(on ? "ok" : "alert")
                      )}
                    />
                  ))}
                </span>
              </SimpleTooltip>
            )}
          </>
        ) : (
          <span className={severityTextClass("alert")}>offline</span>
        )}
      </span>

      <span className="text-border-muted">·</span>

      {/* Client loop metrics */}
      <span className="flex items-center gap-2">
        <Metric
          label="tx"
          value={base.tx_hz}
          title="Radio command packets transmitted per second"
        />
        <Metric
          label="rx"
          value={base.rx_hz}
          title="Robot feedback frames received per second (all robots)"
        />
        <Metric
          label="loop"
          value={base.loop_hz}
          sev={loopSeverity(base.loop_hz)}
          title={`Client IO loop rate (target ${LOOP_TARGET_HZ} Hz)`}
        />
      </span>

      <span className="text-border-muted">·</span>

      {/* Robots: online count, firmware, per-robot freshness dots */}
      <span className="flex items-center gap-1.5">
        <span className="text-text-muted">robots</span>
        <span className={cn("font-mono", severityTextClass(onlineSev))}>
          {onlineCount}/{robots.length}
        </span>
        {fwSummary && (
          <SimpleTooltip title="Firmware version on online robots">
            <span className="font-mono text-text-muted">fw {fwSummary}</span>
          </SimpleTooltip>
        )}
        {robots.length > 0 && (
          <span className="flex items-center gap-1">
            {robots.map(({ team, fb }) => (
              <SimpleTooltip
                key={`${team ?? "?"}-${fb.id}`}
                title={`${team ? team + " " : ""}p${fb.id} · ${
                  fb.online ? "online" : "offline"
                } · ${
                  fb.feedback_hz !== undefined
                    ? `${Math.round(fb.feedback_hz)} Hz`
                    : "— Hz"
                } · ${
                  fb.feedback_age_ms !== undefined
                    ? `${fb.feedback_age_ms} ms`
                    : "— ms"
                } · fw ${fmtVersion(fb.firmware_version)}`}
              >
                <span className="flex items-center gap-0.5">
                  <span
                    className={cn(
                      "w-1.5 h-1.5 rounded-full",
                      severityDotClass(robotSeverity(fb))
                    )}
                  />
                  <span className="font-mono text-text-muted">{fb.id}</span>
                </span>
              </SimpleTooltip>
            ))}
          </span>
        )}
      </span>
    </div>
  );
};

export default BasestationBar;
