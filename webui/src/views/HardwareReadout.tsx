import { PlayerFeedbackMsg, SysStatus, TeamColor } from "@/bindings";
import { Badge } from "@/components/ui/badge";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn, formatNumber } from "@/lib/utils";
import {
  batterySeverity,
  capVoltageSeverity,
  motorTempSeverity,
  Severity,
  severityDotClass,
  severityTextClass,
  sysStatusSeverity,
} from "@/lib/hardware";
import { FC } from "react";
import PatternIcon from "./PatternIcon";

/**
 * Per-player hardware readout, mirroring the standalone basestation viewer.
 * Quiet by default; values/dots color only when a threshold is crossed.
 * Sources data from the `/api/basestation` feedback (live mode only).
 */
const HardwareReadout: FC<{
  id: number;
  team: TeamColor;
  feedback?: PlayerFeedbackMsg;
  /** Ball-in-dribbler from the world stream (works in sim + live). */
  breakbeamBall?: boolean;
  className?: string;
}> = ({ id, team, feedback, breakbeamBall, className }) => {
  const sensorError = feedback?.breakbeam_sensor_ok === false;
  return (
    <div className={cn("flex flex-col gap-2 text-sm", className)}>
      {/* identity row */}
      <div className="flex items-center gap-3">
        <PatternIcon id={id} team={team} size={56} />
        <span className="font-semibold text-text-bright text-lg">#{id}</span>
        <Badge variant={team === TeamColor.Blue ? "team-blue" : "team-yellow"}>
          {team}
        </Badge>
        <div className="ml-auto flex items-center gap-3">
          <BreakbeamBall detected={!!breakbeamBall} sensorError={sensorError} />
          <StatusDot status={feedback?.primary_status} label="Status" />
        </div>
      </div>

      {!feedback ? (
        <div className="text-text-muted text-xs italic">
          No basestation feedback (live mode only).
        </div>
      ) : (
        <>
          {/* power */}
          <Section title="Power">
            <Metric
              label="Battery L"
              value={feedback.pack_voltages?.[0]}
              unit="V"
              severity={batterySeverity(feedback.pack_voltages?.[0])}
            />
            <Metric
              label="Battery R"
              value={feedback.pack_voltages?.[1]}
              unit="V"
              severity={batterySeverity(feedback.pack_voltages?.[1])}
            />
            <Metric
              label="Capacitor"
              value={feedback.kicker_cap_voltage}
              unit="V"
              severity={capVoltageSeverity(feedback.kicker_cap_voltage)}
            />
            <Metric
              label="Kicker temp"
              value={feedback.kicker_temp}
              unit="°C"
            />
          </Section>

          {/* subsystem statuses */}
          <Section title="Status">
            <StatusDot status={feedback.primary_status} label="Primary" />
            <StatusDot status={feedback.imu_status} label="IMU" />
            <StatusDot status={feedback.kicker_status} label="Kicker" />
            <StatusDot
              status={feedback.tof_status}
              label="Time of Flight Sensor"
            />
          </Section>

          {/* imu */}
          {feedback.imu_readings && (
            <Section title="IMU">
              <Metric
                label="θx"
                value={feedback.imu_readings[0]}
                unit=" rad"
              />
              <Metric
                label="θy"
                value={feedback.imu_readings[1]}
                unit=" rad"
              />
              <Metric
                label="θz"
                value={feedback.imu_readings[2]}
                unit=" rad"
              />
              <Metric
                label="ωx"
                value={feedback.imu_readings[3]}
                unit=" rad/s"
              />
              <Metric
                label="ωy"
                value={feedback.imu_readings[4]}
                unit=" rad/s"
              />
              <Metric
                label="ωz"
                value={feedback.imu_readings[5]}
                unit=" rad/s"
              />
            </Section>
          )}

          {/* motors */}
          <Section title="Motors">
            <div className="grid grid-cols-[auto_1fr_1fr_1fr] gap-x-3 gap-y-0.5 text-xs">
              <div className="text-text-dim">#</div>
              <div className="text-text-dim">status</div>
              <div className="text-text-dim">temp</div>
              <div className="text-text-dim">speed</div>
              {[0, 1, 2, 3, 4].map((i) => {
                const status = feedback.motor_statuses?.[i];
                const temp = feedback.motor_temps?.[i];
                const speed = feedback.motor_speeds?.[i];
                return (
                  <MotorRow
                    key={i}
                    label={i === 4 ? "dr" : `m${i}`}
                    status={status}
                    temp={temp}
                    speed={speed}
                  />
                );
              })}
            </div>
          </Section>
        </>
      )}
    </div>
  );
};

export default HardwareReadout;

/** Orange ball indicator for the breakbeam (ball-in-dribbler) flag. */
const BreakbeamBall: FC<{ detected: boolean; sensorError?: boolean }> = ({
  detected,
  sensorError,
}) => (
  <SimpleTooltip
    title={
      sensorError ? "Breakbeam sensor error" : detected ? "Ball" : "No ball"
    }
  >
    <span className="flex items-center">
      <span
        className={cn(
          "w-4 h-4 rounded-full border",
          sensorError ? "border-accent-red" : "border-border-muted",
          !detected && "bg-transparent",
        )}
        style={
          detected
            ? {
                border: "none",
                background:
                  "radial-gradient(circle at 35% 30%, #ffb066, #ff7a00)",
              }
            : undefined
        }
      />
    </span>
  </SimpleTooltip>
);

const Section: FC<{ title: string; children: React.ReactNode }> = ({
  title,
  children,
}) => (
  <div className="flex flex-col gap-1">
    <div className="text-[11px] uppercase tracking-wider text-text-dim">
      {title}
    </div>
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 pl-1">
      {children}
    </div>
  </div>
);

const StatusDot: FC<{ status?: SysStatus; label: string }> = ({
  status,
  label,
}) => {
  const sev = sysStatusSeverity(status);
  return (
    <SimpleTooltip title={status ?? "n/a"}>
      <span className="flex items-center gap-1.5">
        <span
          className={cn("w-1.5 h-1.5 rounded-full", severityDotClass(sev))}
        />
        <span className="text-xs text-text-std">{label}</span>
      </span>
    </SimpleTooltip>
  );
};

const Metric: FC<{
  label: string;
  value?: number;
  unit: string;
  severity?: Severity;
}> = ({ label, value, unit, severity = "ok" }) => (
  <span className="flex items-center gap-1.5 text-xs">
    <span className="text-text-dim">{label}</span>
    <span
      className={cn("font-mono", severityTextClass(severity))}
      title={value !== undefined ? String(value) : undefined}
    >
      {value !== undefined ? `${formatNumber(value)}${unit}` : "—"}
    </span>
  </span>
);

const MotorRow: FC<{
  label: string;
  status?: SysStatus;
  temp?: number;
  speed?: number;
}> = ({ label, status, temp, speed }) => (
  <>
    <div className="font-mono text-text-std">{label}</div>
    <div className="flex items-center gap-1">
      <span
        className={cn(
          "w-1.5 h-1.5 rounded-full",
          severityDotClass(sysStatusSeverity(status)),
        )}
      />
      <span className="text-text-dim" title={status}>
        {status ?? "—"}
      </span>
    </div>
    <div
      className={cn("font-mono", severityTextClass(motorTempSeverity(temp)))}
      title={temp !== undefined ? String(temp) : undefined}
    >
      {temp !== undefined ? `${formatNumber(temp)}°C` : "—"}
    </div>
    <div
      className="font-mono text-text-dim"
      title={speed !== undefined ? String(speed) : undefined}
    >
      {speed !== undefined ? formatNumber(speed) : "—"}
    </div>
  </>
);
