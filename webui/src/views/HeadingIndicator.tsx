import { cn } from "@/lib/utils";
import { FC } from "react";

/**
 * Renders robot heading as a compass dial: the green needle is the measured
 * yaw, the amber needle is the commanded setpoint, and the arc between them is
 * the heading error. Numeric setpoint / value / rate are shown right-aligned
 * with tabular figures so they stay readable while changing quickly.
 *
 * Absolute (dies) frame: yaw 0 points +x (right), +y is up (CCW positive) —
 * the same frame as `player.yaw` and the `heading_setpoint` debug value, so the
 * two needles are directly comparable. (Unlike the team-relative target-vel
 * crosshair above it.)
 */
const SIZE = 76;
const C = SIZE / 2;
const R = C - 6;

/** Wrap to (-pi, pi]. */
const wrap = (a: number) => {
  const t = ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
  return t - Math.PI;
};

const deg = (rad: number) => (rad * 180) / Math.PI;

/** Endpoint on the dial for an angle: yaw 0 -> +x, +y up (screen y inverted). */
const tip = (rad: number, r: number = R): [number, number] => [
  C + Math.cos(rad) * r,
  C - Math.sin(rad) * r,
];

const HeadingIndicator: FC<{
  yaw: number;
  setpoint: number | null;
  rate: number;
  className?: string;
}> = ({ yaw, setpoint, rate, className }) => {
  const [yx, yy] = tip(yaw);
  const hasSp = setpoint != null && Number.isFinite(setpoint);
  const [sx, sy] = hasSp ? tip(setpoint) : [C, C];
  const err = hasSp ? wrap(setpoint - yaw) : null;

  return (
    <div className={cn("flex items-center gap-3", className)}>
      <svg width={SIZE} height={SIZE} className="shrink-0">
        {/* boundary + axes */}
        <circle
          cx={C}
          cy={C}
          r={R}
          className="fill-none stroke-border-subtle"
          strokeWidth={1}
        />
        <line
          x1={C - R}
          y1={C}
          x2={C + R}
          y2={C}
          className="stroke-border-muted"
          strokeWidth={1}
        />
        <line
          x1={C}
          y1={C - R}
          x2={C}
          y2={C + R}
          className="stroke-border-muted"
          strokeWidth={1}
        />
        {/* setpoint needle (amber, dashed) */}
        {hasSp ? (
          <>
            <line
              x1={C}
              y1={C}
              x2={sx}
              y2={sy}
              className="stroke-accent-amber"
              strokeWidth={1.5}
              strokeDasharray="3 2"
            />
            <circle cx={sx} cy={sy} r={2.5} className="fill-accent-amber" />
          </>
        ) : null}
        {/* measured heading needle (green) */}
        <line
          x1={C}
          y1={C}
          x2={yx}
          y2={yy}
          className="stroke-accent-green"
          strokeWidth={2}
        />
        <circle cx={yx} cy={yy} r={3} className="fill-accent-green" />
      </svg>

      <div className="flex flex-col gap-1 font-mono text-sm tabular-nums">
        <ValueRow
          label="sp"
          value={hasSp ? `${Math.round(deg(setpoint!))}°` : "—"}
          className="text-accent-amber"
        />
        <ValueRow
          label="yaw"
          value={`${Math.round(deg(yaw))}°`}
          className="text-accent-green"
        />
        <ValueRow
          label="err"
          value={err != null ? `${Math.round(deg(err))}°` : "—"}
          className="text-text-std"
        />
        <ValueRow
          label="rate"
          value={`${Math.round(deg(rate))}°/s`}
          className="text-text-std"
        />
      </div>
    </div>
  );
};

export default HeadingIndicator;

const ValueRow: FC<{ label: string; value: string; className?: string }> = ({
  label,
  value,
  className,
}) => (
  <div className="flex items-baseline gap-2 w-28" title={value}>
    <span className="text-text-dim w-8 whitespace-nowrap">{label}</span>
    <span className={cn("flex-1 text-right", className)}>{value}</span>
  </div>
);
