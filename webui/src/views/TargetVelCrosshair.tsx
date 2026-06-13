import { cn } from "@/lib/utils";
import { FC } from "react";

/**
 * Renders a target-velocity vec2 as a crosshair: a vector from the center
 * encodes direction and (clamped) magnitude. The signed x/y components and
 * total magnitude are shown right-aligned with tabular figures so they stay
 * readable while changing quickly.
 *
 * Team-relative frame: +x toward the opponent goal (right), +y up.
 */
const SIZE = 76;
const C = SIZE / 2;
const R = C - 6;
const FULL_SCALE = 4000; // mm/s mapped to the boundary radius

const TargetVelCrosshair: FC<{ vx: number; vy: number; className?: string }> = ({
  vx,
  vy,
  className,
}) => {
  const mag = Math.hypot(vx, vy);
  const scale = Math.min(1, mag / FULL_SCALE);
  const nx = mag > 0 ? vx / mag : 0;
  const ny = mag > 0 ? vy / mag : 0;
  const ex = C + nx * R * scale;
  const ey = C - ny * R * scale; // screen y is inverted

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
        {/* vector */}
        {mag > 0 ? (
          <line
            x1={C}
            y1={C}
            x2={ex}
            y2={ey}
            className="stroke-accent-cyan"
            strokeWidth={2}
          />
        ) : null}
        <circle cx={ex} cy={ey} r={3} className="fill-accent-cyan" />
      </svg>

      <div className="flex flex-col gap-1 font-mono text-sm tabular-nums">
        <ValueRow label="x" value={vx} className="text-text-bright" />
        <ValueRow label="y" value={vy} className="text-text-bright" />
        <ValueRow label="|v|" value={mag} className="text-text-std" />
      </div>
    </div>
  );
};

export default TargetVelCrosshair;

const ValueRow: FC<{ label: string; value: number; className?: string }> = ({
  label,
  value,
  className,
}) => (
  <div className="flex items-baseline gap-2 w-24" title={String(value)}>
    <span className="text-text-dim w-8 whitespace-nowrap">{label}</span>
    <span className={cn("flex-1 text-right", className)}>
      {Math.round(value)}
    </span>
  </div>
);
