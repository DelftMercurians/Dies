import { cn } from "@/lib/utils";
import { FC } from "react";

/**
 * Minimal inline-SVG sparkline. Plots a series of numbers as a single
 * polyline, auto-scaled to its own min/max. No axes, no interaction — meant
 * for ultra-dense glanceable rows.
 */
const Sparkline: FC<{
  data: number[];
  width?: number;
  height?: number;
  className?: string;
  /** Stroke color (any CSS color). Defaults to a quiet muted tone. */
  color?: string;
  /** Fixed lower y-limit. If omitted, auto-scales to the data min. */
  min?: number;
  /** Fixed upper y-limit. If omitted, auto-scales to the data max. */
  max?: number;
}> = ({
  data,
  width = 56,
  height = 16,
  className,
  color = "currentColor",
  min,
  max,
}) => {
  if (data.length < 2) {
    return (
      <svg
        width={width}
        height={height}
        className={cn("text-text-muted", className)}
      />
    );
  }

  const lo = min ?? Math.min(...data);
  const hi = max ?? Math.max(...data);
  const range = hi - lo || 1;
  const stepX = width / (data.length - 1);
  // Leave 1px padding top/bottom so the stroke isn't clipped.
  const pad = 1;
  const usable = height - pad * 2;

  const points = data
    .map((v, i) => {
      const x = i * stepX;
      // Clamp to [0,1] so values outside fixed limits don't overflow.
      const t = Math.max(0, Math.min(1, (v - lo) / range));
      const y = pad + usable - t * usable;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <svg
      width={width}
      height={height}
      className={cn("text-text-dim", className)}
      preserveAspectRatio="none"
    >
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth={1}
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
};

export default Sparkline;
