import { TeamColor } from "@/bindings";
import { patternSrc } from "@/lib/hardware";
import { cn } from "@/lib/utils";
import { FC } from "react";

/**
 * Renders the SSL color-pattern icon for a robot (center team dot + 4 corner
 * ID dots). Uses the pre-rendered PNGs in `public/patterns/`.
 */
const PatternIcon: FC<{
  id: number;
  team: TeamColor;
  size?: number;
  className?: string;
}> = ({ id, team, size = 18, className }) => {
  const teamStr = team === TeamColor.Blue ? "blue" : "yellow";
  return (
    <img
      src={patternSrc(id, teamStr)}
      width={size}
      height={size}
      alt={`#${id}`}
      className={cn("shrink-0 select-none object-contain", className)}
      style={{ width: size, height: size }}
      draggable={false}
    />
  );
};

export default PatternIcon;
