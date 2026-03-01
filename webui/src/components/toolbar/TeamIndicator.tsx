import React from "react";
import { ChevronDown, ArrowRightLeft, Repeat } from "lucide-react";
import {
  usePrimaryTeam,
  useExecutorSettings,
  useTeamConfiguration,
} from "@/api";
import { TeamColor, SideAssignment } from "@/bindings";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";

/**
 * Team/side indicator with segmented control.
 *
 * - Two clickable team chips; primary has team-colored bg highlight
 * - Arrow between them shows attack direction (+X)
 * - Chevron dropdown for swap actions (sides, colors)
 */

const TeamIndicator: React.FC = () => {
  const [primaryTeam, setPrimaryTeam] = usePrimaryTeam();
  const { settings } = useExecutorSettings();
  const { swapTeamColors, swapTeamSides } = useTeamConfiguration();

  const sideAssignment =
    settings?.team_configuration.side_assignment ??
    SideAssignment.YellowOnPositive;
  const blueAttacksPositive =
    sideAssignment === SideAssignment.YellowOnPositive;
  const isBlueActive = primaryTeam === TeamColor.Blue;

  return (
    <div className="inline-flex items-center h-7 border border-border-muted">
      {/* Blue team chip */}
      <button
        onClick={() => setPrimaryTeam(TeamColor.Blue)}
        className={cn(
          "h-full px-2.5 text-sm font-semibold uppercase tracking-wider transition-colors",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
          isBlueActive
            ? "bg-team-blue/25 text-team-blue"
            : "text-team-blue/40 hover:text-team-blue/70 hover:bg-bg-overlay",
        )}
      >
        BLU
      </button>

      <div className="w-px h-full bg-border-subtle" />

      {/* Direction arrow */}
      <div className="h-full px-1.5 flex items-center text-text-dim text-xs select-none">
        {blueAttacksPositive ? "\u2192" : "\u2190"}
      </div>

      <div className="w-px h-full bg-border-subtle" />

      {/* Yellow team chip */}
      <button
        onClick={() => setPrimaryTeam(TeamColor.Yellow)}
        className={cn(
          "h-full px-2.5 text-sm font-semibold uppercase tracking-wider transition-colors",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
          !isBlueActive
            ? "bg-team-yellow/25 text-team-yellow"
            : "text-team-yellow/40 hover:text-team-yellow/70 hover:bg-bg-overlay",
        )}
      >
        YEL
      </button>

      <div className="w-px h-full bg-border-subtle" />

      {/* Dropdown for swap actions */}
      <Popover>
        <PopoverTrigger asChild>
          <button
            className={cn(
              "h-full px-1.5 flex items-center",
              "text-text-dim hover:text-text-std hover:bg-bg-overlay transition-colors",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan",
            )}
          >
            <ChevronDown className="w-3 h-3" />
          </button>
        </PopoverTrigger>
        <PopoverContent align="start" className="w-44 p-1">
          <button
            onClick={() => swapTeamSides()}
            className="w-full flex items-center gap-2 px-2 py-1.5 text-sm text-text-std hover:bg-bg-overlay transition-colors"
          >
            <ArrowRightLeft className="w-3.5 h-3.5 text-text-dim" />
            Swap Sides
          </button>
          <button
            onClick={() => swapTeamColors()}
            className="w-full flex items-center gap-2 px-2 py-1.5 text-sm text-text-std hover:bg-bg-overlay transition-colors"
          >
            <Repeat className="w-3.5 h-3.5 text-text-dim" />
            Swap Colors
          </button>
        </PopoverContent>
      </Popover>
    </div>
  );
};

export default TeamIndicator;
