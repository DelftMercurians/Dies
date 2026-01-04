import React from "react";
import { usePrimaryTeam, useExecutorSettings, useTeamConfiguration } from "@/api";
import { TeamColor, SideAssignment } from "@/bindings";
import { SimpleTooltip } from "@/components/ui/tooltip";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { cn } from "@/lib/utils";

/**
 * Compact team/side indicator.
 *
 * Displays: "●BLU→  YEL" or similar
 * - Shows primary team with ● prefix
 * - → indicates which team attacks positive X (right side)
 * - Click: cycles primary team
 * - Right-click: opens quick side-swap menu
 *
 * States:
 * | Primary | Attacking +X | Display       |
 * |---------|--------------|---------------|
 * | Blue    | Blue         | ●BLU→  YEL    |
 * | Blue    | Yellow       | ●BLU  →YEL    |
 * | Yellow  | Blue         | BLU→  ●YEL    |
 * | Yellow  | Yellow       | BLU  →●YEL    |
 */

const TeamIndicator: React.FC = () => {
  const [primaryTeam, setPrimaryTeam] = usePrimaryTeam();
  const { settings } = useExecutorSettings();
  const { swapTeamColors, swapTeamSides } = useTeamConfiguration();

  const sideAssignment =
    settings?.team_configuration.side_assignment ?? SideAssignment.YellowOnPositive;

  // Blue attacks +X means Yellow defends +X
  const blueAttacksPositive = sideAssignment === SideAssignment.YellowOnPositive;

  const handleClick = () => {
    // Cycle primary team
    setPrimaryTeam(primaryTeam === TeamColor.Blue ? TeamColor.Yellow : TeamColor.Blue);
  };

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <SimpleTooltip title="Click to switch primary team. Right-click for options.">
          <button
            onClick={handleClick}
            className={cn(
              "h-5 px-2 flex items-center gap-0.5 border border-border-muted",
              "text-[9px] font-semibold uppercase tracking-wider",
              "transition-colors hover:bg-bg-overlay",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan"
            )}
          >
            {/* Blue team */}
            <span
              className={cn(
                "flex items-center",
                primaryTeam === TeamColor.Blue ? "text-team-blue" : "text-team-blue/60"
              )}
            >
              {primaryTeam === TeamColor.Blue && (
                <span className="text-accent-cyan mr-0.5">●</span>
              )}
              BLU
            </span>

            {/* Arrow indicator - shows who attacks +X */}
            {blueAttacksPositive ? (
              <>
                <span className="text-text-dim">→</span>
                <span className="w-2" />
              </>
            ) : (
              <>
                <span className="w-2" />
                <span className="text-text-dim">→</span>
              </>
            )}

            {/* Yellow team */}
            <span
              className={cn(
                "flex items-center",
                primaryTeam === TeamColor.Yellow ? "text-team-yellow" : "text-team-yellow/60"
              )}
            >
              {primaryTeam === TeamColor.Yellow && (
                <span className="text-accent-cyan mr-0.5">●</span>
              )}
              YEL
            </span>
          </button>
        </SimpleTooltip>
      </ContextMenuTrigger>

      <ContextMenuContent>
        <ContextMenuItem onClick={handleClick}>
          Switch Primary Team
        </ContextMenuItem>
        <ContextMenuSeparator />
        <ContextMenuItem onClick={() => swapTeamSides()}>
          Swap Field Sides
        </ContextMenuItem>
        <ContextMenuItem onClick={() => swapTeamColors()}>
          Swap Team Colors
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  );
};

export default TeamIndicator;

