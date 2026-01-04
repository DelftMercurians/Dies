import React from "react";
import { TeamColor, SideAssignment } from "../bindings";
import { cn } from "@/lib/utils";

/**
 * Game Banner showing team sides and yellow cards.
 * Uses mission control aesthetic with sharp corners and team colors.
 */
interface GameBannerProps {
  blueTeamYellowCards: number;
  yellowTeamYellowCards: number;
  sideAssignment: SideAssignment;
}

const GameBanner: React.FC<GameBannerProps> = ({
  blueTeamYellowCards,
  yellowTeamYellowCards,
  sideAssignment,
}) => {
  // Determine which team is on which side based on side assignment
  const leftTeam =
    sideAssignment === SideAssignment.BlueOnPositive
      ? TeamColor.Yellow
      : TeamColor.Blue;
  const rightTeam =
    sideAssignment === SideAssignment.BlueOnPositive
      ? TeamColor.Blue
      : TeamColor.Yellow;

  const leftCards =
    leftTeam === TeamColor.Blue ? blueTeamYellowCards : yellowTeamYellowCards;
  const rightCards =
    rightTeam === TeamColor.Blue ? blueTeamYellowCards : yellowTeamYellowCards;

  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20 flex items-center gap-6 bg-bg-elevated/90 border border-border-muted px-4 py-2">
      {/* Left side team */}
      <div className="flex items-center gap-2">
        <div
          className={cn(
            "w-4 h-4 border border-text-dim",
            leftTeam === TeamColor.Blue ? "bg-team-blue" : "bg-team-yellow"
          )}
        />
        <span className="text-text-bright font-medium text-[11px] uppercase tracking-wide">
          {leftTeam === TeamColor.Blue ? "Blue" : "Yellow"}
        </span>
        <div className="flex items-center gap-1">
          {Array.from({ length: leftCards }, (_, i) => (
            <div
              key={i}
              className="w-2.5 h-3.5 bg-team-yellow border border-team-yellow-dim"
            />
          ))}
        </div>
      </div>

      {/* Divider */}
      <div className="w-px h-5 bg-border-muted" />

      {/* Right side team */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1">
          {Array.from({ length: rightCards }, (_, i) => (
            <div
              key={i}
              className="w-2.5 h-3.5 bg-team-yellow border border-team-yellow-dim"
            />
          ))}
        </div>
        <span className="text-text-bright font-medium text-[11px] uppercase tracking-wide">
          {rightTeam === TeamColor.Blue ? "Blue" : "Yellow"}
        </span>
        <div
          className={cn(
            "w-4 h-4 border border-text-dim",
            rightTeam === TeamColor.Blue ? "bg-team-blue" : "bg-team-yellow"
          )}
        />
      </div>
    </div>
  );
};

export default GameBanner;
