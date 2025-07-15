import React from "react";
import { TeamColor, SideAssignment } from "../bindings";
import { cn } from "@/lib/utils";

interface YellowCardBannerProps {
  blueTeamYellowCards: number;
  yellowTeamYellowCards: number;
  sideAssignment: SideAssignment;
}

const YellowCardBanner: React.FC<YellowCardBannerProps> = ({
  blueTeamYellowCards,
  yellowTeamYellowCards,
  sideAssignment,
}) => {
  // Determine which team is on which side based on side assignment
  const leftTeam = sideAssignment === SideAssignment.BlueOnPositive ? TeamColor.Yellow : TeamColor.Blue;
  const rightTeam = sideAssignment === SideAssignment.BlueOnPositive ? TeamColor.Blue : TeamColor.Yellow;
  
  const leftCards = leftTeam === TeamColor.Blue ? blueTeamYellowCards : yellowTeamYellowCards;
  const rightCards = rightTeam === TeamColor.Blue ? blueTeamYellowCards : yellowTeamYellowCards;

  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20 flex items-center gap-6 bg-black bg-opacity-80 px-4 py-2 rounded-lg">
      {/* Left side team */}
      <div className="flex items-center gap-2">
        <div
          className={cn(
            "w-4 h-4 rounded-full border-2 border-white",
            leftTeam === TeamColor.Blue ? "bg-blue-500" : "bg-yellow-400"
          )}
        />
        <span className="text-white font-medium">
          {leftTeam === TeamColor.Blue ? "Blue" : "Yellow"}
        </span>
        <div className="flex items-center gap-1">
          {Array.from({ length: leftCards }, (_, i) => (
            <div
              key={i}
              className="w-3 h-4 bg-yellow-400 rounded-sm border border-yellow-300"
            />
          ))}
          {leftCards === 0 && <span className="text-gray-400 text-sm">0</span>}
        </div>
      </div>

      {/* Divider */}
      <div className="w-px h-6 bg-gray-500" />

      {/* Right side team */}
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1">
          {Array.from({ length: rightCards }, (_, i) => (
            <div
              key={i}
              className="w-3 h-4 bg-yellow-400 rounded-sm border border-yellow-300"
            />
          ))}
          {rightCards === 0 && <span className="text-gray-400 text-sm">0</span>}
        </div>
        <span className="text-white font-medium">
          {rightTeam === TeamColor.Blue ? "Blue" : "Yellow"}
        </span>
        <div
          className={cn(
            "w-4 h-4 rounded-full border-2 border-white",
            rightTeam === TeamColor.Blue ? "bg-blue-500" : "bg-yellow-400"
          )}
        />
      </div>
    </div>
  );
};

export default YellowCardBanner;