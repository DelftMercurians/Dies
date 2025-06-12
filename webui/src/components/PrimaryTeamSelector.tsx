import React from "react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Badge } from "@/components/ui/badge";
import { TeamColor } from "@/bindings";

interface PrimaryTeamSelectorProps {
  primaryTeam: TeamColor;
  className?: string;
  setPrimaryTeam: (teamColor: TeamColor) => void;
}

const PrimaryTeamSelector: React.FC<PrimaryTeamSelectorProps> = ({
  primaryTeam,
  className = "",
  setPrimaryTeam,
}) => {
  const teams = [
    {
      name: `Blue Team`,
      color: TeamColor.Blue,
    },
    {
      name: `Yellow Team`,
      color: TeamColor.Yellow,
    },
  ];

  const getColorBadgeClass = (color: TeamColor) => {
    return color === TeamColor.Blue
      ? "bg-blue-600 text-white hover:bg-blue-700"
      : "bg-yellow-500 text-black hover:bg-yellow-600";
  };

  return (
    <ToggleGroup
      type="single"
      value={primaryTeam.toString()}
      onValueChange={(value) => {
        if (value) {
          setPrimaryTeam(value as TeamColor);
        }
      }}
      className="border border-gray-500 rounded-lg"
    >
      {teams.map((team) => (
        <ToggleGroupItem
          key={team.color.toString()}
          value={team.color}
          className="data-[state=on]:bg-green-600 flex items-center gap-2"
        >
          <Badge
            variant="secondary"
            className={`text-xs ${getColorBadgeClass(team.color)}`}
          >
            {team.color}
          </Badge>
          <span>{team.name}</span>
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
};

export default PrimaryTeamSelector;
