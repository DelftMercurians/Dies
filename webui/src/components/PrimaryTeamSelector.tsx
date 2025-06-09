import React from "react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Badge } from "@/components/ui/badge";
import { TeamConfiguration, TeamColor, TeamId } from "@/bindings";
import { useTeamConfiguration } from "@/api";

interface PrimaryTeamSelectorProps {
  currentConfig?: TeamConfiguration;
  primaryTeamId?: TeamId;
  className?: string;
}

const PrimaryTeamSelector: React.FC<PrimaryTeamSelectorProps> = ({
  currentConfig,
  primaryTeamId,
  className = "",
}) => {
  const { setPrimaryTeam } = useTeamConfiguration();

  if (!currentConfig) {
    return (
      <div className={`text-sm text-muted-foreground ${className}`}>
        No team configuration available
      </div>
    );
  }

  const teams = [
    {
      id: currentConfig.team_a_info.id,
      name: currentConfig.team_a_info.name || `Team A`,
      color: currentConfig.team_a_color,
    },
    {
      id: currentConfig.team_b_info.id,
      name: currentConfig.team_b_info.name || `Team B`,
      color: currentConfig.team_b_color,
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
      value={primaryTeamId?.toString()}
      onValueChange={(value) => {
        if (value) {
          setPrimaryTeam(parseInt(value) as TeamId);
        }
      }}
      className="border border-gray-500 rounded-lg"
    >
      {teams.map((team) => (
        <ToggleGroupItem
          key={team.id.toString()}
          value={team.id.toString()}
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
