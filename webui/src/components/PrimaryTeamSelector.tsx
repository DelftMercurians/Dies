import React from "react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Badge } from "@/components/ui/badge";
import { TeamColor } from "@/bindings";
import { usePrimaryTeam } from "@/api";

/**
 * Primary Team Selector for choosing which team to focus on.
 * Uses mission control aesthetic with team colors.
 */
interface PrimaryTeamSelectorProps {
  className?: string;
}

const PrimaryTeamSelector: React.FC<PrimaryTeamSelectorProps> = () => {
  const [primaryTeam, setPrimaryTeam] = usePrimaryTeam();
  const teams = [
    {
      name: `Blue`,
      color: TeamColor.Blue,
    },
    {
      name: `Yellow`,
      color: TeamColor.Yellow,
    },
  ];

  return (
    <ToggleGroup
      type="single"
      value={primaryTeam.toString()}
      onValueChange={(value) => {
        if (value) {
          setPrimaryTeam(value as TeamColor);
        }
      }}
    >
      {teams.map((team) => (
        <ToggleGroupItem
          key={team.color.toString()}
          value={team.color}
          className={`flex items-center gap-2 data-[state=on]:bg-accent-green/30 data-[state=on]:text-accent-green data-[state=on]:border-accent-green ${
            team.color === TeamColor.Blue
              ? "text-team-blue"
              : "text-team-yellow"
          }`}
        >
          <Badge
            variant={team.color === TeamColor.Blue ? "team-blue" : "team-yellow"}
          >
            {team.color}
          </Badge>
          <span className="text-[10px]">{team.name}</span>
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
};

export default PrimaryTeamSelector;
