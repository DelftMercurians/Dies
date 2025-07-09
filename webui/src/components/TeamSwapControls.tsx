import React from "react";
import { Button } from "@/components/ui/button";
import { ArrowLeftRight, Repeat } from "lucide-react";
import { useTeamConfiguration } from "@/api";
import { SimpleTooltip } from "@/components/ui/tooltip";

interface TeamSwapControlsProps {
  className?: string;
  size?: "sm" | "default" | "lg";
  variant?: "default" | "outline" | "ghost";
}

const TeamSwapControls: React.FC<TeamSwapControlsProps> = ({
  className = "",
  size = "sm",
  variant = "outline",
}) => {
  const { swapTeamColors, swapTeamSides } = useTeamConfiguration();

  return (
    <div className={`flex gap-2 ${className}`}>
      <SimpleTooltip title="Swap Blue ↔ Yellow team configurations">
        <Button
          variant={variant}
          size={size}
          onClick={swapTeamColors}
          className="flex items-center gap-2"
        >
          <ArrowLeftRight className="h-4 w-4" />
          Swap Colors
        </Button>
      </SimpleTooltip>

      <SimpleTooltip title="Swap which team defends positive X side">
        <Button
          variant={variant}
          size={size}
          onClick={swapTeamSides}
          className="flex items-center gap-2"
        >
          <Repeat className="h-4 w-4" />
          Swap Sides
        </Button>
      </SimpleTooltip>
    </div>
  );
};

export default TeamSwapControls;
