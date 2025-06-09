import React, { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { Settings } from "lucide-react";
import {
  TeamConfiguration,
  TeamColor,
  SideAssignment,
  TeamId,
} from "@/bindings";
import { useTeamConfiguration } from "@/api";

interface TeamSettingsDialogProps {
  currentConfig?: TeamConfiguration;
  currentSideAssignment?: SideAssignment;
  blueActive: boolean;
  yellowActive: boolean;
}

const TeamSettingsDialog: React.FC<TeamSettingsDialogProps> = ({
  currentConfig,
  currentSideAssignment,
  blueActive,
  yellowActive,
}) => {
  const { updateTeamConfiguration, setActiveTeams } = useTeamConfiguration();
  const [open, setOpen] = useState(false);

  // Local state for the form
  const [teamAName, setTeamAName] = useState(
    currentConfig?.team_a_info.name || ""
  );
  const [teamBName, setTeamBName] = useState(
    currentConfig?.team_b_info.name || ""
  );
  const [teamAColor, setTeamAColor] = useState<TeamColor>(
    currentConfig?.team_a_color || TeamColor.Blue
  );
  const [teamBColor, setTeamBColor] = useState<TeamColor>(
    currentConfig?.team_b_color || TeamColor.Yellow
  );
  const [localBlueActive, setLocalBlueActive] = useState(blueActive);
  const [localYellowActive, setLocalYellowActive] = useState(yellowActive);
  const [sideAssignment, setSideAssignment] = useState<SideAssignment>(
    currentSideAssignment || SideAssignment.BlueOnPositive
  );

  const handleSave = () => {
    // Create team configuration
    const config: TeamConfiguration = {
      team_a_color: teamAColor,
      team_a_info: {
        id: teamAName ? hash(teamAName) : hash("Team A"),
        name: teamAName || undefined,
      },
      team_b_color: teamBColor,
      team_b_info: {
        id: teamBName ? hash(teamBName) : hash("Team B"),
        name: teamBName || undefined,
      },
    };

    updateTeamConfiguration(config);
    setActiveTeams({
      blueActive: localBlueActive,
      yellowActive: localYellowActive,
    });
    setOpen(false);
  };

  // Simple hash function to generate team IDs from names
  const hash = (str: string): TeamId => {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash);
  };

  const handleColorSwap = () => {
    const tempColor = teamAColor;
    setTeamAColor(teamBColor);
    setTeamBColor(tempColor);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Settings className="h-4 w-4 mr-2" />
          Team Settings
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle>Team Configuration</DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Team A Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Team A</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="team-a-name">Team Name</Label>
                <Input
                  id="team-a-name"
                  value={teamAName}
                  onChange={(e) => setTeamAName(e.target.value)}
                  placeholder="Enter team name"
                />
              </div>
              <div className="space-y-2">
                <Label>Color</Label>
                <ToggleGroup
                  type="single"
                  value={teamAColor}
                  onValueChange={(value) =>
                    value && setTeamAColor(value as TeamColor)
                  }
                >
                  <ToggleGroupItem value={TeamColor.Blue}>Blue</ToggleGroupItem>
                  <ToggleGroupItem value={TeamColor.Yellow}>
                    Yellow
                  </ToggleGroupItem>
                </ToggleGroup>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="team-a-active"
                checked={
                  teamAColor === TeamColor.Blue
                    ? localBlueActive
                    : localYellowActive
                }
                onCheckedChange={(checked) => {
                  if (teamAColor === TeamColor.Blue) {
                    setLocalBlueActive(checked);
                  } else {
                    setLocalYellowActive(checked);
                  }
                }}
              />
              <Label htmlFor="team-a-active">Active</Label>
            </div>
          </div>

          {/* Team B Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Team B</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="team-b-name">Team Name</Label>
                <Input
                  id="team-b-name"
                  value={teamBName}
                  onChange={(e) => setTeamBName(e.target.value)}
                  placeholder="Enter team name"
                />
              </div>
              <div className="space-y-2">
                <Label>Color</Label>
                <ToggleGroup
                  type="single"
                  value={teamBColor}
                  onValueChange={(value) =>
                    value && setTeamBColor(value as TeamColor)
                  }
                >
                  <ToggleGroupItem value={TeamColor.Blue}>Blue</ToggleGroupItem>
                  <ToggleGroupItem value={TeamColor.Yellow}>
                    Yellow
                  </ToggleGroupItem>
                </ToggleGroup>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Switch
                id="team-b-active"
                checked={
                  teamBColor === TeamColor.Blue
                    ? localBlueActive
                    : localYellowActive
                }
                onCheckedChange={(checked) => {
                  if (teamBColor === TeamColor.Blue) {
                    setLocalBlueActive(checked);
                  } else {
                    setLocalYellowActive(checked);
                  }
                }}
              />
              <Label htmlFor="team-b-active">Active</Label>
            </div>
          </div>

          {/* Side Assignment */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Field Assignment</h3>
            <div className="space-y-2">
              <Label>Which team defends positive X side</Label>
              <ToggleGroup
                type="single"
                value={sideAssignment}
                onValueChange={(value) =>
                  value && setSideAssignment(value as SideAssignment)
                }
              >
                <ToggleGroupItem value={SideAssignment.BlueOnPositive}>
                  Blue on +X
                </ToggleGroupItem>
                <ToggleGroupItem value={SideAssignment.YellowOnPositive}>
                  Yellow on +X
                </ToggleGroupItem>
              </ToggleGroup>
            </div>
          </div>

          {/* Actions */}
          <div className="flex justify-between">
            <Button variant="outline" onClick={handleColorSwap}>
              Swap Colors
            </Button>
            <div className="space-x-2">
              <Button variant="outline" onClick={() => setOpen(false)}>
                Cancel
              </Button>
              <Button onClick={handleSave}>Save Configuration</Button>
            </div>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default TeamSettingsDialog;
