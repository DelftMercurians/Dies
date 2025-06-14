import React, { useState, useEffect } from "react";
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
import { TeamColor, SideAssignment } from "@/bindings";
import { useTeamConfiguration } from "@/api";

interface TeamSettingsDialogProps {
  currentSideAssignment?: SideAssignment;
  blueActive: boolean;
  yellowActive: boolean;
}

const TeamSettingsDialog: React.FC<TeamSettingsDialogProps> = ({
  currentSideAssignment,
  blueActive,
  yellowActive,
}) => {
  const { setActiveTeams } = useTeamConfiguration();
  const [open, setOpen] = useState(false);

  // Local state for the form
  const [localBlueActive, setLocalBlueActive] = useState(blueActive);
  const [localYellowActive, setLocalYellowActive] = useState(yellowActive);
  const [sideAssignment, setSideAssignment] = useState<SideAssignment>(
    currentSideAssignment || SideAssignment.BlueOnPositive
  );

  useEffect(() => {
    setLocalBlueActive(blueActive);
  }, [blueActive]);

  useEffect(() => {
    setLocalYellowActive(yellowActive);
  }, [yellowActive]);

  useEffect(() => {
    if (currentSideAssignment) {
      setSideAssignment(currentSideAssignment);
    }
  }, [currentSideAssignment]);

  const handleSave = () => {
    setActiveTeams({
      blueActive: localBlueActive,
      yellowActive: localYellowActive,
    });
    setOpen(false);
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
            <h3 className="text-lg font-semibold">Blue Team</h3>
            <div className="flex items-center space-x-2">
              <Switch
                id="team-a-active"
                checked={localBlueActive}
                onCheckedChange={(checked) => {
                  setLocalBlueActive(checked);
                }}
              />
              <Label htmlFor="team-a-active">Active</Label>
            </div>
          </div>

          {/* Team B Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Yellow Team</h3>
            <div className="flex items-center space-x-2">
              <Switch
                id="team-b-active"
                checked={localYellowActive}
                onCheckedChange={(checked) => {
                  setLocalYellowActive(checked);
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
