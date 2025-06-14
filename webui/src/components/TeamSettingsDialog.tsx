import React, { useState, useEffect, useReducer } from "react";
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
import { useExecutorInfo, useTeamConfiguration } from "@/api";
import FilePathSelector from "./FilePathSelector";

interface State {
  blueActive: boolean;
  blueScriptPath?: string;
  yellowActive: boolean;
  yellowScriptPath?: string;
}

type Action =
  | {
      type: "set_blue_active";
      payload: boolean;
    }
  | {
      type: "set_yellow_active";
      payload: boolean;
    }
  | {
      type: "set_blue_script_path";
      payload: string | null;
    }
  | {
      type: "set_yellow_script_path";
      payload: string | null;
    };

const reducer = (state: State, action: Action): State => {
  switch (action.type) {
    case "set_blue_active":
      return {
        ...state,
        blueActive: action.payload,
        blueScriptPath: action.payload ? state.blueScriptPath : undefined,
      };
    case "set_yellow_active":
      return {
        ...state,
        yellowActive: action.payload,
        yellowScriptPath: action.payload ? state.yellowScriptPath : undefined,
      };
    case "set_blue_script_path":
      return {
        ...state,
        blueScriptPath: action.payload ?? undefined,
      };
    case "set_yellow_script_path":
      return {
        ...state,
        yellowScriptPath: action.payload ?? undefined,
      };
  }
};

const TeamSettingsDialog: React.FC = () => {
  const executorInfo = useExecutorInfo();
  const blueActive =
    executorInfo?.active_teams.includes(TeamColor.Blue) ?? false;
  const yellowActive =
    executorInfo?.active_teams.includes(TeamColor.Yellow) ?? false;
  const { setActiveTeams, setTeamScriptPaths } = useTeamConfiguration();
  const [open, setOpen] = useState(false);

  const [state, dispatch] = useReducer(reducer, {
    blueActive,
    yellowActive,
  });

  useEffect(() => {
    dispatch({ type: "set_blue_active", payload: blueActive });
    dispatch({ type: "set_yellow_active", payload: yellowActive });
  }, [blueActive, yellowActive]);

  const handleSave = () => {
    setActiveTeams({
      blueActive: state.blueActive,
      yellowActive: state.yellowActive,
    });
    setTeamScriptPaths({
      blueScriptPath: state.blueScriptPath,
      yellowScriptPath: state.yellowScriptPath,
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
                checked={state.blueActive}
                onCheckedChange={(checked) => {
                  dispatch({ type: "set_blue_active", payload: checked });
                }}
              />
              <Label htmlFor="team-a-active">Active</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Label htmlFor="team-a-script-path">Script Path</Label>
              <div className="flex-1">
                <FilePathSelector
                  value={state.blueScriptPath || ""}
                  onChange={(path) =>
                    dispatch({ type: "set_blue_script_path", payload: path })
                  }
                  placeholder="Select script..."
                />
              </div>
            </div>
          </div>

          {/* Team B Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Yellow Team</h3>
            <div className="flex items-center space-x-2">
              <Switch
                id="team-b-active"
                checked={state.yellowActive}
                onCheckedChange={(checked) => {
                  dispatch({ type: "set_yellow_active", payload: checked });
                }}
              />
              <Label htmlFor="team-b-active">Active</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Label htmlFor="team-b-script-path">Script Path</Label>
              <div className="flex-1">
                <FilePathSelector
                  value={state.yellowScriptPath || ""}
                  onChange={(path) =>
                    dispatch({ type: "set_yellow_script_path", payload: path })
                  }
                  placeholder="Select script..."
                />
              </div>
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
