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
import { Settings, ArrowLeftRight, Repeat } from "lucide-react";
import { TeamColor, SideAssignment } from "@/bindings";
import {
  useExecutorInfo,
  useTeamConfiguration,
  useExecutorSettings,
} from "@/api";
import FilePathSelector from "./FilePathSelector";

interface State {
  blueActive: boolean;
  blueScriptPath?: string;
  yellowActive: boolean;
  yellowScriptPath?: string;
  sideAssignment: SideAssignment;
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
    }
  | {
      type: "set_side_assignment";
      payload: SideAssignment;
    }
  | {
      type: "set_team_configuration";
      payload: State;
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
    case "set_side_assignment":
      return {
        ...state,
        sideAssignment: action.payload,
      };
    case "set_team_configuration":
      return action.payload;
  }
};

const TeamSettingsDialog: React.FC = () => {
  const executorInfo = useExecutorInfo();
  const { settings } = useExecutorSettings();
  const {
    setActiveTeams,
    setTeamScriptPaths,
    setSideAssignment,
    setTeamConfiguration,
    swapTeamColors,
    swapTeamSides,
  } = useTeamConfiguration();
  const [open, setOpen] = useState(false);

  // Initialize state from current settings or defaults
  const initialState: State = {
    blueActive: settings?.team_configuration.blue_active ?? true,
    yellowActive: settings?.team_configuration.yellow_active ?? false,
    blueScriptPath: settings?.team_configuration.blue_script_path,
    yellowScriptPath: settings?.team_configuration.yellow_script_path,
    sideAssignment:
      settings?.team_configuration.side_assignment ??
      SideAssignment.YellowOnPositive,
  };

  const [state, dispatch] = useReducer(reducer, initialState);

  // Update state when settings change
  useEffect(() => {
    if (settings?.team_configuration) {
      dispatch({
        type: "set_team_configuration",
        payload: {
          blueActive: settings.team_configuration.blue_active,
          yellowActive: settings.team_configuration.yellow_active,
          blueScriptPath: settings.team_configuration.blue_script_path,
          yellowScriptPath: settings.team_configuration.yellow_script_path,
          sideAssignment: settings.team_configuration.side_assignment,
        },
      });
    }
  }, [settings]);

  const handleSave = () => {
    // Use the new setTeamConfiguration hook for complete configuration
    setTeamConfiguration({
      blue_active: state.blueActive,
      yellow_active: state.yellowActive,
      blue_script_path: state.blueScriptPath,
      yellow_script_path: state.yellowScriptPath,
      side_assignment: state.sideAssignment,
    });
    setOpen(false);
  };

  const handleSwapColors = () => {
    swapTeamColors();
  };

  const handleSwapSides = () => {
    swapTeamSides();
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Settings className="h-4 w-4 mr-2" />
          Team Settings
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Team Configuration</DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Quick Actions */}
          <div className="space-y-3">
            <h3 className="text-lg font-semibold">Quick Actions</h3>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleSwapColors}
                className="flex items-center gap-2"
              >
                <ArrowLeftRight className="h-4 w-4" />
                Swap Team Colors
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleSwapSides}
                className="flex items-center gap-2"
              >
                <Repeat className="h-4 w-4" />
                Swap Team Sides
              </Button>
            </div>
          </div>

          {/* Side Assignment */}
          <div className="space-y-3">
            <h3 className="text-lg font-semibold">Field Side Assignment</h3>
            <div className="space-y-2">
              <Label>Which team defends the positive X side (+X goal)?</Label>
              <ToggleGroup
                type="single"
                value={state.sideAssignment}
                onValueChange={(value) => {
                  if (value) {
                    dispatch({
                      type: "set_side_assignment",
                      payload: value as SideAssignment,
                    });
                  }
                }}
                className="justify-start"
              >
                <ToggleGroupItem value={SideAssignment.BlueOnPositive}>
                  Blue Team (+X)
                </ToggleGroupItem>
                <ToggleGroupItem value={SideAssignment.YellowOnPositive}>
                  Yellow Team (+X)
                </ToggleGroupItem>
              </ToggleGroup>
            </div>
          </div>

          {/* Blue Team Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-blue-600">Blue Team</h3>
            <div className="flex items-center space-x-2">
              <Switch
                id="team-blue-active"
                checked={state.blueActive}
                onCheckedChange={(checked) => {
                  dispatch({ type: "set_blue_active", payload: checked });
                }}
              />
              <Label htmlFor="team-blue-active">Active</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Label htmlFor="team-blue-script-path">Script Path</Label>
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

          {/* Yellow Team Configuration */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-yellow-600">
              Yellow Team
            </h3>
            <div className="flex items-center space-x-2">
              <Switch
                id="team-yellow-active"
                checked={state.yellowActive}
                onCheckedChange={(checked) => {
                  dispatch({ type: "set_yellow_active", payload: checked });
                }}
              />
              <Label htmlFor="team-yellow-active">Active</Label>
            </div>
            <div className="flex items-center space-x-2">
              <Label htmlFor="team-yellow-script-path">Script Path</Label>
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
