import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { useSendCommand } from "@/api";
import {
  Play,
  Square,
  Pause,
  RotateCcw,
  Target,
  AlertTriangle,
  Clock,
  Trophy,
  Navigation,
} from "lucide-react";

const GameControllerPanel: React.FC = () => {
  const sendCommand = useSendCommand();
  const [lastCommand, setLastCommand] = useState<string | null>(null);

  const sendGcCommand = (command: string) => {
    sendCommand({
      type: "GcCommand",
      data: command,
    });
    setLastCommand(command);
  };

  const commandGroups = {
    "Game Flow": [
      {
        command: "HALT",
        label: "Halt",
        icon: Square,
        variant: "destructive" as const,
        description: "Stop all robots immediately",
      },
      {
        command: "STOP",
        label: "Stop",
        icon: Pause,
        variant: "secondary" as const,
        description: "Stop game, robots may move slowly",
      },
      {
        command: "NORMAL_START",
        label: "Normal Start",
        icon: Play,
        variant: "default" as const,
        description: "Resume normal play",
      },
      {
        command: "FORCE_START",
        label: "Force Start",
        icon: Play,
        variant: "default" as const,
        description: "Force start the game",
      },
    ],
    Kickoffs: [
      {
        command: "PREPARE_KICKOFF_BLUE",
        label: "Blue Kickoff",
        icon: Navigation,
        color: "blue",
        description: "Prepare kickoff for blue team",
      },
      {
        command: "PREPARE_KICKOFF_YELLOW",
        label: "Yellow Kickoff",
        icon: Navigation,
        color: "yellow",
        description: "Prepare kickoff for yellow team",
      },
    ],
    Penalties: [
      {
        command: "PREPARE_PENALTY_BLUE",
        label: "Blue Penalty",
        icon: Target,
        color: "blue",
        description: "Prepare penalty for blue team",
      },
      {
        command: "PREPARE_PENALTY_YELLOW",
        label: "Yellow Penalty",
        icon: Target,
        color: "yellow",
        description: "Prepare penalty for yellow team",
      },
    ],
    "Free Kicks": [
      {
        command: "DIRECT_FREE_BLUE",
        label: "Blue Direct",
        icon: Target,
        color: "blue",
        description: "Direct free kick for blue",
      },
      {
        command: "DIRECT_FREE_YELLOW",
        label: "Yellow Direct",
        icon: Target,
        color: "yellow",
        description: "Direct free kick for yellow",
      },
      {
        command: "INDIRECT_FREE_BLUE",
        label: "Blue Indirect",
        icon: RotateCcw,
        color: "blue",
        description: "Indirect free kick for blue",
      },
      {
        command: "INDIRECT_FREE_YELLOW",
        label: "Yellow Indirect",
        icon: RotateCcw,
        color: "yellow",
        description: "Indirect free kick for yellow",
      },
    ],
    Timeouts: [
      {
        command: "TIMEOUT_BLUE",
        label: "Blue Timeout",
        icon: Clock,
        color: "blue",
        description: "Timeout for blue team",
      },
      {
        command: "TIMEOUT_YELLOW",
        label: "Yellow Timeout",
        icon: Clock,
        color: "yellow",
        description: "Timeout for yellow team",
      },
    ],
    Goals: [
      {
        command: "GOAL_BLUE",
        label: "Blue Goal",
        icon: Trophy,
        color: "blue",
        description: "Goal scored by blue team",
      },
      {
        command: "GOAL_YELLOW",
        label: "Yellow Goal",
        icon: Trophy,
        color: "yellow",
        description: "Goal scored by yellow team",
      },
    ],
    "Ball Placement": [
      {
        command: "BALL_PLACEMENT_BLUE",
        label: "Blue Ball Placement",
        icon: Navigation,
        color: "blue",
        description: "Ball placement by blue team",
      },
      {
        command: "BALL_PLACEMENT_YELLOW",
        label: "Yellow Ball Placement",
        icon: Navigation,
        color: "yellow",
        description: "Ball placement by yellow team",
      },
    ],
  };

  const getButtonVariant = (cmd: any) => {
    if (cmd.variant) return cmd.variant;
    if (cmd.color === "blue") return "default";
    if (cmd.color === "yellow") return "secondary";
    return "outline";
  };

  const getButtonClassName = (cmd: any) => {
    if (cmd.color === "blue") return "bg-blue-600 hover:bg-blue-700 text-white";
    if (cmd.color === "yellow")
      return "bg-yellow-500 hover:bg-yellow-600 text-black";
    return "";
  };

  return (
    <div className="h-full flex flex-col">
      <Card className="flex-1">
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">Game Controller</CardTitle>
          {lastCommand && (
            <Badge variant="outline" className="w-fit">
              Last: {lastCommand.replace(/_/g, " ")}
            </Badge>
          )}
        </CardHeader>
        <CardContent className="space-y-4 overflow-y-auto">
          {Object.entries(commandGroups).map(([groupName, commands]) => (
            <div key={groupName} className="space-y-2">
              <h4 className="text-sm font-medium text-muted-foreground">
                {groupName}
              </h4>
              <div className="grid grid-cols-1 gap-2">
                {commands.map((cmd) => {
                  const Icon = cmd.icon;
                  return (
                    <Button
                      key={cmd.command}
                      variant={getButtonVariant(cmd)}
                      size="sm"
                      className={`justify-start h-auto p-3 ${getButtonClassName(
                        cmd
                      )}`}
                      onClick={() => sendGcCommand(cmd.command)}
                      title={cmd.description}
                    >
                      <Icon className="h-4 w-4 mr-2 flex-shrink-0" />
                      <div className="text-left flex-1">
                        <div className="font-medium">{cmd.label}</div>
                        <div className="text-xs opacity-75 truncate">
                          {cmd.description}
                        </div>
                      </div>
                    </Button>
                  );
                })}
              </div>
              {groupName !== "Ball Placement" && <Separator />}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
};

export default GameControllerPanel;
