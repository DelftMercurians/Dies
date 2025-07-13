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
import { GcSimCommand, TeamColor, Vector2 } from "@/bindings";
import { atom, useAtomValue } from "jotai";

export const BallPlacementPostionAtom = atom<Vector2 | null>(null);

const GameControllerPanel: React.FC = () => {
  const sendCommand = useSendCommand();
  const [lastCommand, setLastCommand] = useState<GcSimCommand | null>(null);

  const sendGcCommand = (command: GcSimCommand) => {
    sendCommand({
      type: "GcCommand",
      data: command,
    });
    setLastCommand(command);
  };

  const ballPlacementPosition = useAtomValue(BallPlacementPostionAtom);
  const commandGroups: Record<
    string,
    {
      command: GcSimCommand;
      label: string;
      icon: React.ElementType;
      variant?: "default" | "secondary" | "destructive" | "outline";
      description: string;
      color?: "blue" | "yellow";
    }[]
  > = {
    "Game Flow": [
      {
        command: { type: "Halt" },
        label: "Halt",
        icon: Square,
        variant: "destructive" as const,
        description: "Stop all robots immediately",
      },
      {
        command: { type: "Stop" },
        label: "Stop",
        icon: Pause,
        variant: "secondary" as const,
        description: "Stop game, robots may move slowly",
      },
      {
        command: { type: "NormalStart" },
        label: "Normal Start",
        icon: Play,
        variant: "default" as const,
        description: "Resume normal play",
      },
      {
        command: { type: "ForceStart" },
        label: "Force Start",
        icon: Play,
        variant: "default" as const,
        description: "Force start the game",
      },
    ],
    Kickoffs: [
      {
        command: { type: "KickOff", data: { team_color: TeamColor.Blue } },
        label: "Blue Kickoff",
        icon: Navigation,
        color: "blue",
        description: "Prepare kickoff for blue team",
      },
      {
        command: { type: "KickOff", data: { team_color: TeamColor.Yellow } },
        label: "Yellow Kickoff",
        icon: Navigation,
        color: "yellow",
        description: "Prepare kickoff for yellow team",
      },
    ],
    Penalties: [
      {
        command: { type: "Penalty", data: { team_color: TeamColor.Blue } },
        label: "Blue Penalty",
        icon: Target,
        color: "blue",
        description: "Prepare penalty for blue team",
      },
      {
        command: { type: "Penalty", data: { team_color: TeamColor.Yellow } },
        label: "Yellow Penalty",
        icon: Target,
        color: "yellow",
        description: "Prepare penalty for yellow team",
      },
    ],
    "Free Kicks": [
      {
        command: { type: "DirectFree", data: { team_color: TeamColor.Blue } },
        label: "Blue Direct",
        icon: Target,
        color: "blue",
        description: "Direct free kick for blue",
      },
      {
        command: { type: "DirectFree", data: { team_color: TeamColor.Yellow } },
        label: "Yellow Direct",
        icon: Target,
        color: "yellow",
        description: "Direct free kick for yellow",
      },
    ],
    // Timeouts: [
    //   {
    //     command: { type: "Timeout", data: { team_color: TeamColor.Blue } },
    //     label: "Blue Timeout",
    //     icon: Clock,
    //     color: "blue",
    //     description: "Timeout for blue team",
    //   },
    //   {
    //     command: "TIMEOUT_YELLOW",
    //     label: "Yellow Timeout",
    //     icon: Clock,
    //     color: "yellow",
    //     description: "Timeout for yellow team",
    //   },
    // ],
    // Goals: [
    //   {
    //     command: "GOAL_BLUE",
    //     label: "Blue Goal",
    //     icon: Trophy,
    //     color: "blue",
    //     description: "Goal scored by blue team",
    //   },
    //   {
    //     command: "GOAL_YELLOW",
    //     label: "Yellow Goal",
    //     icon: Trophy,
    //     color: "yellow",
    //     description: "Goal scored by yellow team",
    //   },
    // ],
    "Ball Placement": [
      {
        command: {
          type: "BallPlacement",
          data: {
            team_color: TeamColor.Blue,
            position: ballPlacementPosition ?? [0, 0],
          },
        },
        label: "Blue Ball Placement",
        icon: Navigation,
        color: "blue",
        description: "Ball placement by blue team",
      },
      {
        command: {
          type: "BallPlacement",
          data: {
            team_color: TeamColor.Yellow,
            position: ballPlacementPosition ?? [0, 0],
          },
        },
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
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Game Controller</CardTitle>
          {lastCommand && (
            <Badge variant="outline" className="w-fit text-xs">
              Last: {lastCommand.type}
            </Badge>
          )}
        </CardHeader>
        <CardContent className="space-y-3 overflow-y-auto p-3">
          {Object.entries(commandGroups).map(
            ([groupName, commands], groupIndex) => (
              <div key={groupName} className="space-y-1.5">
                <h4 className="text-xs font-medium text-muted-foreground">
                  {groupName}
                </h4>
                <div className="grid grid-cols-2 gap-1.5">
                  {commands.map((cmd) => {
                    const Icon = cmd.icon;
                    return (
                      <Button
                        key={cmd.command.type}
                        variant={getButtonVariant(cmd)}
                        size="sm"
                        className={`justify-start h-8 px-2 text-xs ${getButtonClassName(
                          cmd
                        )}`}
                        onClick={() => sendGcCommand(cmd.command)}
                        title={cmd.description}
                      >
                        <Icon className="h-3 w-3 mr-1.5 flex-shrink-0" />
                        <span className="truncate">{cmd.label}</span>
                      </Button>
                    );
                  })}
                </div>
                {groupIndex < Object.entries(commandGroups).length - 1 && (
                  <Separator className="my-2" />
                )}
              </div>
            )
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default GameControllerPanel;
