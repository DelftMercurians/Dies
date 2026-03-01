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
  Target,
  Navigation,
} from "lucide-react";
import { GcSimCommand, TeamColor, Vector2 } from "@/bindings";
import { atom, useAtomValue } from "jotai";

export const BallPlacementPostionAtom = atom<Vector2 | null>(null);

/**
 * Game Controller Panel for sending game controller commands.
 * Uses mission control aesthetic with team-colored buttons.
 */
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
      variant?: "default" | "outline" | "destructive" | "ghost";
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
        variant: "outline" as const,
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

  const getButtonClassName = (cmd: { color?: "blue" | "yellow" }) => {
    if (cmd.color === "blue")
      return "bg-team-blue/30 hover:bg-team-blue/50 text-team-blue border-team-blue/50";
    if (cmd.color === "yellow")
      return "bg-team-yellow/30 hover:bg-team-yellow/50 text-team-yellow border-team-yellow/50";
    return "";
  };

  return (
    <div className="h-full flex flex-col">
      <Card className="flex-1">
        <CardHeader className="pb-2">
          <CardTitle>Game Controller</CardTitle>
          {lastCommand && (
            <Badge variant="outline" className="w-fit">
              Last: {lastCommand.type}
            </Badge>
          )}
        </CardHeader>
        <CardContent className="space-y-3 overflow-y-auto p-3">
          {Object.entries(commandGroups).map(
            ([groupName, commands], groupIndex) => (
              <div key={groupName + groupIndex} className="space-y-1.5">
                <h4 className="text-sm font-semibold uppercase tracking-wider text-text-dim">
                  {groupName}
                </h4>
                <div className="grid grid-cols-2 gap-1.5">
                  {commands.map((cmd, index) => {
                    const Icon = cmd.icon;
                    return (
                      <Button
                        key={cmd.command.type + index}
                        variant={cmd.color ? "outline" : cmd.variant}
                        size="sm"
                        className={`justify-start h-6 px-2 text-sm ${getButtonClassName(cmd)}`}
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
