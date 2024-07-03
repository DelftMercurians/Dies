import { useExecutorInfo, useSendCommand, useWorldState } from "@/api";
import { PlayerData } from "@/bindings";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { FC, useState } from "react";
import TimeSeriesChart from "./TimeSeriesChart";

interface PlayerSidebarProps {
  selectedPlayerId: number | null;
}

const graphableValues = [
  "velocity",
  "angular_speed",
] as const satisfies (keyof PlayerData)[];

type Graphable = (typeof graphableValues)[number];

const PlayerSidebar: FC<PlayerSidebarProps> = ({ selectedPlayerId }) => {
  const world = useWorldState();
  const executorInfo = useExecutorInfo();
  const sendCommand = useSendCommand();
  const [activeGraph, setActiveGraph] = useState<Graphable>("angular_speed");

  if (world.status !== "connected" || selectedPlayerId === null)
    return <div className="bg-slate-950"></div>;

  const selectedPlayer = world.data.own_players.find(
    (p) => p.id === selectedPlayerId
  );
  if (!selectedPlayer)
    throw new Error(`Player with id ${selectedPlayer} not found!`);

  const manualControl =
    executorInfo?.manual_controlled_players.includes(selectedPlayerId);
  const handleToggleManual = (val: boolean) => {
    sendCommand({
      type: "SetManualOverride",
      data: {
        player_id: selectedPlayerId,
        manual_override: val,
      },
    });
  };

  return (
    <div className="flex flex-col gap-6 bg-slate-950 p-6">
      <h1 className="text-lg font-bold mb-2">Player #{selectedPlayerId}</h1>

      <div>
        <div className="flex flex-row gap-2 items-center">
          <Select
            value={activeGraph}
            onValueChange={(val) => setActiveGraph(val as Graphable)}
          >
            <SelectTrigger className="w-64 mb-2 flex-1">
              <SelectValue placeholder="Select graph" />
            </SelectTrigger>

            <SelectContent>
              {graphableValues.map((v, index) => (
                <SelectItem key={index} value={v}>
                  {prettyPrintSnakeCases(v)}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <TimeSeriesChart
          objectId={selectedPlayerId}
          newDataPoint={selectedPlayer}
          selectedKey={activeGraph}
          transform={(val) =>
            Array.isArray(val) ? magnitude(val) : radiansToDegrees(val)
          }
          axisLabels={{ velocity: "mm/s", angular_speed: "deg/s" }}
        />
      </div>

      <div className="flex flex-row gap-2 items-center">
        <Switch
          id="manual-control"
          checked={manualControl}
          disabled={typeof manualControl !== "boolean"}
          onCheckedChange={handleToggleManual}
        />
        <Label htmlFor="manual-control">Manual Control</Label>
      </div>
    </div>
  );
};

export default PlayerSidebar;

const magnitude = ([x, y]: [number, number]) => Math.sqrt(x * x + y * y);

const prettyPrintSnakeCases = (s: string): string =>
  s
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

const radiansToDegrees = (radians: number): number => {
  return (radians * 180) / Math.PI;
};
