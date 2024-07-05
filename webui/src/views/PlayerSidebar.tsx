import {
  useExecutorInfo,
  useKeyboardControl,
  useSendCommand,
  useWorldState,
} from "@/api";
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
import { Button } from "@/components/ui/button";
import { Pause, Play } from "lucide-react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";

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
  const [activeGraph, setActiveGraph] = useState<Graphable>("velocity");
  const [graphPaused, setGraphPaused] = useState(true);
  const [speed, setSpeed] = useState(1000);
  const [angularSpeedDegPerSec, setAngularSpeedDegPerSec] = useState(30);
  const [keyboardControl, setKeyboardControl] = useState(false);

  const manualControl =
    typeof selectedPlayerId === "number" &&
    executorInfo?.manual_controlled_players.includes(selectedPlayerId);
  useKeyboardControl({
    playerId: manualControl && keyboardControl ? selectedPlayerId : null,
    speed,
    angularSpeedDegPerSec,
  });

  if (world.status !== "connected" || typeof selectedPlayerId !== "number")
    return null;

  const selectedPlayer = world.data.own_players.find(
    (p) => p.id === selectedPlayerId
  );
  if (!selectedPlayer)
    return <div>Player with id {selectedPlayerId} not found</div>;

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
    <div className="flex-1 flex flex-col gap-6 p-4">
      <h1 className="text-2xl font-bold mb-2">Player #{selectedPlayerId}</h1>

      <div>
        <div className="flex flex-row gap-2 items-center mb-2">
          <Select
            value={activeGraph}
            onValueChange={(val) => setActiveGraph(val as Graphable)}
          >
            <SelectTrigger className="h-full w-64 flex-1">
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

          <Button variant="ghost" onClick={() => setGraphPaused((p) => !p)}>
            {graphPaused ? <Play /> : <Pause />}
          </Button>
        </div>
        <TimeSeriesChart
          paused={graphPaused}
          objectId={selectedPlayerId}
          newDataPoint={selectedPlayer}
          selectedKey={activeGraph}
          transform={(val) =>
            Array.isArray(val) ? magnitude(val) : radiansToDegrees(val)
          }
          axisLabels={{ velocity: "mm/s", angular_speed: "deg/s" }}
        />
      </div>

      <SimpleTooltip title="Set this player to manual control -- it will stop following the strategy">
        <div className="flex flex-row gap-2 items-center">
          <Switch
            id="manual-control"
            checked={manualControl}
            disabled={typeof manualControl !== "boolean"}
            onCheckedChange={handleToggleManual}
          />
          <Label htmlFor="manual-control">Manual Control</Label>
        </div>
      </SimpleTooltip>

      {manualControl ? (
        <>
          <SimpleTooltip title="Control the robot using the keyboard">
            <div className="flex flex-row gap-2 items-center">
              <Switch
                id="keyboard-control"
                checked={keyboardControl}
                disabled={typeof keyboardControl !== "boolean"}
                onCheckedChange={setKeyboardControl}
              />
              <Label htmlFor="keyboard-control">Keyboard Control</Label>
            </div>
          </SimpleTooltip>

          <div className="flex flex-row gap-2 items-center">
            <div>Speed</div>
            <Input
              type="number"
              min="0"
              max="10000"
              value={speed}
              onChange={(e) => setSpeed(parseInt(e.target.value))}
            />
            <div>mm/s</div>
          </div>

          <div className="flex flex-row gap-2 items-center">
            <div>Angular Speed</div>
            <Input
              type="number"
              min="0"
              max="360"
              value={angularSpeedDegPerSec}
              onChange={(e) =>
                setAngularSpeedDegPerSec(parseInt(e.target.value))
              }
            />
            <div>deg/s</div>
          </div>

          <div className="flex justify-center">
            <div className="inline-block bg-slate-600 p-6 rounded-xl">
              <div className="flex justify-center mb-2 space-x-2">
                <SimpleTooltip title="Turn left">
                  <KeyboardKey letter="q" />
                </SimpleTooltip>
                <SimpleTooltip title="Go forward (global coordinates)">
                  <KeyboardKey letter="w" />
                </SimpleTooltip>
                <SimpleTooltip title="Turn right">
                  <KeyboardKey letter="e" />
                </SimpleTooltip>
              </div>
              <div className="flex justify-center mb-2 space-x-2">
                <SimpleTooltip title="Go left (global coordinates)">
                  <KeyboardKey letter="a" />
                </SimpleTooltip>
                <SimpleTooltip title="Go backward (global coordinates)">
                  <KeyboardKey letter="s" />
                </SimpleTooltip>
                <SimpleTooltip title="Go right (global coordinates)">
                  <KeyboardKey letter="d" />
                </SimpleTooltip>
              </div>
              <div className="flex justify-center space-x-2 w-full">
                <SimpleTooltip title="Dribble" className="w-full">
                  <KeyboardKey letter="Space" className="w-full" />
                </SimpleTooltip>
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
};

export default PlayerSidebar;

const KeyboardKey = ({
  letter,
  className,
}: {
  letter: string;
  className?: string;
}) => (
  <div
    className={cn(
      "w-12 h-12 bg-gray-200 rounded-md shadow-md flex items-center justify-center text-gray-700 font-bold text-lg uppercase border-b-4 border-gray-400 hover:border-b-0  transition-all duration-100 select-none",
      className
    )}
  >
    {letter}
  </div>
);

const magnitude = ([x, y]: [number, number]) => Math.sqrt(x * x + y * y);

const prettyPrintSnakeCases = (s: string): string =>
  s
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

const radiansToDegrees = (radians: number): number => {
  return (radians * 180) / Math.PI;
};
