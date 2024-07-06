import {
  useDebugData,
  useExecutorInfo,
  useKeyboardControl,
  useSendCommand,
  useWorldState,
} from "@/api";
import { DebugValue, PlayerData } from "@/bindings";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { FC, useMemo, useState } from "react";
import TimeSeriesChart from "./TimeSeriesChart";
import { Button } from "@/components/ui/button";
import { ChevronDown, ChevronRight, Pause, Play } from "lucide-react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Input } from "@/components/ui/input";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

interface PlayerSidebarProps {
  selectedPlayerId: number | null;
}

const graphableValues = [
  "velocity",
  "angular_speed",
] as const satisfies (keyof PlayerData)[];

type Graphable = (typeof graphableValues)[number];

type PlayerDebugValues = [string, Exclude<DebugValue, { type: "Shape" }>][];

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

  const debugData = useDebugData();
  const playerDebugData = debugData
    ? (Object.entries(debugData)
        .filter(
          ([key, val]) =>
            key.startsWith(`p${selectedPlayerId}`) && val.type !== "Shape"
        )
        .map(([key, val]) => [
          key.slice(`p${selectedPlayerId}`.length + 1),
          val,
        ]) as PlayerDebugValues)
    : [];

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

      <div>
        <h2 className="text-lg font-semibold mb-2">Debug Data</h2>
        <div className="bg-slate-800 p-2 rounded-xl max-h-[50vh] overflow-auto">
          <HierarchicalList data={playerDebugData} />
        </div>
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

const HierarchicalList: FC<{ data: PlayerDebugValues }> = ({ data }) => {
  const [openKeys, setOpenKeys] = useState<string[]>([]);

  const groupedData = useMemo(() => {
    const grouped: Record<string, any> = {};
    data.forEach(([key, value]) => {
      const parts = key.split(".");
      let current = grouped;
      parts.forEach((part, index) => {
        if (!current[part]) {
          current[part] = index === parts.length - 1 ? value : {};
        }
        current = current[part];
      });
    });
    return grouped;
  }, [data]);

  const formatValue = (value: any) => {
    if (typeof value === "number") {
      return value.toFixed(2);
    }
    return `${value}`;
  };

  const renderGroup = (group: Record<string, any>, key = "", depth = 0) => {
    const isLeaf = typeof group.data !== "undefined";
    if (isLeaf) {
      return (
        <div key={key} className="flex flex-row items-stretch py-1 w-full">
          <div className="font-semibold mr-2 min-w-max">
            {prettyPrintSnakeCases(key)}:
          </div>
          <div className="w-full flex items-center overflow-x-auto">
            <span className="min-w-max font-mono whitespace-nowrap">
              {formatValue(group.data)}
            </span>
          </div>
        </div>
      );
    }

    const isOpen = openKeys.includes(key);
    const handleOpenChange = (isOpen: boolean) => {
      if (isOpen) {
        setOpenKeys((keys) => [...keys, key]);
      } else {
        setOpenKeys((keys) => keys.filter((k) => k !== key));
      }
    };

    return (
      <div key={key} className="flex flex-col">
        <Collapsible
          open={isOpen}
          onOpenChange={handleOpenChange}
          className="w-full"
        >
          <CollapsibleTrigger className="flex items-center py-1 w-full">
            {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            <span className="font-semibold ml-1">
              {prettyPrintSnakeCases(key)}
            </span>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="ml-4 relative">
              <div className="w-4 h-full border-l border-gray-300 absolute -left-4"></div>
              {sortKeys(group).map(([subKey, subGroup]) =>
                renderGroup(subGroup, subKey, depth + 1)
              )}
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    );
  };

  return (
    <div className="p-2">
      {sortKeys(groupedData).map(([key, group]) => renderGroup(group, key))}
    </div>
  );
};

const magnitude = ([x, y]: [number, number]) => Math.sqrt(x * x + y * y);

const sortKeys = (group: Record<string, any>): [string, any][] => {
  return Object.entries(group).sort(([keyA, valueA], [keyB, valueB]) => {
    const isLeafA = typeof valueA.data !== "undefined";
    const isLeafB = typeof valueB.data !== "undefined";
    if (isLeafA !== isLeafB) {
      return isLeafA ? 1 : -1; // Non-leaf nodes first
    }
    return keyA.localeCompare(keyB); // Alphabetical order
  });
};

const prettyPrintSnakeCases = (s: string): string =>
  s
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

const radiansToDegrees = (radians: number): number => {
  return (radians * 180) / Math.PI;
};
