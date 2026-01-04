import {
  useDebugData,
  useExecutorInfo,
  useKeyboardControl,
  useSendCommand,
  useWorldState,
  isPlayerManuallyControlled,
  usePrimaryTeam,
} from "@/api";
import * as math from "mathjs";
import { DebugValue, PlayerData, TeamColor } from "@/bindings";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn, prettyPrintSnakeCases, radiansToDegrees } from "@/lib/utils";
import { Pause, Play, X } from "lucide-react";
import { FC, useState } from "react";
import CodeEditor from "./CodeEditor";
import HierarchicalList from "./HierarchicalList";
import TimeSeriesChart from "./TimeSeriesChart";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import BehaviorTreeView from "./BehaviorTreeView";

/**
 * Player Sidebar showing detailed info for a selected player.
 * Uses mission control aesthetic.
 */

interface PlayerSidebarProps {
  selectedPlayerId: number | null;
  onClose: () => void;
}

interface GraphData {
  player: PlayerData;
  playerDebug: Record<string, number | string>;
}

type PlayerDebugValues = [string, Exclude<DebugValue, { type: "Shape" }>][];

type Graphable = keyof typeof graphableValues | "custom";

type AxisLabels = {
  [key in Graphable]: string;
};

const graphableValues = {
  velocity: (data: GraphData) => magnitude(data.player.velocity),
  angular_speed: (data: GraphData) =>
    radiansToDegrees(data.player.angular_speed),
  filter_error: ({ player }: GraphData) =>
    magnitude([
      player.position[0] - player.raw_position[0],
      player.position[1] - player.raw_position[1],
    ]),
} as const satisfies Record<string, (data: GraphData) => number>;
const graphableLabels = Object.keys(graphableValues);

const functionGlobals = { math };

const axisLabels: AxisLabels = {
  velocity: "mm/s",
  angular_speed: "deg/s",
  filter_error: "mm",
  custom: "custom",
};

const PlayerSidebar: FC<PlayerSidebarProps> = ({
  selectedPlayerId,
  onClose,
}) => {
  const world = useWorldState();
  const executorInfo = useExecutorInfo();
  const sendCommand = useSendCommand();
  const [activeGraph, setActiveGraph] = useState<Graphable>("velocity");
  const [graphPaused, setGraphPaused] = useState(true);
  const [speed, setSpeed] = useState(1000);
  const [angularSpeedDegPerSec, setAngularSpeedDegPerSec] = useState(100);
  const [keyboardControl, setKeyboardControl] = useState(false);
  const [customFunction, setCustomFunction] = useState<
    ((data: GraphData) => number) | null
  >(null);
  const [keyboardMode, setKeyboardMode] = useState<"local" | "global">(
    "global"
  );
  const [fanSpeed, setFanSpeed] = useState(0);
  const [kickSpeed, setKickSpeed] = useState(0);
  const [kick, setKick] = useState(false);
  const [primaryTeam] = usePrimaryTeam();

  const manualControl =
    typeof selectedPlayerId === "number" &&
    isPlayerManuallyControlled(
      selectedPlayerId,
      executorInfo?.manual_controlled_players ?? []
    );
  useKeyboardControl({
    playerId: manualControl && keyboardControl ? selectedPlayerId : null,
    speed,
    angularSpeedDegPerSec,
    mode: keyboardMode,
    fanSpeed,
    kickSpeed,
    kick,
  });

  const debugData = useDebugData();
  const teamColor = primaryTeam === TeamColor.Blue ? "Blue" : "Yellow";
  const playerDebugData = debugData
    ? (Object.entries(debugData)
        .filter(
          ([key, val]) =>
            key.startsWith(`team_${teamColor}.p${selectedPlayerId}`) &&
            val.type !== "Shape"
        )
        .map(([key, val]) => [
          key.slice(`team_${teamColor}.p${selectedPlayerId}`.length + 1),
          val,
        ]) as PlayerDebugValues)
    : [];
  const playerDebugMap = Object.fromEntries(
    playerDebugData.map(([k, v]) => [k, v.data])
  );

  if (typeof selectedPlayerId !== "number")
    return (
      <div className="flex-1 flex flex-col gap-4 p-3 h-full w-full">
        <h1 className="text-[12px] font-semibold uppercase tracking-wider text-text-dim text-center">
          Select a player by clicking
        </h1>
        <div className="bg-bg-elevated border border-border-subtle p-2 h-full overflow-auto">
          <HierarchicalList data={debugData ? Object.entries(debugData) : []} />
        </div>
      </div>
    );

  const selectedPlayer =
    world.status === "connected"
      ? primaryTeam === TeamColor.Blue
        ? world.data.blue_team.find((p) => p.id === selectedPlayerId)
        : world.data.yellow_team.find((p) => p.id === selectedPlayerId)
      : null;

  const handleToggleManual = (val: boolean) => {
    sendCommand({
      type: "SetManualOverride",
      data: {
        team_color: primaryTeam,
        player_id: selectedPlayerId,
        manual_override: val,
      },
    });
  };

  const graphData: GraphData | null = selectedPlayer
    ? {
        player: selectedPlayer,
        playerDebug: playerDebugMap,
      }
    : null;

  return (
    <div className="flex-1 flex flex-col h-full">
      <div className="flex flex-row p-3 pb-0 items-center">
        <h1 className="text-[14px] font-semibold uppercase tracking-wider text-text-bright">
          Player #{selectedPlayerId}
        </h1>
        <Button className="ml-auto" variant="ghost" size="icon-sm" onClick={onClose}>
          <X className="h-3 w-3" />
        </Button>
      </div>

      <Tabs
        defaultValue="info"
        className="flex-1 flex flex-col overflow-hidden"
        size="sm"
      >
        <TabsList className="mx-3 mt-2">
          <TabsTrigger value="info">Info</TabsTrigger>
          <TabsTrigger value="bt">Behavior Tree</TabsTrigger>
          <TabsTrigger value="debug">Debug</TabsTrigger>
        </TabsList>
        <TabsContent value="info" className="flex-1 overflow-y-auto">
          <div className="flex flex-col gap-4 p-3">
            <div className="flex flex-col gap-2">
              <div className="flex flex-row gap-2 items-center">
                <Select
                  value={activeGraph}
                  onValueChange={(val) => setActiveGraph(val as Graphable)}
                >
                  <SelectTrigger className="h-6 w-64 flex-1">
                    <SelectValue placeholder="Select graph" />
                  </SelectTrigger>

                  <SelectContent>
                    {graphableLabels.map((v) => (
                      <SelectItem key={v} value={v}>
                        {prettyPrintSnakeCases(v)}
                      </SelectItem>
                    ))}

                    <SelectItem value={"custom"}>Custom function</SelectItem>
                  </SelectContent>
                </Select>

                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => setGraphPaused((p) => !p)}
                >
                  {graphPaused ? <Play className="h-3 w-3" /> : <Pause className="h-3 w-3" />}
                </Button>
              </div>

              {(activeGraph as string) === "custom" && graphData ? (
                <div>
                  <CodeEditor
                    globals={graphData}
                    onRun={(code) => {
                      setCustomFunction(() =>
                        createCustomFunction(code, Object.keys(graphData))
                      );
                    }}
                  />
                </div>
              ) : null}

              {graphData && selectedPlayer ? (
                <TimeSeriesChart
                  paused={graphPaused}
                  objectId={`${selectedPlayerId}${activeGraph}`}
                  newDataPoint={{
                    timestamp: selectedPlayer.timestamp,
                    ...graphData,
                  }}
                  getData={(data) => {
                    let value =
                      activeGraph === "custom"
                        ? customFunction?.(data) ?? 0
                        : graphableValues[activeGraph](data);
                    return value;
                  }}
                  axisLabel={axisLabels[activeGraph]}
                />
              ) : null}
            </div>

            <div>
              <h2 className="text-[11px] font-semibold uppercase tracking-wider text-text-dim mb-2">
                Player Debug Data
              </h2>
              <div className="bg-bg-elevated border border-border-subtle p-2 max-h-[40vh] overflow-auto">
                <HierarchicalList data={playerDebugData} expandAll />
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

                <SimpleTooltip title="Whether to control the robot in local or global frame">
                  <div className="flex flex-row gap-2 items-center">
                    <Switch
                      id="keyboard-mode"
                      checked={keyboardMode === "global"}
                      disabled={keyboardControl !== true}
                      onCheckedChange={(checked) =>
                        setKeyboardMode(checked ? "global" : "local")
                      }
                    />
                    <Label htmlFor="keyboard-mode">
                      Global keyboard control
                    </Label>
                  </div>
                </SimpleTooltip>

                <div className="flex flex-row gap-2 items-center text-[10px]">
                  <div>Speed</div>
                  <Input
                    type="number"
                    min="0"
                    max="10000"
                    value={speed}
                    onChange={(e) => setSpeed(parseInt(e.target.value))}
                  />
                  <div className="text-text-dim">mm/s</div>
                </div>

                <div className="flex flex-row gap-2 items-center text-[10px]">
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
                  <div className="text-text-dim">deg/s</div>
                </div>

                <div className="flex flex-row gap-2 items-center text-[10px]">
                  <div>Fan Speed</div>
                  <Slider
                    min={0}
                    max={100}
                    value={[fanSpeed]}
                    onValueChange={([newValue]) => setFanSpeed(newValue)}
                  />
                  <div className="text-text-dim">%</div>
                </div>

                <div className="flex flex-row gap-2 items-center text-[10px]">
                  <div>Kick Speed</div>
                  <Input
                    type="number"
                    min="0"
                    max="10000"
                    value={kickSpeed}
                    onChange={(e) => setKickSpeed(parseInt(e.target.value))}
                  />
                  <div className="text-text-dim">mm/s</div>
                </div>

                <div className="flex flex-row gap-2 items-center text-[10px]">
                  <div>Kick</div>
                  <Button
                    size="sm"
                    onClick={() => {
                      setKick(true);
                      setTimeout(() => setKick(false), 1000 / 10);
                    }}
                  >
                    Kick
                  </Button>
                </div>

                <div className="flex justify-center">
                  <div className="inline-block bg-bg-elevated border border-border-muted p-4">
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
        </TabsContent>
        <TabsContent value="bt" className="flex-1 overflow-y-auto">
          <BehaviorTreeView
            selectedPlayerId={selectedPlayerId}
            className="h-full p-3"
          />
        </TabsContent>
        <TabsContent value="debug" className="flex-1 overflow-y-auto p-2">
          <div className="bg-bg-elevated border border-border-subtle p-2 h-full overflow-auto">
            <HierarchicalList
              data={debugData ? Object.entries(debugData) : []}
            />
          </div>
        </TabsContent>
      </Tabs>
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
      "w-10 h-10 bg-bg-overlay border border-border-std flex items-center justify-center text-text-std font-semibold text-[11px] uppercase hover:bg-bg-elevated hover:text-text-bright transition-all duration-100 select-none",
      className
    )}
  >
    {letter}
  </div>
);

const magnitude = ([x, y]: [number, number]) => Math.sqrt(x * x + y * y);

const createCustomFunction = (code: string, args: string[]) => {
  try {
    const globalKeys = Object.keys(functionGlobals);
    const globalValues = Object.values(functionGlobals);
    const fun = new Function(...args, ...globalKeys, `return (${code});`);
    let errored = false;
    return (d: Record<string, any>) => {
      if (errored) return 0;
      try {
        return fun(...args.map((a) => d[a]), ...globalValues);
      } catch (e) {
        alert(`Error in custom function: ${e}`);
        errored = true;
        return 0;
      }
    };
  } catch (e) {
    alert(`Error in custom function: ${e}`);
    return null;
  }
};
