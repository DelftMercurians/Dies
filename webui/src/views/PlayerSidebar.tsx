import {
  useBasestationInfo,
  useDebugData,
  useExecutorInfo,
  useKeyboardControl,
  useSendCommand,
  useWorldState,
  isPlayerManuallyControlled,
  usePrimaryTeam,
  keyboardControlAtom,
  keyboardModeAtom,
} from "@/api";
import { useAtom } from "jotai";
import * as math from "mathjs";
import { DebugValue, PlayerData, TeamColor } from "@/bindings";
import { Button } from "@/components/ui/button";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Switch } from "@/components/ui/switch";
import { SimpleTooltip } from "@/components/ui/tooltip";
import {
  cn,
  formatDebugString,
  formatNumber,
  prettyPrintSnakeCases,
  radiansToDegrees,
} from "@/lib/utils";
import { ChevronDown, ChevronRight, Pause, Play, X } from "lucide-react";
import { FC, useState } from "react";
import CodeEditor from "./CodeEditor";
import HardwareReadout from "./HardwareReadout";
import TargetVelCrosshair from "./TargetVelCrosshair";
import TimeSeriesChart from "./TimeSeriesChart";

/** Parse a vec2 debug string ("x y") into [x, y], or null. */
const parseVec2 = (s: unknown): [number, number] | null => {
  if (typeof s !== "string") return null;
  const m = /^\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$/.exec(s);
  return m ? [Number(m[1]), Number(m[2])] : null;
};

/**
 * Player Sidebar showing detailed info for a selected player.
 * Layout (top -> bottom): quick switcher, hardware readout, debug table,
 * collapsible plot (collapsed by default), collapsible behavior tree, and
 * manual control.
 */

interface PlayerSidebarProps {
  selectedPlayerId: number;
  onClose: () => void;
  onSelectPlayer: (id: number) => void;
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
  onSelectPlayer,
}) => {
  const world = useWorldState();
  const executorInfo = useExecutorInfo();
  const sendCommand = useSendCommand();
  const { data: bsInfo } = useBasestationInfo();
  const [activeGraph, setActiveGraph] = useState<Graphable>("velocity");
  const [graphPaused, setGraphPaused] = useState(true);
  const [speed, setSpeed] = useState(1000);
  const [angularSpeedDegPerSec, setAngularSpeedDegPerSec] = useState(100);
  const [keyboardControl, setKeyboardControl] = useAtom(keyboardControlAtom);
  const [customFunction, setCustomFunction] = useState<
    ((data: GraphData) => number) | null
  >(null);
  const [keyboardMode, setKeyboardMode] = useAtom(keyboardModeAtom);
  const [fanSpeed, setFanSpeed] = useState(0);
  const [kickSpeed, setKickSpeed] = useState(0);
  const [kick, setKick] = useState(false);
  const [primaryTeam] = usePrimaryTeam();

  const manualControl = isPlayerManuallyControlled(
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
  const prefix = `team_${teamColor}.p${selectedPlayerId}`;
  const playerDebugData = debugData
    ? (Object.entries(debugData)
        .filter(
          ([key, val]) => key.startsWith(prefix) && val.type !== "Shape"
        )
        .map(([key, val]) => [key.slice(prefix.length + 1), val]) as PlayerDebugValues)
    : [];
  const playerDebugMap = Object.fromEntries(
    playerDebugData.map(([k, v]) => [k, v.data])
  );

  const own_players =
    world.status === "connected"
      ? primaryTeam === TeamColor.Blue
        ? world.data.blue_team
        : world.data.yellow_team
      : [];
  const sortedPlayers = [...own_players].sort((a, b) => a.id - b.id);
  const selectedPlayer = own_players.find((p) => p.id === selectedPlayerId);

  const bsPlayers = bsInfo
    ? bsInfo.blue_team.length === 0 && bsInfo.yellow_team.length === 0
      ? bsInfo.unknown_team
      : primaryTeam === TeamColor.Blue
      ? bsInfo.blue_team
      : bsInfo.yellow_team
    : [];
  const feedback = bsPlayers.find((p) => p.id === selectedPlayerId);

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
    ? { player: selectedPlayer, playerDebug: playerDebugMap }
    : null;

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden">
      {/* Header: quick switcher + close */}
      <div className="flex flex-row gap-2 p-2 items-center border-b border-border-subtle shrink-0">
        <Select
          value={String(selectedPlayerId)}
          onValueChange={(val) => {
            if (val === "__overview") onClose();
            else onSelectPlayer(Number(val));
          }}
        >
          <SelectTrigger className="h-7 flex-1">
            <SelectValue placeholder="Select player" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__overview">← Team overview</SelectItem>
            {sortedPlayers.map((p) => (
              <SelectItem key={p.id} value={String(p.id)}>
                Player #{p.id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button variant="ghost" size="icon-sm" onClick={onClose}>
          <X className="h-3 w-3" />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto flex flex-col gap-3 p-3">
        {/* Hardware readout (above debug table) */}
        <HardwareReadout
          id={selectedPlayerId}
          team={primaryTeam}
          feedback={feedback}
          breakbeamBall={selectedPlayer?.breakbeam_ball_detected}
        />

        {/* Target velocity crosshair */}
        {(() => {
          const tv = parseVec2(playerDebugMap["target_vel"]);
          return tv ? (
            <div className="flex flex-col gap-1">
              <div className="text-[11px] uppercase tracking-wider text-text-dim">
                Target Velocity
              </div>
              <TargetVelCrosshair vx={tv[0]} vy={tv[1]} />
            </div>
          ) : null;
        })()}

        {/* Debug data table */}
        <div className="flex flex-col gap-1">
          <div className="text-[11px] uppercase tracking-wider text-text-dim">
            Debug Data
          </div>
          <DebugTable data={playerDebugData} />
        </div>

        {/* Plot — collapsible, collapsed by default */}
        <CollapsibleSection title="Plot">
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
                {graphPaused ? (
                  <Play className="h-3 w-3" />
                ) : (
                  <Pause className="h-3 w-3" />
                )}
              </Button>
            </div>

            {(activeGraph as string) === "custom" && graphData ? (
              <CodeEditor
                globals={graphData}
                onRun={(code) => {
                  setCustomFunction(() =>
                    createCustomFunction(code, Object.keys(graphData))
                  );
                }}
              />
            ) : null}

            {graphData && selectedPlayer ? (
              <TimeSeriesChart
                paused={graphPaused}
                objectId={`${selectedPlayerId}${activeGraph}`}
                newDataPoint={{
                  timestamp: selectedPlayer.timestamp,
                  ...graphData,
                }}
                getData={(data) =>
                  activeGraph === "custom"
                    ? customFunction?.(data) ?? 0
                    : graphableValues[activeGraph](data)
                }
                axisLabel={axisLabels[activeGraph]}
              />
            ) : null}
          </div>
        </CollapsibleSection>

        {/* Manual control */}
        <div className="flex flex-col gap-3">
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
                  <Label htmlFor="keyboard-mode">Global keyboard control</Label>
                </div>
              </SimpleTooltip>

              <div className="flex flex-row gap-2 items-center text-sm">
                <div>Speed</div>
                <NumberInput value={speed} onChange={setSpeed} />
                <div className="text-text-dim">mm/s</div>
              </div>

              <div className="flex flex-row gap-2 items-center text-sm">
                <div>Angular Speed</div>
                <NumberInput
                  value={angularSpeedDegPerSec}
                  onChange={setAngularSpeedDegPerSec}
                />
                <div className="text-text-dim">deg/s</div>
              </div>

              <div className="flex flex-row gap-2 items-center text-sm">
                <div>Fan Speed</div>
                <NumberInput value={fanSpeed} onChange={setFanSpeed} />
                <div className="text-text-dim">%</div>
              </div>

              <div className="flex flex-row gap-2 items-center text-sm">
                <div>Kick Speed</div>
                <NumberInput value={kickSpeed} onChange={setKickSpeed} />
                <div className="text-text-dim">mm/s</div>
              </div>

              <div className="flex flex-row gap-2 items-center text-sm">
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
      </div>
    </div>
  );
};

export default PlayerSidebar;

/** A collapsible section, collapsed by default. */
const CollapsibleSection: FC<{
  title: string;
  children: React.ReactNode;
}> = ({ title, children }) => {
  const [open, setOpen] = useState(false);
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="flex flex-col">
      <CollapsibleTrigger className="flex items-center gap-1 text-[11px] uppercase tracking-wider text-text-dim hover:text-text-std py-1">
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        {title}
      </CollapsibleTrigger>
      <CollapsibleContent className="pt-1">{children}</CollapsibleContent>
    </Collapsible>
  );
};

/** Flat two-column key|value table of player debug data (bt.* excluded). */
const DebugTable: FC<{ data: PlayerDebugValues }> = ({ data }) => {
  const rows = data
    .filter(
      ([key]) =>
        key !== "bt" && !key.startsWith("bt.") && key !== "target_vel"
    )
    .sort(([a], [b]) => a.localeCompare(b));

  if (rows.length === 0) {
    return (
      <div className="bg-bg-elevated border border-border-subtle p-2 text-xs text-text-dim">
        No debug data
      </div>
    );
  }

  return (
    <div className="bg-bg-elevated border border-border-subtle overflow-x-auto">
      <table className="w-full text-xs">
        <tbody>
          {rows.map(([key, val]) => (
            <tr key={key} className="border-b border-border-subtle last:border-0">
              <td className="px-2 py-1 align-top text-text-dim whitespace-nowrap">
                {key}
              </td>
              <td className="px-2 py-1 font-mono text-text-std break-all">
                <DebugCell value={val} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const DebugCell: FC<{ value: Exclude<DebugValue, { type: "Shape" }> }> = ({
  value,
}) => {
  if (value.type === "Number") {
    return <span title={String(value.data)}>{formatNumber(value.data)}</span>;
  }
  return <span title={value.data}>{formatDebugString(value.data)}</span>;
};

const KeyboardKey = ({
  letter,
  className,
}: {
  letter: string;
  className?: string;
}) => (
  <div
    className={cn(
      "w-10 h-10 bg-bg-overlay border border-border-std flex items-center justify-center text-text-std font-semibold uppercase hover:bg-bg-elevated hover:text-text-bright transition-all duration-100 select-none",
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
