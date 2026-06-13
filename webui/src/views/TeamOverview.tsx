import {
  useBasestationInfo,
  useExecutorInfo,
  useWorldState,
  isPlayerManuallyControlled,
  usePrimaryTeam,
  useDebugData,
} from "@/api";
import { PlayerData, PlayerFeedbackMsg, TeamColor, DebugMap } from "@/bindings";
import { Badge } from "@/components/ui/badge";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn, magnitude2 } from "@/lib/utils";
import {
  batterySeverity,
  capVoltageSeverity,
  motorTempSeverity,
  playerHealth,
  Severity,
  severityDotClass,
  severityTextClass,
  sysStatusSeverity,
  worstSeverity,
} from "@/lib/hardware";
import { pinnedDebugKeysAtom, formatPlayerDebugValue } from "@/lib/pinnedDebug";
import { useAtomValue } from "jotai";
import { FC, useEffect, useRef } from "react";
import PatternIcon from "./PatternIcon";
import Sparkline from "./Sparkline";

/**
 * Superdense, glanceable team overview shown in the Inspector when no player is
 * selected. One compact row per own-team player: pattern icon, worst-of health
 * dot (quiet unless there's an issue), speed + acceleration sparklines, and the
 * strategy role. Clicking a row selects that player.
 */

interface TeamOverviewProps {
  className?: string;
  onSelectPlayer: (id: number) => void;
  selectedPlayerId: number | null;
}

const getRole = (
  debugData: DebugMap | null,
  playerId: number,
  teamColor: TeamColor
): string | null => {
  if (!debugData) return null;
  const teamColorStr = teamColor === TeamColor.Blue ? "Blue" : "Yellow";
  const v = debugData[`team_${teamColorStr}.p${playerId}.role`];
  return v?.type === "String" ? (v.data as string) : null;
};

// ---- per-player rolling history for sparklines ----

interface HistEntry {
  speed: number[];
  accel: number[];
  lastSpeed: number;
  lastT: number;
}

const MAX_SAMPLES = 50;

/**
 * Maintains a rolling buffer of speed & acceleration magnitude per player,
 * fed from the world stream. Mutates a ref (no re-render side effects); the
 * component re-renders on the next world tick and reads the updated buffers,
 * which is plenty smooth for sparklines.
 */
function usePlayerHistories(
  players: PlayerData[]
): Record<number, { speed: number[]; accel: number[] }> {
  const ref = useRef<Record<number, HistEntry>>({});

  useEffect(() => {
    const store = ref.current;
    for (const p of players) {
      const speed = magnitude2(p.velocity as [number, number]);
      let e = store[p.id];
      if (!e) {
        e = store[p.id] = {
          speed: [],
          accel: [],
          lastSpeed: speed,
          lastT: p.timestamp,
        };
      }
      // Skip duplicate frames (same timestamp re-render).
      if (p.timestamp === e.lastT && e.speed.length > 0) continue;
      const dt = p.timestamp - e.lastT;
      const accel = dt > 0 ? Math.abs(speed - e.lastSpeed) / dt : 0;
      e.speed = [...e.speed, speed].slice(-MAX_SAMPLES);
      e.accel = [...e.accel, accel].slice(-MAX_SAMPLES);
      e.lastSpeed = speed;
      e.lastT = p.timestamp;
    }
  });

  return ref.current;
}

const TeamOverview: FC<TeamOverviewProps> = ({
  className,
  onSelectPlayer,
  selectedPlayerId,
}) => {
  const worldState = useWorldState();
  const { data: bsInfo } = useBasestationInfo();
  const debugData = useDebugData();
  const executorInfo = useExecutorInfo();
  const [primaryTeam] = usePrimaryTeam();
  const pinnedKeys = useAtomValue(pinnedDebugKeysAtom);

  const connected = worldState.status === "connected";
  const blue_team = connected ? worldState.data.blue_team : [];
  const yellow_team = connected ? worldState.data.yellow_team : [];
  const own_players = primaryTeam === TeamColor.Blue ? blue_team : yellow_team;
  const sorted_players = [...own_players].sort((a, b) => a.id - b.id);

  const histories = usePlayerHistories(sorted_players);

  const bsPlayers = bsInfo
    ? bsInfo.blue_team.length === 0 && bsInfo.yellow_team.length === 0
      ? bsInfo.unknown_team
      : primaryTeam === TeamColor.Blue
      ? bsInfo.blue_team
      : bsInfo.yellow_team
    : [];

  if (!connected) {
    return (
      <div className={cn("p-2 bg-bg-surface text-text-std", className)}>
        <div className="text-center text-text-muted text-sm">
          Waiting for world state...
        </div>
      </div>
    );
  }

  return (
    <div className={cn("relative", className)}>
      <div className="absolute inset-0 overflow-y-auto bg-bg-surface">
        <div className="px-2 py-1.5 text-[11px] uppercase tracking-wider text-text-dim border-b border-border-subtle">
          {primaryTeam} team — {sorted_players.length} players
        </div>
        {sorted_players.length > 0 ? (
          <div className="divide-y divide-border-subtle">
            {sorted_players.map((player) => (
              <OverviewRow
                key={player.id}
                player={player}
                feedback={bsPlayers.find((p) => p.id === player.id)}
                role={getRole(debugData, player.id, primaryTeam)}
                team={primaryTeam}
                isManual={isPlayerManuallyControlled(
                  player.id,
                  executorInfo?.manual_controlled_players ?? []
                )}
                isSelected={player.id === selectedPlayerId}
                history={histories[player.id]}
                debugData={debugData}
                pinnedKeys={pinnedKeys}
                onClick={() => onSelectPlayer(player.id)}
              />
            ))}
          </div>
        ) : (
          <div className="text-center text-text-muted text-sm p-4">
            No players in world state.
          </div>
        )}
      </div>
    </div>
  );
};

export default TeamOverview;

const OverviewRow: FC<{
  player: PlayerData;
  feedback?: PlayerFeedbackMsg;
  role: string | null;
  team: TeamColor;
  isManual: boolean;
  isSelected: boolean;
  history?: { speed: number[]; accel: number[] };
  debugData: DebugMap | null;
  pinnedKeys: string[];
  onClick: () => void;
}> = ({
  player,
  feedback,
  role,
  team,
  isManual,
  isSelected,
  history,
  debugData,
  pinnedKeys,
  onClick,
}) => {
  const health = playerHealth(feedback);
  const speed = magnitude2(player.velocity as [number, number]);

  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex flex-col gap-1 px-2 py-1.5 text-left hover:bg-bg-overlay transition-colors",
        isSelected && "bg-bg-overlay ring-1 ring-inset ring-accent-cyan"
      )}
    >
      {/* top line: icon, id, role, manual, health */}
      <div className="flex items-center gap-2 w-full">
        <PatternIcon id={player.id} team={team} size={20} />
        <span className="font-mono text-sm text-text-bright w-5 shrink-0">
          {player.id}
        </span>
        <span className="text-xs text-text-std truncate flex-1 min-w-0">
          {role ?? "—"}
        </span>
        {isManual && (
          <Badge variant="destructive" className="px-1 py-0 text-[10px]">
            M
          </Badge>
        )}
        <SimpleTooltip
          title={
            health.online
              ? health.issues.length > 0
                ? `Issues: ${health.issues.join(", ")}`
                : "OK"
              : "No basestation feedback"
          }
        >
          <span
            className={cn(
              "w-2 h-2 rounded-full shrink-0",
              severityDotClass(health.severity)
            )}
          />
        </SimpleTooltip>
      </div>

      {/* hardware strip */}
      <HardwareStrip feedback={feedback} />

      {/* pinned debug values */}
      {pinnedKeys.length > 0 ? (
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] font-mono">
          {pinnedKeys.map((key) => {
            const val = formatPlayerDebugValue(debugData, player.id, team, key);
            return (
              <span key={key} className="flex items-center gap-1">
                <span className="text-text-muted">{key}</span>
                <span className="text-text-std">{val ?? "—"}</span>
              </span>
            );
          })}
        </div>
      ) : null}

      {/* sparklines — fixed y-limits: speed 0-3000 mm/s, accel 0-3000 mm/s² */}
      <div className="flex items-center gap-2">
        <SimpleTooltip title={`Speed ${Math.round(speed)} mm/s`}>
          <span className="flex items-center text-text-dim">
            <Sparkline
              data={history?.speed ?? []}
              width={56}
              height={14}
              min={0}
              max={3000}
            />
          </span>
        </SimpleTooltip>
        <SimpleTooltip title="Acceleration magnitude (mm/s²)">
          <span className="flex items-center text-text-muted">
            <Sparkline
              data={history?.accel ?? []}
              width={56}
              height={14}
              min={0}
              max={3000}
            />
          </span>
        </SimpleTooltip>
        <span className="font-mono text-[10px] text-text-dim w-12 text-right">
          {Math.round(speed)}
        </span>
      </div>
    </button>
  );
};

/**
 * Compact one-line hardware status: 5 motor dots (worst of status+temp),
 * battery & capacitor voltages, IMU and kicker status dots. Quiet when
 * healthy; colored only on issues. Shows a muted placeholder without feedback.
 */
const HardwareStrip: FC<{ feedback?: PlayerFeedbackMsg }> = ({ feedback }) => {
  if (!feedback) {
    return (
      <div className="text-[10px] text-text-muted italic">no telemetry</div>
    );
  }

  const battery =
    feedback.pack_voltages && feedback.pack_voltages.length > 0
      ? Math.min(...feedback.pack_voltages)
      : undefined;
  const batterySev = feedback.pack_voltages
    ? worstSeverity(...feedback.pack_voltages.map(batterySeverity))
    : "idle";
  const cap = feedback.kicker_cap_voltage;

  return (
    <div className="flex items-center gap-2 text-[10px] text-text-dim">
      {/* motors */}
      <span className="flex items-center gap-1">
        <span className="text-text-muted">M</span>
        <span className="flex items-center gap-0.5">
          {[0, 1, 2, 3, 4].map((i) => {
            const sev = worstSeverity(
              sysStatusSeverity(feedback.motor_statuses?.[i]),
              motorTempSeverity(feedback.motor_temps?.[i])
            );
            return (
              <SimpleTooltip
                key={i}
                title={`${i === 4 ? "Dribbler" : `Motor ${i}`}: ${
                  feedback.motor_statuses?.[i] ?? "n/a"
                }${
                  feedback.motor_temps?.[i] !== undefined
                    ? ` · ${Math.round(feedback.motor_temps[i])}°C`
                    : ""
                }`}
              >
                <span
                  className={cn(
                    "w-1.5 h-1.5 rounded-full",
                    severityDotClass(sev)
                  )}
                />
              </SimpleTooltip>
            );
          })}
        </span>
      </span>

      {/* battery */}
      <Stat label="bat" sev={batterySev}>
        {battery !== undefined ? `${battery.toFixed(1)}V` : "—"}
      </Stat>

      {/* capacitor */}
      <Stat label="cap" sev={capVoltageSeverity(cap)}>
        {cap !== undefined ? `${Math.round(cap)}V` : "—"}
      </Stat>

      {/* imu + kicker dots */}
      <DotStat label="imu" sev={sysStatusSeverity(feedback.imu_status)} />
      <DotStat label="kick" sev={sysStatusSeverity(feedback.kicker_status)} />
    </div>
  );
};

const Stat: FC<{ label: string; sev: Severity; children: React.ReactNode }> = ({
  label,
  sev,
  children,
}) => (
  <span className="flex items-center gap-1">
    <span className="text-text-muted">{label}</span>
    <span className={cn("font-mono", severityTextClass(sev))}>{children}</span>
  </span>
);

const DotStat: FC<{ label: string; sev: Severity }> = ({ label, sev }) => (
  <span className="flex items-center gap-1">
    <span className="text-text-muted">{label}</span>
    <span className={cn("w-1.5 h-1.5 rounded-full", severityDotClass(sev))} />
  </span>
);
