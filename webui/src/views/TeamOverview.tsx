import {
  useBasestationInfo,
  useExecutorInfo,
  useWorldState,
  isPlayerManuallyControlled,
  usePrimaryTeam,
  useDebugData,
} from "@/api";
import { PlayerData, PlayerFeedbackMsg, TeamColor, DebugMap } from "@/bindings";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { FC } from "react";

/**
 * Team Overview showing all players on the primary team.
 * Uses mission control aesthetic with compact player cards.
 */

interface TeamOverviewProps {
  className?: string;
  onSelectPlayer: (id: number) => void;
  selectedPlayerId: number | null;
}

// Helper functions to extract debug data for a player
const getPlayerDebugValue = (
  debugData: DebugMap | null,
  playerId: number,
  teamColor: TeamColor,
  key: string
): string | null => {
  if (!debugData) return null;
  const teamColorStr = teamColor === TeamColor.Blue ? "Blue" : "Yellow";
  const debugKey = `team_${teamColorStr}.p${playerId}.${key}`;
  const debugValue = debugData[debugKey];
  return debugValue?.type === "String" ? (debugValue.data as string) : null;
};

const getPlayerRole = (
  debugData: DebugMap | null,
  playerId: number,
  teamColor: TeamColor
): string | null => {
  return getPlayerDebugValue(debugData, playerId, teamColor, "role");
};

const getPlayerSkill = (
  debugData: DebugMap | null,
  playerId: number,
  teamColor: TeamColor
): string | null => {
  return getPlayerDebugValue(debugData, playerId, teamColor, "skill");
};

const getPlayerMotorStatus = (
  debugData: DebugMap | null,
  playerId: number,
  teamColor: TeamColor
): string | null => {
  return getPlayerDebugValue(debugData, playerId, teamColor, "motor_driver");
};

const getPlayerIMUStatus = (
  debugData: DebugMap | null,
  playerId: number,
  teamColor: TeamColor
): string | null => {
  return getPlayerDebugValue(debugData, playerId, teamColor, "imu");
};

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

  if (worldState.status !== "connected") {
    return (
      <div className={cn("p-2 bg-bg-surface text-text-std", className)}>
        <div className="text-center text-text-muted text-[10px]">
          Waiting for world state...
        </div>
      </div>
    );
  }

  const { blue_team, yellow_team } = worldState.data;
  const own_players = primaryTeam === TeamColor.Blue ? blue_team : yellow_team;
  const sorted_players = [...own_players].sort((a, b) => a.id - b.id);
  const bsPlayers = bsInfo
    ? bsInfo.blue_team.length === 0 && bsInfo.yellow_team.length === 0
      ? bsInfo.unknown_team
      : primaryTeam === TeamColor.Blue
      ? bsInfo.blue_team
      : bsInfo.yellow_team
    : [];

  return (
    <div className={cn("relative", className)}>
      <div className="absolute inset-0 overflow-y-auto p-1 bg-bg-surface">
        <div className="grid grid-cols-1 gap-1">
          {sorted_players.length > 0 ? (
            sorted_players.map((player) => {
              const basestationData = bsPlayers.find((p) => p.id === player.id);
              const isManual = isPlayerManuallyControlled(
                player.id,
                executorInfo?.manual_controlled_players ?? []
              );
              const isSelected = player.id === selectedPlayerId;

              return (
                <PlayerCard
                  key={player.id}
                  player={player}
                  basestationData={basestationData}
                  isManual={isManual}
                  isSelected={isSelected}
                  onClick={() => onSelectPlayer(player.id)}
                  debugData={debugData}
                  teamColor={primaryTeam}
                />
              );
            })
          ) : (
            <div className="text-center text-text-muted text-[10px] p-4">
              No players in world state.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default TeamOverview;

interface PlayerCardProps {
  player: PlayerData;
  basestationData?: PlayerFeedbackMsg;
  isManual: boolean;
  isSelected: boolean;
  onClick: () => void;
  debugData: DebugMap | null;
  teamColor: TeamColor;
}

const PlayerCard: FC<PlayerCardProps> = ({
  player,
  basestationData,
  isManual,
  isSelected,
  onClick,
  debugData,
  teamColor,
}) => {
  const bsStatus = basestationData?.primary_status;
  const hasBsInfo = !!basestationData;

  // Extract debug information
  const role = getPlayerRole(debugData, player.id, teamColor);
  const skill = getPlayerSkill(debugData, player.id, teamColor);
  const motorStatus = getPlayerMotorStatus(debugData, player.id, teamColor);
  const imuStatus = getPlayerIMUStatus(debugData, player.id, teamColor);

  // Determine status colors
  const getStatusColor = (status: string | null | undefined) => {
    if (!status) return "bg-text-muted";
    const lowerStatus = status.toLowerCase();
    if (
      lowerStatus.includes("error") ||
      lowerStatus.includes("fail") ||
      lowerStatus.includes("timeout")
    ) {
      return "bg-accent-red";
    }
    if (lowerStatus.includes("warn") || lowerStatus.includes("degraded")) {
      return "bg-accent-amber";
    }
    if (
      lowerStatus.includes("ok") ||
      lowerStatus.includes("good") ||
      lowerStatus.includes("connected")
    ) {
      return "bg-accent-green";
    }
    return "bg-accent-blue";
  };

  const breakbeamDetected = basestationData?.breakbeam_ball_detected;

  return (
    <Card
      onClick={onClick}
      className={cn(
        "cursor-pointer hover:bg-bg-overlay transition-colors",
        isSelected && "ring-1 ring-accent-cyan border-accent-cyan"
      )}
    >
      <CardContent className="px-2 py-1.5">
        <div className="flex justify-between items-center mb-1">
          <h3 className="text-[11px] font-semibold text-text-bright">
            Robot {player.id}
          </h3>
          {isManual && (
            <Badge variant="destructive">Manual</Badge>
          )}
        </div>

        {/* Role and Skill */}
        <div className="text-[9px] mb-1 flex gap-1">
          {role && (
            <Badge variant="team-blue">{role}</Badge>
          )}
          {skill && (
            <Badge variant="info">{skill}</Badge>
          )}
        </div>

        {/* Status indicators row */}
        <div className="flex items-center gap-3 text-[9px] text-text-dim mb-1">
          {/* Basestation status */}
          <div className="flex items-center gap-1">
            <span>BS:</span>
            <div
              className={cn(
                "w-1.5 h-1.5",
                hasBsInfo ? "bg-accent-green" : "bg-accent-red"
              )}
            />
          </div>

          {/* Breakbeam with orange circle when detected */}
          <div className="flex items-center gap-1">
            <span>BB:</span>
            <div
              className={cn(
                "w-1.5 h-1.5",
                breakbeamDetected === true
                  ? "bg-accent-amber"
                  : breakbeamDetected === false
                  ? "bg-text-muted"
                  : "bg-border-muted"
              )}
            />
          </div>

          {/* Motor driver status */}
          <div className="flex items-center gap-1">
            <span>MTR:</span>
            <div
              className={cn(
                "w-1.5 h-1.5",
                getStatusColor(motorStatus)
              )}
            />
          </div>

          {/* IMU status */}
          <div className="flex items-center gap-1">
            <span>IMU:</span>
            <div className={cn("w-1.5 h-1.5", getStatusColor(imuStatus))} />
          </div>
        </div>

        {/* Position */}
        <div className="text-[9px] flex items-center gap-2 text-text-dim">
          <span>Pos:</span>
          <span className="text-text-std">
            ({player.position[0].toFixed(0)}, {player.position[1].toFixed(0)})
          </span>
        </div>
      </CardContent>
    </Card>
  );
};
