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
import { cn } from "@/lib/utils";
import { FC } from "react";

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
      <div className={cn("p-1 bg-gray-900 text-white", className)}>
        <div className="text-center text-gray-400">
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
      <div className="absolute inset-0 overflow-y-auto p-2 bg-slate-900">
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
            <div className="text-center text-gray-400">
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
    if (!status) return "bg-gray-500";
    const lowerStatus = status.toLowerCase();
    if (
      lowerStatus.includes("error") ||
      lowerStatus.includes("fail") ||
      lowerStatus.includes("timeout")
    ) {
      return "bg-red-500";
    }
    if (lowerStatus.includes("warn") || lowerStatus.includes("degraded")) {
      return "bg-yellow-500";
    }
    if (
      lowerStatus.includes("ok") ||
      lowerStatus.includes("good") ||
      lowerStatus.includes("connected")
    ) {
      return "bg-green-500";
    }
    return "bg-blue-500";
  };

  const breakbeamDetected = basestationData?.breakbeam_ball_detected;

  return (
    <Card
      onClick={onClick}
      className={cn(
        "bg-gray-800 text-white cursor-pointer hover:bg-gray-700",
        isSelected && "ring-2 ring-blue-500"
      )}
    >
      <CardContent className="px-0 py-1 text-sm">
        <div className="flex justify-between items-center mb-1">
          <h3 className="text-sm font-semibold">Robot {player.id}</h3>
          {isManual && (
            <span className="text-xs bg-red-500 text-white px-2 py-1 rounded">
              Manual
            </span>
          )}
        </div>

        {/* Role and Skill */}
        <div className="text-xs mb-1 flex gap-2">
          {role && (
            <span className="bg-blue-600 text-white px-1 rounded">{role}</span>
          )}
          {skill && (
            <span className="bg-purple-600 text-white px-1 rounded">
              {skill}
            </span>
          )}
        </div>

        {/* Status indicators row */}
        <div className="flex items-center gap-3 text-xs mb-1">
          {/* Basestation status */}
          <div className="flex items-center gap-1">
            <span>BS:</span>
            <div
              className={cn(
                "w-2 h-2 rounded-full",
                hasBsInfo ? "bg-green-500" : "bg-red-500"
              )}
            />
          </div>

          {/* Breakbeam with orange circle when detected */}
          <div className="flex items-center gap-1">
            <span>BB:</span>
            <div
              className={cn(
                "w-2 h-2 rounded-full",
                breakbeamDetected === true
                  ? "bg-orange-500"
                  : breakbeamDetected === false
                  ? "bg-gray-500"
                  : "bg-gray-600"
              )}
            />
          </div>

          {/* Motor driver status */}
          <div className="flex items-center gap-1">
            <span>Motor:</span>
            <div
              className={cn(
                "w-2 h-2 rounded-full",
                getStatusColor(motorStatus)
              )}
            />
          </div>

          {/* IMU status */}
          <div className="flex items-center gap-1">
            <span>IMU:</span>
            <div
              className={cn("w-2 h-2 rounded-full", getStatusColor(imuStatus))}
            />
          </div>
        </div>

        {/* Position */}
        <div className="text-xs flex items-center gap-2 font-mono">
          <span>Pos:</span>
          <span>
            ({player.position[0].toFixed(0)}, {player.position[1].toFixed(0)})
          </span>
        </div>
      </CardContent>
    </Card>
  );
};
