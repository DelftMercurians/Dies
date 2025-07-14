import {
  useBasestationInfo,
  useExecutorInfo,
  useWorldState,
  isPlayerManuallyControlled,
  usePrimaryTeam,
} from "@/api";
import { PlayerData, PlayerFeedbackMsg, TeamColor } from "@/bindings";
import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { FC } from "react";

interface TeamOverviewProps {
  className?: string;
  onSelectPlayer: (id: number) => void;
  selectedPlayerId: number | null;
}

const TeamOverview: FC<TeamOverviewProps> = ({
  className,
  onSelectPlayer,
  selectedPlayerId,
}) => {
  const worldState = useWorldState();
  const { data: bsInfo } = useBasestationInfo();
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

  return (
    <div className={cn("relative", className)}>
      <div className="absolute inset-0 overflow-y-auto p-2 bg-slate-900">
        <div className="grid grid-cols-1 gap-1">
          {sorted_players.length > 0 ? (
            sorted_players.map((player) => {
              const basestationData =
                primaryTeam === TeamColor.Blue
                  ? bsInfo?.blue_team?.[player.id]
                  : bsInfo?.yellow_team?.[player.id];
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
}

const PlayerCard: FC<PlayerCardProps> = ({
  player,
  basestationData,
  isManual,
  isSelected,
  onClick,
}) => {
  const bsStatus = basestationData?.primary_status;
  const hasBsInfo = !!basestationData;

  return (
    <Card
      onClick={onClick}
      className={cn(
        "bg-gray-800 text-white cursor-pointer hover:bg-gray-700",
        isSelected && "ring-2 ring-blue-500"
      )}
    >
      <CardContent className="px-0 py-1 text-sm">
        <div className="flex justify-between items-center">
          <h3 className="text-sm font-semibold">Robot {player.id}</h3>
          {isManual && (
            <span className="text-xs bg-red-500 text-white px-2 py-1 rounded">
              Manual
            </span>
          )}
        </div>
        <div className="text-sm mt-0.5 flex items-center gap-2">
          <span>Basestation:</span>
          <div
            className={cn(
              "w-3 h-3 rounded-full",
              hasBsInfo ? "bg-green-500" : "bg-red-500"
            )}
          />
          <span className="text-xs">
            {bsStatus || (hasBsInfo ? "Ok" : "N/A")}
          </span>
        </div>
        <div className="text-sm mt-0.5 flex items-center gap-2">
          <span>Breakbeam:</span>
          <span>
            {basestationData
              ? basestationData.breakbeam_ball_detected
                ? "Yes"
                : "No"
              : "N/A"}
          </span>
        </div>
        <div className="text-sm mt-0.5 flex items-center gap-2 font-mono">
          <span>Pos:</span>
          <span>
            ({player.position[0].toFixed(0)}, {player.position[1].toFixed(0)})
          </span>
        </div>
      </CardContent>
    </Card>
  );
};
