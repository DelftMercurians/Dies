import {
  useBasestationInfo,
  useExecutorInfo,
  useWorldState,
  isPlayerManuallyControlled,
} from "@/api";
import { PlayerData, PlayerFeedbackMsg } from "@/bindings";
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

  if (worldState.status !== "connected") {
    return (
      <div className={cn("p-4 bg-gray-900 text-white", className)}>
        <h2 className="text-2xl font-bold mb-4">Team Overview</h2>
        <div className="text-center text-gray-400">
          Waiting for world state...
        </div>
      </div>
    );
  }

  const { own_players } = worldState.data;
  const sorted_players = [...own_players].sort((a, b) => a.id - b.id);

  return (
    <div className={cn("relative", className)}>
      <div className="absolute inset-0 overflow-y-auto p-2 bg-slate-900">
        <div className="grid grid-cols-1 gap-4">
          {sorted_players.length > 0 ? (
            sorted_players.map((player) => {
              const basestationData = bsInfo?.players[player.id];
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
      <CardContent className="p-4">
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Robot {player.id}</h3>
          {isManual && (
            <span className="text-xs bg-red-500 text-white px-2 py-1 rounded">
              Manual
            </span>
          )}
        </div>
        <div className="text-sm mt-2 flex items-center gap-2">
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
        <div className="text-sm mt-2 flex items-center gap-2 font-mono">
          <span>Pos:</span>
          <span>
            ({player.position[0].toFixed(0)}, {player.position[1].toFixed(0)})
          </span>
        </div>
      </CardContent>
    </Card>
  );
};
