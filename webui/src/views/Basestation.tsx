import { useBasestationInfo } from "@/api";
import { FC } from "react";
import { PlayerFeedbackMsg, SysStatus, TeamColor } from "@/bindings";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { ChevronUp, ChevronDown } from "lucide-react";
import React from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const Basestation: FC<{
  className: string;
  onSelectPlayer: (id: number) => void;
}> = ({ className, onSelectPlayer }) => {
  const { data } = useBasestationInfo();

  if (!data) return null;
  const { blue_team, yellow_team } = data;

  const allPlayers = [
    ...blue_team.map((player) => ({ ...player, teamColor: TeamColor.Blue })),
    ...yellow_team.map((player) => ({
      ...player,
      teamColor: TeamColor.Yellow,
    })),
  ].sort((a, b) => a.id - b.id);

  return (
    <div className={`${className} p-4 bg-gray-900 text-white`}>
      <h2 className="text-2xl font-bold mb-4">Connected Robots</h2>
      <div className="grid grid-cols-1 gap-4">
        {allPlayers.length > 0 ? (
          allPlayers.map((player) => (
            <PlayerStatus
              key={`${player.teamColor}-${player.id}`}
              player={player}
              teamColor={player.teamColor}
              onClick={() => onSelectPlayer(player.id)}
            />
          ))
        ) : (
          <div className="text-center text-gray-400">No robots connected</div>
        )}
      </div>
    </div>
  );
};

export default Basestation;

const StatusIndicator: FC<{
  status?: SysStatus;
  label: string;
  details?: Record<string, string | number>;
}> = ({ status, label, details }) => (
  <Tooltip>
    <TooltipTrigger asChild>
      <div className="flex items-center space-x-2 cursor-help">
        <div
          className={`w-2 h-2 rounded-full ${
            status === SysStatus.Emergency
              ? "bg-red-500"
              : [
                  SysStatus.Stop,
                  SysStatus.NoReply,
                  SysStatus.Overtemp,
                  SysStatus.Armed,
                ].includes(status as SysStatus)
              ? "bg-yellow-500"
              : status === SysStatus.Ok
              ? "bg-green-500"
              : "bg-gray-500"
          }`}
        />
        <span className="text-xs">{label}</span>
      </div>
    </TooltipTrigger>

    <TooltipContent align="start">
      <p>{status}</p>
      {details &&
        Object.entries(details).map(([key, value]) => (
          <p key={key}>
            {key}: {value}
          </p>
        ))}
    </TooltipContent>
  </Tooltip>
);

const PlayerStatus: FC<{
  player: PlayerFeedbackMsg;
  teamColor: TeamColor;
  onClick: () => void;
}> = ({ player, teamColor, onClick }) => {
  const [isMotorsExpanded, setIsMotorsExpanded] = React.useState(false);
  const [isSensorExpanded, setIsSensorExpanded] = React.useState(false);

  const getTeamColorClasses = (team: TeamColor) => {
    return team === TeamColor.Blue
      ? "bg-blue-600 text-white"
      : "bg-yellow-500 text-black";
  };

  return (
    <Card className="bg-gray-800 text-white">
      <CardContent className="p-4">
        <div className="space-y-2">
          <div className="flex flex-row gap-6 items-center">
            <h3
              onClick={onClick}
              className="text-lg font-semibold cursor-pointer"
            >
              Robot {player.id}
            </h3>
            <Badge className={`text-xs ${getTeamColorClasses(teamColor)}`}>
              {teamColor}
            </Badge>
            <StatusIndicator status={player.primary_status} label="Status" />
          </div>

          <Collapsible
            open={isMotorsExpanded}
            onOpenChange={setIsMotorsExpanded}
          >
            <CollapsibleTrigger className="flex items-center space-x-2">
              <span>Motors</span>
              {isMotorsExpanded ? (
                <ChevronUp size={16} />
              ) : (
                <ChevronDown size={16} />
              )}
            </CollapsibleTrigger>
            <CollapsibleContent className="AnimatedCollapsible space-y-2 ml-4 mt-2">
              {[0, 1, 2, 3, 4].map((index) => (
                <StatusIndicator
                  key={index}
                  status={player.motor_statuses?.[index]}
                  label={index === 4 ? "Dribbler" : `Motor ${index}`}
                  details={{
                    Temp: `${player.motor_temps?.[index] || "0.0"}°C`,
                    Speed: `${player.motor_speeds?.[index] || "0.0"} rad/s`,
                  }}
                />
              ))}
            </CollapsibleContent>
          </Collapsible>

          <Collapsible
            open={isSensorExpanded}
            onOpenChange={setIsSensorExpanded}
          >
            <CollapsibleTrigger className="flex items-center space-x-2">
              <span>Sensors</span>
              {isSensorExpanded ? (
                <ChevronUp size={16} />
              ) : (
                <ChevronDown size={16} />
              )}
            </CollapsibleTrigger>

            <CollapsibleContent className="AnimatedCollapsible space-y-2 ml-4 mt-2">
              <StatusIndicator status={player.imu_status} label="IMU" />
              <StatusIndicator
                status={player.kicker_status}
                label="Kicker"
                details={{
                  Temp: `${player.kicker_temp || "0.0"}°C`,
                  CapV: `${player.kicker_cap_voltage || "0.0"}V`,
                }}
              />

              <StatusIndicator status={player.fan_status} label="Fan" />

              <StatusIndicator
                status={
                  player.breakbeam_ball_detected
                    ? SysStatus.Ok
                    : SysStatus.NoReply
                }
                label="BreakBeam"
                details={{
                  "Sensor OK": player.breakbeam_sensor_ok ? "Yes" : "No",
                }}
              />
            </CollapsibleContent>
          </Collapsible>
        </div>
      </CardContent>
    </Card>
  );
};
