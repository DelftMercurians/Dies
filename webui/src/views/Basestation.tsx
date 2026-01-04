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

/**
 * Basestation view showing connected robots and their hardware status.
 * Uses mission control aesthetic.
 */

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
    <div className={`${className} p-2 bg-bg-surface text-text-std`}>
      <h2 className="text-[12px] font-semibold uppercase tracking-wider text-text-bright mb-3">
        Connected Robots
      </h2>
      <div className="grid grid-cols-1 gap-2">
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
          <div className="text-center text-text-muted text-[10px] p-4">
            No robots connected
          </div>
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
          className={`w-1.5 h-1.5 ${
            status === SysStatus.Emergency
              ? "bg-accent-red"
              : [
                  SysStatus.Stop,
                  SysStatus.NoReply,
                  SysStatus.Overtemp,
                  SysStatus.Armed,
                ].includes(status as SysStatus)
              ? "bg-accent-amber"
              : status === SysStatus.Ok
              ? "bg-accent-green"
              : "bg-text-muted"
          }`}
        />
        <span className="text-[9px]">{label}</span>
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

  return (
    <Card className="hover:bg-bg-overlay transition-colors">
      <CardContent className="p-2">
        <div className="space-y-1.5">
          <div className="flex flex-row gap-3 items-center">
            <h3
              onClick={onClick}
              className="text-[11px] font-semibold cursor-pointer text-text-bright hover:text-accent-cyan"
            >
              Robot {player.id}
            </h3>
            <Badge
              variant={
                teamColor === TeamColor.Blue ? "team-blue" : "team-yellow"
              }
            >
              {teamColor}
            </Badge>
            <StatusIndicator status={player.primary_status} label="Status" />
          </div>

          <Collapsible
            open={isMotorsExpanded}
            onOpenChange={setIsMotorsExpanded}
          >
            <CollapsibleTrigger className="flex items-center space-x-1 text-[10px] text-text-dim hover:text-text-std">
              <span>Motors</span>
              {isMotorsExpanded ? (
                <ChevronUp size={12} />
              ) : (
                <ChevronDown size={12} />
              )}
            </CollapsibleTrigger>
            <CollapsibleContent className="AnimatedCollapsible space-y-1 ml-3 mt-1">
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
            <CollapsibleTrigger className="flex items-center space-x-1 text-[10px] text-text-dim hover:text-text-std">
              <span>Sensors</span>
              {isSensorExpanded ? (
                <ChevronUp size={12} />
              ) : (
                <ChevronDown size={12} />
              )}
            </CollapsibleTrigger>

            <CollapsibleContent className="AnimatedCollapsible space-y-1 ml-3 mt-1">
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
