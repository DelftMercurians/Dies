import React from "react";
import { GameState, RawGameStateData, SideAssignment, TeamColor } from "../bindings";
import { cn } from "@/lib/utils";

/**
 * Central game status banner: score, the live game state, and what's coming up.
 * The single "what is the game doing right now" view. Yellow cards are shown per
 * side; the right-edge announcer feed carries the running commentary.
 */
interface GameBannerProps {
  gameState: RawGameStateData;
  sideAssignment: SideAssignment;
}

const gameStateLabel = (gs: GameState): string => {
  switch (gs.type) {
    case "Unknown":
      return "Unknown";
    case "Halt":
      return "Halt";
    case "Timeout":
      return "Timeout";
    case "Stop":
      return "Stop";
    case "PrepareKickoff":
      return "Kick-off setup";
    case "BallReplacement":
      return "Ball placement";
    case "PreparePenalty":
      return "Penalty setup";
    case "Kickoff":
      return "Kick-off";
    case "FreeKick":
      return "Free kick";
    case "Penalty":
      return "Penalty";
    case "PenaltyRun":
      return "Penalty";
    case "Run":
      return "Running";
    default:
      return "—";
  }
};

const teamTextClass = (team: TeamColor): string =>
  team === TeamColor.Blue ? "text-team-blue" : "text-team-yellow";

/** States that actually belong to a team (so the label can be tinted). Neutral
 *  states like Stop/Halt/Run shouldn't pick up the defaulted operating team. */
const isTeamState = (gs: GameState): boolean =>
  gs.type === "FreeKick" ||
  gs.type === "Kickoff" ||
  gs.type === "Penalty" ||
  gs.type === "PenaltyRun" ||
  gs.type === "PrepareKickoff" ||
  gs.type === "PreparePenalty" ||
  gs.type === "BallReplacement";

const GameBanner: React.FC<GameBannerProps> = ({ gameState, sideAssignment }) => {
  const {
    blue_team_yellow_cards: blueTeamYellowCards,
    yellow_team_yellow_cards: yellowTeamYellowCards,
    blue_team_score: blueTeamScore,
    yellow_team_score: yellowTeamScore,
    operating_team: operating,
    action_time_remaining: action,
    next_command: nextCommand,
    blue_team_name: blueTeamName,
    yellow_team_name: yellowTeamName,
  } = gameState;

  // Prefer the GC-reported team name (e.g. "TIGERs"); fall back to the colour.
  const teamLabel = (team: TeamColor): string => {
    const name = team === TeamColor.Blue ? blueTeamName : yellowTeamName;
    return name && name.length > 0 ? name : team === TeamColor.Blue ? "Blue" : "Yellow";
  };

  // Determine which team is on which side based on side assignment
  const leftTeam =
    sideAssignment === SideAssignment.BlueOnPositive
      ? TeamColor.Yellow
      : TeamColor.Blue;
  const rightTeam =
    sideAssignment === SideAssignment.BlueOnPositive
      ? TeamColor.Blue
      : TeamColor.Yellow;

  const leftCards =
    leftTeam === TeamColor.Blue ? blueTeamYellowCards : yellowTeamYellowCards;
  const rightCards =
    rightTeam === TeamColor.Blue ? blueTeamYellowCards : yellowTeamYellowCards;

  const leftScore =
    leftTeam === TeamColor.Blue ? blueTeamScore : yellowTeamScore;
  const rightScore =
    rightTeam === TeamColor.Blue ? blueTeamScore : yellowTeamScore;

  // Once play is running there's no pending action — the lingering free-kick /
  // kick-off command and its countdown are stale, so don't show them.
  const inPlay = gameState.game_state.type === "Run";
  const showAction = !inPlay && action != null && action > -2;
  const showNext = !inPlay && !!nextCommand;

  return (
    <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-20 flex flex-col items-center bg-bg-elevated/90 border border-border-muted px-4 py-1.5">
      {/* Score row: teams + score */}
      <div className="flex items-center gap-5">
        {/* Left side team */}
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "w-4 h-4 border border-text-dim",
              leftTeam === TeamColor.Blue ? "bg-team-blue" : "bg-team-yellow"
            )}
          />
          <span className="text-text-bright font-medium uppercase tracking-wide">
            {teamLabel(leftTeam)}
          </span>
          <div className="flex items-center gap-1">
            {Array.from({ length: leftCards }, (_, i) => (
              <div
                key={i}
                className="w-2.5 h-3.5 bg-team-yellow border border-team-yellow-dim"
              />
            ))}
          </div>
        </div>

        {/* Score */}
        <div className="flex items-center gap-2 font-mono text-lg font-bold tabular-nums">
          <span className={teamTextClass(leftTeam)}>{leftScore}</span>
          <span className="text-text-dim">:</span>
          <span className={teamTextClass(rightTeam)}>{rightScore}</span>
        </div>

        {/* Right side team */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            {Array.from({ length: rightCards }, (_, i) => (
              <div
                key={i}
                className="w-2.5 h-3.5 bg-team-yellow border border-team-yellow-dim"
              />
            ))}
          </div>
          <span className="text-text-bright font-medium uppercase tracking-wide">
            {teamLabel(rightTeam)}
          </span>
          <div
            className={cn(
              "w-4 h-4 border border-text-dim",
              rightTeam === TeamColor.Blue ? "bg-team-blue" : "bg-team-yellow"
            )}
          />
        </div>
      </div>

      {/* Status row: current state + what's next / countdown */}
      <div className="flex items-center gap-2 text-sm leading-tight">
        <span
          className={cn(
            "font-semibold",
            isTeamState(gameState.game_state)
              ? teamTextClass(operating)
              : "text-text-bright"
          )}
        >
          {gameStateLabel(gameState.game_state)}
        </span>
        {showNext && (
          <span className="text-text-dim">· Next: {nextCommand}</span>
        )}
        {showAction && (
          <span className="font-mono text-accent-cyan">
            {Math.max(0, action as number).toFixed(1)}s
          </span>
        )}
      </div>
    </div>
  );
};

export default GameBanner;
