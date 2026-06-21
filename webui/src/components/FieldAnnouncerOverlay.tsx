import React, { useEffect, useRef, useState } from "react";
import { useAtomValue } from "jotai";

import {
  AnnouncementFeedItem,
  announcementsAtom,
  useSendCommand,
  useStatus,
} from "@/api";
import {
  AnnouncementCategory,
  GameState,
  GcSimCommand,
  RawGameStateData,
  TeamColor,
  UiMode,
} from "@/bindings";

/** Opacity of the overlay when idle/un-hovered (the "rest of the overlay"). */
const BASELINE_OPACITY = 0.6;
/** How long a fresh announcer line stays fully opaque before fading. */
const HOLD_MS = 4000;
/** Fade duration from full opacity down to the baseline. */
const FADE_MS = 1500;

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

const teamTextClass = (team: TeamColor | undefined): string => {
  if (team === TeamColor.Blue) return "text-sky-400";
  if (team === TeamColor.Yellow) return "text-yellow-300";
  return "text-gray-200";
};

const teamLabel = (team: TeamColor | undefined): string =>
  team === TeamColor.Blue ? "Blue" : team === TeamColor.Yellow ? "Yellow" : "";

const categoryAccent = (cat: AnnouncementCategory): string => {
  switch (cat) {
    case AnnouncementCategory.Goal:
      return "border-l-emerald-400";
    case AnnouncementCategory.Foul:
      return "border-l-orange-400";
    case AnnouncementCategory.Card:
      return "border-l-red-500";
    case AnnouncementCategory.FreeKick:
    case AnnouncementCategory.Kickoff:
    case AnnouncementCategory.Penalty:
      return "border-l-cyan-400";
    case AnnouncementCategory.Placement:
      return "border-l-violet-400";
    case AnnouncementCategory.Stoppage:
      return "border-l-amber-400";
    default:
      return "border-l-slate-500";
  }
};

/** Format seconds as a compact clock (mm:ss), clamped at zero. */
const formatClock = (secs: number): string => {
  const s = Math.max(0, Math.floor(secs));
  const m = Math.floor(s / 60);
  return `${m}:${(s % 60).toString().padStart(2, "0")}`;
};

/** The current-state panel: what's happening, who's acting, what's next. */
const StatePanel: React.FC<{
  gameState: RawGameStateData;
  baseOpacity: number;
}> = ({ gameState, baseOpacity }) => {
  const operating = gameState.operating_team;
  const action = gameState.action_time_remaining;
  // Once the ball is in open play there's no pending action — the lingering
  // free-kick/kickoff command and its countdown are stale, so suppress them.
  const inPlay = gameState.game_state.type === "Run";
  const showAction = !inPlay && action != null && action > -2;
  const showNext = !inPlay && gameState.next_command;
  const kicker = gameState.freekick_kicker;

  return (
    <div
      className="bg-bg-elevated/85 border border-border-muted backdrop-blur-sm rounded px-3 py-2 text-xs"
      style={{ opacity: baseOpacity, transition: "opacity 200ms ease" }}
    >
      {/* Stage + stage clock */}
      {(gameState.stage || gameState.stage_time_left != null) && (
        <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-gray-400 mb-1">
          <span>{gameState.stage ?? ""}</span>
          {gameState.stage_time_left != null && (
            <span className="font-mono">
              {formatClock(gameState.stage_time_left)}
            </span>
          )}
        </div>
      )}

      {/* Current state, tinted by the operating team */}
      <div className="flex items-baseline justify-between gap-2">
        <span
          className={`text-sm font-semibold ${teamTextClass(operating)}`}
        >
          {gameStateLabel(gameState.game_state)}
        </span>
        {teamLabel(operating) && (
          <span className={`text-[11px] ${teamTextClass(operating)}`}>
            {teamLabel(operating)}
          </span>
        )}
      </div>

      {/* What's next + action countdown */}
      {(showNext || showAction) && (
        <div className="mt-1 flex items-center justify-between gap-2 text-gray-300">
          <span className="truncate">
            {showNext ? `Next: ${gameState.next_command}` : ""}
          </span>
          {showAction && (
            <span className="font-mono text-cyan-300">
              {Math.max(0, action as number).toFixed(1)}s
            </span>
          )}
        </div>
      )}

      {/* Double-touch watch */}
      {kicker && (
        <div className="mt-1 text-[11px] text-amber-300/90">
          Kicker locked: {teamLabel(kicker.team_color)} #{kicker.player_id}
        </div>
      )}

      {/* Stoppage reason */}
      {gameState.status_message && (
        <div className="mt-1 text-[11px] italic text-gray-400">
          {gameState.status_message}
        </div>
      )}
    </div>
  );
};

/** Game-controller quick actions (sim only). */
const GcQuickActions: React.FC<{
  gameState: RawGameStateData;
  baseOpacity: number;
}> = ({ gameState, baseOpacity }) => {
  const sendCommand = useSendCommand();

  const send = (command: GcSimCommand) =>
    sendCommand({ type: "GcCommand", data: command });

  // "Continue" resumes the right way: a prepared kick-off/penalty needs a
  // normal start, anything else just force-starts open play.
  const continueCommand = (): GcSimCommand => {
    const t = gameState.game_state.type;
    if (t === "PrepareKickoff" || t === "PreparePenalty") {
      return { type: "NormalStart" };
    }
    return { type: "ForceStart" };
  };

  return (
    <div
      className="flex gap-2"
      style={{ opacity: baseOpacity, transition: "opacity 200ms ease" }}
    >
      <button
        onClick={() => send({ type: "Stop" })}
        className="flex-1 bg-bg-elevated/85 hover:bg-red-900/70 border border-border-muted backdrop-blur-sm rounded px-2 py-1.5 text-xs font-medium text-red-300 transition-colors"
      >
        Stop
      </button>
      <button
        onClick={() => send(continueCommand())}
        className="flex-1 bg-bg-elevated/85 hover:bg-emerald-900/70 border border-border-muted backdrop-blur-sm rounded px-2 py-1.5 text-xs font-medium text-emerald-300 transition-colors"
      >
        Continue
      </button>
    </div>
  );
};

/** A single announcer line: pops in at full opacity, then fades to baseline. */
const AnnouncerLine: React.FC<{
  item: AnnouncementFeedItem;
  hovered: boolean;
  baseOpacity: number;
}> = ({ item, hovered, baseOpacity }) => {
  const [settled, setSettled] = useState(false);

  useEffect(() => {
    const age = Date.now() - item.arrivedAt;
    if (age >= HOLD_MS) {
      setSettled(true);
      return;
    }
    const id = setTimeout(() => setSettled(true), HOLD_MS - age);
    return () => clearTimeout(id);
  }, [item.arrivedAt]);

  const opacity = hovered ? 1 : settled ? baseOpacity : 1;
  const transition = hovered ? "opacity 150ms ease" : `opacity ${FADE_MS}ms ease`;

  return (
    <div
      className={`border-l-2 ${categoryAccent(
        item.category
      )} pl-2 py-0.5 leading-snug`}
      style={{ opacity, transition }}
    >
      <span className={teamTextClass(item.team ?? undefined)}>{item.text}</span>
    </div>
  );
};

/** Scrolling commentary feed, newest at the bottom. */
const Announcer: React.FC<{
  items: AnnouncementFeedItem[];
  hovered: boolean;
}> = ({ items, hovered }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep the newest line in view.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [items.length]);

  if (items.length === 0) return null;

  return (
    <div className="flex-1 min-h-0 bg-bg-elevated/70 border border-border-muted backdrop-blur-sm rounded">
      <div
        ref={scrollRef}
        className="h-full overflow-y-auto px-1.5 py-1 text-[11px] flex flex-col gap-px"
        style={{
          // Fade older lines out into the top edge.
          maskImage:
            "linear-gradient(to bottom, transparent, black 1.75rem)",
          WebkitMaskImage:
            "linear-gradient(to bottom, transparent, black 1.75rem)",
        }}
      >
        {items.map((item) => (
          <AnnouncerLine
            key={item.clientKey}
            item={item}
            hovered={hovered}
            baseOpacity={BASELINE_OPACITY}
          />
        ))}
      </div>
    </div>
  );
};

/**
 * Field overlay anchored to the right edge: a live game-state panel, sim-only GC
 * quick actions, and a scrolling announcer feed. Translucent when idle, fully
 * opaque on hover; fresh announcer lines pop in at full opacity then fade back.
 * Deliberately does not duplicate the top bar (score / yellow cards).
 */
const FieldAnnouncerOverlay: React.FC<{ gameState: RawGameStateData }> = ({
  gameState,
}) => {
  const { data: backendState } = useStatus();
  const isSim = backendState?.ui_mode === UiMode.Simulation;
  const announcements = useAtomValue(announcementsAtom);
  const [hovered, setHovered] = useState(false);

  const baseOpacity = hovered ? 1 : BASELINE_OPACITY;
  const hoverProps = {
    onMouseEnter: () => setHovered(true),
    onMouseLeave: () => setHovered(false),
  };

  return (
    // Container is click-through; only the panels capture pointer events so the
    // field underneath stays interactive between/around them.
    <div className="absolute right-2 top-12 bottom-16 z-20 w-64 flex flex-col gap-2 pointer-events-none select-none">
      <div className="pointer-events-auto" {...hoverProps}>
        <StatePanel gameState={gameState} baseOpacity={baseOpacity} />
      </div>
      {isSim && (
        <div className="pointer-events-auto" {...hoverProps}>
          <GcQuickActions gameState={gameState} baseOpacity={baseOpacity} />
        </div>
      )}
      {/* Only capture pointer events where there's actually a feed to show, so
          the empty lower field stays click-through. */}
      {announcements.length > 0 && (
        <div className="flex-1 min-h-0 pointer-events-auto" {...hoverProps}>
          <Announcer items={announcements} hovered={hovered} />
        </div>
      )}
    </div>
  );
};

export default FieldAnnouncerOverlay;
