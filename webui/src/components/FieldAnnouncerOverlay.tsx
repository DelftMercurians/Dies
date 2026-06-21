import React, { useEffect, useRef, useState } from "react";
import { useAtomValue } from "jotai";

import { AnnouncementFeedItem, announcementsAtom } from "@/api";
import { AnnouncementCategory, TeamColor } from "@/bindings";

/** Opacity of an idle (settled, un-hovered) announcer line. Mostly transparent
 *  so the feed floats unobtrusively over the field. */
const BASELINE_OPACITY = 0.3;
/** How long a fresh line stays fully opaque before fading. */
const HOLD_MS = 4000;
/** Fade duration from full opacity down to the baseline. */
const FADE_MS = 1500;

const teamTextClass = (team: TeamColor | undefined): string => {
  if (team === TeamColor.Blue) return "text-sky-300";
  if (team === TeamColor.Yellow) return "text-yellow-200";
  return "text-gray-100";
};

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

/** A single announcer line: pops in at full opacity, then fades to baseline. */
const AnnouncerLine: React.FC<{
  item: AnnouncementFeedItem;
  hovered: boolean;
}> = ({ item, hovered }) => {
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

  const opacity = hovered ? 1 : settled ? BASELINE_OPACITY : 1;
  const transition = hovered ? "opacity 150ms ease" : `opacity ${FADE_MS}ms ease`;

  return (
    <div
      className={`border-l-2 ${categoryAccent(
        item.category
      )} pl-2 py-0.5 leading-snug`}
      style={{
        opacity,
        transition,
        // Legibility over the field without a panel background.
        textShadow: "0 1px 3px rgba(0,0,0,0.9)",
      }}
    >
      <span className={teamTextClass(item.team ?? undefined)}>{item.text}</span>
    </div>
  );
};

/**
 * Right-edge announcer feed: a backgroundless, mostly-transparent scrolling
 * commentary log. Newest line pops in fully opaque, then fades back; hovering
 * lifts the whole feed to full opacity. Game state + score live in the top
 * banner; GC actions live in the toolbar.
 */
const FieldAnnouncerOverlay: React.FC = () => {
  const announcements = useAtomValue(announcementsAtom);
  const [hovered, setHovered] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep the newest line in view.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [announcements.length]);

  if (announcements.length === 0) return null;

  return (
    <div
      ref={scrollRef}
      className="absolute right-2 bottom-16 z-20 w-60 max-h-[55%] overflow-y-auto pointer-events-auto select-none flex flex-col justify-end gap-px text-[13px]"
      style={{
        // Fade older lines out into the top edge.
        maskImage: "linear-gradient(to bottom, transparent, black 2rem)",
        WebkitMaskImage: "linear-gradient(to bottom, transparent, black 2rem)",
        // Hide the scrollbar (the feed is backgroundless; a bar would be noise).
        scrollbarWidth: "none",
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {announcements.map((item) => (
        <AnnouncerLine key={item.clientKey} item={item} hovered={hovered} />
      ))}
    </div>
  );
};

export default FieldAnnouncerOverlay;
