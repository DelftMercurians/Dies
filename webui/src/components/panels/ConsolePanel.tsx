import React, { useEffect, useMemo, useRef, useState } from "react";
import { IDockviewPanelProps } from "dockview";
import { Trash2, ArrowDownToLine } from "lucide-react";

import { useConsoleLogs, useClearConsoleLogs } from "@/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { ConsoleLogLevel, ConsoleLogMessage } from "@/bindings";

/** Severity order, low → high. Index doubles as the filter rank. */
const LEVEL_ORDER: ConsoleLogLevel[] = [
  ConsoleLogLevel.Trace,
  ConsoleLogLevel.Debug,
  ConsoleLogLevel.Info,
  ConsoleLogLevel.Warn,
  ConsoleLogLevel.Error,
];

const levelRank = (level: ConsoleLogLevel): number =>
  LEVEL_ORDER.indexOf(level);

/**
 * Console Panel — streams the backend's `log`-crate output live. Lines arrive
 * over the WS (`ConsoleLog` messages) into a ring-buffered atom. Supports a
 * minimum-level filter, free-text filter, autoscroll (pauses when the user
 * scrolls up), and clear.
 */
const ConsolePanel: React.FC<IDockviewPanelProps> = () => {
  const logs = useConsoleLogs();
  const clear = useClearConsoleLogs();
  const [minLevel, setMinLevel] = useState<ConsoleLogLevel>(
    ConsoleLogLevel.Debug
  );
  const [filter, setFilter] = useState("");
  const [autoScroll, setAutoScroll] = useState(true);

  const minRank = levelRank(minLevel);
  const filtered = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    return logs.filter((e) => {
      if (levelRank(e.level) < minRank) return false;
      if (needle && !matches(e, needle)) return false;
      return true;
    });
  }, [logs, minRank, filter]);

  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Stick to the bottom while autoscroll is on and new lines arrive.
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [filtered, autoScroll]);

  // Pause autoscroll when the user scrolls up; resume at the bottom.
  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom =
      el.scrollHeight - el.scrollTop - el.clientHeight < 24;
    setAutoScroll(atBottom);
  };

  const jumpToBottom = () => {
    setAutoScroll(true);
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  };

  return (
    <div className="w-full h-full bg-bg-surface flex flex-col text-sm">
      <div className="flex items-center gap-2 px-2 py-1.5 border-b border-bg-muted">
        <Select
          value={minLevel}
          onValueChange={(v) => setMinLevel(v as ConsoleLogLevel)}
        >
          <SelectTrigger className="h-6 w-[88px] text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {LEVEL_ORDER.map((lvl) => (
              <SelectItem key={lvl} value={lvl} className="text-xs">
                {lvl}+
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter…"
          className="h-6 flex-1 min-w-0 text-xs"
        />

        <span className="text-xs text-text-dim tabular-nums shrink-0">
          {filtered.length}
          {filtered.length !== logs.length ? `/${logs.length}` : ""}
        </span>

        <Button
          size="icon-sm"
          variant={autoScroll ? "primary" : "ghost"}
          onClick={jumpToBottom}
          title={autoScroll ? "Autoscroll on" : "Jump to bottom"}
        >
          <ArrowDownToLine className="w-3 h-3" />
        </Button>
        <Button
          size="icon-sm"
          variant="ghost"
          onClick={clear}
          disabled={logs.length === 0}
          title="Clear console"
        >
          <Trash2 className="w-3 h-3" />
        </Button>
      </div>

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="flex-1 min-h-0 overflow-auto font-mono text-xs leading-snug bg-bg-base"
      >
        {filtered.length === 0 ? (
          <div className="px-3 py-2 text-text-dim italic">
            {logs.length === 0 ? "No log output." : "No lines match the filter."}
          </div>
        ) : (
          filtered.map((entry, i) => (
            <div
              key={i}
              className={cn(
                "px-3 py-0.5 border-b border-bg-muted/30 whitespace-pre-wrap break-words",
                levelColor(entry.level)
              )}
            >
              <span className="text-text-dim">{formatTs(entry.ts_ms)}</span>{" "}
              <span className="uppercase">{shortLevel(entry.level)}</span>{" "}
              <span className="text-text-muted">{shortTarget(entry.target)}</span>{" "}
              <span>{entry.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

function matches(e: ConsoleLogMessage, needle: string): boolean {
  return (
    e.message.toLowerCase().includes(needle) ||
    e.target.toLowerCase().includes(needle)
  );
}

function levelColor(level: ConsoleLogLevel): string {
  switch (level) {
    case ConsoleLogLevel.Error:
      return "text-accent-red";
    case ConsoleLogLevel.Warn:
      return "text-accent-amber";
    case ConsoleLogLevel.Info:
      return "text-text-std";
    case ConsoleLogLevel.Debug:
    case ConsoleLogLevel.Trace:
    default:
      return "text-text-dim";
  }
}

function shortLevel(level: ConsoleLogLevel): string {
  switch (level) {
    case ConsoleLogLevel.Error:
      return "ERR";
    case ConsoleLogLevel.Warn:
      return "WRN";
    case ConsoleLogLevel.Info:
      return "INF";
    case ConsoleLogLevel.Debug:
      return "DBG";
    case ConsoleLogLevel.Trace:
      return "TRC";
  }
}

/** Drop the leading `dies_` and keep the last path segment to save width. */
function shortTarget(target: string): string {
  const last = target.split("::").pop() ?? target;
  return `[${last}]`;
}

function formatTs(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

export default ConsolePanel;
