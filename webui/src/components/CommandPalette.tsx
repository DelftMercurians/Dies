import React, { useEffect, useMemo, useRef, useState } from "react";
import { useAtom } from "jotai";
import { DockviewApi } from "dockview";
import { commandPaletteOpenAtom } from "@/api";
import { useCommandContext } from "@/lib/useGlobalShortcuts";
import {
  buildPaletteEntries,
  EXTRA_SHORTCUTS,
  PaletteEntry,
} from "@/lib/commands";
import { cn } from "@/lib/utils";

/**
 * macOS-/VSCode-style command palette (⌘K). Reuses the shared command registry
 * and {@link CommandContext}, so every keyboard shortcut is also runnable here,
 * plus dynamic entries (robots, panels). Fuzzy filter + arrow/enter navigation.
 */

/** Subsequence fuzzy score; returns -1 for no match (higher = better). */
const fuzzyScore = (query: string, text: string): number => {
  if (!query) return 0;
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  const idx = t.indexOf(q);
  if (idx !== -1) return 1000 - idx; // contiguous match, prefer early
  // subsequence
  let qi = 0;
  let score = 0;
  let lastHit = -1;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      score += lastHit === ti - 1 ? 5 : 1; // reward adjacency
      lastHit = ti;
      qi++;
    }
  }
  return qi === q.length ? score : -1;
};

const CommandPalette: React.FC<{
  getDockviewApi: () => DockviewApi | null;
}> = ({ getDockviewApi }) => {
  const [open, setOpen] = useAtom(commandPaletteOpenAtom);
  const ctx = useCommandContext(getDockviewApi);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState(0);

  // Rebuild entries each time the palette opens (captures current state).
  const entries = useMemo(
    () => (open ? buildPaletteEntries(ctx) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [open, ctx.selectedPlayerId, ctx.ownPlayerIds.join(",")]
  );

  const filtered = useMemo(() => {
    const scored = entries
      .map((e) => ({ e, s: fuzzyScore(query, e.title) }))
      .filter((x) => x.s >= 0);
    scored.sort((a, b) => b.s - a.s);
    return scored.map((x) => x.e);
  }, [entries, query]);

  // Reset on open; focus input.
  useEffect(() => {
    if (open) {
      setQuery("");
      setIndex(0);
      // Defer focus until the input is mounted.
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Keep selection in range and scrolled into view.
  useEffect(() => {
    if (index >= filtered.length) setIndex(Math.max(0, filtered.length - 1));
  }, [filtered.length, index]);

  useEffect(() => {
    const el = listRef.current?.querySelector(`[data-idx="${index}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [index]);

  if (!open) return null;

  const run = (entry: PaletteEntry | undefined) => {
    if (!entry) return;
    setOpen(false);
    entry.run();
  };

  const onKeyDown = (ev: React.KeyboardEvent) => {
    if (ev.key === "Escape") {
      ev.preventDefault();
      setOpen(false);
    } else if (ev.key === "ArrowDown") {
      ev.preventDefault();
      setIndex((i) => Math.min(i + 1, filtered.length - 1));
    } else if (ev.key === "ArrowUp") {
      ev.preventDefault();
      setIndex((i) => Math.max(i - 1, 0));
    } else if (ev.key === "Enter") {
      ev.preventDefault();
      run(filtered[index]);
    }
  };

  // Flat list with section headers; index maps to filtered[] order.
  let lastSection = "";

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[12vh] bg-black/40"
      onMouseDown={() => setOpen(false)}
    >
      <div
        className="w-[560px] max-w-[92vw] bg-bg-elevated border border-border-std shadow-2xl flex flex-col overflow-hidden"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setIndex(0);
          }}
          onKeyDown={onKeyDown}
          placeholder="Type a command, robot, or panel…"
          className="w-full bg-transparent px-3 py-2.5 text-sm text-text-bright placeholder:text-text-muted outline-none border-b border-border-subtle"
        />
        <div ref={listRef} className="max-h-[50vh] overflow-y-auto py-1">
          {filtered.length === 0 ? (
            <div className="px-3 py-4 text-sm text-text-dim text-center">
              No matching commands
            </div>
          ) : (
            filtered.map((entry, i) => {
              const showHeader = entry.section !== lastSection;
              lastSection = entry.section;
              return (
                <React.Fragment key={entry.id}>
                  {showHeader ? (
                    <div className="px-3 pt-2 pb-0.5 text-[10px] uppercase tracking-wider text-text-muted">
                      {entry.section}
                    </div>
                  ) : null}
                  <button
                    data-idx={i}
                    onMouseEnter={() => setIndex(i)}
                    onClick={() => run(entry)}
                    className={cn(
                      "w-full flex items-center justify-between gap-3 px-3 py-1.5 text-left text-sm",
                      i === index
                        ? "bg-accent-cyan/15 text-text-bright"
                        : "text-text-std hover:bg-bg-overlay"
                    )}
                  >
                    <span className="truncate">{entry.title}</span>
                    {entry.keys ? (
                      <kbd className="font-mono text-xs text-text-dim bg-bg-base border border-border-muted px-1.5 py-0.5 whitespace-nowrap shrink-0">
                        {entry.keys}
                      </kbd>
                    ) : null}
                  </button>
                </React.Fragment>
              );
            })
          )}
        </div>

        {/* Footer: non-runnable shortcut hints */}
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-3 py-1.5 border-t border-border-subtle text-[11px] text-text-muted">
          {EXTRA_SHORTCUTS.map((s) => (
            <span key={s.keys + s.title} className="flex items-center gap-1">
              <kbd className="font-mono text-text-dim bg-bg-base border border-border-muted px-1 py-0.5">
                {s.keys}
              </kbd>
              {s.title}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CommandPalette;
