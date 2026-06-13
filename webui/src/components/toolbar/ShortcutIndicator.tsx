import React, { useEffect, useState } from "react";
import { Keyboard } from "lucide-react";
import { useAtomValue } from "jotai";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { lastShortcutAtom } from "@/api";
import { COMMANDS, EXTRA_SHORTCUTS } from "@/lib/commands";

/**
 * VSCode-style keyboard indicator on the toolbar: a small icon that opens a
 * cheat sheet of all shortcuts, plus a transient flash showing the most
 * recently triggered command for quick feedback.
 */
const ShortcutIndicator: React.FC = () => {
  const last = useAtomValue(lastShortcutAtom);
  const [flash, setFlash] = useState<string | null>(null);

  useEffect(() => {
    if (!last) return;
    setFlash(last.label);
    const t = setTimeout(() => setFlash(null), 1300);
    return () => clearTimeout(t);
  }, [last?.ts]);

  const rows = [
    ...COMMANDS.filter((c) => c.keys).map((c) => ({
      keys: c.keys!,
      title: c.title,
    })),
    ...EXTRA_SHORTCUTS,
  ];

  return (
    <div className="flex items-center gap-1.5">
      {flash ? (
        <span className="text-xs text-accent-cyan font-mono whitespace-nowrap animate-in fade-in-0 slide-in-from-right-1 duration-150">
          {flash}
        </span>
      ) : null}

      <Popover>
        <SimpleTooltip title="Keyboard shortcuts">
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size="icon-sm"
              className="text-text-dim hover:text-text-std"
            >
              <Keyboard className="w-4 h-4" />
            </Button>
          </PopoverTrigger>
        </SimpleTooltip>
        <PopoverContent align="end" className="w-72 p-2">
          <div className="text-[11px] uppercase tracking-wider text-text-dim mb-1.5 px-1">
            Keyboard Shortcuts
          </div>
          <div className="flex flex-col">
            {rows.map((r) => (
              <div
                key={r.keys + r.title}
                className="flex items-center justify-between gap-3 px-1 py-0.5 text-sm"
              >
                <span className="text-text-std">{r.title}</span>
                <kbd className="font-mono text-xs text-text-bright bg-bg-base border border-border-muted px-1.5 py-0.5 whitespace-nowrap">
                  {r.keys}
                </kbd>
              </div>
            ))}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  );
};

export default ShortcutIndicator;
