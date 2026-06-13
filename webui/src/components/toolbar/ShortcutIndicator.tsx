import React, { useEffect, useState } from "react";
import { Keyboard } from "lucide-react";
import { useAtomValue, useSetAtom } from "jotai";
import { Button } from "@/components/ui/button";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { lastShortcutAtom, commandPaletteOpenAtom } from "@/api";

/**
 * VSCode-style keyboard indicator on the toolbar: clicking it opens the command
 * palette (the discovery surface for all commands + keybindings), and it shows a
 * transient flash of the most recently triggered command for quick feedback.
 */
const ShortcutIndicator: React.FC = () => {
  const last = useAtomValue(lastShortcutAtom);
  const setPaletteOpen = useSetAtom(commandPaletteOpenAtom);
  const [flash, setFlash] = useState<string | null>(null);

  useEffect(() => {
    if (!last) return;
    setFlash(last.label);
    const t = setTimeout(() => setFlash(null), 1300);
    return () => clearTimeout(t);
  }, [last?.ts]);

  return (
    <div className="flex items-center gap-1.5">
      {flash ? (
        <span className="text-xs text-accent-cyan font-mono whitespace-nowrap animate-in fade-in-0 slide-in-from-right-1 duration-150">
          {flash}
        </span>
      ) : null}

      <SimpleTooltip title="Command palette (⌘K)">
        <Button
          variant="ghost"
          size="icon-sm"
          className="text-text-dim hover:text-text-std"
          onClick={() => setPaletteOpen(true)}
        >
          <Keyboard className="w-4 h-4" />
        </Button>
      </SimpleTooltip>
    </div>
  );
};

export default ShortcutIndicator;
