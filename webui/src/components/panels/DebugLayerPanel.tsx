import React, { useMemo, useCallback } from "react";
import { IDockviewPanelProps } from "dockview";
import { useAtom } from "jotai";
import { useDebugData } from "@/api";
import { pinnedFieldKeysAtom } from "@/lib/pinnedDebug";
import HierarchicalList from "@/views/HierarchicalList";

/** True for keys that belong to a specific player (e.g. `team_Blue.p3.role`). */
const isPlayerKey = (key: string): boolean =>
  /^team_(Blue|Yellow)\.p\d+(\.|$)/.test(key);

/**
 * Debug panel — lists global + team-level debug values. Each value can be
 * pinned to the floating field overlay. The category/layer visibility controls
 * now live in the field's view-settings popover.
 */
const DebugLayerPanel: React.FC<IDockviewPanelProps> = () => {
  const debugMap = useDebugData();
  const [pinnedKeys, setPinnedKeys] = useAtom(pinnedFieldKeysAtom);

  // Non-player, non-shape tag values (global + team-level scalars/strings).
  // Player tags are shown in the right-hand Inspector instead.
  const nonPlayerValues = useMemo(
    () =>
      debugMap
        ? Object.entries(debugMap).filter(
            ([key, val]) => !isPlayerKey(key) && val.type !== "Shape",
          )
        : [],
    [debugMap],
  );

  const togglePin = useCallback(
    (key: string) =>
      setPinnedKeys((prev) =>
        prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
      ),
    [setPinnedKeys],
  );

  return (
    <div className="w-full h-full bg-bg-surface flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border-dim shrink-0">
        <span className="text-sm font-semibold text-text-std">Debug Values</span>
      </div>

      <div className="flex-1 overflow-auto px-1 py-1">
        {nonPlayerValues.length === 0 ? (
          <div className="px-2 py-1 text-sm text-text-dim">No values</div>
        ) : (
          <HierarchicalList
            data={nonPlayerValues}
            className="p-1 text-sm"
            pinnedKeys={pinnedKeys}
            onTogglePin={togglePin}
          />
        )}
      </div>
    </div>
  );
};

export default DebugLayerPanel;
