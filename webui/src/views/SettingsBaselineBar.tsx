import React, { useState } from "react";
import { useExecutorSettings, useSettingsSnapshots } from "@/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  diffSettings,
  revertField,
  revertSection,
  sectionLabel,
  fieldLabel,
  formatValue,
  SettingsDiffEntry,
} from "@/lib/settingsDiff";
import { toast } from "sonner";

const relativeTime = (ts: number): string => {
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
};

/**
 * Explore/revert bar shown above the settings tabs: shows how far the live
 * config has drifted from the known-good baseline, lets you mark a new
 * baseline, revert per-field / per-section / all, and restore any auto-snapshot.
 */
const SettingsBaselineBar: React.FC = () => {
  const { settings } = useExecutorSettings();
  const { baseline, history, markBaseline, restore } = useSettingsSnapshots();
  const [expanded, setExpanded] = useState(false);

  if (!settings) return null;

  const diff: SettingsDiffEntry[] = baseline
    ? diffSettings(settings, baseline.settings)
    : [];
  const dirty = diff.length > 0;

  // Group diff entries by section, preserving encounter order.
  const sections: { section: string; entries: SettingsDiffEntry[] }[] = [];
  for (const e of diff) {
    let g = sections.find((s) => s.section === e.section);
    if (!g) {
      g = { section: e.section, entries: [] };
      sections.push(g);
    }
    g.entries.push(e);
  }

  const handleMarkBaseline = () => {
    markBaseline();
    toast.success("Marked current settings as baseline");
  };

  const handleRevertAll = () => {
    if (!baseline) return;
    restore(baseline.settings);
    toast.success("Reverted all settings to baseline");
    setExpanded(false);
  };

  return (
    <div className="flex flex-col-reverse border-t border-border-muted bg-bg-surface text-xs shrink-0">
      <div className="flex items-center gap-1.5 px-2 py-1">
        <button
          type="button"
          className="flex items-center gap-2 min-w-0"
          onClick={() => dirty && setExpanded((v) => !v)}
        >
          {!baseline ? (
            <span className="text-text-muted">No baseline set</span>
          ) : dirty ? (
            <Badge variant="warning">
              {diff.length} changed {expanded ? "▾" : "▸"}
            </Badge>
          ) : (
            <span className="text-accent-green">Matches baseline ✓</span>
          )}
        </button>

        <div className="ml-auto flex items-center gap-1">
          <Button size="xs" variant="success" onClick={handleMarkBaseline}>
            Mark good
          </Button>
          <Button
            size="xs"
            variant="outline"
            disabled={!dirty}
            onClick={handleRevertAll}
          >
            Revert all
          </Button>
          <Popover>
            <PopoverTrigger asChild>
              <Button size="xs" variant="ghost">
                History
              </Button>
            </PopoverTrigger>
            <PopoverContent align="end" side="top" className="w-72 p-0">
              <div className="max-h-80 overflow-y-auto">
                {history.length === 0 ? (
                  <div className="px-3 py-2 text-text-muted">
                    No snapshots yet
                  </div>
                ) : (
                  history.map((snap, i) => (
                    <div
                      key={`${snap.ts}-${i}`}
                      className="flex items-center gap-2 px-3 py-1.5 border-b border-border-muted last:border-b-0"
                    >
                      <div className="flex flex-col min-w-0">
                        <span className="text-text-std">
                          {relativeTime(snap.ts)}
                        </span>
                        <span className="text-text-muted capitalize">
                          {snap.kind}
                        </span>
                      </div>
                      <Button
                        size="xs"
                        variant="outline"
                        className="ml-auto"
                        onClick={() => {
                          restore(snap.settings);
                          toast.success(`Restored snapshot (${snap.kind})`);
                        }}
                      >
                        Restore
                      </Button>
                    </div>
                  ))
                )}
              </div>
            </PopoverContent>
          </Popover>
        </div>
      </div>

      {expanded && dirty && baseline && (
        <div className="max-h-48 overflow-y-auto border-b border-border-muted px-2 py-1.5 space-y-2">
          {sections.map((g) => (
            <div key={g.section}>
              <div className="flex items-center gap-2 mb-1">
                <span className="font-medium text-text-bright">
                  {sectionLabel(g.section)}
                </span>
                <Button
                  size="xs"
                  variant="ghost"
                  className="ml-auto"
                  onClick={() =>
                    restore(revertSection(settings, baseline.settings, g.section))
                  }
                >
                  Revert section
                </Button>
              </div>
              <div className="space-y-0.5">
                {g.entries.map((e) => (
                  <div
                    key={e.field}
                    className="flex items-center gap-2 text-text-std"
                  >
                    <span className="min-w-0 flex-1 truncate">
                      {fieldLabel(e.field)}
                    </span>
                    <span className="text-text-muted tabular-nums">
                      {formatValue(e.baseline)}
                    </span>
                    <span className="text-text-muted">→</span>
                    <span className="text-accent-cyan tabular-nums">
                      {formatValue(e.current)}
                    </span>
                    <Button
                      size="icon-xs"
                      variant="ghost"
                      title="Revert this field to baseline"
                      onClick={() =>
                        restore(
                          revertField(
                            settings,
                            baseline.settings,
                            e.section,
                            e.field,
                          ),
                        )
                      }
                    >
                      ↺
                    </Button>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SettingsBaselineBar;
