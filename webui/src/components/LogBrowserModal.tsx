import { useMemo, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { useLogs, useSendCommand } from "../api";
import { LogInfo } from "../bindings";

function fmtTime(unix: number): string {
  return new Date(unix * 1000).toLocaleString();
}

function fmtDuration(s: number | undefined): string {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  const sec = Math.round(s % 60);
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

/** Centered modal listing recorded logs with metadata + quick search. */
export function LogBrowserModal({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const logs = useLogs(open);
  const sendCommand = useSendCommand();
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const needle = query.toLowerCase();
    return logs.filter(
      (l) =>
        l.name.toLowerCase().includes(needle) ||
        (l.blue_strategy ?? "").toLowerCase().includes(needle) ||
        (l.yellow_strategy ?? "").toLowerCase().includes(needle)
    );
  }, [logs, query]);

  const load = (l: LogInfo) => {
    sendCommand({ type: "LoadLog", data: { path: l.path } });
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle>Load recorded log</DialogTitle>
        </DialogHeader>
        <Input
          autoFocus
          placeholder="Search by name or strategy…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="mt-2 max-h-96 divide-y divide-border-subtle overflow-y-auto">
          {filtered.length === 0 ? (
            <div className="py-6 text-center text-sm text-text-dim">
              No logs found
            </div>
          ) : (
            filtered.map((l) => (
              <button
                key={l.path}
                onClick={() => load(l)}
                className="block w-full px-2 py-2 text-left hover:bg-bg-overlay"
              >
                <div className="font-mono text-sm text-text-std">
                  {l.name}
                  {l.is_zip ? " (zip)" : ""}
                </div>
                <div className="text-[11px] text-text-dim">
                  {fmtTime(l.session_start_unix)} ·{" "}
                  {fmtDuration(l.duration_s ?? undefined)} ·{" "}
                  {l.frame_count ?? "?"} frames · {l.is_simulation ? "sim" : "live"}
                  {l.blue_strategy ? ` · ${l.blue_strategy}` : ""}
                </div>
              </button>
            ))
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
