import { FC } from "react";
import { useDebugData } from "@/api";
import { DebugValue } from "@/bindings";
import { cn } from "@/lib/utils";

/**
 * Pass Inspector — a dedicated right-sidebar mode that surfaces the full internal
 * state of every active pass coordinator. Reads the same debug stream as the
 * field overlays (keys `team_{Color}.pass.{passer}-{receiver}.*`).
 */

interface PassEntry {
  team: string;
  passer: number;
  receiver: number;
  fields: Record<string, DebugValue>;
}

const KEY_RE = /^team_(\w+)\.pass\.(\d+)-(\d+)\.(.+)$/;

const collectPasses = (debug: Record<string, DebugValue> | null): PassEntry[] => {
  if (!debug) return [];
  const byPair = new Map<string, PassEntry>();
  for (const [key, val] of Object.entries(debug)) {
    const m = KEY_RE.exec(key);
    if (!m) continue;
    const [, team, p, r, field] = m;
    // Skip the field-overlay shapes; this panel shows scalars/strings only.
    if (val.type === "Shape") continue;
    const id = `${team}:${p}-${r}`;
    let entry = byPair.get(id);
    if (!entry) {
      entry = { team, passer: Number(p), receiver: Number(r), fields: {} };
      byPair.set(id, entry);
    }
    entry.fields[field] = val;
  }
  return Array.from(byPair.values()).sort((a, b) => a.passer - b.passer);
};

const str = (v: DebugValue | undefined): string | null =>
  v && v.type === "String" ? (v.data as string) : null;
const num = (v: DebugValue | undefined): number | null =>
  v && v.type === "Number" ? (v.data as number) : null;

const Chip: FC<{ label: string; on: boolean | null }> = ({ label, on }) => (
  <span
    className={cn(
      "px-2 py-0.5 rounded text-xs font-medium border",
      on === null
        ? "border-border text-text-muted"
        : on
          ? "bg-green-500/20 border-green-500/50 text-green-300"
          : "bg-red-500/20 border-red-500/50 text-red-300"
    )}
  >
    {label}
  </span>
);

const Stat: FC<{ label: string; value: string }> = ({ label, value }) => (
  <div className="flex flex-col">
    <span className="text-[10px] uppercase tracking-wide text-text-muted">{label}</span>
    <span className="text-sm tabular-nums">{value}</span>
  </div>
);

const boolOf = (v: DebugValue | undefined): boolean | null => {
  const s = str(v);
  if (s === "true") return true;
  if (s === "false") return false;
  return null;
};

const PassCard: FC<{ entry: PassEntry }> = ({ entry }) => {
  const f = entry.fields;
  const phase = str(f["phase"]) ?? "?";
  const result = str(f["result"]);
  const isFailure =
    result && !["Success"].includes(result);

  const fmt = (k: string, digits = 0) => {
    const n = num(f[k]);
    return n === null ? "—" : n.toFixed(digits);
  };

  return (
    <div className="rounded-lg border border-border bg-bg-base p-3 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <div className="text-sm font-semibold">
          <span className="text-text-muted">{entry.team}</span> · p{entry.passer} →
          p{entry.receiver}
        </div>
        <div
          className={cn(
            "px-2 py-0.5 rounded text-xs font-bold",
            result
              ? isFailure
                ? "bg-red-500/25 text-red-300"
                : "bg-green-500/25 text-green-300"
              : "bg-blue-500/25 text-blue-300"
          )}
        >
          {result ? result : phase}
        </div>
      </div>

      {!result && (
        <div className="flex flex-wrap gap-1.5">
          <Chip label="has ball" on={boolOf(f["passer_has_ball"])} />
          <Chip label="aligned" on={boolOf(f["passer_aligned"])} />
          <Chip label="ready" on={boolOf(f["receiver_ready"])} />
          <Chip label="barrier" on={boolOf(f["barrier_satisfied"])} />
        </div>
      )}

      <div className="grid grid-cols-3 gap-2">
        <Stat label="phase" value={phase} />
        <Stat label="elapsed" value={`${fmt("phase_elapsed_s", 2)}s`} />
        <Stat label="total" value={`${fmt("total_elapsed_s", 2)}s`} />
        <Stat label="distance" value={`${fmt("pass_distance_mm")}mm`} />
        <Stat label="recv dist" value={`${fmt("receiver_dist_mm")}mm`} />
        <Stat label="head err" value={`${fmt("heading_error_rad", 3)}rad`} />
        <Stat label="kick" value={fmt("kick_speed")} />
        <Stat label="poss frames" value={fmt("possession_frames")} />
      </div>
    </div>
  );
};

const PassInspector: FC = () => {
  const debug = useDebugData();
  const passes = collectPasses(debug);

  return (
    <div className="w-full h-full overflow-y-auto p-3 flex flex-col gap-3">
      <div className="text-sm font-semibold text-text-muted">Pass coordinators</div>
      {passes.length === 0 ? (
        <div className="text-sm text-text-muted">No active passes.</div>
      ) : (
        passes.map((e) => (
          <PassCard key={`${e.team}-${e.passer}-${e.receiver}`} entry={e} />
        ))
      )}
    </div>
  );
};

export default PassInspector;
