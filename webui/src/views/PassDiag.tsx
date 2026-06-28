import { FC } from "react";
import { DebugMap, DebugValue, TeamColor } from "@/bindings";
import { cn } from "@/lib/utils";

/**
 * Pass diagnostics, surfaced inline in the team overview instead of a separate
 * panel. The pass coordinator emits its full internal state under
 * `team_{Color}.pass.{passer}-{receiver}.*` (the same stream that draws the
 * field overlays). We fold that into the player list: the passer's row shows
 * the full readout, the receiver's row shows the receive-side subset.
 */

export interface PassDiag {
  passer: number;
  receiver: number;
  fields: Record<string, DebugValue>;
}

const KEY_RE = /^team_(\w+)\.pass\.(\d+)-(\d+)\.(.+)$/;

/** Collect every active pass for `team`, keyed by passer→receiver pair. */
export const collectTeamPasses = (
  debug: DebugMap | null,
  team: TeamColor
): PassDiag[] => {
  if (!debug) return [];
  const teamStr = team === TeamColor.Blue ? "Blue" : "Yellow";
  const byPair = new Map<string, PassDiag>();
  for (const [key, val] of Object.entries(debug)) {
    const m = KEY_RE.exec(key);
    if (!m) continue;
    const [, t, p, r, field] = m;
    if (t !== teamStr) continue;
    // Field overlays are Shapes; this view shows scalars/strings only.
    if (val.type === "Shape") continue;
    const id = `${p}-${r}`;
    let entry = byPair.get(id);
    if (!entry) {
      entry = { passer: Number(p), receiver: Number(r), fields: {} };
      byPair.set(id, entry);
    }
    entry.fields[field] = val;
  }
  return Array.from(byPair.values());
};

const str = (v: DebugValue | undefined): string | null =>
  v && v.type === "String" ? (v.data as string) : null;
const num = (v: DebugValue | undefined): number | null =>
  v && v.type === "Number" ? (v.data as number) : null;
const boolOf = (v: DebugValue | undefined): boolean | null => {
  const s = str(v);
  if (s === "true") return true;
  if (s === "false") return false;
  return null;
};

const fmt = (
  f: Record<string, DebugValue>,
  k: string,
  digits = 0
): string => {
  const n = num(f[k]);
  return n === null ? "—" : n.toFixed(digits);
};

/** Tiny on/off pill, matching the dense overview aesthetic. */
const Chip: FC<{ label: string; on: boolean | null }> = ({ label, on }) => (
  <span
    className={cn(
      "px-1 py-0 rounded text-[10px] font-medium border",
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

/** Colored phase/result badge. */
const PhaseBadge: FC<{ diag: PassDiag }> = ({ diag }) => {
  const phase = str(diag.fields["phase"]) ?? "?";
  const result = str(diag.fields["result"]);
  const isFailure = result !== null && result !== "Success";
  return (
    <span
      className={cn(
        "px-1 py-0 rounded text-[10px] font-bold",
        result
          ? isFailure
            ? "bg-red-500/25 text-red-300"
            : "bg-green-500/25 text-green-300"
          : "bg-blue-500/25 text-blue-300"
      )}
    >
      {result ?? phase}
    </span>
  );
};

/**
 * Passer-side readout: phase/result, the four commit-gate chips, and the key
 * scalars (distance, heading error, kick speed, elapsed).
 */
export const PasserStrip: FC<{ diag: PassDiag }> = ({ diag }) => {
  const f = diag.fields;
  const result = str(f["result"]);
  return (
    <div className="flex flex-col gap-0.5 rounded bg-bg-overlay/40 px-1.5 py-1">
      <div className="flex items-center gap-1.5 text-[10px]">
        <span className="text-text-muted">pass →</span>
        <span className="font-mono text-text-bright">p{diag.receiver}</span>
        <PhaseBadge diag={diag} />
        <span className="ml-auto text-text-dim font-mono">
          {fmt(f, "total_elapsed_s", 1)}s
        </span>
      </div>
      {!result && (
        <div className="flex flex-wrap gap-1">
          <Chip label="ball" on={boolOf(f["passer_has_ball"])} />
          <Chip label="aligned" on={boolOf(f["passer_aligned"])} />
          <Chip label="ready" on={boolOf(f["receiver_ready"])} />
          <Chip label="barrier" on={boolOf(f["barrier_satisfied"])} />
        </div>
      )}
      <div className="flex flex-wrap gap-x-3 text-[10px] font-mono text-text-dim">
        <span>dist {fmt(f, "pass_distance_mm")}mm</span>
        <span>head {fmt(f, "heading_error_rad", 2)}rad</span>
        <span>kick {fmt(f, "kick_speed")}</span>
      </div>
    </div>
  );
};

/**
 * Receiver-side readout: who's passing, phase, ready chip and distance to the
 * intercept point.
 */
export const ReceiverStrip: FC<{ diag: PassDiag }> = ({ diag }) => {
  const f = diag.fields;
  return (
    <div className="flex items-center gap-1.5 rounded bg-bg-overlay/40 px-1.5 py-1 text-[10px]">
      <span className="text-text-muted">recv ←</span>
      <span className="font-mono text-text-bright">p{diag.passer}</span>
      <PhaseBadge diag={diag} />
      <Chip label="ready" on={boolOf(f["receiver_ready"])} />
      <span className="ml-auto font-mono text-text-dim">
        dist {fmt(f, "receiver_dist_mm")}mm
      </span>
    </div>
  );
};
