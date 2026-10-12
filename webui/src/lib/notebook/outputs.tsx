/**
 * Renderers for notebook cell outputs: Observable Plot replays, tables, text,
 * JSON, errors.
 */
import React from "react";
import { cn } from "@/lib/utils";
import { PlotOutput } from "./plot";
import { CellOutput, ShowData } from "./store";

const MAX_TABLE_ROWS = 300;
const MAX_TABLE_COLS = 40;

function fmtCell(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    const a = Math.abs(v);
    return a !== 0 && (a < 1e-3 || a >= 1e6) ? v.toExponential(3) : v.toPrecision(4);
  }
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

const TableOutput: React.FC<{ rows: Record<string, unknown>[] }> = ({ rows }) => {
  const cols: string[] = [];
  const seen = new Set<string>();
  for (const r of rows.slice(0, 50)) {
    for (const k of Object.keys(r)) {
      if (!seen.has(k) && cols.length < MAX_TABLE_COLS) {
        seen.add(k);
        cols.push(k);
      }
    }
  }
  const shown = rows.slice(0, MAX_TABLE_ROWS);
  return (
    <div className="max-h-80 overflow-auto border border-border-subtle">
      <table className="text-xs font-mono border-collapse w-max min-w-full">
        <thead className="sticky top-0 bg-bg-elevated">
          <tr>
            {cols.map((c) => (
              <th key={c} className="text-left font-semibold text-text-dim px-2 py-1 border-b border-border-subtle">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {shown.map((r, i) => (
            <tr key={i} className="odd:bg-bg-base/40">
              {cols.map((c) => (
                <td
                  key={c}
                  className={cn("px-2 py-0.5 whitespace-nowrap", typeof r[c] === "number" && "text-right tabular-nums")}
                >
                  {fmtCell(r[c])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > shown.length && (
        <div className="text-xs text-text-dim px-2 py-1">… {rows.length - shown.length} more rows</div>
      )}
    </div>
  );
};

const ShowOutput: React.FC<{ data: ShowData }> = ({ data }) => {
  switch (data?.type) {
    case "plot":
      return <PlotOutput spec={data.spec} />;
    case "table":
      return <TableOutput rows={data.rows} />;
    case "text":
      return <pre className="text-xs text-text-std whitespace-pre-wrap">{data.text}</pre>;
    case "json":
      return (
        <pre className="text-xs text-text-std whitespace-pre-wrap max-h-80 overflow-auto">
          {JSON.stringify(data.value, null, 2)}
        </pre>
      );
    default:
      return <pre className="text-xs text-text-dim">{JSON.stringify(data)}</pre>;
  }
};

export const OutputView: React.FC<{ output: CellOutput }> = ({ output }) => {
  if (output.kind === "show") return <ShowOutput data={output.data} />;
  return (
    <pre
      className={cn(
        "text-xs whitespace-pre-wrap",
        output.kind === "error" ? "text-accent-red" : "text-accent-amber"
      )}
    >
      {output.text}
    </pre>
  );
};
