import React, { useEffect, useMemo, useRef, useState } from "react";
import { IDockviewPanelProps } from "dockview";
import { Play, Square, RefreshCw, Trash2 } from "lucide-react";

import {
  useScenarios,
  useScenarioStatus,
  useScenarioLogs,
  useClearScenarioLogs,
  useSendCommand,
  useStatus,
  usePrimaryTeam,
} from "@/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { TestLogEntry, TestLogLevel, TestStatus } from "@/bindings";

/**
 * Scenario Panel — pick a JS scenario from `scenarios/` and run it against the
 * currently active executor (live or simulation). Shows scenario status and
 * the running log stream.
 */
const ScenarioPanel: React.FC<IDockviewPanelProps> = () => {
  const { data: backendState } = useStatus();
  const sendCommand = useSendCommand();
  const status = useScenarioStatus();
  const logs = useScenarioLogs();
  const clearLogs = useClearScenarioLogs();
  const scenariosQuery = useScenarios();
  const [primaryTeam] = usePrimaryTeam();
  const [selected, setSelected] = useState<string | null>(null);

  const isRunning = status.state === "Running" || status.state === "Starting";
  const executorRunning = backendState?.executor?.type === "RunningExecutor";

  const scenarios = scenariosQuery.data?.scenarios ?? [];
  const runningName =
    status.state === "Running" ? status.data.name : null;

  // Auto-select the first scenario once the list loads.
  useEffect(() => {
    if (selected === null && scenarios.length > 0) {
      setSelected(scenarios[0].name);
    }
  }, [scenarios, selected]);

  const start = () => {
    if (!selected) return;
    if (!executorRunning) return;
    sendCommand({
      type: "StartScenario",
      data: { scenario: selected, team: primaryTeam },
    });
  };

  const stop = () => {
    sendCommand({ type: "StopScenario" });
  };

  return (
    <div className="w-full h-full bg-bg-surface flex flex-col text-sm">
      <div className="flex items-center justify-between px-3 py-2 border-b border-bg-muted">
        <StatusBadge status={status} />
        <Button
          size="xs"
          variant="ghost"
          onClick={() => scenariosQuery.refetch()}
          title="Reload scenario list"
        >
          <RefreshCw className="w-3 h-3" />
        </Button>
      </div>

      <div className="flex-1 min-h-0 grid grid-cols-[minmax(180px,1fr)_2fr]">
        {/* Scenario list */}
        <div className="border-r border-bg-muted overflow-auto">
          {scenarios.length === 0 ? (
            <div className="p-3 text-text-dim italic">
              No scenarios found. Drop `.js` files into the `scenarios/`
              directory.
            </div>
          ) : (
            scenarios.map((s) => {
              const isSelected = selected === s.name;
              const isActive = runningName === s.name;
              return (
                <button
                  key={s.name}
                  onClick={() => setSelected(s.name)}
                  className={cn(
                    "w-full text-left px-3 py-1.5 border-b border-bg-muted/50 transition-colors",
                    "hover:bg-bg-overlay",
                    isSelected && "bg-bg-overlay",
                    isActive && "text-accent-cyan"
                  )}
                >
                  <span className="font-mono">{s.name}</span>
                </button>
              );
            })
          )}
        </div>

        {/* Right pane: actions + log */}
        <div className="flex flex-col min-h-0">
          <div className="flex items-center gap-2 px-3 py-2 border-b border-bg-muted">
            <Button
              size="sm"
              variant="success"
              onClick={start}
              disabled={!selected || isRunning || !executorRunning}
              title={
                !executorRunning
                  ? "Start the executor first"
                  : isRunning
                    ? "A scenario is already running"
                    : `Run ${selected ?? ""}`
              }
            >
              <Play className="w-3 h-3 mr-1.5 fill-current" />
              Run
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={stop}
              disabled={!isRunning}
            >
              <Square className="w-3 h-3 mr-1.5" />
              Stop
            </Button>
            <div className="flex-1" />
            <Button
              size="xs"
              variant="ghost"
              onClick={clearLogs}
              disabled={logs.length === 0}
              title="Clear log"
            >
              <Trash2 className="w-3 h-3" />
            </Button>
          </div>

          {!executorRunning && (
            <div className="px-3 py-2 text-text-dim border-b border-bg-muted">
              Executor is not running — start it from the toolbar to enable
              scenarios.
            </div>
          )}

          <LogView logs={logs} />
        </div>
      </div>
    </div>
  );
};

const StatusBadge: React.FC<{ status: TestStatus }> = ({ status }) => {
  const { label, color } = useMemo(() => describeStatus(status), [status]);
  return (
    <div className="flex items-center gap-2">
      <span
        className={cn(
          "w-2 h-2 rounded-full",
          color === "green" && "bg-accent-green animate-pulse",
          color === "amber" && "bg-accent-amber animate-pulse",
          color === "red" && "bg-accent-red",
          color === "cyan" && "bg-accent-cyan",
          color === "muted" && "bg-text-muted"
        )}
      />
      <span className="font-mono text-xs uppercase tracking-wide">{label}</span>
    </div>
  );
};

function describeStatus(status: TestStatus): {
  label: string;
  color: "green" | "amber" | "red" | "cyan" | "muted";
} {
  switch (status.state) {
    case "Idle":
      return { label: "Idle", color: "muted" };
    case "Starting":
      return { label: "Starting", color: "amber" };
    case "Running":
      return { label: `Running: ${status.data.name}`, color: "green" };
    case "Completed":
      return { label: "Completed", color: "cyan" };
    case "Failed":
      return { label: `Failed: ${status.data.error}`, color: "red" };
    case "Aborted":
      return { label: "Aborted", color: "red" };
  }
}

const LogView: React.FC<{ logs: TestLogEntry[] }> = ({ logs }) => {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [logs]);
  return (
    <div
      ref={ref}
      className="flex-1 min-h-0 overflow-auto font-mono text-xs leading-snug bg-bg-base"
    >
      {logs.length === 0 ? (
        <div className="px-3 py-2 text-text-dim italic">No log output.</div>
      ) : (
        logs.map((entry, i) => (
          <div
            key={i}
            className={cn(
              "px-3 py-0.5 border-b border-bg-muted/30",
              levelColor(entry.level)
            )}
          >
            <span className="text-text-dim">
              {formatTs(entry.ts_ms)}
            </span>{" "}
            <span className="uppercase">{entry.level}</span>{" "}
            {entry.tag && (
              <span className="text-text-muted">[{entry.tag}]</span>
            )}{" "}
            <span>{entry.message}</span>
            {entry.value_json && (
              <span className="text-text-dim"> {entry.value_json}</span>
            )}
          </div>
        ))
      )}
    </div>
  );
};

function levelColor(level: TestLogLevel): string {
  switch (level) {
    case TestLogLevel.Error:
      return "text-accent-red";
    case TestLogLevel.Warn:
      return "text-accent-amber";
    case TestLogLevel.Record:
      return "text-accent-cyan";
    case TestLogLevel.Info:
    default:
      return "text-text-std";
  }
}

function formatTs(ms: number): string {
  const d = new Date(ms);
  const pad = (n: number, w = 2) => String(n).padStart(w, "0");
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}

export default ScenarioPanel;
