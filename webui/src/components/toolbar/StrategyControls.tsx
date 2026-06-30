import React, { FC, useEffect, useState } from "react";
import { FlaskConical, Play, Square, RotateCw } from "lucide-react";
import { toast } from "sonner";
import {
  useStrategies,
  useExecutorSettings,
  useSendCommand,
  useStatus,
} from "@/api";
import { SideAssignment, TeamConfiguration } from "@/bindings";
import { Button } from "@/components/ui/button";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from "@/components/ui/dialog";
import SnapshotManager from "@/components/SnapshotManager";

const NONE = "";

const TeamPicker: FC<{
  label: string;
  value: string;
  onChange: (v: string) => void;
  strategies: string[];
  scenarios: string[];
}> = ({ label, value, onChange, strategies, scenarios }) => (
  <div className="flex flex-col gap-1">
    <label className="text-xs text-text-muted">{label}</label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="h-8 px-2 text-sm bg-bg-overlay border border-border-muted rounded text-text-std focus:outline-none focus:ring-1 focus:ring-accent-cyan"
    >
      <option value={NONE}>(none)</option>
      {strategies.length > 0 && (
        <optgroup label="Strategies">
          {strategies.map((s) => (
            <option key={`s-${s}`} value={s}>
              {s}
            </option>
          ))}
        </optgroup>
      )}
      {scenarios.length > 0 && (
        <optgroup label="Scenarios">
          {scenarios.map((s) => (
            <option key={`c-${s}`} value={s}>
              {s}
            </option>
          ))}
        </optgroup>
      )}
    </select>
  </div>
);

/**
 * Strategy / scenario picker, opened as a modal from the toolbar. Assign any
 * strategy (concerto, …) or skill-test scenario (handle_ball, …) to
 * each team, then Run. Both are plain Strategy binaries launched over the strategy
 * IPC — scenarios are just strategies that script a robot through skills.
 *
 * Restart-per-run: Run writes the team configuration and restarts the executor
 * (which resets the simulated world), so use Field setup to seed a snapshot after
 * Run when you need a specific starting situation.
 */
const StrategyControls: React.FC = () => {
  const { strategies, scenarios } = useStrategies();
  const { settings } = useExecutorSettings();
  const sendCommand = useSendCommand();
  const status = useStatus();

  const isRunning = status.data?.executor?.type === "RunningExecutor";
  const tc = settings?.team_configuration;

  const [open, setOpen] = useState(false);
  const [blue, setBlue] = useState<string>(NONE);
  const [yellow, setYellow] = useState<string>(NONE);

  // Sync the dropdowns to the live configuration whenever settings load/change.
  useEffect(() => {
    if (tc) {
      setBlue(tc.blue_strategy ?? NONE);
      setYellow(tc.yellow_strategy ?? NONE);
    }
  }, [tc?.blue_strategy, tc?.yellow_strategy]);

  const run = () => {
    if (!blue && !yellow) {
      toast.error("Pick a strategy or scenario for at least one team");
      return;
    }
    const configuration: TeamConfiguration = {
      blue_active: blue !== NONE,
      yellow_active: yellow !== NONE,
      side_assignment: tc?.side_assignment ?? SideAssignment.BlueOnPositive,
      blue_strategy: blue || undefined,
      yellow_strategy: yellow || undefined,
    };
    // One channel, in order: persist the config, then restart the executor so it
    // rebuilds with the new strategies. SetTeamConfiguration is a no-op on a
    // running executor but always updates the settings the next Start reads.
    sendCommand({ type: "SetTeamConfiguration", data: { configuration } });
    if (isRunning) sendCommand({ type: "Stop" });
    sendCommand({ type: "Start" });
    toast.success(isRunning ? "Restarting with new strategies" : "Starting");
    // Close the picker on run — the field/inspector is what you want to watch.
    setOpen(false);
  };

  const stop = () => sendCommand({ type: "Stop" });

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <SimpleTooltip title="Strategies & scenarios — assign per team and run" className="h-full">
        <DialogTrigger asChild>
          <button
            className={cn(
              "h-7 w-7 flex items-center justify-center border border-border-muted transition-colors",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
              isRunning
                ? "text-accent-green hover:bg-bg-overlay"
                : "text-text-muted hover:text-text-std hover:bg-bg-overlay"
            )}
          >
            <FlaskConical className="w-3.5 h-3.5" />
          </button>
        </DialogTrigger>
      </SimpleTooltip>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Strategies & scenarios</DialogTitle>
          <DialogDescription>
            Assign a strategy or skill-test scenario to each team, then run. Step
            results are logged to the Console panel (✓/✗).
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">Assignment</span>
            <span
              className={cn(
                "text-xs",
                isRunning ? "text-accent-green" : "text-text-dim"
              )}
            >
              {isRunning ? "running" : "stopped"}
            </span>
          </div>

          <TeamPicker
            label="Blue team"
            value={blue}
            onChange={setBlue}
            strategies={strategies}
            scenarios={scenarios}
          />
          <TeamPicker
            label="Yellow team"
            value={yellow}
            onChange={setYellow}
            strategies={strategies}
            scenarios={scenarios}
          />

          <div className="flex gap-2">
            <Button onClick={run} className="flex-1 h-8 gap-1.5 text-xs">
              {isRunning ? <RotateCw size={14} /> : <Play size={14} />}
              {isRunning ? "Restart" : "Run"}
            </Button>
            <Button
              onClick={stop}
              disabled={!isRunning}
              variant="outline"
              className="h-8 gap-1.5 text-xs"
            >
              <Square size={14} /> Stop
            </Button>
          </div>

          <p className="text-xs text-text-dim leading-snug">
            Run restarts the executor and resets the field. Seed a snapshot below
            to set up a situation, then watch the scenario loop.
          </p>

          <div className="border-t border-border-muted pt-3">
            <div className="text-xs text-text-muted mb-2">Field setup</div>
            <SnapshotManager />
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default StrategyControls;
