import React from "react";
import { IDockviewPanelProps } from "dockview";
import { useDebugData, usePrimaryTeam } from "@/api";
import { PlanData, PlanStep, TeamColor } from "@/bindings";
import { cn } from "@/lib/utils";

/**
 * Plan Panel — renders the active team's current strategy plan (the
 * `DebugValue::Plan` primitive emitted by the strategy under
 * `team_{Color}.strategy.plan`). Shows the driving robot and an ordered list of
 * steps, with the active step highlighted.
 */

/** Accent color per waypoint kind. */
const kindColor = (kind: string): string => {
  switch (kind) {
    case "Shoot":
      return "text-accent-red";
    case "Pass":
      return "text-accent-cyan";
    case "Capture":
      return "text-accent-amber";
    case "Dribble":
      return "text-accent-green";
    default:
      return "text-accent-blue";
  }
};

const StepRow: React.FC<{ step: PlanStep; index: number }> = ({
  step,
  index,
}) => (
  <div
    className={cn(
      "flex items-start gap-2 px-3 py-2 border-l-2",
      step.active
        ? "border-accent-cyan bg-bg-overlay"
        : "border-transparent opacity-70"
    )}
  >
    <span className="font-mono text-[10px] text-text-muted mt-0.5 w-4 shrink-0">
      {index + 1}
    </span>
    <div className="flex flex-col min-w-0">
      <span className={cn("text-sm font-medium", kindColor(step.kind))}>
        {step.label}
        {step.active && (
          <span className="ml-2 text-[10px] uppercase tracking-wide text-accent-cyan">
            active
          </span>
        )}
      </span>
      {step.detail && (
        <span className="text-xs text-text-dim font-mono">{step.detail}</span>
      )}
    </div>
  </div>
);

const PlanPanel: React.FC<IDockviewPanelProps> = () => {
  const debugData = useDebugData();
  const [primaryTeam] = usePrimaryTeam();
  const teamStr = primaryTeam === TeamColor.Blue ? "Blue" : "Yellow";

  const entry = debugData?.[`team_${teamStr}.strategy.plan`];
  const plan: PlanData | null =
    entry?.type === "Plan" ? (entry.data as PlanData) : null;

  return (
    <div className="w-full h-full bg-bg-surface flex flex-col overflow-auto">
      <div className="flex items-center justify-between px-3 py-1.5 text-[11px] uppercase tracking-wider text-text-dim border-b border-border-subtle">
        <span>{primaryTeam} plan</span>
        {plan?.active_robot !== undefined && plan?.active_robot !== null && (
          <span className="font-mono normal-case text-text-std">
            active: p{plan.active_robot}
          </span>
        )}
      </div>
      {plan && plan.steps.length > 0 ? (
        <div className="flex flex-col divide-y divide-border-subtle">
          {plan.steps.map((step, i) => (
            <StepRow key={i} step={step} index={i} />
          ))}
        </div>
      ) : (
        <div className="p-4 text-sm text-text-muted">
          No active plan — the strategy is defending or idle.
        </div>
      )}
    </div>
  );
};

export default PlanPanel;
