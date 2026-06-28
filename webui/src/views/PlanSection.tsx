import { FC } from "react";
import { useDebugData, usePrimaryTeam } from "@/api";
import { PlanData, PlanStep, TeamColor } from "@/bindings";
import { cn } from "@/lib/utils";

/**
 * Compact plan readout, shown at the top of the Inspector above the team list
 * whenever the active team has a plan. The strategy emits its plan (at most a
 * couple of steps) as the `DebugValue::Plan` primitive under
 * `team_{Color}.strategy.plan`. Renders nothing when there is no plan.
 */

/** Accent color per step kind. */
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

const StepRow: FC<{ step: PlanStep; index: number }> = ({ step, index }) => (
  <div
    className={cn(
      "flex items-baseline gap-1.5 px-2 py-0.5 border-l-2",
      step.active
        ? "border-accent-cyan bg-bg-overlay"
        : "border-transparent opacity-70"
    )}
  >
    <span className="font-mono text-[10px] text-text-muted w-3 shrink-0">
      {index + 1}
    </span>
    <span className={cn("text-xs font-medium shrink-0", kindColor(step.kind))}>
      {step.label}
    </span>
    {step.detail && (
      <span className="text-[10px] text-text-dim font-mono truncate min-w-0">
        {step.detail}
      </span>
    )}
    {step.active && (
      <span className="ml-auto text-[9px] uppercase tracking-wide text-accent-cyan shrink-0">
        active
      </span>
    )}
  </div>
);

const PlanSection: FC = () => {
  const debugData = useDebugData();
  const [primaryTeam] = usePrimaryTeam();
  const teamStr = primaryTeam === TeamColor.Blue ? "Blue" : "Yellow";

  const entry = debugData?.[`team_${teamStr}.strategy.plan`];
  const plan: PlanData | null =
    entry?.type === "Plan" ? (entry.data as PlanData) : null;

  // Nothing to show — keep the inspector clean when the strategy is idle.
  if (!plan || plan.steps.length === 0) return null;

  return (
    <div className="shrink-0 border-b border-border bg-bg-surface">
      <div className="flex items-center justify-between px-2 py-1 text-[11px] uppercase tracking-wider text-text-dim">
        <span>plan</span>
        {plan.active_robot !== undefined && plan.active_robot !== null && (
          <span className="font-mono normal-case text-text-std">
            p{plan.active_robot}
          </span>
        )}
      </div>
      <div className="flex flex-col pb-1">
        {plan.steps.map((step, i) => (
          <StepRow key={i} step={step} index={i} />
        ))}
      </div>
    </div>
  );
};

export default PlanSection;
