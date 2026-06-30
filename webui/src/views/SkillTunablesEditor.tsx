import React from "react";
import { ChevronDown, RotateCcw } from "lucide-react";
import { Label } from "@/components/ui/label";
import { NumberInput } from "@/components/ui/number-input";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { useExecutorInfo, useExecutorSettings } from "@/api";
import { ExecutorSettings, TunableSpec } from "@/bindings";

/** A single knob: number input (+ optional slider), unit, help, reset-to-default. */
const TunableControl: React.FC<{
  spec: TunableSpec;
  value: number;
  overridden: boolean;
  onChange: (value: number) => void;
  onReset: () => void;
}> = ({ spec, value, overridden, onChange, onReset }) => {
  const label = (
    <Label htmlFor={spec.key} className="font-medium flex items-center gap-1.5">
      {overridden && (
        <span
          className="w-1.5 h-1.5 rounded-full bg-accent-cyan shrink-0"
          title="overridden (differs from code default)"
        />
      )}
      <span className="truncate">{spec.label}</span>
    </Label>
  );

  return (
    <div className="px-4 py-2 border-b border-bg-muted">
      <div className="flex items-center justify-between gap-3">
        {spec.help ? (
          <SimpleTooltip title={spec.help} className="w-fit cursor-help min-w-0">
            {label}
          </SimpleTooltip>
        ) : (
          label
        )}
        <div className="flex items-center gap-1.5 shrink-0">
          <NumberInput
            id={spec.key}
            value={value}
            onChange={onChange}
            className="w-24"
          />
          {spec.unit ? (
            <span className="text-xs text-text-muted w-10">{spec.unit}</span>
          ) : (
            <span className="w-10" />
          )}
          <button
            type="button"
            onClick={onReset}
            disabled={!overridden}
            title={`Reset to default (${spec.default})`}
            className="p-1 text-text-muted hover:text-text-base disabled:opacity-30 disabled:pointer-events-none"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};

/** One collapsible group of knobs sharing a `section`. */
const Section: React.FC<{
  title: string;
  specs: TunableSpec[];
  overrides: Record<string, number>;
  setOverrides: (next: Record<string, number>) => void;
}> = ({ title, specs, overrides, setOverrides }) => {
  const overriddenCount = specs.filter((s) => s.key in overrides).length;

  return (
    <Collapsible defaultOpen className="border-b border-bg-muted">
      <CollapsibleTrigger className="group w-full flex items-center justify-between px-4 py-1.5 bg-bg-muted text-xs uppercase tracking-wide text-text-muted hover:text-text-base">
        <span className="flex items-center gap-2">
          <ChevronDown className="w-3.5 h-3.5 transition-transform group-data-[state=closed]:-rotate-90" />
          {title}
        </span>
        {overriddenCount > 0 && (
          <span className="text-accent-cyan normal-case tracking-normal">
            {overriddenCount} changed
          </span>
        )}
      </CollapsibleTrigger>
      <CollapsibleContent>
        {specs.map((spec) => {
          const overridden = spec.key in overrides;
          const value = overridden ? overrides[spec.key] : spec.default;
          return (
            <TunableControl
              key={spec.key}
              spec={spec}
              value={value}
              overridden={overridden}
              onChange={(v) => setOverrides({ ...overrides, [spec.key]: v })}
              onReset={() => {
                const next = { ...overrides };
                delete next[spec.key];
                setOverrides(next);
              }}
            />
          );
        })}
      </CollapsibleContent>
    </Collapsible>
  );
};

/**
 * Skill tunables — runtime knobs declared in the executor's skills with the
 * `tunables!` macro. Metadata (label/help/range/unit/section) is auto-discovered
 * from `ExecutorInfo.skill_tunable_specs`; the current overrides live in
 * `ExecutorSettings.skill_tunables` and are written through the normal settings
 * POST, so they persist and ride the baseline/revert bar like every other setting.
 */
const SkillTunablesEditor: React.FC = () => {
  const info = useExecutorInfo();
  const { settings, updateSettings } = useExecutorSettings();

  if (!settings) {
    return <div className="p-4 text-sm text-text-muted">Loading…</div>;
  }
  const specs = info?.skill_tunable_specs ?? [];
  if (specs.length === 0) {
    return (
      <div className="p-4 text-sm text-text-muted">
        No skill tunables declared (is an executor running?).
      </div>
    );
  }

  const overrides = settings.skill_tunables ?? {};
  const setOverrides = (next: Record<string, number>) =>
    updateSettings({
      ...(settings as ExecutorSettings),
      skill_tunables: next,
    });

  // Group by section, preserving first-seen order.
  const order: string[] = [];
  const groups: Record<string, TunableSpec[]> = {};
  for (const spec of specs) {
    const section = spec.section ?? "Other";
    if (!(section in groups)) {
      groups[section] = [];
      order.push(section);
    }
    groups[section].push(spec);
  }

  return (
    <div className="h-full overflow-auto">
      {order.map((section) => (
        <Section
          key={section}
          title={section}
          specs={groups[section]}
          overrides={overrides}
          setOverrides={setOverrides}
        />
      ))}
    </div>
  );
};

export default SkillTunablesEditor;
