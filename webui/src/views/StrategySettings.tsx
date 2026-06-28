import React from "react";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { useExecutorInfo, useSendCommand } from "@/api";
import {
  ParamKind,
  ParamSpec,
  ParamValue,
  TeamColor,
  TeamStrategyParams,
} from "@/bindings";

// Read helpers for the externally-tagged `ParamValue` union.
const asBool = (v?: ParamValue): boolean | undefined =>
  v && "Bool" in v ? v.Bool : undefined;
const asNumber = (v?: ParamValue): number | undefined =>
  v && "Float" in v ? v.Float : v && "Int" in v ? v.Int : undefined;
const asText = (v?: ParamValue): string | undefined =>
  v && "Text" in v ? v.Text : undefined;

/** A single declared parameter rendered as the control matching its kind. */
const ParamControl: React.FC<{
  team: TeamColor;
  spec: ParamSpec;
  value: ParamValue | undefined;
}> = ({ team, spec, value }) => {
  const send = useSendCommand();
  const current = value ?? spec.default;
  const setValue = (next: ParamValue) =>
    send({
      type: "SetStrategyParam",
      data: { team_color: team, key: spec.key, value: next },
    });

  const id = `${team}-${spec.key}`;
  return (
    <div className="flex items-center justify-between gap-3 px-4 py-2 border-b border-bg-muted">
      <Label htmlFor={id} className="font-medium">
        {spec.label}
      </Label>
      {spec.kind === ParamKind.Bool && (
        <Switch
          id={id}
          checked={asBool(current) ?? false}
          onCheckedChange={(checked) => setValue({ Bool: checked })}
        />
      )}
      {spec.kind === ParamKind.Float && (
        <Input
          id={id}
          type="number"
          className="w-28"
          value={asNumber(current) ?? 0}
          onChange={(e) => setValue({ Float: Number(e.target.value) })}
        />
      )}
      {spec.kind === ParamKind.Int && (
        <Input
          id={id}
          type="number"
          step={1}
          className="w-28"
          value={asNumber(current) ?? 0}
          onChange={(e) => setValue({ Int: Math.round(Number(e.target.value)) })}
        />
      )}
      {spec.kind === ParamKind.Text && (
        <Input
          id={id}
          type="text"
          className="w-40"
          value={asText(current) ?? ""}
          onChange={(e) => setValue({ Text: e.target.value })}
        />
      )}
    </div>
  );
};

/** All declared parameters for one team's strategy. */
const TeamParams: React.FC<{ tsp: TeamStrategyParams }> = ({ tsp }) => (
  <div className="flex flex-col">
    <div className="px-4 py-1.5 text-xs uppercase tracking-wide text-text-muted bg-bg-muted">
      {tsp.team} strategy
    </div>
    {tsp.specs.length === 0 ? (
      <div className="px-4 py-2 text-sm text-text-muted">
        No parameters declared.
      </div>
    ) : (
      tsp.specs.map((spec) => (
        <ParamControl
          key={spec.key}
          team={tsp.team}
          spec={spec}
          value={tsp.values[spec.key]}
        />
      ))
    )}
  </div>
);

/**
 * Strategy settings — runtime parameter controls declared by each active
 * strategy (auto-discovered via `ExecutorInfo.strategy_params`). Changes are
 * pushed live to the running strategy process. Rendered as a tab inside the
 * Settings panel.
 */
const StrategySettings: React.FC = () => {
  const info = useExecutorInfo();
  const params = info?.strategy_params ?? [];
  return (
    <div className="h-full overflow-auto">
      {params.length === 0 ? (
        <div className="p-4 text-sm text-text-muted">
          No active strategy, or it declares no parameters.
        </div>
      ) : (
        params.map((tsp) => <TeamParams key={tsp.team} tsp={tsp} />)
      )}
    </div>
  );
};

export default StrategySettings;
