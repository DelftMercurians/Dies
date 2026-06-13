import React from "react";
import { useExecutorSettings } from "@/api";
import { AvoidanceConfig } from "@/bindings";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

/**
 * Dedicated editor for the collision-avoidance tuning
 * (`ExecutorSettings.avoidance`): the shared obstacle margins, the global
 * planner, and ORCA. Writes whole-`AvoidanceConfig` updates back through the
 * standard settings mutation so edits live-update the running executor.
 */

// Mirrors `AvoidanceConfig::default()` — used to fill in fields if the backend
// sends a partial object (serde marks the whole struct optional).
const DEFAULTS: AvoidanceConfig = {
  robot_clearance: 50.0,
  wall_margin: 120.0,
  defense_margin: 50.0,
  ball_stop_radius: 800.0,
  ball_base_radius: 100.0,
  ball_care_scale: 100.0,
  robot_extrapolation: 0.5,
  time_horizon: 1.0,
  stationary_speed: 50.0,
  prefer_steering: false,
  neighbor_dist: 2000.0,
  max_neighbors: 8,
  grid_resolution: 100.0,
  planner_margin: 100.0,
  waypoint_tolerance: 200.0,
  replan_target_tol: 150.0,
  planner_enabled: true,
  orca_enabled: true,
};

interface NumberFieldProps {
  id: string;
  label: string;
  value: number;
  onChange: (value: number) => void;
  unit?: string;
  hint?: string;
}

const NumberField: React.FC<NumberFieldProps> = ({
  id,
  label,
  value,
  onChange,
  unit,
  hint,
}) => (
  <div className="space-y-1">
    <div className="flex items-center justify-between gap-2">
      <Label htmlFor={id} className="font-medium text-sm">
        {label}
      </Label>
      <div className="flex items-center">
        <NumberInput id={id} value={value} onChange={onChange} className="w-28" />
        {unit ? (
          <span className="text-sm text-text-muted ml-2 w-14">{unit}</span>
        ) : (
          <span className="ml-2 w-14" />
        )}
      </div>
    </div>
    {hint ? <p className="text-xs text-text-muted">{hint}</p> : null}
  </div>
);

const BoolField: React.FC<{
  id: string;
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  hint?: string;
}> = ({ id, label, value, onChange, hint }) => (
  <div className="space-y-1">
    <div className="flex items-center justify-between gap-2">
      <Label htmlFor={id} className="font-medium text-sm">
        {label}
      </Label>
      <Switch id={id} checked={value} onCheckedChange={onChange} />
    </div>
    {hint ? <p className="text-xs text-text-muted">{hint}</p> : null}
  </div>
);

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({
  title,
  children,
}) => (
  <div className="space-y-3">
    <h2 className="text-sm font-bold uppercase tracking-wide text-text-muted">
      {title}
    </h2>
    {children}
  </div>
);

const AvoidanceSettingsEditor: React.FC<{ className?: string }> = ({
  className = "",
}) => {
  const { settings, updateSettings } = useExecutorSettings();
  if (!settings) return null;

  const cfg: AvoidanceConfig = settings.avoidance ?? DEFAULTS;
  const update = (next: AvoidanceConfig) =>
    updateSettings({ ...settings, avoidance: next });
  const set = <K extends keyof AvoidanceConfig>(
    key: K,
    v: AvoidanceConfig[K],
  ) => update({ ...cfg, [key]: v });

  return (
    <div className={`h-full relative ${className}`}>
      <div className="absolute inset-0 overflow-y-auto flex flex-col gap-6 p-4">
        <Section title="Layers">
          <BoolField
            id="avoid-planner-enabled"
            label="Global planner"
            value={cfg.planner_enabled}
            onChange={(v) => set("planner_enabled", v)}
            hint="Theta* path planner. When off, MTP steers straight at the target (ORCA still avoids collisions)."
          />
          <BoolField
            id="avoid-orca-enabled"
            label="ORCA"
            value={cfg.orca_enabled}
            onChange={(v) => set("orca_enabled", v)}
            hint="Reciprocal velocity-space avoidance. When off, the raw MTP velocity passes through."
          />
          <BoolField
            id="avoid-prefer-steering"
            label="Prefer steering"
            value={cfg.prefer_steering}
            onChange={(v) => set("prefer_steering", v)}
            hint="Deflect around obstacles instead of braking for them (drops ORCA's cut-off projection). Avoids the sticky crawl near obstacles."
          />
        </Section>

        <Section title="Obstacle margins">
          <NumberField
            id="avoid-robot-clearance"
            label="Robot clearance"
            value={cfg.robot_clearance}
            onChange={(v) => set("robot_clearance", v)}
            unit="mm"
            hint="Extra gap on top of the two-robot contact distance."
          />
          <NumberField
            id="avoid-wall-margin"
            label="Wall margin"
            value={cfg.wall_margin}
            onChange={(v) => set("wall_margin", v)}
            unit="mm"
          />
          <NumberField
            id="avoid-defense-margin"
            label="Defense margin"
            value={cfg.defense_margin}
            onChange={(v) => set("defense_margin", v)}
            unit="mm"
          />
          <NumberField
            id="avoid-ball-stop-radius"
            label="Ball radius (Stop)"
            value={cfg.ball_stop_radius}
            onChange={(v) => set("ball_stop_radius", v)}
            unit="mm"
            hint="Keep-out radius around the ball during Stop (rule: 500 mm clear)."
          />
          <NumberField
            id="avoid-ball-base-radius"
            label="Ball radius (base)"
            value={cfg.ball_base_radius}
            onChange={(v) => set("ball_base_radius", v)}
            unit="mm"
          />
          <NumberField
            id="avoid-ball-care-scale"
            label="Ball radius (care scale)"
            value={cfg.ball_care_scale}
            onChange={(v) => set("ball_care_scale", v)}
            unit="mm"
            hint="Added to the base ball radius, scaled by the skill's avoid-ball care."
          />
          <NumberField
            id="avoid-robot-extrapolation"
            label="Robot extrapolation"
            value={cfg.robot_extrapolation}
            onChange={(v) => set("robot_extrapolation", v)}
            unit="s"
          />
        </Section>

        <Section title="ORCA">
          <NumberField
            id="avoid-time-horizon"
            label="Time horizon τ"
            value={cfg.time_horizon}
            onChange={(v) => set("time_horizon", v)}
            unit="s"
            hint="How far ahead collisions are resolved. Larger ⇒ reacts earlier but brakes harder near obstacles."
          />
          <NumberField
            id="avoid-stationary-speed"
            label="Stationary speed"
            value={cfg.stationary_speed}
            onChange={(v) => set("stationary_speed", v)}
            unit="mm/s"
            hint="Below this a robot counts as stationary: it won't yield to movers, and movers take the full avoidance burden around it."
          />
          <NumberField
            id="avoid-neighbor-dist"
            label="Neighbour distance"
            value={cfg.neighbor_dist}
            onChange={(v) => set("neighbor_dist", v)}
            unit="mm"
          />
          <NumberField
            id="avoid-max-neighbors"
            label="Max neighbours"
            value={cfg.max_neighbors}
            onChange={(v) => set("max_neighbors", Math.round(v))}
          />
        </Section>

        <Section title="Planner">
          <NumberField
            id="avoid-grid-resolution"
            label="Grid resolution"
            value={cfg.grid_resolution}
            onChange={(v) => set("grid_resolution", v)}
            unit="mm"
            hint="Theta* cell size. Smaller ⇒ finer paths, more compute."
          />
          <NumberField
            id="avoid-planner-margin"
            label="Planner margin"
            value={cfg.planner_margin}
            onChange={(v) => set("planner_margin", v)}
            unit="mm"
            hint="Extra clearance the planner leaves around robots on top of ORCA's hard radius, so paths sit outside ORCA's braking band."
          />
          <NumberField
            id="avoid-waypoint-tolerance"
            label="Waypoint tolerance"
            value={cfg.waypoint_tolerance}
            onChange={(v) => set("waypoint_tolerance", v)}
            unit="mm"
            hint="How close to an intermediate corner counts as reached. Wider ⇒ the robot cuts corners and flows through; the final target still decelerates."
          />
          <NumberField
            id="avoid-replan-target-tol"
            label="Replan tolerance"
            value={cfg.replan_target_tol}
            onChange={(v) => set("replan_target_tol", v)}
            unit="mm"
            hint="Reuse the cached path until the target moves more than this (suppresses flicker)."
          />
        </Section>
      </div>
    </div>
  );
};

export default AvoidanceSettingsEditor;
