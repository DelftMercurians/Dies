import React from "react";
import { useExecutorSettings } from "@/api";
import {
  CostWeights,
  ObstacleConfig,
  RobotParams,
} from "@/bindings";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";

/**
 * Dedicated editor for the iLQR / MPC tuning parameters (`ExecutorSettings.ilqr_params`).
 *
 * The generic `SettingsEditor` only renders flat number/boolean fields, but
 * `RobotParams` is nested (per-axis `tau`/`accel_max` pairs, `weights`,
 * `obstacles`). This component renders those by hand as plain number inputs
 * (no sliders) and writes whole-`RobotParams` updates back through the standard
 * settings mutation, so edits live-update the running executor.
 */

// Defaults mirror `RobotParams::default_hand_tuned()` — used only to fill in
// fields if the backend ever sends a partial object (serde marks them optional).
const DEFAULT_WEIGHTS: CostWeights = {
  position: 1.0e-3,
  velocity: 0.0,
  control: 5.0e-5,
  control_smoothness: 1.0e-4,
  heading: 10.0,
  heading_control: 1.0,
};

const DEFAULT_OBSTACLES: ObstacleConfig = {
  weight: 5.0e-2,
  influence: 250.0,
  robot_clearance: 50.0,
  robot_extrapolation: 0.5,
  wall_margin: 120.0,
  defense_margin: 50.0,
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
        <NumberInput
          id={id}
          value={value}
          onChange={onChange}
          className="w-28"
        />
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

const IlqrSettingsEditor: React.FC<{ className?: string }> = ({
  className = "",
}) => {
  const { settings, updateSettings } = useExecutorSettings();
  if (!settings) return null;

  const params: RobotParams = settings.ilqr_params ?? {
    tau: [0.08, 0.1],
    accel_max: [3500.0, 3500.0],
    tau_yaw: 0.1,
    omega_max: 20.0,
    obstacles: DEFAULT_OBSTACLES,
    weights: DEFAULT_WEIGHTS,
  };
  const weights = params.weights ?? DEFAULT_WEIGHTS;
  const obstacles = params.obstacles ?? DEFAULT_OBSTACLES;

  const update = (next: RobotParams) =>
    updateSettings({ ...settings, ilqr_params: next });
  const setTau = (axis: 0 | 1, v: number) => {
    const tau: [number, number] =
      axis === 0 ? [v, params.tau[1]] : [params.tau[0], v];
    update({ ...params, tau });
  };
  const setAccel = (axis: 0 | 1, v: number) => {
    const accel_max: [number, number] =
      axis === 0 ? [v, params.accel_max[1]] : [params.accel_max[0], v];
    update({ ...params, accel_max });
  };
  const setWeight = (key: keyof CostWeights, v: number) =>
    update({ ...params, weights: { ...weights, [key]: v } });
  const setObstacle = (key: keyof ObstacleConfig, v: number) =>
    update({ ...params, obstacles: { ...obstacles, [key]: v } });

  return (
    <div className={`h-full relative ${className}`}>
      <div className="absolute inset-0 overflow-y-auto flex flex-col gap-6 p-4">
        <Section title="Dynamics model">
          <NumberField
            id="ilqr-tau-fwd"
            label="τ forward"
            value={params.tau[0]}
            onChange={(v) => setTau(0, v)}
            unit="s"
            hint="Velocity-lag time constant. Larger ⇒ model brakes earlier ⇒ less overshoot."
          />
          <NumberField
            id="ilqr-tau-strafe"
            label="τ strafe"
            value={params.tau[1]}
            onChange={(v) => setTau(1, v)}
            unit="s"
          />
          <NumberField
            id="ilqr-accel-fwd"
            label="a_max forward"
            value={params.accel_max[0]}
            onChange={(v) => setAccel(0, v)}
            unit="mm/s²"
            hint="Modelled accel ceiling. Lower ⇒ more conservative braking plan."
          />
          <NumberField
            id="ilqr-accel-strafe"
            label="a_max strafe"
            value={params.accel_max[1]}
            onChange={(v) => setAccel(1, v)}
            unit="mm/s²"
          />
          <NumberField
            id="ilqr-tau-yaw"
            label="τ yaw"
            value={params.tau_yaw ?? 0.1}
            onChange={(v) => update({ ...params, tau_yaw: v })}
            unit="s"
            hint="Heading-lag time constant. Models the onboard yaw loop slewing toward the commanded heading."
          />
          <NumberField
            id="ilqr-omega-max"
            label="ω max"
            value={params.omega_max ?? 20.0}
            onChange={(v) => update({ ...params, omega_max: v })}
            unit="rad/s"
            hint="Heading slew-rate ceiling (tanh saturation on θ̇)."
          />
        </Section>

        <Section title="Cost weights">
          <NumberField
            id="ilqr-w-position"
            label="Position"
            value={weights.position}
            onChange={(v) => setWeight("position", v)}
            hint="Pulls trajectory to target. Raise relative to control ⇒ harder braking."
          />
          <NumberField
            id="ilqr-w-velocity"
            label="Velocity"
            value={weights.velocity}
            onChange={(v) => setWeight("velocity", v)}
            hint="Penalises arrival speed (target vel = 0). The main anti-overshoot knob; 0 disables."
          />
          <NumberField
            id="ilqr-w-control"
            label="Control"
            value={weights.control}
            onChange={(v) => setWeight("control", v)}
            hint="Control-effort regulariser. Keep small but nonzero (stabilises Q_uu)."
          />
          <NumberField
            id="ilqr-w-control-smoothness"
            label="Control smoothness"
            value={weights.control_smoothness}
            onChange={(v) => setWeight("control_smoothness", v)}
            hint="Penalises command changes. High values delay braking ⇒ more overshoot."
          />
          <NumberField
            id="ilqr-w-heading"
            label="Heading"
            value={weights.heading ?? 10.0}
            onChange={(v) => setWeight("heading", v)}
            hint="Attracts heading to the desired yaw (1 − cos error). 0 ⇒ heading left free to optimise translation."
          />
          <NumberField
            id="ilqr-w-heading-control"
            label="Heading control"
            value={weights.heading_control ?? 1.0}
            onChange={(v) => setWeight("heading_control", v)}
            hint="Turn-effort regulariser on the heading setpoint; keeps Q_uu non-degenerate in the θ_cmd axis."
          />
        </Section>

        <Section title="Obstacle barriers">
          <NumberField
            id="ilqr-obs-weight"
            label="Weight"
            value={obstacles.weight}
            onChange={(v) => setObstacle("weight", v)}
          />
          <NumberField
            id="ilqr-obs-influence"
            label="Influence"
            value={obstacles.influence}
            onChange={(v) => setObstacle("influence", v)}
            unit="mm"
          />
          <NumberField
            id="ilqr-obs-robot-clearance"
            label="Robot clearance"
            value={obstacles.robot_clearance}
            onChange={(v) => setObstacle("robot_clearance", v)}
            unit="mm"
          />
          <NumberField
            id="ilqr-obs-robot-extrapolation"
            label="Robot extrapolation"
            value={obstacles.robot_extrapolation}
            onChange={(v) => setObstacle("robot_extrapolation", v)}
            unit="s"
          />
          <NumberField
            id="ilqr-obs-wall-margin"
            label="Wall margin"
            value={obstacles.wall_margin}
            onChange={(v) => setObstacle("wall_margin", v)}
            unit="mm"
          />
          <NumberField
            id="ilqr-obs-defense-margin"
            label="Defense margin"
            value={obstacles.defense_margin}
            onChange={(v) => setObstacle("defense_margin", v)}
            unit="mm"
          />
        </Section>
      </div>
    </div>
  );
};

export default IlqrSettingsEditor;
