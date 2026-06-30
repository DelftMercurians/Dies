import { useExecutorSettings } from "@/api";
import { ExecutorSettings, FieldMask } from "@/bindings";
import { NumberInput } from "@/components/ui/number-input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { useAtom } from "jotai";
import { maskEditModeAtom } from "@/lib/fieldEditing";
import { cn } from "@/lib/utils";

const FULL_FIELD_MASK: FieldMask = { x_min: -1, x_max: 1, y_min: -1, y_max: 1 };

/**
 * Field-mask editor: Edit toggles drag-to-draw mode on the field canvas; Clear
 * resets to the full field. The four fractions remain numerically editable.
 */
const FieldMaskEditor: React.FC<{
  value: FieldMask;
  onChange: (v: FieldMask) => void;
}> = ({ value, onChange }) => {
  const [editing, setEditing] = useAtom(maskEditModeAtom);
  const num = (k: keyof FieldMask, label: string) => (
    <div className="flex items-center gap-1">
      <span className="text-xs text-text-muted w-9">{label}</span>
      <NumberInput
        value={value[k]}
        onChange={(v) => onChange({ ...value, [k]: v as number })}
        className="w-16"
      />
    </div>
  );
  return (
    <div className="w-full space-y-2">
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant={editing ? "default" : "outline"}
          className={cn(editing && "ring-1 ring-accent-cyan")}
          onClick={() => setEditing((e) => !e)}
        >
          {editing ? "Drawing… (drag on field)" : "Edit on field"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            onChange(FULL_FIELD_MASK);
            setEditing(false);
          }}
        >
          Clear
        </Button>
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
        {num("x_min", "X min")}
        {num("x_max", "X max")}
        {num("y_min", "Y min")}
        {num("y_max", "Y max")}
      </div>
    </div>
  );
};

type FieldConfig = {
  [K in keyof ExecutorSettings]?: {
    [M in keyof ExecutorSettings[K]]?: {
      hidden?: boolean;
      /** Retained for the field_mask slider custom component; ignored by the
       * default number-input renderer. */
      min?: number;
      max?: number;
      step?: number;
      unit?: string;
      isAngle?: boolean;
      /** Tooltip shown on the field label: what it does + which way to tune. */
      help?: string;
      customComponent?: React.ComponentType<{
        id: string;
        value: ExecutorSettings[K][M];
        onChange: (value: ExecutorSettings[K][M]) => void;
      }>;
    };
  };
};

interface SettingsEditorProps<K extends keyof ExecutorSettings> {
  settingsKey: K;
  className?: string;
  /** Restrict (and order) which fields are shown; defaults to all. */
  include?: (keyof ExecutorSettings[K])[];
}

const fieldConfig: FieldConfig = {
  controller_settings: {
    max_acceleration: {
      min: 0, max: 10000, step: 50, unit: "mm/s²",
      help: "Cap on acceleration when speeding up. Higher = snappier starts but more wheel slip (which vision can't see → tracking error). Lower = gentler launches.",
    },
    max_deceleration: {
      min: 0, max: 10000, step: 50, unit: "mm/s²",
      help: "Cap on braking, and the braking authority the path follower plans its stop against. Higher = later, harder stops (overshoot if the robot can't actually brake that hard).",
    },
    max_velocity: {
      min: 0, max: 6000, step: 50, unit: "mm/s",
      help: "Top commanded speed magnitude. Keep at/below what the robots can actually hold.",
    },
    lateral_acceleration: {
      min: 0, max: 10000, step: 50, unit: "mm/s²",
      help: "Cornering acceleration budget — how much speed the robot carries through a path corner. Higher = faster corners but wider drift; lower = slows more for turns.",
    },
    approach_kp: {
      min: 0, max: 20, step: 0.1, unit: "1/s",
      help: "Arrival gain: commanded speed eases off as kp × remaining distance into corners and the goal. Lower = gentler/earlier braking; higher = later/snappier stops (risk overshoot).",
    },
    brake_gain: {
      min: 0, max: 5, step: 0.1,
      help: "Terminal active braking. When overspeeding into the goal, the command is pushed below the arrival profile (and may reverse) by gain × overspeed, bypassing the accel clamp so the firmware reverse-thrusts to a crisp stop. 0 = off (gentle proportional only); ~1 mirrors overspeed into a reverse command; higher = more aggressive (risk of over-braking/jerk).",
    },
    lookahead_min: {
      min: 0, max: 1000, step: 10, unit: "mm",
      help: "Minimum pure-pursuit lookahead, used at low speed. Higher = smoother but cuts corners; lower = tighter path tracking but can wobble.",
    },
    lookahead_time: {
      min: 0, max: 1, step: 0.01, unit: "s",
      help: "Lookahead = clamp(time × speed, min, max). Higher = smoother at speed and wider corner cuts; lower = hugs the path more closely.",
    },
    max_angular_velocity: {
      min: 0, max: 720, unit: "deg/s", isAngle: true,
      help: "Maximum turn rate.",
    },
    max_angular_acceleration: {
      min: 0, max: 7200, unit: "deg/s²", isAngle: true,
      help: "Maximum angular acceleration. Higher = snappier heading changes.",
    },
    angle_kp: {
      min: 0, max: 10, step: 0.05,
      help: "Proportional gain for heading tracking. Higher = snappier heading, but too high oscillates/hunts.",
    },
    angle_cutoff_distance: {
      min: 0, max: 30, step: 0.1, unit: "deg", isAngle: true,
      help: "Heading deadzone: yaw control outputs zero inside this band. Raise to stop hunting/jitter near the target heading; too high leaves a visible heading error.",
    },
  },
  tracker_settings: {
    player_use_acceleration: {
      help: "Player motion model. ON = constant-acceleration: the filter tracks acceleration, so it doesn't lag during hard starts/stops. OFF = constant-velocity: smoother but lags any acceleration. Leave ON; toggle OFF only to compare.",
    },
    player_use_command_feedforward: {
      help: "Feed the commanded velocity into the filter's predict step so the estimate anticipates commanded moves (less lag, leads the vision). EXPERIMENTAL — depends on a good Command Tau. Turn OFF if the estimate overshoots or oscillates.",
    },
    player_command_tau: {
      min: 0, max: 1, step: 0.01, unit: "s",
      help: "Feedforward time constant: how fast the estimate is pulled toward the command (~the robot's velocity-loop response time). Lower = more aggressive anticipation (less lag, more noise/overshoot); higher = gentler. Start ~0.15 and raise if it overshoots.",
    },
    player_measurement_var: {
      min: 0, max: 10, step: 0.01, unit: "mm²",
      help: "Assumed vision position-noise variance. Set near the real noise floor (~0.4 mm²). Lower = trust vision more (snappier, noisier); higher = trust the model more (smoother, laggier).",
    },
    player_unit_transition_var: {
      unit: "mm²/s³ (CV)",
      help: "Constant-velocity process noise (acceleration PSD). Used only when CA is OFF. Higher = filter accepts velocity changes faster (less lag, noisier); lower = smoother/laggier.",
    },
    player_ca_unit_transition_var: {
      unit: "mm²/s⁵ (CA)",
      help: "Constant-acceleration process noise (jerk PSD). Used when CA is ON. Higher = accepts acceleration changes faster (less lag, more velocity noise); lower = smoother. ~1e6 nominal. (Different scale from the CV value — don't copy between them.)",
    },
    player_yaw_lpf_alpha: {
      min: 0, max: 1, step: 0.01,
      help: "Yaw low-pass smoothing (0–1). 0 = no smoothing (responsive but noisy heading); higher = smoother but laggier; 1 = frozen.",
    },
    ball_measurement_var: {
      min: 0, max: 10, step: 0.01, unit: "mm²",
      help: "Assumed ball vision-noise variance. Lower = trust vision more (snappier, noisier); higher = smoother/laggier.",
    },
    ball_unit_transition_var: {
      min: 0, max: 500, step: 1, unit: "mm²/s",
      help: "Ball process noise. Higher = tracks abrupt speed changes (kicks, deflections) faster but noisier; lower = smoother but lags sudden ball motion.",
    },
    ball_confidence_threshold: {
      min: 0, max: 1, step: 0.01,
      help: "Minimum vision confidence to accept a ball detection. Raise to reject false balls IF your vision reports trustworthy confidence; many setups report ~0 even for a clean ball, so default 0 accepts all.",
    },
    field_mask: {
      help: "Vision crop as fractions of the field half-extent — detections outside the box are ignored. Click 'Edit on field' and drag a rectangle, or set the fractions directly.",
      customComponent: ({ value, onChange }) => (
        <FieldMaskEditor value={value} onChange={onChange} />
      ),
    },
  },
};

function SettingsEditor<K extends keyof ExecutorSettings>({
  settingsKey,
  className = "",
  include,
}: SettingsEditorProps<K>) {
  type SettingsKey = keyof ExecutorSettings[K];
  type Value = ExecutorSettings[K][SettingsKey];
  const { settings: allSettings, updateSettings } = useExecutorSettings();

  if (!allSettings)
    return (
      <div className={`space-y-6 ${className}`}>
        <h1 className="text-lg font-bold mb-4">Controller Settings</h1>
      </div>
    );

  const settings = allSettings[settingsKey];
  if (!settings || typeof settings !== "object") return null;
  const handleChange = (key: SettingsKey, value: Value) => {
    if (allSettings && settings) {
      const config = fieldConfig[settingsKey]?.[key];
      if (config?.isAngle && typeof value === "number") {
        // Convert degrees to radians before updating
        value = (value * (Math.PI / 180)) as Value;
      }
      updateSettings({
        ...allSettings,
        [settingsKey]: { ...(settings as any), [key]: value },
      });
    }
  };

  return (
    <div className="h-full relative">
      <div className="absolute inset-0 overflow-y-auto grid grid-cols-2 gap-x-6 gap-y-4 p-4 auto-rows-max">
        {(include
          ? include
              .filter((k) => k in (settings as object))
              .map((k) => [k as string, (settings as any)[k]] as [string, unknown])
          : Object.entries(settings as unknown as Record<string, unknown>)
        ).map(
          ([key, _value]) => {
            const value = _value as Value;
            const config = fieldConfig[settingsKey]?.[key as SettingsKey] || {};
            const {
              hidden = false,
              unit = null,
              isAngle = false,
              help = null,
              customComponent: CustomComponent,
            } = config;

            if (hidden) return null;

            const displayValue =
              isAngle && typeof value === "number"
                ? value * (180 / Math.PI)
                : value;

            const label = (
              <Label htmlFor={key} className="font-medium">
                {key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase())}
              </Label>
            );

            return (
              <div key={key} className="space-y-2 min-w-0">
                {help ? (
                  <SimpleTooltip title={help} className="w-fit cursor-help">
                    {label}
                  </SimpleTooltip>
                ) : (
                  label
                )}
                <div className="flex items-center gap-2 flex-wrap">
                  {typeof CustomComponent === "function" ? (
                    <CustomComponent
                      id={key}
                      value={value}
                      onChange={(value) =>
                        handleChange(key as SettingsKey, value)
                      }
                    />
                  ) : typeof displayValue === "number" ? (
                    <div className="flex items-center">
                      <NumberInput
                        id={key}
                        value={displayValue}
                        onChange={(newValue) =>
                          handleChange(key as SettingsKey, newValue as Value)
                        }
                        className="w-24"
                      />
                      {unit ? (
                        <span className="text-sm text-text-muted ml-2">
                          {unit}
                        </span>
                      ) : null}
                    </div>
                  ) : typeof displayValue === "boolean" ? (
                    <Switch
                      id={key}
                      checked={displayValue}
                      onCheckedChange={(newValue) =>
                        handleChange(key as SettingsKey, newValue as Value)
                      }
                    />
                  ) : (
                    <span>{`${displayValue}`}</span>
                  )}
                </div>
              </div>
            );
          },
        )}
      </div>
    </div>
  );
}

export default SettingsEditor;
