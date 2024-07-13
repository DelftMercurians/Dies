import { useExecutorSettings } from "@/api";
import { ExecutorSettings } from "@/bindings";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { MoveLeft, MoveRight } from "lucide-react";

type FieldConfig = {
  [K in keyof ExecutorSettings]?: {
    [M in keyof ExecutorSettings[K]]?: {
      hidden?: boolean;
      min?: number;
      max?: number;
      step?: number;
      unit?: string;
      disableSlider?: boolean;
      isAngle?: boolean;
      customComponent?: React.ComponentType<{
        id: string;
        value: ExecutorSettings[K][M];
        onChange: (value: ExecutorSettings[K][M]) => void;
      }>;
    };
  };
};

interface SettingsEditorProps<K> {
  settingsKey: K;
  className?: string;
}

const fieldConfig: FieldConfig = {
  controller_settings: {
    max_acceleration: { min: 0, max: 50000, step: 10, unit: "mm/s²" },
    max_velocity: { min: 0, max: 10000, step: 10, unit: "mm/s" },
    max_deceleration: { min: 0, max: 50000, step: 10, unit: "mm/s²" },
    max_angular_velocity: { min: 0, max: 360*4, unit: "deg/s", isAngle: true },
    max_angular_acceleration: {
      min: 0,
      max: 360*4,
      unit: "deg/s²",
      isAngle: true,
    },
    max_angular_deceleration: {
      min: 0,
      max: 360*4,
      unit: "deg/s²",
      isAngle: true,
    },
    position_kp: { min: 0, max: 10, unit: "" },
    position_proportional_time_window: { min: 0, max: 10, unit: "s" },
    angle_kp: { min: 0, max: 10, unit: "" },
    angle_proportional_time_window: { min: 0, max: 10, unit: "s" },
    angle_cutoff_distance: { min: 0, max: 180, unit: "deg", isAngle: true },
    position_cutoff_distance: { min: 0, max: 2000, unit: "mm" },
  },
  tracker_settings: {
    is_blue: { hidden: true },
    initial_opp_goal_x: {
      customComponent: ({ value, onChange }) => (
        <ToggleGroup
          type="single"
          value={value == -1 ? "-1" : "1"}
          onValueChange={(value) => onChange(value === "-1" ? -1 : 1)}
          className="border border-gray-500 rounded-lg"
        >
          <ToggleGroupItem value="-1">
            <MoveLeft />
          </ToggleGroupItem>
          <ToggleGroupItem value="1">
            <MoveRight />
          </ToggleGroupItem>
        </ToggleGroup>
      ),
    },
    player_measurement_var: { min: 0.01, max: 200, step: 0.01, unit: "s" },
    player_unit_transition_var: { min: 0.01, max: 200, step: 0.01, unit: "s" },
    ball_measurement_var: { min: 0.01, max: 200, step: 0.01, unit: "s" },
    ball_unit_transition_var: { min: 0.01, max: 200, step: 0.01, unit: "s" },
    player_yaw_lpf_alpha: { min: 0, max: 1, step: 0.01 },
  },
};

function SettingsEditor<K extends keyof ExecutorSettings>({
  settingsKey,
  className = "",
}: SettingsEditorProps<K>) {
  type SettingsKey = keyof ExecutorSettings[K];
  type Value = ExecutorSettings[K][SettingsKey];
  const { settings: allSettings, updateSettings } = useExecutorSettings();

  if (!allSettings)
    return (
      <div className={`space-y-6 ${className}`}>
        <h1 className="text-2xl font-bold mb-4">Controller Settings</h1>
      </div>
    );

  const settings = allSettings[settingsKey];
  const handleChange = (key: SettingsKey, value: Value) => {
    if (allSettings && settings) {
      const config = fieldConfig[settingsKey]?.[key];
      if (config?.isAngle && typeof value === "number") {
        // Convert degrees to radians before updating
        value = (value * (Math.PI / 180)) as Value;
      }
      updateSettings({
        ...allSettings,
        [settingsKey]: { ...settings, [key]: value },
      });
    }
  };

  return (
    <div className="h-full relative">
      <div className="absolute inset-0 overflow-y-auto flex flex-col gap-4 p-4">
        {Object.entries(settings).map(([key, _value]) => {
          const value = _value as Value;
          const config = fieldConfig[settingsKey]?.[key as SettingsKey] || {};
          const {
            hidden = false,
            min = 0,
            max = 100,
            step,
            unit = null,
            disableSlider = false,
            isAngle = false,
            customComponent: CustomComponent,
          } = config;

          if (hidden) return null;

          const displayValue =
            isAngle && typeof value === "number"
              ? value * (180 / Math.PI)
              : value;

          return (
            <div key={key} className="space-y-2">
              <Label htmlFor={key} className="text-sm font-medium">
                {key
                  .replace(/_/g, " ")
                  .replace(/\b\w/g, (l) => l.toUpperCase())}
              </Label>
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
                  <>
                    {!disableSlider ? (
                      <Slider
                        id={`${key}-slider`}
                        min={min}
                        max={max}
                        step={step ?? (max - min) / 100}
                        value={[displayValue]}
                        onValueChange={([newValue]) =>
                          handleChange(key as SettingsKey, newValue as Value)
                        }
                        className="flex-1 min-w-24"
                      />
                    ) : null}
                    <div className="flex items-center">
                      <Input
                        id={key}
                        type="number"
                        value={displayValue}
                        onChange={(e) =>
                          handleChange(
                            key as SettingsKey,
                            parseFloat(e.target.value) as Value
                          )
                        }
                        className="w-24"
                      />
                      {unit ? (
                        <span className="text-sm text-gray-500 ml-2">
                          {unit}
                        </span>
                      ) : null}
                    </div>
                  </>
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
        })}
      </div>
    </div>
  );
}

export default SettingsEditor;
