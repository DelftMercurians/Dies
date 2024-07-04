import React, { FC, useState } from "react";
import { useControllerSettings } from "@/api";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { ControllerSettings } from "@/bindings";

interface ControllerSettingsProps {
  className?: string;
}

type FieldConfig = {
  [K in keyof ControllerSettings]?: {
    min?: number;
    max?: number;
    unit?: string;
    disableSlider?: boolean;
    isAngle?: boolean;
  };
};

// Custom configuration for fields
const FIELD_CONFIG: FieldConfig = {
  max_acceleration: { min: 0, max: 10000, unit: "mm/s²" },
  max_velocity: { min: 0, max: 5000, unit: "mm/s" },
  max_deceleration: { min: 0, max: 10000, unit: "mm/s²" },
  max_angular_velocity: { min: 0, max: 10, unit: "deg/s", isAngle: true },
  max_angular_acceleration: { min: 0, max: 20, unit: "deg/s²", isAngle: true },
  max_angular_deceleration: { min: 0, max: 20, unit: "deg/s²", isAngle: true },
  position_kp: { min: 0, max: 10, unit: "", disableSlider: true },
  position_propotional_time_window: { min: 0, max: 10, unit: "s" },
  angle_kp: { min: 0, max: 10, unit: "", disableSlider: true },
  angle_propotional_time_window: { min: 0, max: 10, unit: "s" },
};

const ControllerSettingsEditor: FC<ControllerSettingsProps> = ({
  className = "",
}) => {
  const { settings, updateSettings } = useControllerSettings();

  if (!settings)
    return (
      <div className={`space-y-6 ${className}`}>
        <h1 className="text-2xl font-bold mb-4">Controller Settings</h1>
      </div>
    );

  const handleChange = (key: string, value: number) => {
    if (settings) {
      const config = FIELD_CONFIG[key as keyof typeof FIELD_CONFIG];
      if (config?.isAngle) {
        // Convert degrees to radians before updating
        value = value * (Math.PI / 180);
      }
      updateSettings({ ...settings, [key]: value });
    }
  };

  const formatValue = (value: number) => value.toFixed(2);

  return (
    <div className={`flex flex-col space-y-6 p-4 ${className}`}>
      <h1 className="text-2xl font-bold">Controller Settings</h1>

      <div className="h-full overflow-y-auto flex flex-col gap-4">
        {Object.entries(settings).map(([key, value]) => {
          const config = FIELD_CONFIG[key as keyof typeof FIELD_CONFIG] || {};
          const {
            min = 0,
            max = 100,
            unit = "",
            disableSlider = false,
            isAngle = false,
          } = config;

          const displayValue = isAngle ? value * (180 / Math.PI) : value;
          const displayUnit = isAngle ? unit.replace("rad", "deg") : unit;

          return (
            <div key={key} className="space-y-2">
              <Label htmlFor={key} className="text-sm font-medium">
                {key
                  .replace(/_/g, " ")
                  .replace(/\b\w/g, (l) => l.toUpperCase())}
              </Label>
              <div className="flex items-center space-x-4">
                {!disableSlider && (
                  <Slider
                    id={`${key}-slider`}
                    min={min}
                    max={max}
                    step={(max - min) / 100}
                    value={[displayValue]}
                    onValueChange={([newValue]) => handleChange(key, newValue)}
                    className="flex-grow"
                  />
                )}
                <div className="flex items-center space-x-2">
                  <Input
                    id={key}
                    type="number"
                    value={formatValue(displayValue)}
                    onChange={(e) =>
                      handleChange(key, parseFloat(e.target.value))
                    }
                    className="w-24"
                  />
                  {displayUnit && (
                    <span className="text-sm text-gray-500">{displayUnit}</span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ControllerSettingsEditor;
