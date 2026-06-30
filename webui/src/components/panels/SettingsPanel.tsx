import React from "react";
import { IDockviewPanelProps } from "dockview";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import SettingsEditor from "@/views/SettingsEditor";
import AvoidanceSettingsEditor from "@/views/AvoidanceSettingsEditor";
import SettingsBaselineBar from "@/views/SettingsBaselineBar";
import StrategySettings from "@/views/StrategySettings";
import SkillTunablesEditor from "@/views/SkillTunablesEditor";
import { useExecutorSettings } from "@/api";

const GoalAreaAvoidanceToggle: React.FC = () => {
  const { settings, updateSettings } = useExecutorSettings();
  if (!settings) return null;
  return (
    <div className="flex items-center justify-between px-4 py-2 border-b border-bg-muted">
      <Label htmlFor="goal_area_avoidance" className="font-medium">
        Goal area avoidance
      </Label>
      <Switch
        id="goal_area_avoidance"
        checked={settings.goal_area_avoidance}
        onCheckedChange={(checked) =>
          updateSettings({ ...settings, goal_area_avoidance: checked })
        }
      />
    </div>
  );
};

/**
 * Settings Panel - Controller, Avoidance, Tracker, and Skill settings.
 * Contains tabbed interface for different settings categories.
 */
const SettingsPanel: React.FC<IDockviewPanelProps> = () => {
  return (
    <div className="w-full h-full bg-bg-surface flex flex-col">
      <Tabs
        size="sm"
        defaultValue="strategy"
        className="flex-1 min-h-0 w-full flex flex-col gap-2 p-2"
      >
        <TabsList className="w-full">
          <TabsTrigger value="strategy">Strategy</TabsTrigger>
          <TabsTrigger value="controller">Controller</TabsTrigger>
          <TabsTrigger value="avoidance">Avoidance</TabsTrigger>
          <TabsTrigger value="player">Player</TabsTrigger>
          <TabsTrigger value="ball">Ball</TabsTrigger>
          <TabsTrigger value="field_mask">Field Mask</TabsTrigger>
          <TabsTrigger value="skill">Skill</TabsTrigger>
        </TabsList>

        <TabsContent value="strategy" className="flex-1 overflow-hidden">
          <StrategySettings />
        </TabsContent>
        <TabsContent value="controller" className="flex-1 overflow-hidden">
          <div className="h-full flex flex-col">
            <GoalAreaAvoidanceToggle />
            <div className="flex-1 overflow-hidden">
              <SettingsEditor settingsKey="controller_settings" />
            </div>
          </div>
        </TabsContent>
        <TabsContent value="avoidance" className="flex-1 overflow-hidden">
          <AvoidanceSettingsEditor />
        </TabsContent>
        <TabsContent value="player" className="flex-1 overflow-hidden">
          <SettingsEditor
            settingsKey="tracker_settings"
            include={[
              "player_use_acceleration",
              "player_use_command_feedforward",
              "player_command_tau",
              "player_measurement_var",
              "player_unit_transition_var",
              "player_ca_unit_transition_var",
              "player_yaw_lpf_alpha",
            ]}
          />
        </TabsContent>
        <TabsContent value="ball" className="flex-1 overflow-hidden">
          <SettingsEditor
            settingsKey="tracker_settings"
            include={[
              "ball_measurement_var",
              "ball_unit_transition_var",
              "ball_confidence_threshold",
            ]}
          />
        </TabsContent>
        <TabsContent value="field_mask" className="flex-1 overflow-hidden">
          <SettingsEditor settingsKey="tracker_settings" include={["field_mask"]} />
        </TabsContent>
        <TabsContent value="skill" className="flex-1 overflow-hidden">
          <SkillTunablesEditor />
        </TabsContent>
      </Tabs>
      <SettingsBaselineBar />
    </div>
  );
};

export default SettingsPanel;
