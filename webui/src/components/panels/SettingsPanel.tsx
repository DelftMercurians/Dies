import React from "react";
import { IDockviewPanelProps } from "dockview";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import SettingsEditor from "@/views/SettingsEditor";
import AvoidanceSettingsEditor from "@/views/AvoidanceSettingsEditor";
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
    <div className="w-full h-full bg-bg-surface">
      <Tabs
        size="sm"
        defaultValue="controller"
        className="h-full w-full flex flex-col gap-2 p-2"
      >
        <TabsList className="w-full">
          <TabsTrigger value="controller">Controller</TabsTrigger>
          <TabsTrigger value="avoidance">Avoidance</TabsTrigger>
          <TabsTrigger value="tracker">Tracker</TabsTrigger>
          <TabsTrigger value="skill">Skill</TabsTrigger>
        </TabsList>

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
        <TabsContent value="tracker" className="flex-1 overflow-hidden">
          <SettingsEditor settingsKey="tracker_settings" />
        </TabsContent>
        <TabsContent value="skill" className="flex-1 overflow-hidden">
          <SettingsEditor settingsKey="skill_settings" />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SettingsPanel;
