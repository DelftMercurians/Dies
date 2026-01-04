import React, { useState } from "react";
import { IDockviewPanelProps } from "dockview";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import SettingsEditor from "@/views/SettingsEditor";

/**
 * Settings Panel - Controller, Tracker, and Skill settings.
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
          <TabsTrigger value="tracker">Tracker</TabsTrigger>
          <TabsTrigger value="skill">Skill</TabsTrigger>
        </TabsList>

        <TabsContent value="controller" className="flex-1 overflow-hidden">
          <SettingsEditor settingsKey="controller_settings" />
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

