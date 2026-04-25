import React from "react";
import { IDockviewPanelProps } from "dockview";
import { useAtom } from "jotai";
import Field from "@/views/Field";
import { selectedPlayerIdAtom } from "@/api";

/**
 * Field Panel - Main visualization of robots, ball, field, debug shapes.
 * Central viewport in the Dockview layout.
 */
const FieldPanel: React.FC<IDockviewPanelProps> = () => {
  const [selectedPlayerId, setSelectedPlayerId] = useAtom(selectedPlayerIdAtom);

  return (
    <div className="w-full h-full bg-bg-void p-2">
      <div className="flex justify-center items-center w-full h-full overflow-auto">
        <Field
          selectedPlayerId={selectedPlayerId}
          onSelectPlayer={setSelectedPlayerId}
        />
      </div>
    </div>
  );
};

export default FieldPanel;
