import React from "react";
import { IDockviewPanelProps } from "dockview";
import { useSetAtom } from "jotai";
import Basestation from "@/views/Basestation";
import { selectedPlayerIdAtom } from "@/api";

/**
 * Basestation Panel - Hardware status for live mode.
 * Shows connected robots, battery levels, connection quality.
 */
const BasestationPanel: React.FC<IDockviewPanelProps> = () => {
  const setSelectedPlayerId = useSetAtom(selectedPlayerIdAtom);

  return (
    <div className="w-full h-full bg-bg-surface overflow-auto">
      <Basestation
        onSelectPlayer={setSelectedPlayerId}
        className="h-full"
      />
    </div>
  );
};

export default BasestationPanel;
