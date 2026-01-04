import React from "react";
import { IDockviewPanelProps } from "dockview";
import Basestation from "@/views/Basestation";

/**
 * Basestation Panel - Hardware status for live mode.
 * Shows connected robots, battery levels, connection quality.
 */
const BasestationPanel: React.FC<IDockviewPanelProps> = ({ params }) => {
  const onSelectPlayer = params?.onSelectPlayer as (id: number | null) => void;

  return (
    <div className="w-full h-full bg-bg-surface overflow-auto">
      <Basestation
        onSelectPlayer={onSelectPlayer ?? (() => {})}
        className="h-full"
      />
    </div>
  );
};

export default BasestationPanel;

