import React from "react";
import { IDockviewPanelProps } from "dockview";
import PlayerSidebar from "@/views/PlayerSidebar";

/**
 * Player Inspector Panel - Detailed view of selected player.
 * Shows position, velocity, heading, current skill, debug values.
 */
const PlayerInspectorPanel: React.FC<IDockviewPanelProps> = ({ params }) => {
  const selectedPlayerId = params?.selectedPlayerId as number | null;
  const onSelectPlayer = params?.onSelectPlayer as (id: number | null) => void;

  const handleClose = () => {
    onSelectPlayer?.(null);
  };

  return (
    <div className="w-full h-full bg-bg-surface overflow-auto">
      <PlayerSidebar
        selectedPlayerId={selectedPlayerId ?? null}
        onClose={handleClose}
      />
    </div>
  );
};

export default PlayerInspectorPanel;

