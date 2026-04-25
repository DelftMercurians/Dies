import React from "react";
import { IDockviewPanelProps } from "dockview";
import { useAtom } from "jotai";
import PlayerSidebar from "@/views/PlayerSidebar";
import { selectedPlayerIdAtom } from "@/api";

/**
 * Player Inspector Panel - Detailed view of selected player.
 * Shows position, velocity, heading, current skill, debug values.
 */
const PlayerInspectorPanel: React.FC<IDockviewPanelProps> = () => {
  const [selectedPlayerId, setSelectedPlayerId] = useAtom(selectedPlayerIdAtom);

  const handleClose = () => {
    setSelectedPlayerId(null);
  };

  return (
    <div className="w-full h-full bg-bg-surface overflow-auto">
      <PlayerSidebar
        selectedPlayerId={selectedPlayerId}
        onClose={handleClose}
      />
    </div>
  );
};

export default PlayerInspectorPanel;
