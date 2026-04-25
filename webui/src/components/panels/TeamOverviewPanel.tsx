import React from "react";
import { IDockviewPanelProps } from "dockview";
import { useAtom } from "jotai";
import TeamOverview from "@/views/TeamOverview";
import { selectedPlayerIdAtom } from "@/api";

/**
 * Team Overview Panel - Team roster and status at a glance.
 * Shows list of players with ID, status indicator, quick stats.
 */
const TeamOverviewPanel: React.FC<IDockviewPanelProps> = () => {
  const [selectedPlayerId, setSelectedPlayerId] = useAtom(selectedPlayerIdAtom);

  return (
    <div className="w-full h-full bg-bg-surface">
      <TeamOverview
        onSelectPlayer={setSelectedPlayerId}
        selectedPlayerId={selectedPlayerId}
        className="h-full"
      />
    </div>
  );
};

export default TeamOverviewPanel;
