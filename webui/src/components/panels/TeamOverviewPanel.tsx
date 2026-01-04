import React from "react";
import { IDockviewPanelProps } from "dockview";
import TeamOverview from "@/views/TeamOverview";

/**
 * Team Overview Panel - Team roster and status at a glance.
 * Shows list of players with ID, status indicator, quick stats.
 */
const TeamOverviewPanel: React.FC<IDockviewPanelProps> = ({ params }) => {
  const selectedPlayerId = params?.selectedPlayerId as number | null;
  const onSelectPlayer = params?.onSelectPlayer as (id: number | null) => void;

  return (
    <div className="w-full h-full bg-bg-surface">
      <TeamOverview
        onSelectPlayer={onSelectPlayer ?? (() => {})}
        selectedPlayerId={selectedPlayerId ?? null}
        className="h-full"
      />
    </div>
  );
};

export default TeamOverviewPanel;

