import React from "react";
import { IDockviewPanelProps } from "dockview";
import GameController from "@/components/GameControllerPanel";

/**
 * Game Controller Panel - Send game controller commands.
 * Contains button groups for Halt, Stop, Start, Kickoffs, Penalties, etc.
 */
const GameControllerPanelWrapper: React.FC<IDockviewPanelProps> = () => {
  return (
    <div className="w-full h-full bg-bg-surface p-2 overflow-auto">
      <GameController />
    </div>
  );
};

export default GameControllerPanelWrapper;

