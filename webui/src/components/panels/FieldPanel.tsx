import React from "react";
import { IDockviewPanelProps } from "dockview";
import Field from "@/views/Field";

/**
 * Field Panel - Main visualization of robots, ball, field, debug shapes.
 * Central viewport in the Dockview layout.
 */
const FieldPanel: React.FC<IDockviewPanelProps> = ({ params }) => {
  const selectedPlayerId = params?.selectedPlayerId as number | null;
  const onSelectPlayer = params?.onSelectPlayer as (id: number | null) => void;

  return (
    <div className="w-full h-full bg-bg-void p-2">
      <div className="flex justify-center items-center w-full h-full overflow-auto">
        <Field
          selectedPlayerId={selectedPlayerId ?? null}
          onSelectPlayer={onSelectPlayer ?? (() => {})}
        />
      </div>
    </div>
  );
};

export default FieldPanel;

