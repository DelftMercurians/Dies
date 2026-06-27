import React from "react";
import { Move, Bookmark } from "lucide-react";
import { useSetAtom, useAtom } from "jotai";

import { useStatus } from "@/api";
import { UiMode } from "@/bindings";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogTrigger,
} from "@/components/ui/dialog";
import SnapshotManager from "@/components/SnapshotManager";
import { simEditModeAtom, maskEditModeAtom } from "@/lib/fieldEditing";

/**
 * Sim Edit toolbar cluster (simulation only): a drag-to-edit mode toggle and a
 * button that opens the snapshot manager dialog.
 */
const SimEditControls: React.FC = () => {
  const { data: backendState } = useStatus();
  const [simEdit, setSimEdit] = useAtom(simEditModeAtom);
  const setMaskEdit = useSetAtom(maskEditModeAtom);

  if (backendState?.ui_mode !== UiMode.Simulation) return null;

  const toggle = () => {
    setSimEdit((on) => {
      if (!on) setMaskEdit(false); // mutually exclusive with mask editing
      return !on;
    });
  };

  return (
    <div className="inline-flex items-center border border-border-muted h-7">
      <SimpleTooltip
        title="Sim Edit — drag robots/ball, shift-drag ball to kick, drag the ring to rotate"
        className="h-full"
      >
        <button
          onClick={toggle}
          className={cn(
            "h-full px-2 flex items-center gap-1 text-xs transition-colors",
            "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
            simEdit
              ? "bg-accent-amber/20 text-accent-amber"
              : "text-text-muted hover:text-accent-amber hover:bg-accent-amber/10"
          )}
        >
          <Move className="w-3.5 h-3.5" />
          Edit
        </button>
      </SimpleTooltip>

      <div className="w-px h-full bg-border-subtle" />

      <Dialog>
        <SimpleTooltip title="Field snapshots (save / load board state)" className="h-full">
          <DialogTrigger asChild>
            <button
              className={cn(
                "h-full w-7 flex items-center justify-center transition-colors",
                "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
                "text-text-muted hover:text-text-std hover:bg-bg-overlay"
              )}
            >
              <Bookmark className="w-3.5 h-3.5" />
            </button>
          </DialogTrigger>
        </SimpleTooltip>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Field snapshots</DialogTitle>
            <DialogDescription>
              Save the current robot/ball layout and reload it later. Positions
              and yaw only.
            </DialogDescription>
          </DialogHeader>
          <SnapshotManager />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default SimEditControls;
