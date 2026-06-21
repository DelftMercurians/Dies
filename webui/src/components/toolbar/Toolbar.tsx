import React from "react";
import { Settings, PanelBottom } from "lucide-react";
import { DockviewApi } from "dockview";
import { Button } from "@/components/ui/button";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { toggleDrawer } from "@/lib/drawer";
import ModeToggle from "./ModeToggle";
import ExecutorControls from "./ExecutorControls";
import GcControls from "./GcControls";
import TeamIndicator from "./TeamIndicator";
import StatusCluster from "./StatusCluster";
import LayoutSelector from "./LayoutSelector";
import AddPanelMenu from "./AddPanelMenu";
import ShortcutIndicator from "./ShortcutIndicator";
import ReplayButton from "./ReplayButton";
import logo from "@/assets/mercury-logo.svg";

/**
 * Compact 32px toolbar following Dies mission control aesthetic.
 *
 * Layout:
 * [Logo] | [SIM|LIV] | [▶ ⏸ ⏹] | [Team/Side] | ═══ spacer ═══ | [+] [Layout ▼] | [⚙] | [● STATUS dt]
 */

interface ToolbarProps {
  onOpenSettings?: () => void;
  onExecutorStart?: () => void;
  onExecutorStop?: () => void;
  onResetLayout?: () => void;
  getDockviewApi?: () => DockviewApi | null;
}

const Toolbar: React.FC<ToolbarProps> = ({
  onOpenSettings,
  onExecutorStart,
  onExecutorStop,
  onResetLayout,
  getDockviewApi,
}) => {
  return (
    <div className="relative h-10 min-h-10 flex items-center gap-2.5 bg-bg-surface border-b border-border-subtle px-3 select-none">
      {/* Game-controller quick actions (Stop/Continue) — sim only, centered in
          the bar directly above the score banner */}
      <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-10">
        <GcControls />
      </div>

      {/* Logo */}
      <div className="flex items-center justify-center w-7 h-7 shrink-0">
        <img src={logo} alt="Dies" className="w-6 h-6" />
      </div>

      {/* Separator */}
      <div className="w-px h-5 bg-border-muted" />

      {/* Mode Toggle (SIM/LIV) */}
      <ModeToggle />

      {/* Separator */}
      <div className="w-px h-5 bg-border-muted" />

      {/* Executor Controls (Play/Pause/Stop) */}
      <ExecutorControls onStart={onExecutorStart} onStop={onExecutorStop} />

      {/* Separator */}
      <div className="w-px h-5 bg-border-muted" />

      {/* Team/Side Indicator */}
      <TeamIndicator />

      {/* Spacer */}
      <div className="flex-1" />

      {/* Keyboard shortcut indicator + cheat sheet */}
      <ShortcutIndicator />

      {/* Load log / replay control */}
      <ReplayButton />

      {/* Add Panel Menu */}
      {getDockviewApi && <AddPanelMenu getApi={getDockviewApi} />}

      {/* Layout Selector */}
      <LayoutSelector onResetToDefault={onResetLayout} />

      {/* Toggle bottom drawer (⌘J) */}
      {getDockviewApi && (
        <SimpleTooltip title="Toggle bottom drawer (⌘J)">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => {
              const api = getDockviewApi();
              if (api) toggleDrawer(api);
            }}
            className="text-text-dim hover:text-text-std"
          >
            <PanelBottom className="w-4 h-4" />
          </Button>
        </SimpleTooltip>
      )}

      {/* Separator */}
      <div className="w-px h-5 bg-border-muted" />

      {/* Settings Button */}
      <SimpleTooltip title="Settings (Motion controller, tracker, skills)">
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={onOpenSettings}
          className="text-text-dim hover:text-text-std"
        >
          <Settings className="w-4 h-4" />
        </Button>
      </SimpleTooltip>

      {/* Separator */}
      <div className="w-px h-5 bg-border-muted" />

      {/* Status Cluster */}
      <StatusCluster />
    </div>
  );
};

export default Toolbar;

