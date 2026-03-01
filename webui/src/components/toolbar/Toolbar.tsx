import React from "react";
import { Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SimpleTooltip } from "@/components/ui/tooltip";
import ModeToggle from "./ModeToggle";
import ExecutorControls from "./ExecutorControls";
import TeamIndicator from "./TeamIndicator";
import StatusCluster from "./StatusCluster";
import LayoutSelector from "./LayoutSelector";
import logo from "@/assets/mercury-logo.svg";

/**
 * Compact 32px toolbar following Dies mission control aesthetic.
 *
 * Layout:
 * [Logo] | [SIM|LIV] | [▶ ⏸ ⏹] | [Team/Side] | ═══ spacer ═══ | [Layout ▼] | [⚙] | [● STATUS dt]
 */

interface ToolbarProps {
  onOpenSettings?: () => void;
  onExecutorStart?: () => void;
  onExecutorStop?: () => void;
  onResetLayout?: () => void;
}

const Toolbar: React.FC<ToolbarProps> = ({
  onOpenSettings,
  onExecutorStart,
  onExecutorStop,
  onResetLayout,
}) => {
  return (
    <div className="h-10 min-h-10 flex items-center gap-2.5 bg-bg-surface border-b border-border-subtle px-3 select-none">
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

      {/* Layout Selector */}
      <LayoutSelector onResetToDefault={onResetLayout} />

      {/* Separator */}
      <div className="w-px h-5 bg-border-muted" />

      {/* Settings Button */}
      <SimpleTooltip title="Settings">
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

