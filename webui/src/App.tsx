import { Loader } from "lucide-react";
import { useState, useRef, useEffect, useCallback } from "react";
import {
  useStatus,
  useWorldState,
  usePrimaryTeam,
  useWsConnectionStatus,
  useScriptError,
} from "./api";
import { Button } from "./components/ui/button";
import { cn, useRobotCountAlerts } from "./lib/utils";
import {
  ScriptConsoleWithRef,
  ScriptConsoleRef,
} from "./components/ScriptConsole";
import { Toolbar } from "./components/toolbar";
import DockviewWrapper, {
  DockviewWrapperRef,
} from "./components/DockviewWrapper";
import { panelComponents, PANEL_IDS } from "./components/panels";
import { DockviewApi } from "dockview";

const App: React.FC = () => {
  const { data: backendState, status: backendLoadingState } = useStatus();
  const worldState = useWorldState();
  const [selectedPlayerId, setSelectedPlayerId] = useState<number | null>(null);
  const [primaryTeam] = usePrimaryTeam();
  const [wsConnectionStatus] = useWsConnectionStatus();
  const dockviewRef = useRef<DockviewWrapperRef>(null);

  // Robot count sound alerts
  useRobotCountAlerts(worldState, primaryTeam);

  // Script error handling state
  const [scriptError, setScriptError] = useScriptError();
  const scriptConsoleRef = useRef<ScriptConsoleRef>(null);
  useEffect(() => {
    if (scriptError) {
      scriptConsoleRef.current?.addError(scriptError);
    }
  }, [scriptError]);

  // Handle executor start - clear script console
  const handleExecutorStart = () => {
    scriptConsoleRef.current?.clearConsole();
    setScriptError(null);
  };

  // Handle executor stop - deselect player
  const handleExecutorStop = () => {
    setSelectedPlayerId(null);
  };

  // Create default layout
  const createDefaultLayout = useCallback((api: DockviewApi) => {
    // Clear any existing panels
    api.clear();

    // Add Game Controller panel (left column)
    api.addPanel({
      id: PANEL_IDS.GAME_CONTROLLER,
      component: PANEL_IDS.GAME_CONTROLLER,
      title: "GAME CTRL",
    });

    // Add Team Overview panel (left column, below game controller)
    api.addPanel({
      id: PANEL_IDS.TEAM_OVERVIEW,
      component: PANEL_IDS.TEAM_OVERVIEW,
      title: "TEAM",
      position: {
        referencePanel: PANEL_IDS.GAME_CONTROLLER,
        direction: "below",
      },
    });

    // Add Basestation as a tab in the same group as Team Overview
    api.addPanel({
      id: PANEL_IDS.BASESTATION,
      component: PANEL_IDS.BASESTATION,
      title: "BASESTATION",
      position: {
        referencePanel: PANEL_IDS.TEAM_OVERVIEW,
        direction: "within",
      },
    });

    // Add Field panel (center, main viewport)
    api.addPanel({
      id: PANEL_IDS.FIELD,
      component: PANEL_IDS.FIELD,
      title: "FIELD",
      position: {
        referencePanel: PANEL_IDS.GAME_CONTROLLER,
        direction: "right",
      },
    });

    // Add Settings panel (below field)
    api.addPanel({
      id: PANEL_IDS.SETTINGS,
      component: PANEL_IDS.SETTINGS,
      title: "SETTINGS",
      position: {
        referencePanel: PANEL_IDS.FIELD,
        direction: "below",
      },
    });

    // Add Player Inspector panel (right column)
    api.addPanel({
      id: PANEL_IDS.PLAYER_INSPECTOR,
      component: PANEL_IDS.PLAYER_INSPECTOR,
      title: "PLAYER",
      position: {
        referencePanel: PANEL_IDS.FIELD,
        direction: "right",
      },
    });
  }, []);

  // Handle reset layout
  const handleResetLayout = useCallback(() => {
    dockviewRef.current?.resetToDefault();
  }, []);

  if (!backendState) {
    return (
      <div className="w-full h-full flex justify-center items-center bg-bg-base">
        {backendLoadingState === "error" ? (
          <div className="flex flex-col items-center gap-4">
            <h1 className="text-accent-red">
              Failed to connect to the backend
            </h1>
            <Button onClick={() => window.location.reload()}>Retry</Button>
          </div>
        ) : (
          <Loader className="animate-spin h-16 w-16 text-text-dim" />
        )}
      </div>
    );
  }

  const { executor: executorStatus } = backendState;

  return (
    <main className="w-full h-full flex flex-col bg-bg-base text-text-std">
      {/* Compact Toolbar */}
      <Toolbar
        onExecutorStart={handleExecutorStart}
        onExecutorStop={handleExecutorStop}
        onResetLayout={handleResetLayout}
      />

      {/* Main content - Dockview */}
      <div className="flex-1 overflow-hidden">
        <DockviewWrapper
          ref={dockviewRef}
          components={panelComponents}
          onCreateDefaultLayout={createDefaultLayout}
          selectedPlayerId={selectedPlayerId}
          onSelectPlayer={setSelectedPlayerId}
        />
      </div>

      {/* Statusbar - simplified, most status now in toolbar */}
      <div
        className={cn(
          "w-full text-[10px] px-3 py-0.5 select-none flex justify-between items-center",
          "bg-bg-surface border-t border-border-subtle font-medium uppercase tracking-wider",
          executorStatus.type === "RunningExecutor" &&
            worldState.status === "connected" &&
            "bg-accent-green/10 border-accent-green/30",
          (backendLoadingState === "error" ||
            executorStatus.type === "Failed") &&
            "bg-accent-red/10 border-accent-red/30"
        )}
      >
        <div className="text-text-dim">
          {backendLoadingState === "error"
            ? "Backend connection failed"
            : executorStatus.type === "Failed"
            ? "Executor failed"
            : executorStatus.type === "RunningExecutor"
            ? "Executor running"
            : "Ready"}
        </div>
        <div className="flex items-center gap-3 text-text-dim">
          <span>
            WS: {wsConnectionStatus.connected ? "Connected" : "Disconnected"}
          </span>
          {wsConnectionStatus.connected && wsConnectionStatus.dt !== null && (
            <span>dt: {(wsConnectionStatus.dt * 1000).toFixed(1)}ms</span>
          )}
        </div>
      </div>

      {/* Script Console */}
      <ScriptConsoleWithRef ref={scriptConsoleRef} />
    </main>
  );
};

export default App;
