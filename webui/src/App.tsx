import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LaptopMinimal,
  Loader,
  Pause,
  Play,
  Radio,
  Square,
} from "lucide-react";
import { useState, useRef, useEffect } from "react";
import { toast } from "sonner";
import {
  useBasestationInfo,
  useSendCommand,
  useSetMode,
  useStatus,
  useWorldState,
  useExecutorInfo,
  useRawWorldData,
  useScriptError,
} from "./api";
import logo from "./assets/mercury-logo.svg";
import { Button } from "./components/ui/button";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { ToggleGroup, ToggleGroupItem } from "./components/ui/toggle-group";
import { SimpleTooltip } from "./components/ui/tooltip";
import { cn, useWarningSound } from "./lib/utils";
import Field from "./views/Field";
import PlayerSidebar from "./views/PlayerSidebar";
import SettingsEditor from "./views/SettingsEditor";
import Basestation from "./views/Basestation";
import TeamOverview from "./views/TeamOverview";
import { PlayerFeedbackMsg, ScriptError } from "./bindings";
import TeamSettingsDialog from "./components/TeamSettingsDialog";
import GameControllerPanel from "./components/GameControllerPanel";
import PrimaryTeamSelector from "./components/PrimaryTeamSelector";
import TeamSwapControls from "./components/TeamSwapControls";
import { ScriptErrorDialog } from "./components/ScriptErrorDialog";
import {
  ScriptConsoleWithRef,
  ScriptConsoleRef,
} from "./components/ScriptConsole";

type Panel = "left" | "right" | "left-bottom" | "game-controller";

const App: React.FC = () => {
  const { data: backendState, status: backendLoadingState } = useStatus();
  const worldState = useWorldState();
  const rawWorldData = useRawWorldData();
  const { mutate: setMode, status: setModeStatus } = useSetMode();
  const sendCommand = useSendCommand();
  const [selectedPlayerId, setSelectedPlayerId] = useState<number | null>(null);
  const [collapsed, setCollapsed] = useState<Panel[]>(["left-bottom"]);
  const executorInfo = useExecutorInfo();

  // Script error handling state
  const [scriptError, setScriptError] = useScriptError();
  const scriptConsoleRef = useRef<ScriptConsoleRef>(null);
  useEffect(() => {
    if (scriptError) {
      scriptConsoleRef.current?.addError(scriptError);
    }
  }, [scriptError]);

  if (!backendState) {
    return (
      <div className="w-full h-full flex justify-center items-center bg-slate-100">
        {backendLoadingState === "error" ? (
          <div className="flex flex-col items-center gap-4">
            <h1 className="text-red-900">Failed to connect to the backend</h1>
            <Button onClick={() => window.location.reload()}>Retry</Button>
          </div>
        ) : (
          <Loader className="animate-spin h-16 w-16" />
        )}
      </div>
    );
  }
  const {
    executor: executorStatus,
    is_live_available: isLiveAvailable,
    ui_mode: uiMode,
  } = backendState;
  const playingState =
    executorStatus.type === "None" || executorStatus.type === "Failed"
      ? "stop"
      : "play";
  const handleSetPlayState = (val: string) => {
    if (val === "play" && playingState !== "play") {
      // Clear script errors when starting a new run
      scriptConsoleRef.current?.clearConsole();
      setScriptError(null);

      sendCommand({
        type: "Start",
      });
    } else if (val === "stop" && playingState !== "stop") {
      setSelectedPlayerId(null);
      sendCommand({ type: "Stop" });
    } else {
      toast.error(`Unhandled state ${val}`);
    }
  };

  return (
    <main className="w-full h-full flex flex-col bg-background bg-slate-100">
      {/* Toolbar */}
      <div className="flex flex-row gap-6 bg-slate-800 p-4">
        <img src={logo} width="45px" height="41px" />

        <ToggleGroup
          type="single"
          value={uiMode}
          onValueChange={setMode}
          className="border border-gray-500 rounded-lg"
        >
          <SimpleTooltip title="Simulation">
            <ToggleGroupItem
              value="Simulation"
              className="data-[state=on]:bg-green-600"
            >
              <LaptopMinimal />
            </ToggleGroupItem>
          </SimpleTooltip>

          <SimpleTooltip title="Live">
            <ToggleGroupItem
              value="Live"
              disabled={!isLiveAvailable}
              className="data-[state=on]:bg-green-600"
            >
              <Radio />
            </ToggleGroupItem>
          </SimpleTooltip>
        </ToggleGroup>

        <ToggleGroup
          type="single"
          value={playingState}
          onValueChange={handleSetPlayState}
          className="border border-gray-500 rounded-lg"
        >
          <SimpleTooltip title="Start">
            <ToggleGroupItem
              value="play"
              className="data-[state=on]:bg-green-400 data-[state=on]:opacity-100  data-[state=on]:text-gray-500"
            >
              <Play />
            </ToggleGroupItem>
          </SimpleTooltip>

          <SimpleTooltip title="Pause">
            <ToggleGroupItem
              value="pause"
              disabled={playingState !== "play"}
              className="data-[state=on]:bg-yellow-400 data-[state=on]:opacity-100  data-[state=on]:text-gray-500"
            >
              <Pause />
            </ToggleGroupItem>
          </SimpleTooltip>

          <SimpleTooltip title="Terminate executor">
            <ToggleGroupItem
              className="hover:bg-red-500 hover:text-accent-foreground"
              value="stop"
              disabled={playingState !== "play"}
            >
              <Square />
            </ToggleGroupItem>
          </SimpleTooltip>
        </ToggleGroup>

        {/* Team Configuration */}
        <div className="flex items-center gap-4">
          <TeamSettingsDialog />
          <TeamSwapControls />
          <PrimaryTeamSelector />
        </div>
      </div>

      {/* Main content */}
      <ResizablePanelGroup autoSaveId="main-layout" direction="horizontal">
        {/* Game Controller Panel */}
        <ResizablePanel
          defaultSize={15}
          minSize={12}
          maxSize={25}
          collapsible
          className="h-full bg-slate-950 p-2"
          onCollapse={() =>
            setCollapsed((prev) =>
              !prev.includes("game-controller")
                ? [...prev, "game-controller"]
                : prev
            )
          }
          onExpand={() =>
            setCollapsed((prev) => prev.filter((v) => v !== "game-controller"))
          }
        >
          {!collapsed.includes("game-controller") ? (
            <GameControllerPanel />
          ) : (
            <div className="text-center text-slate-400 p-2 text-sm transform -rotate-90">
              Game Controller
            </div>
          )}
        </ResizablePanel>
        <ResizableHandle withHandle />

        {/* Team Overview Panel */}
        <ResizablePanel
          defaultSize={20}
          minSize={10}
          collapsible
          className="h-full bg-slate-950 p-2"
          onCollapse={() =>
            setCollapsed((prev) =>
              !prev.includes("left") ? [...prev, "left"] : prev
            )
          }
          onExpand={() =>
            setCollapsed((prev) => prev.filter((v) => v !== "left"))
          }
        >
          {!collapsed.includes("left") ? (
            <ResizablePanelGroup direction="vertical">
              <ResizablePanel defaultSize={70}>
                <Tabs
                  size="sm"
                  defaultValue="team_overview"
                  className="h-full w-full flex flex-col gap-2 p-2"
                >
                  <TabsList>
                    <TabsTrigger value="team_overview">Team</TabsTrigger>
                    <TabsTrigger value="basestation">Basestation</TabsTrigger>
                  </TabsList>
                  <TabsContent value="team_overview" className="flex-1">
                    <TeamOverview
                      onSelectPlayer={setSelectedPlayerId}
                      selectedPlayerId={selectedPlayerId}
                      className="h-full"
                    />
                  </TabsContent>
                  <TabsContent value="basestation" asChild>
                    <Basestation
                      onSelectPlayer={(id) => setSelectedPlayerId(id)}
                      className="h-full"
                    />
                  </TabsContent>
                </Tabs>
              </ResizablePanel>
              <ResizableHandle withHandle />
              <ResizablePanel
                defaultSize={5}
                collapsible
                collapsedSize={5}
                onCollapse={() =>
                  setCollapsed((prev) =>
                    !prev.includes("left-bottom")
                      ? [...prev, "left-bottom"]
                      : prev
                  )
                }
                onExpand={() =>
                  setCollapsed((prev) =>
                    prev.filter((v) => v !== "left-bottom")
                  )
                }
              >
                {!collapsed.includes("left-bottom") ? (
                  <Tabs
                    size="sm"
                    defaultValue="controller"
                    className="h-full w-full flex flex-col gap-2 p-2"
                  >
                    <TabsList className="w-full">
                      <TabsTrigger value="controller">Controller</TabsTrigger>
                      <TabsTrigger value="tracker">Tracker</TabsTrigger>
                    </TabsList>

                    <TabsContent value="controller" asChild>
                      <SettingsEditor settingsKey="controller_settings" />
                    </TabsContent>
                    <TabsContent value="tracker" asChild>
                      <SettingsEditor settingsKey="tracker_settings" />
                    </TabsContent>
                  </Tabs>
                ) : (
                  <div className="text-center text-slate-400 p-2 text-sm">
                    Settings
                  </div>
                )}
              </ResizablePanel>
            </ResizablePanelGroup>
          ) : null}
        </ResizablePanel>
        <ResizableHandle withHandle />

        <ResizablePanel className="bg-green-800 p-6" minSize={30}>
          <div className="flex justify-center items-center w-full h-full min-w-[600px] overflow-auto">
            <Field
              selectedPlayerId={selectedPlayerId}
              onSelectPlayer={(id) => setSelectedPlayerId(id)}
            />
          </div>
        </ResizablePanel>
        <ResizableHandle withHandle />

        <ResizablePanel
          defaultSize={20}
          minSize={10}
          collapsible
          className=" bg-slate-950 flex flex-col"
          onCollapse={() =>
            setCollapsed((prev) =>
              !prev.includes("right") ? [...prev, "right"] : prev
            )
          }
          onExpand={() =>
            setCollapsed((prev) => prev.filter((v) => v !== "right"))
          }
        >
          {!collapsed.includes("right") ? (
            <PlayerSidebar
              selectedPlayerId={selectedPlayerId}
              onClose={() => setSelectedPlayerId(null)}
            />
          ) : null}
        </ResizablePanel>
      </ResizablePanelGroup>

      {/* Statusbar */}
      <div
        className={cn(
          "w-full text-sm px-4 py-1 select-none",
          "bg-slate-800",
          executorStatus.type === "RunningExecutor" &&
            worldState.status === "connected" &&
            "bg-green-500",
          (backendLoadingState === "error" ||
            executorStatus.type === "Failed") &&
            "bg-red-500"
        )}
      >
        {backendLoadingState === "error"
          ? "Failed to connect to backend"
          : executorStatus.type === "Failed"
          ? "Executor failed"
          : executorStatus.type === "RunningExecutor"
          ? "Running"
          : "Idle"}
      </div>

      {/* Script Error Dialog */}
      {/* <ScriptErrorDialog
        open={syntaxErrorDialogOpen}
        onClose={() => setSyntaxErrorDialogOpen(false)}
        error={scriptError}
      /> */}

      {/* Script Console */}
      <ScriptConsoleWithRef ref={scriptConsoleRef} />
    </main>
  );
};

export default App;
