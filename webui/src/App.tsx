import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  LaptopMinimal,
  Loader,
  Pause,
  Play,
  Radio,
  Square,
} from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";
import {
  useBasestationInfo,
  useDebugData,
  useScenarios,
  useSendCommand,
  useSetMode,
  useStatus,
  useWorldState,
} from "./api";
import logo from "./assets/mercury-logo.svg";
import { Button } from "./components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { ToggleGroup, ToggleGroupItem } from "./components/ui/toggle-group";
import { SimpleTooltip } from "./components/ui/tooltip";
import { cn, useIsOverflow, useWarningSound } from "./lib/utils";
import Field from "./views/Field";
import PlayerSidebar from "./views/PlayerSidebar";
import SettingsEditor from "./views/SettingsEditor";
import Basestation from "./views/Basestation";
import HierarchicalList from "./views/HierarchicalList";
import { PlayerFeedbackMsg } from "./bindings";

type Panel = "left" | "right";

const App: React.FC = () => {
  const scenarios = useScenarios() ?? [];
  const [selectedScenario, setSelectedScenario] = useState<null | string>(null);
  const { data: backendState, status: backendLoadingState } = useStatus();
  const worldState = useWorldState();
  const { mutate: setMode, status: setModeStatus } = useSetMode();
  const sendCommand = useSendCommand();
  const [selectedPlayerId, setSelectedPlayerId] = useState<number | null>(null);
  const [collapsed, setCollapsed] = useState<Panel[]>([]);
  const debugData = useDebugData();
  const tabListRef = useRef<HTMLDivElement>(null);
  const isTabListOverflowing = useIsOverflow(tabListRef, "horizontal");

  const bsInfo = useBasestationInfo().data;
  const allMotorsOk = Object.values(bsInfo?.players ?? {}).every((p: PlayerFeedbackMsg) => p.motor_statuses?.find((m) => m === "NoReply") === undefined);
  useWarningSound(!allMotorsOk);

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
    if (val === "play" && playingState !== "play" && selectedScenario) {
      sendCommand({
        type: "StartScenario",
        data: { scenario: selectedScenario },
      });
    } else if (val === "stop" && playingState !== "stop") {
      setSelectedPlayerId(null);
      sendCommand({ type: "Stop" });
    } else {
      toast.error(`Unhandled state ${val}`);
    }
  };
  const runningScenario =
    executorStatus.type === "RunningExecutor"
      ? executorStatus.data.scenario
      : null;

  return (
    <main className="w-full h-full flex flex-col bg-background bg-slate-100">
      {/* Toolbar */}
      <div className="flex flex-row gap-6 bg-slate-800 p-4">
        <img src={logo} width="45px" height="41px" />

        <ToggleGroup
          type="single"
          value={uiMode}
          onValueChange={setMode}
          // disabled={setModeStatus === "pending"}
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

        <Select
          value={
            runningScenario ? runningScenario : selectedScenario ?? undefined
          }
          onValueChange={(val) => setSelectedScenario(val)}
          disabled={!!runningScenario}
        >
          <SelectTrigger className="w-64">
            <SelectValue placeholder="Select Scenario" />
          </SelectTrigger>

          <SelectContent>
            {scenarios.map((scenario) => (
              <SelectItem key={scenario} value={scenario}>
                {scenario}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <ToggleGroup
          type="single"
          value={playingState}
          onValueChange={handleSetPlayState}
          disabled={executorStatus.type === "StartingScenario"}
          className="border border-gray-500 rounded-lg"
        >
          <SimpleTooltip
            title={
              selectedScenario
                ? "Start selected scenario"
                : "Select a scenario first"
            }
          >
            <ToggleGroupItem
              value="play"
              disabled={!selectedScenario}
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
      </div>

      {/* Main content */}
      <ResizablePanelGroup autoSaveId="main-layout" direction="horizontal">
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
            <Tabs
              defaultValue="controller"
              className="h-full w-full flex flex-col gap-2"
              orientation={isTabListOverflowing ? "vertical" : "horizontal"}
            >
              <TabsList
                ref={tabListRef}
                className="w-full data-[orientation=vertical]:flex-col data-[orientation=vertical]:h-auto data-[orientation=vertical]:w-auto"
              >
                <TabsTrigger value="controller">Controller</TabsTrigger>
                <TabsTrigger value="tracker">Tracker</TabsTrigger>
                <TabsTrigger value="basestation">Basestation</TabsTrigger>
                <TabsTrigger value="debug">Debug Values</TabsTrigger>
              </TabsList>

              <TabsContent value="controller" asChild>
                <SettingsEditor settingsKey="controller_settings" />
              </TabsContent>
              <TabsContent value="tracker" asChild>
                <SettingsEditor settingsKey="tracker_settings" />
              </TabsContent>
              <TabsContent value="basestation" asChild>
                <Basestation className="h-full" />
              </TabsContent>
              <TabsContent value="debug" asChild>
                <div className="bg-slate-800 p-2 rounded-xl h-full overflow-auto">
                  <HierarchicalList
                    data={debugData ? Object.entries(debugData) : []}
                    className="h-full"
                  />
                </div>
              </TabsContent>
            </Tabs>
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
          executorStatus.type === "StartingScenario" && "bg-yellow-500",
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
              : executorStatus.type === "StartingScenario"
                ? "Starting scenario"
                : "Idle"}
      </div>
    </main>
  );
};

export default App;
