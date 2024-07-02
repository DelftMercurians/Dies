import { useState } from "react";
import {
  useScenarios,
  useSendCommand,
  useSetMode,
  useStatus,
  useWorldState,
} from "./api";
import Field from "./views/Field";
import {
  Select,
  SelectItem,
  SelectContent,
  SelectTrigger,
  SelectValue,
} from "./components/ui/select";
import { Button } from "./components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "./components/ui/toggle-group";
import {
  LaptopMinimal,
  Loader,
  Pause,
  Play,
  Radio,
  Square,
} from "lucide-react";
import { toast } from "sonner";
import {
  SimpleTooltip,
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "./components/ui/tooltip";
import logo from "./assets/mercury-logo.svg";
import { log } from "console";
import { cn } from "./lib/utils";

const App: React.FC = () => {
  const scenarios = useScenarios() ?? [];
  const [selectedScenario, setSelectedScenario] = useState<null | string>(null);
  const { data: backendState, status: backendLoadingState } = useStatus();
  const worldState = useWorldState();
  const { mutate: setMode, status: setModeStatus } = useSetMode();
  const sendCommand = useSendCommand();

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
      sendCommand({ type: "Stop" });
    } else {
      toast.error(`Unhandled state ${val}`);
    }
  };

  return (
    <main className="w-full h-full flex flex-col bg-background bg-slate-100">
      {/* Toolbar */}
      <div className="flex flex-row gap-6 bg-slate-800 p-4 text-accent">
        <img src={logo} width="45px" height="41px" />

        <ToggleGroup
          type="single"
          value={uiMode}
          onValueChange={setMode}
          disabled={setModeStatus === "pending"}
          className="border border-gray-500 rounded-lg"
        >
          <SimpleTooltip title="Simulation">
            <ToggleGroupItem value="Simulation">
              <LaptopMinimal />
            </ToggleGroupItem>
          </SimpleTooltip>

          <SimpleTooltip title="Live">
            <ToggleGroupItem value="Live" disabled={!isLiveAvailable}>
              <Radio />
            </ToggleGroupItem>
          </SimpleTooltip>
        </ToggleGroup>

        <Select
          value={selectedScenario ?? undefined}
          onValueChange={(val) => setSelectedScenario(val)}
        >
          <SelectTrigger className="text-accent-foreground w-64">
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
            <ToggleGroupItem value="play" disabled={!selectedScenario}>
              <Play />
            </ToggleGroupItem>
          </SimpleTooltip>

          <SimpleTooltip title="Pause">
            <ToggleGroupItem value="pause" disabled={playingState !== "play"}>
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

      <Field />

      {/* Statusbar */}
      <div
        className={cn(
          "w-full text-accent text-sm px-4 py-1 select-none",
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
