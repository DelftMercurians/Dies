import React, {
  useEffect,
  PropsWithChildren,
  FC,
  useRef,
  useContext,
  createContext,
  useState,
} from "react";
import {
  QueryClient,
  QueryClientProvider,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  UiMode,
  UiStatus,
  UiCommand,
  WorldData,
  PostUiModeBody,
  PostUiCommandBody,
  UiWorldState,
  ExecutorInfoResponse,
  ExecutorInfo,
  ExecutorSettingsResponse,
  ExecutorSettings,
  PostExecutorSettingsBody,
  BasestationResponse,
  WsMessage,
  DebugMap,
  GetDebugMapResponse,
} from "./bindings";
import { toast } from "sonner";

export type Status =
  | { status: "loading" }
  | { status: "connected"; data: UiStatus }
  | { status: "error" };

export type WorldStatus =
  | { status: "none" }
  | { status: "loading" }
  | { status: "connected"; data: WorldData };

const queryClient = new QueryClient();
const WsConnectedContext = createContext(false);

const getWorldState = (): Promise<UiWorldState> =>
  fetch("/api/world-state").then((res) => res.json());

const getUiStatus = (): Promise<UiStatus> =>
  fetch("/api/ui-status").then((res) => res.json());

const getScenarios = (): Promise<string[]> =>
  fetch("/api/scenarios").then((res) => res.json());

const getExecutorInfo = (): Promise<ExecutorInfo | null> =>
  fetch("/api/executor")
    .then((res) => res.json())
    .then((data) => (data as ExecutorInfoResponse).info ?? null);

const getExecutorSettings = (): Promise<ExecutorSettings> =>
  fetch("/api/settings")
    .then((res) => res.json() as Promise<ExecutorSettingsResponse>)
    .then((data) => data.settings);

const getBasestationInfo = (): Promise<BasestationResponse> =>
  fetch("/api/basestation").then((res) => res.json());

const postExecutorSettings = (settings: ExecutorSettings) =>
  fetch("/api/settings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ settings } satisfies PostExecutorSettingsBody),
  });

const getDebugMap = (): Promise<DebugMap> =>
  fetch("/api/debug")
    .then((res) => res.json())
    .then((data: GetDebugMapResponse) => data.debug_map);

const postUiMode = (mode: UiMode) =>
  fetch("/api/ui-mode", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ mode } satisfies PostUiModeBody),
  });

async function postCommand(command: UiCommand) {
  await fetch("/api/command", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ command } satisfies PostUiCommandBody),
  });
}

export const useStatus = () =>
  useQuery({
    queryKey: ["status"],
    queryFn: getUiStatus,
    refetchInterval: 1500,
  });

export const useScenarios = (): string[] | null => {
  const query = useQuery({
    queryKey: ["scenarios"],
    queryFn: getScenarios,
    staleTime: Infinity,
  });

  return query.data ?? null;
};

export const useExecutorInfo = (): ExecutorInfo | null => {
  const query = useQuery({
    queryKey: ["executor-info"],
    queryFn: getExecutorInfo,
    refetchInterval: 1000,
  });

  return query.data ?? null;
};

export const useSetMode = () => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: postUiMode,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });
};

export const useBasestationInfo = () =>
  useQuery({
    queryKey: ["basestation"],
    queryFn: getBasestationInfo,
    refetchInterval: 1000,
  });

export const useSendCommand = () => {
  const queryClient = useQueryClient();
  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["status"] });
    queryClient.invalidateQueries({ queryKey: ["executor-info"] });
    queryClient.invalidateQueries({ queryKey: ["world-state"] });
  };
  const isWsConnected = useContext(WsConnectedContext);
  const mutation = useMutation({
    mutationFn: postCommand,
    onSuccess: invalidate,
  });

  if (isWsConnected) {
    return (command: UiCommand) => {
      sendWsCommand(command);
      invalidate();
    };
  }

  return (command: UiCommand) => mutation.mutate(command);
};

export const useWorldState = (): WorldStatus => {
  const wsConnected = useContext(WsConnectedContext);
  const query = useQuery({
    queryKey: ["world-state"],
    queryFn: getWorldState,
    refetchInterval: 100,
    enabled: !wsConnected,
  });

  if (query.isSuccess) {
    if (query.data.type === "Loaded") {
      return { status: "connected", data: query.data.data };
    } else {
      return { status: "none" };
    }
  }
  return { status: "loading" };
};

export const useDebugData = (): DebugMap | null => {
  const wsConnected = useContext(WsConnectedContext);
  const query = useQuery({
    queryKey: ["debug-map"],
    queryFn: getDebugMap,
    refetchInterval: 100,
    enabled: !wsConnected,
  });

  return query.data ?? null;
};

export const useExecutorSettings = () => {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["controller-settings"],
    queryFn: getExecutorSettings,
  });

  const mutation = useMutation({
    mutationFn: postExecutorSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["controller-settings"] });
    },
  });

  return {
    settings: query.data,
    updateSettings: mutation.mutate,
  };
};

let ws: WebSocket | null = null;
const onWsConnectedChange: ((connected: boolean) => void)[] = [];
const addWsConnectedListener = (cb: (connected: boolean) => void) => {
  onWsConnectedChange.push(cb);
};
const removeWsConnectedListener = (cb: (connected: boolean) => void) => {
  const idx = onWsConnectedChange.indexOf(cb);
  if (idx >= 0) onWsConnectedChange.splice(idx, 1);
};
export const sendWsCommand = (command: UiCommand) => {
  if (ws) {
    ws.send(JSON.stringify(command));
  } else {
    throw new Error("Websocket not connected");
  }
};
export function startWsClient() {
  const notify = (connected: boolean) => {
    onWsConnectedChange.forEach((cb) => cb(connected));
  };

  const connectAndListen = (): Promise<void> => {
    if (ws) return Promise.resolve();
    return new Promise((_, reject) => {
      ws = new WebSocket(`ws://127.0.0.1:5555/api/ws`);

      ws.onopen = () => {
        toast.success("WebSocket connected");
        notify(true);
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as WsMessage;
        if (msg.type === "WorldUpdate") {
          queryClient.setQueryData(["world-state"], {
            type: "Loaded",
            data: msg.data,
          } satisfies UiWorldState);
        } else if (msg.type === "Debug") {
          queryClient.setQueryData(["debug-map"], msg.data satisfies DebugMap);
        }
      };
      ws.onerror = (err) => {
        if (ws) {
          ws.onmessage = () => {};
          ws.onclose = () => {};
          ws.close();
        }
        notify(false);
        reject(err);
      };
      ws.onclose = () => {
        notify(false);
        reject(new Error("Websocket closed"));
      };
    });
  };

  const run = async () => {
    while (true) {
      try {
        await connectAndListen();
      } catch (err) {
        toast.error("WebSocket unexpectadly closed");
        console.error("Error in WebSocket connection", err);
      }
      if (ws) {
        ws.close();
        ws = null;
      }
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  };
  run();
}

export const useKeyboardControl = ({
  playerId,
  angularSpeedDegPerSec,
  speed,
  mode = "global",
  fanSpeed,
  kickSpeed,
  kick
}: {
  playerId: number | null;
  speed: number;
  angularSpeedDegPerSec: number;
  mode: "local" | "global";
  fanSpeed: number;
  kickSpeed: number;
  kick: boolean;
}) => {
  const sendCommand = useSendCommand();

  const speedRef = useRef(speed);
  speedRef.current = speed;
  const angularSpeedRef = useRef(angularSpeedDegPerSec);
  angularSpeedRef.current = angularSpeedDegPerSec;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const fanSpeedRef = useRef(fanSpeed);
  fanSpeedRef.current = fanSpeed;
  const kickRef = useRef(kick);
  kickRef.current = kick;
  const kickSpeedRef = useRef(kickSpeed);
  kickSpeedRef.current = kickSpeed;
  useEffect(() => {
    if (playerId === null) return;
    const pressedKeys = new Set<string>();
    const handleKeyDown = (ev: KeyboardEvent) => {
      if (ev.repeat) return;
      pressedKeys.add(ev.key);
    };
    const handleKeyUp = (ev: KeyboardEvent) => {
      pressedKeys.delete(ev.key);
    };

    const interval = setInterval(() => {
      if (playerId === null) return;

      const command = {
        type: "OverrideCommand",
        data: {
          player_id: playerId,
          command: {
            type:
              modeRef.current === "global" ? "GlobalVelocity" : "LocalVelocity",
            data: {
              velocity: [0, 0] as [number, number],
              angular_velocity: 0,
              arm_kick: false,
              dribble_speed: 0,
            },
          },
        },
      } satisfies UiCommand;

      let velocity = [0, 0] as [number, number];
      if (pressedKeys.has("w")) velocity[1] += 1;
      if (pressedKeys.has("s")) velocity[1] -= 1;
      if (pressedKeys.has("a")) velocity[0] -= 1;
      if (pressedKeys.has("d")) velocity[0] += 1;
      const vel_mag = Math.sqrt(velocity[0] ** 2 + velocity[1] ** 2);
      if (vel_mag > 0) {
        velocity = velocity.map((v) => (v / vel_mag) * speedRef.current) as [
          number,
          number,
        ];
      }
      command.data.command.data.velocity = velocity;

      const angularSpeedRadPerSec = (angularSpeedRef.current * Math.PI) / 180;
      let angular_velocity = 0;
      if (pressedKeys.has("q")) angular_velocity += angularSpeedRadPerSec;
      if (pressedKeys.has("e")) angular_velocity -= angularSpeedRadPerSec;
      command.data.command.data.angular_velocity = angular_velocity;

      let dribble_speed = 0;
      if (pressedKeys.has(" ")) {
        dribble_speed = 1;
      }
      command.data.command.data.dribble_speed = dribble_speed;

      if (pressedKeys.has("c")) {
        const command = {
          type: "OverrideCommand",
          data: {
            player_id: playerId,
            command: {
              type: "Kick",
              data: {
                speed: kickSpeedRef.current,
              },
            },
          },
        } satisfies UiCommand;
        sendCommand(command as UiCommand);
      } 

      if (vel_mag > 0 || angular_velocity !== 0 || dribble_speed > 0) {
        sendCommand(command as UiCommand);
      }

      if (fanSpeed) {
        const command = {
          type: "OverrideCommand",
          data: {
            player_id: playerId,
            command: {
              type: "SetFanSpeed",
              data: {
                speed: fanSpeedRef.current,
              },
            },
          },
        } satisfies UiCommand;
        // sendCommand(command as UiCommand);
      }
      if (kickRef.current) {
        console.log("kick");
        // kickRef.current = false; // doesn't work :/
        const command = {
          type: "OverrideCommand",
          data: {
            player_id: playerId,
            command: {
              type: "Kick",
              data: {
                speed: kickSpeedRef.current,
              },
            },
          },
        } satisfies UiCommand;
        // sendCommand(command as UiCommand);
      }
    }, 1000 / 30);

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      clearInterval(interval);
    };
  }, [playerId]);
};

export const WorldDataProvider: FC<PropsWithChildren> = ({ children }) => {
  const [connected, setConnected] = useState(false);
  useEffect(() => {
    addWsConnectedListener(setConnected);
    return () => {
      removeWsConnectedListener(setConnected);
    };
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <WsConnectedContext.Provider value={connected}>
        {children}
      </WsConnectedContext.Provider>
    </QueryClientProvider>
  );
};
