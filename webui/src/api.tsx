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
  skipToken,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  UiMode,
  UiStatus,
  UiCommand,
  TeamData,
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
  WorldData,
  TeamColor,
  TeamPlayerId,
  PlayerId,
  PlayerOverrideCommand,
  TeamConfiguration,
  SideAssignment,
  ScriptError,
} from "./bindings";
import { toast } from "sonner";
import { atom, getDefaultStore, useAtom } from "jotai";

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

const convertWorldDataToTeamData = (
  worldData: WorldData,
  primaryTeamColor: TeamColor = TeamColor.Blue
): TeamData => {
  const isBlue = primaryTeamColor === TeamColor.Blue;
  return {
    t_received: worldData.t_received,
    t_capture: worldData.t_capture,
    dt: worldData.dt,
    own_players: isBlue ? worldData.blue_team : worldData.yellow_team,
    opp_players: isBlue ? worldData.yellow_team : worldData.blue_team,
    ball: worldData.ball,
    field_geom: worldData.field_geom,
    current_game_state: {
      game_state: worldData.game_state.game_state,
      us_operating: worldData.game_state.operating_team === primaryTeamColor,
      yellow_cards: 0, // TODO: Get actual yellow cards from worldData
    },
  };
};

// Helper function to extract player ID from TeamPlayerId
const extractPlayerId = (teamPlayerId: TeamPlayerId): PlayerId => {
  return teamPlayerId.player_id;
};

// Helper function to check if a player is in manual control
const isPlayerManuallyControlled = (
  playerId: PlayerId,
  manualControlledPlayers: TeamPlayerId[]
): boolean => {
  return manualControlledPlayers.some((tp) => tp.player_id === playerId);
};

export const useTeamConfiguration = () => {
  const queryClient = useQueryClient();

  const setActiveTeams = useMutation({
    mutationFn: ({
      blueActive,
      yellowActive,
    }: {
      blueActive: boolean;
      yellowActive: boolean;
    }) =>
      postCommand({
        type: "SetActiveTeams",
        data: { blue_active: blueActive, yellow_active: yellowActive },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["executor-info"] });
      queryClient.invalidateQueries({ queryKey: ["controller-settings"] });
    },
  });

  const setTeamScriptPaths = useMutation({
    mutationFn: ({
      blueScriptPath,
      yellowScriptPath,
    }: {
      blueScriptPath?: string;
      yellowScriptPath?: string;
    }) =>
      postCommand({
        type: "SetTeamScriptPaths",
        data: {
          blue_script_path: blueScriptPath ?? undefined,
          yellow_script_path: yellowScriptPath ?? undefined,
        },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["executor-info"] });
      queryClient.invalidateQueries({ queryKey: ["controller-settings"] });
    },
  });

  const setSideAssignment = useMutation({
    mutationFn: (sideAssignment: SideAssignment) =>
      postCommand({
        type: "SetSideAssignment",
        data: { side_assignment: sideAssignment },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["executor-info"] });
      queryClient.invalidateQueries({ queryKey: ["controller-settings"] });
    },
  });

  const setTeamConfiguration = useMutation({
    mutationFn: (configuration: TeamConfiguration) =>
      postCommand({
        type: "SetTeamConfiguration",
        data: { configuration },
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["executor-info"] });
      queryClient.invalidateQueries({ queryKey: ["controller-settings"] });
    },
  });

  const swapTeamColors = useMutation({
    mutationFn: () => postCommand({ type: "SwapTeamColors" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["executor-info"] });
      queryClient.invalidateQueries({ queryKey: ["controller-settings"] });
    },
  });

  const swapTeamSides = useMutation({
    mutationFn: () => postCommand({ type: "SwapTeamSides" }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["executor-info"] });
      queryClient.invalidateQueries({ queryKey: ["controller-settings"] });
    },
  });

  return {
    setActiveTeams: setActiveTeams.mutate,
    setTeamScriptPaths: setTeamScriptPaths.mutate,
    setSideAssignment: setSideAssignment.mutate,
    setTeamConfiguration: setTeamConfiguration.mutate,
    swapTeamColors: () => swapTeamColors.mutate(),
    swapTeamSides: () => swapTeamSides.mutate(),
  };
};

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

const PrimaryTeamContext = createContext<
  [TeamColor, (team: TeamColor) => void]
>([TeamColor.Blue, () => {}]);

export const PrimaryTeamProvider: FC<PropsWithChildren> = ({ children }) => {
  const [primaryTeam, setPrimaryTeam] = useState<TeamColor>(TeamColor.Blue);
  return (
    <PrimaryTeamContext.Provider value={[primaryTeam, setPrimaryTeam]}>
      {children}
    </PrimaryTeamContext.Provider>
  );
};

export const usePrimaryTeam = () => {
  return useContext(PrimaryTeamContext);
};

export const useWorldState = (): WorldStatus => {
  const wsConnected = useContext(WsConnectedContext);
  const query = useQuery({
    queryKey: ["world-state"],
    queryFn: getWorldState,
    refetchInterval: () => (wsConnected ? false : 100),
    enabled: !wsConnected,
  });
  const [primaryTeam] = useContext(PrimaryTeamContext);

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

export const useRawWorldData = () => {
  const wsConnected = useContext(WsConnectedContext);
  const query = useQuery({
    queryKey: ["raw-world-state"],
    queryFn: getWorldState,
    refetchInterval: () => (wsConnected ? false : 100),
    enabled: !wsConnected,
  });

  if (query.isSuccess && query.data.type === "Loaded") {
    return query.data.data;
  }
  return null;
};

const scriptErrorAtom = atom<ScriptError | null>(null);
export const useScriptError = () => {
  return useAtom(scriptErrorAtom);
};

// WebSocket connection status and dt tracking
const wsConnectionStatusAtom = atom<{
  connected: boolean;
  lastUpdateTime: number | null;
  dt: number | null;
}>({
  connected: false,
  lastUpdateTime: null,
  dt: null,
});

export const useWsConnectionStatus = () => {
  return useAtom(wsConnectionStatusAtom);
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
        const store = getDefaultStore();
        store.set(wsConnectionStatusAtom, {
          connected: true,
          lastUpdateTime: null,
          dt: null,
        });
      };
      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data) as WsMessage;
        const store = getDefaultStore();

        if (msg.type === "WorldUpdate") {
          // Update world state
          queryClient.setQueryData(["world-state"], {
            type: "Loaded",
            data: msg.data,
          } satisfies UiWorldState);

          // Track dt from WebSocket updates
          const currentTime = Date.now();
          const currentStatus = store.get(wsConnectionStatusAtom);
          const dt = currentStatus.lastUpdateTime
            ? (currentTime - currentStatus.lastUpdateTime) / 1000
            : null;

          store.set(wsConnectionStatusAtom, {
            connected: true,
            lastUpdateTime: currentTime,
            dt: dt,
          });
        } else if (msg.type === "Debug") {
          queryClient.setQueryData(["debug-map"], msg.data satisfies DebugMap);
        } else if (msg.type === "ScriptError") {
          const scriptError = msg.data as ScriptError;
          store.set(scriptErrorAtom, scriptError);
        }
      };
      ws.onerror = (err) => {
        if (ws) {
          ws.onmessage = () => {};
          ws.onclose = () => {};
          ws.close();
        }
        notify(false);
        const store = getDefaultStore();
        store.set(wsConnectionStatusAtom, {
          connected: false,
          lastUpdateTime: null,
          dt: null,
        });
        reject(err);
      };
      ws.onclose = () => {
        notify(false);
        const store = getDefaultStore();
        store.set(wsConnectionStatusAtom, {
          connected: false,
          lastUpdateTime: null,
          dt: null,
        });
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
  kick,
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

  const [primaryTeam] = usePrimaryTeam();
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

      // Default team ID - this should be replaced with actual primary team selection
      const defaultTeamId = 1; // TODO: Get from primary team selection

      const command = {
        type: "OverrideCommand",
        data: {
          team_color: primaryTeam,
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
          number
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
        const kickCommand = {
          type: "OverrideCommand",
          data: {
            team_color: primaryTeam,
            player_id: playerId,
            command: {
              type: "Kick",
              data: {
                speed: kickSpeedRef.current,
              },
            },
          },
        } satisfies UiCommand;
        sendCommand(kickCommand);
      }

      if (vel_mag > 0 || angular_velocity !== 0 || dribble_speed > 0) {
        sendCommand(command);
      }

      if (fanSpeed) {
        const fanCommand = {
          type: "OverrideCommand",
          data: {
            team_color: primaryTeam,
            player_id: playerId,
            command: {
              type: "SetFanSpeed",
              data: {
                speed: fanSpeedRef.current,
              },
            },
          },
        } satisfies UiCommand;
        // sendCommand(fanCommand);
      }
      if (kickRef.current) {
        console.log("kick");
        // kickRef.current = false; // doesn't work :/
        const kickCommand = {
          type: "OverrideCommand",
          data: {
            team_color: primaryTeam,
            player_id: playerId,
            command: {
              type: "Kick",
              data: {
                speed: kickSpeedRef.current,
              },
            },
          },
        } satisfies UiCommand;
        // sendCommand(kickCommand);
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

export const TeamDataProvider: FC<PropsWithChildren> = ({ children }) => {
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

// Export helper functions for use in components
export { extractPlayerId, isPlayerManuallyControlled };
