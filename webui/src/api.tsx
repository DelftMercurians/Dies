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
  BenchMotionMode,
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
  Announcement,
  TeamColor,
  TeamPlayerId,
  PlayerId,
  PlayerOverrideCommand,
  TeamConfiguration,
  SideAssignment,
  ReplayState,
  LogsResponse,
  LogInfo,
  ConsoleLogMessage,
  SettingsSnapshot,
  SettingsSnapshotsResponse,
  SnapshotsResponse,
  StrategiesResponse,
  FieldSnapshot,
  SaveSnapshotBody,
} from "./bindings";
import { toast } from "sonner";
import { atom, getDefaultStore, useAtom } from "jotai";

export const selectedPlayerIdAtom = atom<number | null>(null);

/** Keyboard-driving toggle (lifted so global shortcuts can toggle it). */
export const keyboardControlAtom = atom<boolean>(false);
/** Keyboard-driving frame: global vs local. */
export const keyboardModeAtom = atom<"local" | "global">("global");

/** Transient feedback for the last triggered shortcut/command (toolbar flash). */
export interface ShortcutFeedback {
  label: string;
  ts: number;
}
export const lastShortcutAtom = atom<ShortcutFeedback | null>(null);

/** Whether the command palette (⌘K) is open. */
export const commandPaletteOpenAtom = atom<boolean>(false);

/** Current world-frame id, from each WorldUpdate (live + replay). */
export const currentFrameIdAtom = atom<number>(0);

/**
 * An announcer line as held in the UI. Extends the backend {@link Announcement}
 * with client-side bookkeeping: backend ids reset when the executor restarts, so
 * we mint our own stable key, and we stamp wall-clock arrival time to drive the
 * fade-in/translucency animation independent of backend/world time.
 */
export interface AnnouncementFeedItem extends Announcement {
  clientKey: number;
  arrivedAt: number;
}

/** Rolling announcer commentary feed (newest last). Capped to avoid growth. */
export const announcementsAtom = atom<AnnouncementFeedItem[]>([]);
const MAX_ANNOUNCEMENTS = 60;
let announcementSeq = 0;

/** Latest replay-player state (null when not replaying). */
export const replayStateAtom = atom<ReplayState | null>(null);

/** Whether a recorded log is loaded for replay. */
export const isReplayingAtom = atom((get) => !!get(replayStateAtom)?.loaded);

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

const getSettingsSnapshots = (): Promise<SettingsSnapshotsResponse> =>
  fetch("/api/settings/snapshots").then((res) => res.json());

const postSettingsBaseline = (): Promise<SettingsSnapshot> =>
  fetch("/api/settings/baseline", { method: "POST" }).then((res) =>
    res.json(),
  );

const getFieldSnapshots = (): Promise<SnapshotsResponse> =>
  fetch("/api/snapshots").then((res) => res.json());

export const getFieldSnapshot = (name: string): Promise<FieldSnapshot> =>
  fetch(`/api/snapshots/${encodeURIComponent(name)}`).then((res) => {
    if (!res.ok) throw new Error(`Snapshot "${name}" not found`);
    return res.json();
  });

const postFieldSnapshot = (body: SaveSnapshotBody): Promise<Response> =>
  fetch("/api/snapshots", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

const deleteFieldSnapshot = (name: string): Promise<Response> =>
  fetch(`/api/snapshots/${encodeURIComponent(name)}`, { method: "DELETE" });

const getStrategies = (): Promise<StrategiesResponse> =>
  fetch("/api/strategies").then((res) => res.json());

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
    refetchInterval: 250,
  });

/** Whether the Test Bench modal is open (lifted so the toolbar can toggle it). */
export const benchOpenAtom = atom<boolean>(false);

/**
 * Telemetry feed for the Test Bench. Same endpoint as {@link useBasestationInfo}
 * but on its own query key with a faster poll while the modal is open, so the
 * grid/focus readouts (cap voltage, motor speeds, breakbeam) stay smooth.
 */
export const useBenchTelemetry = (enabled: boolean) =>
  useQuery({
    queryKey: ["bench-telemetry"],
    queryFn: getBasestationInfo,
    refetchInterval: enabled ? 80 : false,
    enabled,
  });

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
  const [primaryTeam, setPrimaryTeamState] = useState<TeamColor>(TeamColor.Blue);
  // Whether the user has picked a team explicitly this session. In-memory (a ref,
  // like `primaryTeam`'s state) so both reset on page reload — a fresh load
  // re-auto-focuses from scratch.
  const userChoseRef = useRef(false);
  const setPrimaryTeam = (team: TeamColor) => {
    userChoseRef.current = true;
    setPrimaryTeamState(team);
  };

  return (
    <PrimaryTeamContext.Provider value={[primaryTeam, setPrimaryTeam]}>
      {/* Isolated in a leaf so the high-frequency debug subscription driving the
          auto-focus never re-renders the app tree under this provider. */}
      <TeamAutoFocus userChoseRef={userChoseRef} setTeam={setPrimaryTeamState} />
      {children}
    </PrimaryTeamContext.Provider>
  );
};

/**
 * Auto-focuses the controlled team — the one emitting `team_<Color>.*` debug —
 * uniformly across live, sim, and replay (a replay's log only carries the
 * controlled team's debug). No-op once the user has chosen a team manually, and
 * when both (or neither) teams are controlled/logged. Renders nothing.
 */
const TeamAutoFocus: FC<{
  userChoseRef: React.MutableRefObject<boolean>;
  setTeam: (t: TeamColor) => void;
}> = ({ userChoseRef, setTeam }) => {
  const debugMap = useDebugData();
  useEffect(() => {
    if (userChoseRef.current || !debugMap) return;
    const has = (t: TeamColor) =>
      Object.keys(debugMap).some((k) => k.startsWith(`team_${t}.`));
    const blue = has(TeamColor.Blue);
    const yellow = has(TeamColor.Yellow);
    if (blue !== yellow) setTeam(blue ? TeamColor.Blue : TeamColor.Yellow);
  }, [debugMap, userChoseRef, setTeam]);
  return null;
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
      // The diff-vs-baseline depends on current settings; refresh the history
      // too (a debounced auto-snapshot may have just landed).
      queryClient.invalidateQueries({ queryKey: ["settings-snapshots"] });
    },
  });

  return {
    settings: query.data,
    updateSettings: mutation.mutate,
  };
};

/**
 * Settings explore/revert state: the known-good baseline + auto-captured
 * history, plus actions to mark a new baseline and restore any snapshot.
 */
export const useSettingsSnapshots = () => {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["settings-snapshots"],
    queryFn: getSettingsSnapshots,
    // Poll lightly so debounced auto-snapshots show up without a manual edit.
    refetchInterval: 3000,
  });

  const setBaseline = useMutation({
    mutationFn: postSettingsBaseline,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["settings-snapshots"] });
    },
  });

  // Restoring a snapshot is just re-applying its settings through the normal path.
  const restore = useMutation({
    mutationFn: postExecutorSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["controller-settings"] });
      queryClient.invalidateQueries({ queryKey: ["settings-snapshots"] });
    },
  });

  return {
    baseline: query.data?.baseline ?? null,
    history: query.data?.history ?? [],
    markBaseline: setBaseline.mutate,
    restore: restore.mutate,
  };
};

/** Saved simulator field-state snapshots (list + save/delete). */
export const useFieldSnapshots = () => {
  const queryClient = useQueryClient();
  const query = useQuery({
    queryKey: ["field-snapshots"],
    queryFn: getFieldSnapshots,
  });

  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: ["field-snapshots"] });

  const save = useMutation({
    mutationFn: postFieldSnapshot,
    onSuccess: invalidate,
  });

  const remove = useMutation({
    mutationFn: deleteFieldSnapshot,
    onSuccess: invalidate,
  });

  return {
    names: query.data?.names ?? [],
    save: save.mutate,
    remove: remove.mutate,
  };
};

/** Binaries the strategy picker can assign to a team: full strategies + scenarios. */
export const useStrategies = () => {
  const query = useQuery({
    queryKey: ["strategies"],
    queryFn: getStrategies,
  });
  return {
    strategies: query.data?.strategies ?? [],
    scenarios: query.data?.scenarios ?? [],
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

/** Fetch the list of recorded logs available for replay. */
export const useLogs = (enabled: boolean) => {
  const query = useQuery({
    queryKey: ["logs"],
    queryFn: async (): Promise<LogInfo[]> => {
      const res = await fetch("/api/logs");
      const body = (await res.json()) as LogsResponse;
      return body.logs;
    },
    enabled,
    refetchInterval: enabled ? 5000 : false,
  });
  return query.data ?? [];
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

// Console state — backend log lines streamed over the WS for the console panel.
const consoleLogsAtom = atom<ConsoleLogMessage[]>([]);
const CONSOLE_LOG_LIMIT = 1000;

export const useConsoleLogs = () => useAtom(consoleLogsAtom)[0];
export const useClearConsoleLogs = () => {
  const set = useAtom(consoleLogsAtom)[1];
  return () => set([]);
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
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(command));
  } else {
    throw new Error("Websocket not connected");
  }
};

/** Last-resort backstop: force-close a socket still stuck in CONNECTING after
 * this long so a wedged handshake eventually retries. Kept long (Firefox can
 * legitimately take tens of seconds to open a localhost WS when proxy
 * resolution is in play) so it never kills a slow-but-viable connect. */
const WS_CONNECT_TIMEOUT_MS = 60000;
const WS_BACKOFF_MIN_MS = 1000;
const WS_BACKOFF_MAX_MS = 10000;

/** Same-origin WS URL derived from the page location (correct scheme + host +
 * port), so it works behind the vite dev proxy and over https/wss in prod
 * instead of a hardcoded `ws://127.0.0.1:5555`. */
const wsUrl = (): string => {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${window.location.host}/api/ws`;
};

export function startWsClient() {
  const notify = (connected: boolean) => {
    onWsConnectedChange.forEach((cb) => cb(connected));
  };
  const markDisconnected = () => {
    notify(false);
    getDefaultStore().set(wsConnectionStatusAtom, {
      connected: false,
      lastUpdateTime: null,
      dt: null,
    });
  };

  // Resolves with whether the socket ever reached OPEN, so the run loop can
  // reset its backoff after a real connection and grow it only on outright
  // connect failures. A stuck CONNECTING socket is force-closed by the
  // watchdog so we never block on the browser's own connect timeout.
  const connectAndListen = (): Promise<boolean> =>
    new Promise((resolve) => {
      const socket = new WebSocket(wsUrl());
      ws = socket;
      let settled = false;
      let opened = false;
      const done = () => {
        if (settled) return;
        settled = true;
        clearTimeout(watchdog);
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        if (ws === socket) ws = null;
        try {
          socket.close();
        } catch {
          /* already closing/closed */
        }
        markDisconnected();
        resolve(opened);
      };

      const watchdog = setTimeout(() => {
        if (socket.readyState !== WebSocket.OPEN) {
          console.warn("WebSocket connect timed out, retrying");
          done();
        }
      }, WS_CONNECT_TIMEOUT_MS);

      socket.onopen = () => {
        opened = true;
        clearTimeout(watchdog);
        toast.success("WebSocket connected");
        notify(true);
        const store = getDefaultStore();
        store.set(wsConnectionStatusAtom, {
          connected: true,
          lastUpdateTime: null,
          dt: null,
        });
      };
      socket.onmessage = (event) => {
        const msg = JSON.parse(event.data) as WsMessage;
        const store = getDefaultStore();

        if (msg.type === "WorldUpdate") {
          // msg.data is now the full WorldUpdate { world_data, frame_id }.
          queryClient.setQueryData(["world-state"], {
            type: "Loaded",
            data: msg.data.world_data,
          } satisfies UiWorldState);
          store.set(currentFrameIdAtom, msg.data.frame_id);

          // Announcer feed: the backend re-sends a rolling window every frame
          // (the watch channel to us coalesces, so deltas would be lost).
          // Reconcile by id — keep arrival time/key for already-seen lines so
          // their fade animation continues, mint fresh ones for new ids.
          const incoming = msg.data.announcements ?? [];
          const prev = store.get(announcementsAtom);
          const prevMaxId = prev.length ? prev[prev.length - 1].id : -1;
          const incomingMaxId = incoming.length
            ? incoming[incoming.length - 1].id
            : -1;
          // Skip the common case: window unchanged since last frame.
          const unchanged =
            incoming.length === prev.length && incomingMaxId === prevMaxId;
          if (!unchanged) {
            // Executor restart (ids reset): drop stale arrival bookkeeping.
            const restart = incomingMaxId < prevMaxId;
            const byId = new Map<number, AnnouncementFeedItem>();
            if (!restart) for (const i of prev) byId.set(i.id, i);
            const now = Date.now();
            const next = incoming.slice(-MAX_ANNOUNCEMENTS).map((a) => {
              const existing = byId.get(a.id);
              return (
                existing ?? {
                  ...a,
                  clientKey: announcementSeq++,
                  arrivedAt: now,
                }
              );
            });
            store.set(announcementsAtom, next);
          }

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
        } else if (msg.type === "ConsoleLog") {
          const cur = store.get(consoleLogsAtom);
          const next = cur.length >= CONSOLE_LOG_LIMIT
            ? [...cur.slice(cur.length - CONSOLE_LOG_LIMIT + 1), msg.data]
            : [...cur, msg.data];
          store.set(consoleLogsAtom, next);
        } else if (msg.type === "ReplayState") {
          store.set(replayStateAtom, msg.data.loaded ? msg.data : null);
        }
      };
      socket.onerror = () => done();
      socket.onclose = () => done();
    });

  const run = async () => {
    let backoff = WS_BACKOFF_MIN_MS;
    while (true) {
      const opened = await connectAndListen();
      // A connection that actually opened resets the backoff; a failed
      // connect grows it (capped) so a down server isn't hammered.
      if (opened) backoff = WS_BACKOFF_MIN_MS;
      await new Promise((resolve) => setTimeout(resolve, backoff));
      if (!opened) backoff = Math.min(backoff * 2, WS_BACKOFF_MAX_MS);
    }
  };

  // Firefox aborts WebSockets opened while the document is still loading
  // ("interrupted while the page was loading"). Defer the first connect until
  // after the load event so the initial attempt isn't thrown away.
  if (document.readyState === "complete") {
    run();
  } else {
    window.addEventListener("load", () => run(), { once: true });
  }
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
  const worldState = useWorldState();

  const worldStateRef = useRef(worldState);
  worldStateRef.current = worldState;
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
      if (playerId === null || worldStateRef.current?.status !== "connected")
        return;
      const players =
        primaryTeam === TeamColor.Blue
          ? worldStateRef.current?.data.blue_team
          : worldStateRef.current?.data.yellow_team;
      const player = players?.find((p) => p.id === playerId);

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
              yaw: undefined as number | undefined,
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
      const newYaw = player
        ? player.yaw + angular_velocity * (1000 / 30)
        : undefined;
      command.data.command.data.yaw = newYaw;

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

      sendCommand(command);

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
  }, [playerId, primaryTeam]);
};

/**
 * Direct keyboard driving for the Test Bench. Unlike {@link useKeyboardControl}
 * this sends `Bench` `SetMotion` commands straight to the basestation (no
 * executor, no vision). Active only while `robotId` is non-null (i.e. the robot
 * has been "taken"). WASD = translate, Q/E = rotate, space = dribble.
 */
export const useBenchKeyboardControl = ({
  robotId,
  mode,
  speed,
  angularSpeedDegPerSec,
  dribbleSpeed,
}: {
  robotId: number | null;
  mode: BenchMotionMode;
  speed: number;
  angularSpeedDegPerSec: number;
  dribbleSpeed: number;
}) => {
  const sendCommand = useSendCommand();
  const speedRef = useRef(speed);
  speedRef.current = speed;
  const angRef = useRef(angularSpeedDegPerSec);
  angRef.current = angularSpeedDegPerSec;
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const dribbleRef = useRef(dribbleSpeed);
  dribbleRef.current = dribbleSpeed;

  useEffect(() => {
    if (robotId === null) return;
    const pressed = new Set<string>();
    let heading = 0; // accumulated heading setpoint for global mode
    const down = (ev: KeyboardEvent) => {
      if (ev.repeat) return;
      pressed.add(ev.key.toLowerCase());
    };
    const up = (ev: KeyboardEvent) => pressed.delete(ev.key.toLowerCase());

    const dtMs = 1000 / 30;
    const interval = setInterval(() => {
      let fwd = 0;
      let right = 0;
      let rot = 0;
      if (pressed.has("w")) fwd += 1;
      if (pressed.has("s")) fwd -= 1;
      if (pressed.has("d")) right += 1;
      if (pressed.has("a")) right -= 1;
      if (pressed.has("q")) rot += 1;
      if (pressed.has("e")) rot -= 1;

      const mag = Math.hypot(fwd, right);
      let vx = 0;
      let vy = 0;
      if (mag > 0) {
        vx = (fwd / mag) * speedRef.current;
        vy = (right / mag) * speedRef.current;
      }

      const angRad = (angRef.current * Math.PI) / 180;
      let w_or_heading: number;
      if (modeRef.current === BenchMotionMode.Local) {
        w_or_heading = rot * angRad; // rad/s
      } else {
        heading += rot * angRad * (dtMs / 1000); // integrate to a heading setpoint
        w_or_heading = heading;
      }

      const dribble_speed = pressed.has(" ") ? dribbleRef.current : 0;

      sendCommand({
        type: "Bench",
        data: {
          type: "SetMotion",
          data: {
            robot_id: robotId,
            mode: modeRef.current,
            vx,
            vy,
            w_or_heading,
            dribble_speed,
          },
        },
      } satisfies UiCommand);
    }, dtMs);

    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      clearInterval(interval);
    };
  }, [robotId]);
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
