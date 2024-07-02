import React, {
  useEffect,
  PropsWithChildren,
  FC,
  useRef,
  useContext,
  createContext,
  useState,
} from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  UiMode,
  UiStatus,
  UiCommand,
  WorldData,
  PostUiModeBody,
  PostUiCommandBody,
  UiWorldState,
} from "./bindings";

export type Status =
  | { status: "loading" }
  | { status: "connected"; data: UiStatus }
  | { status: "error" };

export type WorldStatus =
  | { status: "none" }
  | { status: "loading" }
  | { status: "connected"; data: WorldData };

const WsConnectedContext = createContext(false);

const getWorldState = (): Promise<UiWorldState> =>
  fetch("/api/world-state").then((res) => res.json());

const getUiStatus = (): Promise<UiStatus> =>
  fetch("/api/status").then((res) => res.json());

const getScenarios = (): Promise<string[]> =>
  fetch("/api/scenarios").then((res) => res.json());

const postUiMode = (mode: UiMode) =>
  fetch("/api/mode", {
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

export const useStatus = (): Status => {
  const query = useQuery({
    queryKey: ["status"],
    queryFn: getUiStatus,
    refetchInterval: 500,
  });

  if (query.data && query.isSuccess) {
    return { status: "connected", data: query.data };
  } else if (query.isError) {
    return { status: "error" };
  } else {
    return { status: "loading" };
  }
};

export const useScenarios = (): string[] | null => {
  const query = useQuery({
    queryKey: ["scenarios"],
    queryFn: getScenarios,
    staleTime: Infinity,
  });

  return query.data ?? null;
};

export const useSetMode = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: postUiMode,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });

  return (mode: UiMode) => mutation.mutate(mode);
};

export const useSendCommand = () => {
  const queryClient = useQueryClient();
  const mutation = useMutation({
    mutationFn: postCommand,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["status"] });
    },
  });

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

export const WorldDataProvider: FC<PropsWithChildren> = ({ children }) => {
  const ws = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const queryClient = useQueryClient();

  useEffect(() => {
    const host = window.location.host.replace("localhost", "127.0.0.1");

    const connectAndListen = (): Promise<WebSocket> => {
      return new Promise((_, reject) => {
        ws.current = new WebSocket(`ws://127.0.0.1:5555/api/ws`);

        ws.current.onopen = () => {
          console.log("Connected to websocket");
          setConnected(true);
        };
        ws.current.onmessage = (event) => {
          const data = JSON.parse(event.data) as WorldData;
          queryClient.setQueryData(["world-state"], {
            type: "Loaded",
            data,
          } satisfies UiWorldState);
        };
        ws.current.onerror = (err) => {
          if (ws.current) {
            ws.current.onmessage = () => {};
            ws.current.onclose = () => {};
            ws.current.close();
          }
          setConnected(false);
          reject(err);
        };
        ws.current.onclose = () => {
          setConnected(false);
          reject(new Error("Websocket closed"));
        };
      });
    };

    let closing = false;
    const startWs = async () => {
      while (true) {
        try {
          await connectAndListen();
        } catch (err) {
          console.error("Error in WebSocket connection", err);
        }
        if (closing) {
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 1000));
      }
    };

    startWs();

    return () => {
      closing = true;
      if (ws.current) {
        ws.current.onmessage = () => {};
        ws.current.close();
      }
      ws.current = null;
    };
  }, []);

  return (
    <WsConnectedContext.Provider value={connected}>
      {children}
    </WsConnectedContext.Provider>
  );
};
