import { useState, useEffect, useCallback } from "react";
import type { UiCommand, World } from "./types";

let socket: WebSocket | null = null;

export function useWebSocket({
  onUpdate,
}: { onUpdate?: (world: World) => void } = {}) {
  const connect = useCallback(() => {
    if (!socket || socket.readyState === WebSocket.CLOSED) {
      console.log("Connecting to WebSocket...");
      socket = new WebSocket("ws://127.0.0.1:5555/api/ws");

      socket.onerror = (error) => {
        console.error("WebSocket error:", error);
        if (socket?.readyState === WebSocket.CLOSED) {
          console.log("Reconnecting...");
          setTimeout(connect, 1000);
        }
      };

      socket.onopen = () => {
        console.log("WebSocket connection established.");
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data) as World;
        onUpdate?.(data);
      };

      socket.onclose = () => {
        console.log("WebSocket connection closed. Reconnecting...");
        setTimeout(connect, 1000);
      };
    }
  }, []);

  const sendCommand = useCallback(
    (command: UiCommand) => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(command));
      } else if (socket && socket.readyState === WebSocket.CLOSED) {
        console.error("WebSocket connection closed");
      }
    },
    [socket]
  );

  useEffect(() => {
    connect();

    // Fetch initial state
    fetch("/api/state")
      .then((response) => response.json())
      .then((data: World) => {
        onUpdate?.(data);
      });
  }, [connect]);

  return {
    sendCommand,
  };
}
