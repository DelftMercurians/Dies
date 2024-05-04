import { writable } from "svelte/store";
import type { UiCommand, World } from "./types";
import { onDestroy, onMount } from "svelte";

export function connectWs() {
  const worldState = writable<World | null>(null);
  let socket: WebSocket;
  let queue: UiCommand[] = [];

  function connect() {
    console.log("Connecting to WebSocket...");
    socket = new WebSocket(`ws://localhost:5555/api/ws`);

    socket.onopen = () => {
      console.log("WebSocket connection established.");
      queue.forEach((command) => {
        sendCommand(command);
      });
      queue = [];
    };

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data) as World;
      worldState.set(data);
    };

    socket.onclose = () => {
      console.log("WebSocket connection closed. Reconnecting...");
      setTimeout(connect, 1000);
    };

    socket.onerror = (error) => {
      console.error("WebSocket error:", error);
    };
  }

  function sendCommand(command: UiCommand) {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(command));
    } else if (socket.readyState === WebSocket.CLOSED) {
      if (queue.length === 0) {
        connect();
      }
      // Reconnect and try again
      queue.push(command);
    }
  }

  onMount(() => {
    connect();
    fetch("/api/state")
      .then((response) => response.json())
      .then((data: World) => {
        worldState.set(data);
      });
  });

  onDestroy(() => {
    if (socket) {
      socket.close();
    }
  });

  return {
    worldState,
    sendCommand,
  };
}
