import { writable } from "svelte/store";
import type { UiCommand, World } from "./types";
import { onDestroy, onMount } from "svelte";

export function connectWs() {
  const worldState = writable<World | null>(null);
  let socket: WebSocket;

  function connect() {
    socket = new WebSocket(`ws://localhost:5555/api/ws`);

    socket.onopen = () => {
      console.log("WebSocket connection established.");
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
    } else {
      console.error("WebSocket connection is not open.");
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
