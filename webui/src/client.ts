import { writable } from "svelte/store";
import type { UiCommand, World } from "./types";
import { onMount } from "svelte";

export function connectWs() {
  const worldState = writable<World | null>(null);
  let socket: WebSocket;

  function connect() {
    socket = new WebSocket("ws://localhost:5555/api/ws");

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

  onMount(connect);

  return {
    worldState,
    sendCommand,
  };
}
