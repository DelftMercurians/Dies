import React from "react";
import { useStatus, useWsConnectionStatus } from "@/api";
import { cn } from "@/lib/utils";

/**
 * Status cluster displaying connection and executor state.
 *
 * Specs:
 * - Connection indicator dot (● green/red)
 * - Status text: RUNNING / STOPPED / ERROR
 * - Latency: dt in milliseconds
 * - Width: ~100px
 */

const StatusCluster: React.FC = () => {
  const { data: backendState, status: backendLoadingState } = useStatus();
  const [wsConnectionStatus] = useWsConnectionStatus();

  const executorStatus = backendState?.executor;
  const isRunning = executorStatus?.type === "RunningExecutor";
  const isFailed = executorStatus?.type === "Failed";
  const isConnected = wsConnectionStatus.connected;

  // Determine status text and color
  let statusText = "IDLE";
  let statusColorClass = "text-text-muted";

  if (backendLoadingState === "error") {
    statusText = "ERROR";
    statusColorClass = "text-accent-red";
  } else if (isFailed) {
    statusText = "FAILED";
    statusColorClass = "text-accent-red";
  } else if (isRunning) {
    statusText = "RUNNING";
    statusColorClass = "text-accent-green";
  }

  // Format latency
  const latencyMs =
    wsConnectionStatus.dt !== null
      ? (wsConnectionStatus.dt * 1000).toFixed(0)
      : null;

  return (
    <div className="flex items-center gap-1.5 text-sm font-medium uppercase tracking-wider min-w-[90px]">
      {/* Connection indicator */}
      <div
        className={cn(
          "w-2 h-2 shrink-0",
          isConnected ? "bg-accent-green" : "bg-accent-red",
        )}
      />

      {/* Status text */}
      <span className={cn("shrink-0", statusColorClass)}>{statusText}</span>
    </div>
  );
};

export default StatusCluster;
