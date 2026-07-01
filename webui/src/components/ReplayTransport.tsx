import { useAtomValue } from "jotai";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Rewind,
  FastForward,
  Flag,
} from "lucide-react";
import { useState } from "react";
import { isReplayingAtom, replayStateAtom, useSendCommand } from "../api";
import { Slider } from "@/components/ui/slider";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

const SPEEDS = [0.25, 0.5, 1, 2, 4];
const TIME_STEPS = [0.1, 0.5, 1, 5];

/**
 * Replay transport bar, pinned to the bottom of the field and shown only when a
 * log is loaded. Recolored (cyan) to signal replay mode. Play/pause, frame- and
 * time-stepping, a scrubber with marker ticks, a marker list, a speed selector,
 * and a time/frame readout.
 */
export const ReplayTransport = () => {
  const isReplaying = useAtomValue(isReplayingAtom);
  const replay = useAtomValue(replayStateAtom);
  const sendCommand = useSendCommand();
  const [timeStep, setTimeStep] = useState(1);

  if (!isReplaying || !replay) return null;

  const {
    playing,
    speed,
    t_min,
    t_max,
    current_t,
    current_frame_id,
    frame_count,
    dt,
    markers,
  } = replay;
  const span = Math.max(t_max - t_min, 1e-6);
  const fps = dt > 0 ? 1 / dt : 0;

  const stepFrame = (delta: number) =>
    sendCommand({ type: "ReplayStep", data: { delta } });
  const stepTime = (d: number) =>
    sendCommand({ type: "ReplayStepTime", data: { dt: d } });
  const seek = (t: number) => sendCommand({ type: "ReplaySeek", data: { t } });

  // Marker the playhead currently sits on (by frame id), if any.
  const activeMarker = markers.find((m) => m.frame_id === current_frame_id);

  return (
    <div className="absolute bottom-0 left-0 right-0 z-20 flex items-center gap-2 border-t border-accent-cyan/40 bg-cyan-950/80 px-3 py-2 backdrop-blur-sm">
      <button
        className="text-accent-cyan hover:text-white"
        onClick={() =>
          sendCommand({ type: playing ? "ReplayPause" : "ReplayPlay" })
        }
        title={playing ? "Pause (space)" : "Play (space)"}
      >
        {playing ? <Pause size={18} /> : <Play size={18} />}
      </button>

      {/* time step back / frame back */}
      <TransportButton
        onClick={() => stepTime(-timeStep)}
        title={`Back ${timeStep}s (⇧←)`}
      >
        <Rewind size={15} />
      </TransportButton>
      <TransportButton
        onClick={() => stepFrame(-1)}
        title="Previous frame (←)"
      >
        <SkipBack size={15} />
      </TransportButton>

      {/* frame forward / time step forward */}
      <TransportButton onClick={() => stepFrame(1)} title="Next frame (→)">
        <SkipForward size={15} />
      </TransportButton>
      <TransportButton
        onClick={() => stepTime(timeStep)}
        title={`Forward ${timeStep}s (⇧→)`}
      >
        <FastForward size={15} />
      </TransportButton>

      {/* time-step size */}
      <select
        className="border border-accent-cyan/40 bg-cyan-950 px-1 py-0.5 text-[11px] text-cyan-100"
        value={timeStep}
        onChange={(e) => setTimeStep(Number(e.target.value))}
        title="Time-step size"
      >
        {TIME_STEPS.map((s) => (
          <option key={s} value={s}>
            {s}s
          </option>
        ))}
      </select>

      <div className="relative flex-1">
        <Slider
          min={t_min}
          max={t_max}
          step={span / 1000}
          value={[current_t]}
          onValueChange={([t]) => seek(t)}
        />
        {/* Marker ticks overlaid on the scrubber track. */}
        {markers.map((m, i) => (
          <Tooltip key={i}>
            <TooltipTrigger asChild>
              <button
                onClick={() => seek(m.t)}
                className="absolute top-1/2 h-3 w-0.5 -translate-x-1/2 -translate-y-1/2 bg-amber-400 hover:bg-amber-200"
                style={{ left: `${((m.t - t_min) / span) * 100}%` }}
              />
            </TooltipTrigger>
            <TooltipContent side="top">
              {m.label
                ? `${m.label} · ${m.t.toFixed(2)}s`
                : `marker @ ${m.t.toFixed(2)}s`}
            </TooltipContent>
          </Tooltip>
        ))}
      </div>

      {/* markers list */}
      <Popover>
        <PopoverTrigger asChild>
          <button
            className="flex items-center gap-1 text-amber-300 hover:text-amber-100 disabled:opacity-40"
            disabled={markers.length === 0}
            title="Markers"
          >
            <Flag size={15} />
            <span className="font-mono text-[11px] tabular-nums">
              {markers.length}
            </span>
          </button>
        </PopoverTrigger>
        <PopoverContent
          side="top"
          align="end"
          className="max-h-72 w-64 overflow-y-auto p-1"
        >
          {markers.length === 0 ? (
            <div className="px-2 py-1 text-xs text-text-muted">No markers</div>
          ) : (
            <ul className="flex flex-col">
              {markers.map((m, i) => (
                <li key={i}>
                  <button
                    onClick={() => seek(m.t)}
                    className={
                      "flex w-full items-baseline gap-2 px-2 py-1 text-left text-xs hover:bg-bg-overlay " +
                      (m.frame_id === current_frame_id
                        ? "text-amber-300"
                        : "text-text-std")
                    }
                  >
                    <span className="font-mono tabular-nums text-text-dim">
                      {m.t.toFixed(2)}s
                    </span>
                    <span className="truncate">
                      {m.label ?? <span className="italic">unlabeled</span>}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </PopoverContent>
      </Popover>

      <select
        className="border border-border-std bg-bg-overlay px-1 py-0.5 text-xs text-text-std"
        value={speed}
        onChange={(e) =>
          sendCommand({
            type: "ReplaySetSpeed",
            data: { speed: Number(e.target.value) },
          })
        }
      >
        {SPEEDS.map((s) => (
          <option key={s} value={s}>
            {s}×
          </option>
        ))}
      </select>

      <div className="flex items-center gap-2 whitespace-nowrap font-mono text-[11px] tabular-nums text-cyan-100">
        {activeMarker && (
          <span className="max-w-40 truncate rounded-sm bg-amber-500/20 px-1.5 py-0.5 text-amber-200">
            {activeMarker.label ?? "marker"}
          </span>
        )}
        <span>
          {current_t.toFixed(2)}/{t_max.toFixed(1)}s · #{current_frame_id}/
          {frame_count}
          {fps > 0 && ` · ${Math.round(fps)}fps`}
        </span>
      </div>
    </div>
  );
};

const TransportButton: React.FC<{
  onClick: () => void;
  title: string;
  children: React.ReactNode;
}> = ({ onClick, title, children }) => (
  <button
    className="text-cyan-200 hover:text-white"
    onClick={onClick}
    title={title}
  >
    {children}
  </button>
);
