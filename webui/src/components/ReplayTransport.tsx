import { useAtomValue } from "jotai";
import { Play, Pause } from "lucide-react";
import { isReplayingAtom, replayStateAtom, useSendCommand } from "../api";
import { Slider } from "@/components/ui/slider";

const SPEEDS = [0.25, 0.5, 1, 2, 4];

/**
 * Replay transport bar, pinned to the bottom of the field and shown only when a
 * log is loaded. Recolored (cyan) to signal replay mode. Play/pause, a scrubber
 * with clickable marker ticks, a speed selector, and a time/frame readout.
 */
export const ReplayTransport = () => {
  const isReplaying = useAtomValue(isReplayingAtom);
  const replay = useAtomValue(replayStateAtom);
  const sendCommand = useSendCommand();

  if (!isReplaying || !replay) return null;

  const {
    playing,
    speed,
    t_min,
    t_max,
    current_t,
    current_frame_id,
    frame_count,
    markers,
  } = replay;
  const span = Math.max(t_max - t_min, 1e-6);

  return (
    <div className="absolute bottom-0 left-0 right-0 z-20 flex items-center gap-3 border-t border-accent-cyan/40 bg-cyan-950/80 px-3 py-2 backdrop-blur-sm">
      <button
        className="text-accent-cyan hover:text-white"
        onClick={() =>
          sendCommand({ type: playing ? "ReplayPause" : "ReplayPlay" })
        }
        title={playing ? "Pause" : "Play"}
      >
        {playing ? <Pause size={18} /> : <Play size={18} />}
      </button>

      <div className="relative flex-1">
        <Slider
          min={t_min}
          max={t_max}
          step={span / 1000}
          value={[current_t]}
          onValueChange={([t]) =>
            sendCommand({ type: "ReplaySeek", data: { t } })
          }
        />
        {/* Marker ticks overlaid on the scrubber track. */}
        {markers.map((m, i) => (
          <button
            key={i}
            title={m.label ?? `marker @ ${m.t.toFixed(1)}s`}
            onClick={() => sendCommand({ type: "ReplaySeek", data: { t: m.t } })}
            className="absolute top-1/2 h-3 w-0.5 -translate-x-1/2 -translate-y-1/2 bg-amber-400 hover:bg-amber-200"
            style={{ left: `${((m.t - t_min) / span) * 100}%` }}
          />
        ))}
      </div>

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

      <div className="whitespace-nowrap font-mono text-[11px] tabular-nums text-cyan-100">
        {current_t.toFixed(1)}/{t_max.toFixed(1)}s · #{current_frame_id}/
        {frame_count}
      </div>
    </div>
  );
};
