import { useState } from "react";
import { useAtomValue } from "jotai";
import { History } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SimpleTooltip } from "@/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { isReplayingAtom, replayStateAtom, useSendCommand } from "@/api";
import { LogBrowserModal } from "@/components/LogBrowserModal";

/**
 * Toolbar control for replay. A "Load log" button normally; while a log is
 * loaded it becomes a pulsing "Replay" indicator whose popover holds stop /
 * switch-log / jump-to-moment controls.
 */
export default function ReplayButton() {
  const isReplaying = useAtomValue(isReplayingAtom);
  const replay = useAtomValue(replayStateAtom);
  const sendCommand = useSendCommand();
  const [modalOpen, setModalOpen] = useState(false);
  const [jumpFrame, setJumpFrame] = useState("");

  const seekToFrame = () => {
    const f = Number(jumpFrame);
    if (Number.isNaN(f) || !replay || replay.frame_count <= 0) return;
    const span = replay.t_max - replay.t_min;
    const t = replay.t_min + (f / replay.frame_count) * span;
    sendCommand({ type: "ReplaySeek", data: { t } });
  };

  return (
    <>
      {!isReplaying ? (
        <SimpleTooltip title="Load recorded log for replay">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setModalOpen(true)}
            className="text-text-dim hover:text-text-std"
          >
            <History className="h-4 w-4" />
          </Button>
        </SimpleTooltip>
      ) : (
        <Popover>
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="animate-pulse gap-1 text-accent-cyan"
            >
              <History className="h-4 w-4" /> Replay
            </Button>
          </PopoverTrigger>
          <PopoverContent className="flex w-60 flex-col gap-2 text-sm">
            <button
              onClick={() => sendCommand({ type: "Stop" })}
              className="px-1 py-1 text-left text-text-std hover:bg-bg-overlay"
            >
              Stop replay (back to live)
            </button>
            <button
              onClick={() => setModalOpen(true)}
              className="px-1 py-1 text-left text-text-std hover:bg-bg-overlay"
            >
              Switch log…
            </button>
            <div className="border-t border-border-subtle pt-2">
              <div className="mb-1 text-[11px] uppercase tracking-wider text-text-dim">
                Jump to moment
              </div>
              <div className="max-h-40 overflow-y-auto">
                {(replay?.markers ?? []).length === 0 ? (
                  <div className="text-xs text-text-dim">No markers</div>
                ) : (
                  replay!.markers.map((m, i) => (
                    <button
                      key={i}
                      onClick={() =>
                        sendCommand({ type: "ReplaySeek", data: { t: m.t } })
                      }
                      className="block w-full px-1 py-0.5 text-left text-xs hover:bg-bg-overlay"
                    >
                      {m.label ?? `#${m.frame_id}`}{" "}
                      <span className="text-text-dim">{m.t.toFixed(1)}s</span>
                    </button>
                  ))
                )}
              </div>
              <div className="mt-2 flex items-center gap-1">
                <input
                  type="number"
                  value={jumpFrame}
                  onChange={(e) => setJumpFrame(e.target.value)}
                  placeholder="frame"
                  className="w-20 border border-border-std bg-bg-overlay px-1 py-0.5 text-xs text-text-std outline-none"
                />
                <button
                  onClick={seekToFrame}
                  className="border border-border-std px-2 py-0.5 text-xs hover:bg-bg-overlay"
                >
                  Go
                </button>
              </div>
            </div>
          </PopoverContent>
        </Popover>
      )}
      <LogBrowserModal open={modalOpen} onOpenChange={setModalOpen} />
    </>
  );
}
