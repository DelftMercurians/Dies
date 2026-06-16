import { useAtomValue } from "jotai";
import { currentFrameIdAtom } from "../api";

/** Rounding for the displayed frame number, so it's stable enough to read and
 * jot down by hand mid-match. */
const FRAME_DISPLAY_ROUNDING = 50;

/**
 * Subtle always-visible frame-number readout (top-right of the field). Rounded
 * so it doesn't flicker; used for hand-noting interesting frames during a live
 * match when the keyboard isn't reachable.
 */
export const FrameCounter = () => {
  const frameId = useAtomValue(currentFrameIdAtom);
  const rounded =
    Math.round(frameId / FRAME_DISPLAY_ROUNDING) * FRAME_DISPLAY_ROUNDING;
  return (
    <div className="absolute top-0 right-0 z-10 px-2 py-0.5 font-mono text-[11px] text-text-dim bg-slate-950 bg-opacity-50 rounded-bl select-none tabular-nums pointer-events-none">
      f≈{rounded}
    </div>
  );
};
