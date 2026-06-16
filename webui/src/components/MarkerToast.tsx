import { useState } from "react";
import { toast } from "sonner";

function MarkerToastForm({
  toastId,
  onSubmit,
}: {
  toastId: string | number;
  onSubmit: (label: string | null) => void;
}) {
  const [value, setValue] = useState("");
  return (
    <div className="flex items-center gap-2 bg-bg-elevated border border-border-std shadow-lg px-3 py-2 rounded min-w-72">
      <span className="text-[11px] uppercase tracking-wider text-text-dim shrink-0">
        Marker
      </span>
      <input
        autoFocus
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            onSubmit(value.trim() || null);
            toast.dismiss(toastId);
          } else if (e.key === "Escape") {
            toast.dismiss(toastId);
          }
          // Don't let typed keys reach the global shortcut handler.
          e.stopPropagation();
        }}
        placeholder="optional label · Enter to drop · Esc cancels"
        className="flex-1 bg-bg-overlay border border-border-std px-2 py-1 text-sm text-text-std outline-none focus:border-cyan-500"
      />
    </div>
  );
}

/**
 * Open a transient toast with an auto-focused text input for labelling a
 * point-of-interest marker. Enter drops the marker (empty = unlabelled), Esc
 * cancels.
 */
export function openMarkerToast(onSubmit: (label: string | null) => void) {
  toast.custom((id) => <MarkerToastForm toastId={id} onSubmit={onSubmit} />, {
    duration: Infinity,
  });
}
