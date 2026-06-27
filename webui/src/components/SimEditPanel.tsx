import { FC } from "react";
import { toast } from "sonner";
import { Trash2, Download, Save } from "lucide-react";
import {
  useSendCommand,
  useFieldSnapshots,
  getFieldSnapshot,
} from "@/api";
import { TeamColor, WorldData, FieldSnapshot } from "@/bindings";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface SimEditPanelProps {
  enabled: boolean;
  onToggle: (on: boolean) => void;
  worldData: WorldData | null;
  primaryTeam: TeamColor;
}

/**
 * Sim Edit controls: a mode toggle (drag robots/ball, shift-drag to kick) plus
 * save/load/delete of named field-state snapshots (positions + yaw only).
 */
const SimEditPanel: FC<SimEditPanelProps> = ({
  enabled,
  onToggle,
  worldData,
}) => {
  const sendCommand = useSendCommand();
  const { names, save, remove } = useFieldSnapshots();

  const handleSave = () => {
    if (!worldData) {
      toast.error("No world state to snapshot");
      return;
    }
    const name = window.prompt("Snapshot name")?.trim();
    if (!name) return;
    const snapshot: FieldSnapshot = {
      blue: worldData.blue_team.map((p) => ({
        id: p.id,
        position: p.position,
        yaw: p.yaw,
      })),
      yellow: worldData.yellow_team.map((p) => ({
        id: p.id,
        position: p.position,
        yaw: p.yaw,
      })),
      ball: worldData.ball ? [worldData.ball.position[0], worldData.ball.position[1]] : undefined,
    };
    save(
      { name, snapshot },
      { onSuccess: () => toast.success(`Saved "${name}"`) }
    );
  };

  const handleLoad = async (name: string) => {
    if (!worldData) return;
    let snapshot: FieldSnapshot;
    try {
      snapshot = await getFieldSnapshot(name);
    } catch (e) {
      toast.error(String(e));
      return;
    }
    const teams = [
      { color: TeamColor.Blue, current: worldData.blue_team, snap: snapshot.blue },
      {
        color: TeamColor.Yellow,
        current: worldData.yellow_team,
        snap: snapshot.yellow,
      },
    ];
    for (const { color, current, snap } of teams) {
      const currentIds = new Set(current.map((p) => p.id));
      const snapIds = new Set(snap.map((r) => r.id));
      // Teleport existing robots, add missing ones.
      for (const r of snap) {
        const inner = {
          team_color: color,
          player_id: r.id,
          position: r.position,
          yaw: r.yaw,
        };
        sendCommand({
          type: "SimulatorCmd",
          data: currentIds.has(r.id)
            ? { type: "TeleportRobot", data: inner }
            : { type: "AddRobot", data: inner },
        });
      }
      // Remove robots not present in the snapshot so the setup reproduces exactly.
      for (const p of current) {
        if (!snapIds.has(p.id)) {
          sendCommand({
            type: "SimulatorCmd",
            data: {
              type: "RemoveRobot",
              data: { team_color: color, player_id: p.id },
            },
          });
        }
      }
    }
    if (snapshot.ball) {
      sendCommand({
        type: "SimulatorCmd",
        data: { type: "TeleportBall", data: { position: snapshot.ball } },
      });
    }
    toast.success(`Loaded "${name}"`);
  };

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-row items-center justify-between">
        <div className="text-sm">Sim Edit</div>
        <button
          onClick={() => onToggle(!enabled)}
          className={cn(
            "px-2 py-0.5 text-xs rounded border transition-colors",
            enabled
              ? "bg-accent-amber/20 text-accent-amber border-accent-amber/50"
              : "text-text-muted border-border-muted hover:text-text-std"
          )}
        >
          {enabled ? "On" : "Off"}
        </button>
      </div>

      {enabled ? (
        <div className="text-[11px] text-text-dim leading-snug">
          Drag robots/ball to place. Shift-drag the ball to kick.
        </div>
      ) : null}

      <div className="flex flex-row items-center justify-between">
        <div className="text-xs text-text-muted">Snapshots</div>
        <Button
          onClick={handleSave}
          className="h-6 px-2 text-xs gap-1"
          title="Save current field state"
        >
          <Save size={12} /> Save
        </Button>
      </div>

      {names.length === 0 ? (
        <div className="text-[11px] text-text-dim">No saved snapshots</div>
      ) : (
        <div className="flex flex-col gap-1 max-h-40 overflow-y-auto">
          {names.map((name) => (
            <div
              key={name}
              className="flex flex-row items-center gap-2 group/snap"
            >
              <span className="flex-1 truncate text-sm">{name}</span>
              <button
                onClick={() => handleLoad(name)}
                title="Load"
                className="text-text-muted hover:text-accent-green"
              >
                <Download size={14} />
              </button>
              <button
                onClick={() =>
                  remove(name, {
                    onSuccess: () => toast.success(`Deleted "${name}"`),
                  })
                }
                title="Delete"
                className="text-text-muted hover:text-accent-red opacity-0 group-hover/snap:opacity-100"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SimEditPanel;
