import { FC } from "react";
import { toast } from "sonner";
import { Trash2, Download, Save } from "lucide-react";
import {
  useSendCommand,
  useFieldSnapshots,
  getFieldSnapshot,
  useWorldState,
} from "@/api";
import { TeamColor, FieldSnapshot } from "@/bindings";
import { Button } from "@/components/ui/button";

/**
 * Save / load / delete named simulator field-state snapshots (robot poses + ball
 * position). Built from the live world state on save and replayed as teleport
 * commands on load. Rendered inside the Sim Edit dialog.
 */
const SnapshotManager: FC = () => {
  const sendCommand = useSendCommand();
  const { names, save, remove } = useFieldSnapshots();
  const world = useWorldState();
  const worldData = world.status === "connected" ? world.data : null;

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
      ball: worldData.ball
        ? [worldData.ball.position[0], worldData.ball.position[1]]
        : undefined,
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
    <div className="flex flex-col gap-3">
      <div className="flex flex-row items-center justify-between">
        <div className="text-sm text-text-muted">
          {names.length} saved snapshot{names.length === 1 ? "" : "s"}
        </div>
        <Button onClick={handleSave} className="h-7 px-2 text-xs gap-1">
          <Save size={13} /> Save current
        </Button>
      </div>

      {names.length === 0 ? (
        <div className="text-sm text-text-dim py-4 text-center">
          No saved snapshots yet
        </div>
      ) : (
        <div className="flex flex-col gap-0.5 max-h-72 overflow-y-auto">
          {names.map((name) => (
            <div
              key={name}
              className="flex flex-row items-center gap-2 group/snap px-2 py-1.5 rounded hover:bg-bg-overlay"
            >
              <span className="flex-1 truncate text-sm">{name}</span>
              <button
                onClick={() => handleLoad(name)}
                title="Load onto field"
                className="text-text-muted hover:text-accent-green"
              >
                <Download size={15} />
              </button>
              <button
                onClick={() =>
                  remove(name, {
                    onSuccess: () => toast.success(`Deleted "${name}"`),
                  })
                }
                title="Delete"
                className="text-text-muted hover:text-accent-red"
              >
                <Trash2 size={15} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SnapshotManager;
