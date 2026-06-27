import React, { FC, useEffect, useRef, useCallback, useState, useMemo } from "react";
import {
  useDebugData,
  useExecutorInfo,
  useExecutorSettings,
  useSendCommand,
  useStatus,
  useWorldState,
  extractPlayerId,
  usePrimaryTeam,
  lastShortcutAtom,
} from "../api";
import { isTypingTarget } from "@/lib/commands";
import { TeamColor, Vector2, WorldData } from "../bindings";
import { useResizeObserver } from "@/lib/useResizeObserver";
import {
  CANVAS_PADDING,
  DEFAULT_FIELD_SIZE,
  FieldRenderer,
  ManualTargetMarker,
  PositionDisplayMode,
  SIM_EDIT_RING_RADIUS,
} from "./FieldRenderer";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Settings } from "lucide-react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn, radiansToDegrees, prettyPrintSnakeCases } from "@/lib/utils";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { BallPlacementPostionAtom } from "@/components/GameControllerPanel";
import GameBanner from "@/components/GameBanner";
import FieldAnnouncerOverlay from "@/components/FieldAnnouncerOverlay";
import { FrameCounter } from "@/components/FrameCounter";
import { ReplayTransport } from "@/components/ReplayTransport";
import {
  debugLayerVisibilityAtom,
  debugCategoryVisibilityAtom,
  filterDebugMap,
} from "@/lib/debugLayers";
import DebugVisibilityControls from "./DebugVisibilityControls";
import {
  pinnedFieldKeysAtom,
  formatFieldDebugValue,
  pinnedDebugKeysAtom,
  formatPlayerDebugValue,
} from "@/lib/pinnedDebug";
import {
  maskEditModeAtom,
  simEditModeAtom,
  manualTargetsAtom,
  manualTargetKey,
} from "@/lib/fieldEditing";
import { X } from "lucide-react";

const CONT_PADDING_PX = 8;

// --- Sim Edit tuning -------------------------------------------------------
/** Grab radius (mm) for picking up the ball with the cursor. */
const BALL_GRAB_RADIUS = 200;
/** Grab radius (mm) around the rotation handle knob. */
const HANDLE_GRAB_RADIUS = 90;
/** Min teleport spacing (ms) while dragging, to throttle command spam. */
const TELEPORT_THROTTLE_MS = 40;
/** Pull distance (mm) → ball exit speed (mm/s). Ball mass is 1, so impulse = Δv. */
const KICK_SCALE = 3.0;
/** Max kick impulse magnitude (mm/s), roughly SSL max ball speed. */
const KICK_MAX = 6500;

/** An in-progress Sim Edit drag. */
type SimDrag =
  | { kind: "robot"; teamColor: TeamColor; playerId: number; yaw: number }
  | { kind: "robot-rotate"; teamColor: TeamColor; playerId: number; pos: Vector2 }
  | { kind: "ball-move" }
  | { kind: "ball-kick"; ball: Vector2 };

/** Which robot to draw rotation controls around. */
type SimEditRobotRef = { teamColor: TeamColor; playerId: number };

interface FieldProps {
  selectedPlayerId: null | number;
  onSelectPlayer: (playerId: null | number) => void;
}

interface PlayerTooltip {
  position: [number, number];
  color: TeamColor;
  playerId: number;
}

const Field: FC<FieldProps> = ({ selectedPlayerId, onSelectPlayer }) => {
  const contRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<FieldRenderer | null>(null);
  const [mouseField, setMouseField] = useState<Vector2>([0, 0]);
  const [playerTooltip, setPlayerTooltip] = useState<PlayerTooltip | null>(
    null
  );
  const contextMenuPosRef = useRef([0, 0] as [number, number]);
  const isOverFieldRef = useRef(false);
  const mouseContRef = useRef<[number, number]>([0, 0]);
  // Popover for choosing a player to target when none is selected (H key).
  const [targetPicker, setTargetPicker] = useState<{
    field: [number, number];
    at: [number, number];
  } | null>(null);
  const setLastShortcut = useSetAtom(lastShortcutAtom);

  const executorInfo = useExecutorInfo();
  const manualControlledPlayerIds =
    executorInfo?.manual_controlled_players.map(extractPlayerId) ?? [];
  const world = useWorldState();
  const worldData = world.status === "connected" ? world.data : null;
  const sendCommand = useSendCommand();

  const { data: backendState } = useStatus();
  const isSim = backendState?.ui_mode === "Simulation";

  const mouseFieldRef = useRef(mouseField);
  mouseFieldRef.current = mouseField;
  const ballRef = useRef(worldData?.ball);
  ballRef.current = worldData?.ball;

  // --- Sim Edit mode (drag robots/ball, slingshot kick) -------------------
  const [simEditMode, setSimEditMode] = useAtom(simEditModeAtom);
  const simEditRef = useRef(simEditMode);
  simEditRef.current = simEditMode;
  const worldDataRef = useRef(worldData);
  worldDataRef.current = worldData;
  const sendCommandRef = useRef(sendCommand);
  sendCommandRef.current = sendCommand;
  const simDragRef = useRef<SimDrag | null>(null);
  const lastTeleportRef = useRef(0);
  // In-progress slingshot kick, drawn by the renderer.
  const [simKickDraft, setSimKickDraft] = useState<{
    ball: Vector2;
    pull: Vector2;
  } | null>(null);
  // Robot to show rotation controls around: the one being edited, else hovered.
  const [simEditHover, setSimEditHover] = useState<SimEditRobotRef | null>(null);
  const [simEditActiveRobot, setSimEditActiveRobot] =
    useState<SimEditRobotRef | null>(null);
  const simEditRobotMarker = useMemo<{ pos: Vector2; yaw: number } | null>(() => {
    if (!simEditMode) return null;
    const sel = simEditActiveRobot ?? simEditHover;
    if (!sel) return null;
    const team =
      sel.teamColor === TeamColor.Blue
        ? worldData?.blue_team
        : worldData?.yellow_team;
    const p = team?.find((pp) => pp.id === sel.playerId);
    return p ? { pos: p.position, yaw: p.yaw } : null;
  }, [simEditMode, simEditActiveRobot, simEditHover, worldData]);

  const [primaryTeam] = usePrimaryTeam();
  const debugMap = useDebugData();
  const [layerVis] = useAtom(debugLayerVisibilityAtom);
  const [categoryVis] = useAtom(debugCategoryVisibilityAtom);
  const filteredDebugMap = useMemo(() => {
    if (!debugMap) return null;
    return filterDebugMap(debugMap, layerVis, categoryVis, primaryTeam);
  }, [debugMap, layerVis, categoryVis, primaryTeam]);

  const [positionDisplayMode, setPositionDisplayMode] =
    useState<PositionDisplayMode>("filtered");

  // --- Field mask (vision crop) -------------------------------------------
  const { settings: executorSettings, updateSettings } = useExecutorSettings();
  const fieldMask = executorSettings?.tracker_settings.field_mask ?? null;
  const [maskEditMode, setMaskEditMode] = useAtom(maskEditModeAtom);
  const maskEditRef = useRef(maskEditMode);
  maskEditRef.current = maskEditMode;
  // In-progress drag rectangle in field mm: [x1, y1, x2, y2].
  const [maskDraft, setMaskDraft] = useState<
    [number, number, number, number] | null
  >(null);
  const maskDragStartRef = useRef<[number, number] | null>(null);

  // Field half-extents in mm (matches the backend FieldMask scaling).
  const halfExtents = useMemo<[number, number]>(() => {
    const geom = worldData?.field_geom;
    if (geom) {
      return [
        geom.field_length / 2 + geom.boundary_width,
        geom.field_width / 2 + geom.boundary_width,
      ];
    }
    return [DEFAULT_FIELD_SIZE[0] / 2, DEFAULT_FIELD_SIZE[1] / 2];
  }, [worldData?.field_geom]);

  const commitMask = useCallback(
    (rect: [number, number, number, number]) => {
      if (!executorSettings) return;
      const [hx, hy] = halfExtents;
      const clamp = (v: number) => Math.max(-1, Math.min(1, v));
      const mask = {
        x_min: clamp(Math.min(rect[0], rect[2]) / hx),
        x_max: clamp(Math.max(rect[0], rect[2]) / hx),
        y_min: clamp(Math.min(rect[1], rect[3]) / hy),
        y_max: clamp(Math.max(rect[1], rect[3]) / hy),
      };
      updateSettings({
        ...executorSettings,
        tracker_settings: { ...executorSettings.tracker_settings, field_mask: mask },
      });
    },
    [executorSettings, halfExtents, updateSettings]
  );

  // --- Manual MoveTo targets (tracked client-side) ------------------------
  const [manualTargets, setManualTargets] = useAtom(manualTargetsAtom);
  // Prune targets for players no longer under manual control.
  const manualControlKeys = (executorInfo?.manual_controlled_players ?? []).map(
    (p) => manualTargetKey(p.team_color, p.player_id)
  );
  const manualControlKeysStr = manualControlKeys.join(",");
  useEffect(() => {
    setManualTargets((prev) => {
      const next: typeof prev = {};
      for (const k of Object.keys(prev)) {
        if (manualControlKeys.includes(k)) next[k] = prev[k];
      }
      return Object.keys(next).length === Object.keys(prev).length ? prev : next;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manualControlKeysStr]);

  const manualTargetMarkers = useMemo<ManualTargetMarker[]>(() => {
    const team =
      primaryTeam === TeamColor.Blue
        ? worldData?.blue_team
        : worldData?.yellow_team;
    return Object.entries(manualTargets)
      .filter(([key]) => key.startsWith(`${primaryTeam}:`))
      .map(([key, to]) => {
        const id = Number(key.split(":")[1]);
        const player = team?.find((p) => p.id === id);
        return { from: player ? player.position : null, to };
      });
  }, [manualTargets, primaryTeam, worldData]);

  // --- Floating overlay pins ----------------------------------------------
  const [pinnedFieldKeys, setPinnedFieldKeys] = useAtom(pinnedFieldKeysAtom);
  const pinnedPlayerKeys = useAtomValue(pinnedDebugKeysAtom);

  const { width: contWidth = 0, height: contHeight = 0 } = useResizeObserver({
    ref: contRef,
  });
  const { canvasWidth, canvasHeight } = useCanvasSize(
    worldData,
    contWidth,
    contHeight
  );
  const [manualBallPlacementPosition, setManualBallPlacementPosition] = useAtom(
    BallPlacementPostionAtom
  );

  useEffect(() => {
    if (!canvasRef.current) return;

    if (!rendererRef.current) {
      rendererRef.current = new FieldRenderer(canvasRef.current);
    }

    if (filteredDebugMap) {
      rendererRef.current.setDebugData(filteredDebugMap);
    }
    rendererRef.current.setPositionDisplayMode(positionDisplayMode);
    rendererRef.current.setWorldData(worldData);
    rendererRef.current.setFieldMask(fieldMask);
    rendererRef.current.setMaskDraft(maskDraft);
    rendererRef.current.setManualTargets(manualTargetMarkers);
    rendererRef.current.setSimKickDraft(simKickDraft);
    rendererRef.current.setSimEditRobot(simEditRobotMarker);
    rendererRef.current.render(
      selectedPlayerId,
      primaryTeam,
      manualControlledPlayerIds,
      manualBallPlacementPosition
    );
  }, [
    filteredDebugMap,
    worldData,
    canvasWidth,
    canvasHeight,
    manualControlledPlayerIds,
    positionDisplayMode,
    selectedPlayerId,
    manualBallPlacementPosition,
    fieldMask,
    maskDraft,
    manualTargetMarkers,
    simKickDraft,
    simEditRobotMarker,
  ]);

  const selectedPlayerData =
    primaryTeam === TeamColor.Blue
      ? worldData?.blue_team.find((p) => p.id === selectedPlayerId) ?? null
      : worldData?.yellow_team.find((p) => p.id === selectedPlayerId) ?? null;

  const ownPlayers =
    (primaryTeam === TeamColor.Blue
      ? worldData?.blue_team
      : worldData?.yellow_team) ?? [];

  // Send a MoveTo target to a player, enabling manual control first (MoveTo
  // overrides only take effect for manually-controlled robots). Kept in a ref
  // so the global 'h' key handler always sees fresh state.
  const sendTargetRef = useRef<(id: number, pos: [number, number]) => void>(
    () => {}
  );
  sendTargetRef.current = (playerId, pos) => {
    if (!manualControlledPlayerIds.includes(playerId)) {
      sendCommand({
        type: "SetManualOverride",
        data: {
          team_color: primaryTeam,
          player_id: playerId,
          manual_override: true,
        },
      });
    }
    sendCommand({
      type: "OverrideCommand",
      data: {
        team_color: primaryTeam,
        player_id: playerId,
        command: {
          type: "MoveTo",
          data: { position: pos, yaw: undefined, dribble_speed: 0, arm_kick: false },
        },
      },
    });
    setManualTargets((prev) => ({
      ...prev,
      [manualTargetKey(primaryTeam, playerId)]: pos,
    }));
  };

  const selectedPlayerIdRef = useRef(selectedPlayerId);
  selectedPlayerIdRef.current = selectedPlayerId;

  // 'H' over the field: set target for the current player, or open a picker.
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (
        ev.key.toLowerCase() !== "h" ||
        ev.shiftKey ||
        ev.ctrlKey ||
        ev.metaKey ||
        ev.altKey
      )
        return;
      if (isTypingTarget(document.activeElement)) return;
      if (!isOverFieldRef.current) return;
      ev.preventDefault();
      const pos = mouseFieldRef.current;
      const sel = selectedPlayerIdRef.current;
      if (sel !== null) {
        sendTargetRef.current(sel, pos);
        setLastShortcut({ label: `Target → #${sel}`, ts: Date.now() });
      } else {
        setTargetPicker({ field: pos, at: mouseContRef.current });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Esc cancels mask / sim editing.
  useEffect(() => {
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key !== "Escape") return;
      if (maskEditRef.current) {
        setMaskEditMode(false);
        maskDragStartRef.current = null;
        setMaskDraft(null);
      }
      if (simEditRef.current) {
        setSimEditMode(false);
        simDragRef.current = null;
        setSimKickDraft(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setMaskEditMode, setSimEditMode]);

  /** Canvas-pixel event → field-mm coordinates. */
  const eventToField = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>): [number, number] | null => {
      if (!canvasRef.current || !rendererRef.current) return null;
      const rect = canvasRef.current.getBoundingClientRect();
      return rendererRef.current.canvasToField([
        event.clientX - rect.left,
        event.clientY - rect.top,
      ]);
    },
    []
  );

  // --- Sim Edit handlers --------------------------------------------------
  const sendTeleportRobot = useCallback(
    (teamColor: TeamColor, playerId: number, position: Vector2, yaw: number) => {
      sendCommandRef.current({
        type: "SimulatorCmd",
        data: {
          type: "TeleportRobot",
          data: { team_color: teamColor, player_id: playerId, position, yaw },
        },
      });
    },
    []
  );
  const sendTeleportBall = useCallback((position: Vector2) => {
    sendCommandRef.current({
      type: "SimulatorCmd",
      data: { type: "TeleportBall", data: { position } },
    });
  }, []);

  /** Try to begin a Sim Edit drag. Returns true if one started. */
  const onSimMouseDown = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>): boolean => {
      if (!simEditRef.current || !rendererRef.current) return false;
      const f = eventToField(event);
      if (!f) return false;
      const ball = worldDataRef.current?.ball;
      const overBall =
        !!ball &&
        Math.hypot(ball.position[0] - f[0], ball.position[1] - f[1]) <
          BALL_GRAB_RADIUS;

      // Shift-drag on the ball → slingshot kick.
      if (event.shiftKey && overBall && ball) {
        event.preventDefault();
        const origin: Vector2 = [ball.position[0], ball.position[1]];
        simDragRef.current = { kind: "ball-kick", ball: origin };
        setSimKickDraft({ ball: origin, pull: f });
        return true;
      }

      // Robot under (or near) the cursor → rotate via the handle, else move.
      const player = rendererRef.current.getPlayerAt(f[0], f[1]);
      if (player) {
        const [teamColor, playerId] = player;
        const team =
          teamColor === TeamColor.Blue
            ? worldDataRef.current?.blue_team
            : worldDataRef.current?.yellow_team;
        const pd = team?.find((p) => p.id === playerId);
        const pos: Vector2 = pd ? pd.position : f;
        const yaw = pd?.yaw ?? 0;
        event.preventDefault();
        setSimEditActiveRobot({ teamColor, playerId });

        // Near the heading handle on the ring → rotate (position fixed).
        const handle: Vector2 = [
          pos[0] + Math.cos(yaw) * SIM_EDIT_RING_RADIUS,
          pos[1] + Math.sin(yaw) * SIM_EDIT_RING_RADIUS,
        ];
        if (Math.hypot(handle[0] - f[0], handle[1] - f[1]) < HANDLE_GRAB_RADIUS) {
          simDragRef.current = { kind: "robot-rotate", teamColor, playerId, pos };
          return true;
        }

        simDragRef.current = { kind: "robot", teamColor, playerId, yaw };
        sendTeleportRobot(teamColor, playerId, f, yaw);
        return true;
      }

      // Otherwise, plain drag on the ball → move it.
      if (overBall) {
        event.preventDefault();
        simDragRef.current = { kind: "ball-move" };
        sendTeleportBall(f);
        return true;
      }
      return false;
    },
    [eventToField, sendTeleportRobot, sendTeleportBall]
  );

  /** Drive an in-progress Sim Edit drag from a mouse-move. */
  const onSimDrag = useCallback(
    (f: Vector2) => {
      const d = simDragRef.current;
      if (!d) return;
      if (d.kind === "ball-kick") {
        setSimKickDraft({ ball: d.ball, pull: f });
        return;
      }
      const now = performance.now();
      if (now - lastTeleportRef.current < TELEPORT_THROTTLE_MS) return;
      lastTeleportRef.current = now;
      if (d.kind === "robot") {
        sendTeleportRobot(d.teamColor, d.playerId, f, d.yaw);
      } else if (d.kind === "robot-rotate") {
        const yaw = Math.atan2(f[1] - d.pos[1], f[0] - d.pos[0]);
        sendTeleportRobot(d.teamColor, d.playerId, d.pos, yaw);
      } else {
        sendTeleportBall(f);
      }
    },
    [sendTeleportRobot, sendTeleportBall]
  );

  /** Finish a Sim Edit drag. Returns true if one was active. */
  const onSimMouseUp = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>): boolean => {
      const d = simDragRef.current;
      if (!d) return false;
      simDragRef.current = null;
      setSimEditActiveRobot(null);
      const f = eventToField(event);
      if (d.kind === "ball-kick") {
        setSimKickDraft(null);
        if (f) {
          let fx = (d.ball[0] - f[0]) * KICK_SCALE;
          let fy = (d.ball[1] - f[1]) * KICK_SCALE;
          const mag = Math.hypot(fx, fy);
          if (mag > KICK_MAX) {
            fx = (fx / mag) * KICK_MAX;
            fy = (fy / mag) * KICK_MAX;
          }
          if (mag > 1) {
            sendCommandRef.current({
              type: "SimulatorCmd",
              data: { type: "ApplyBallForce", data: { force: [fx, fy] } },
            });
          }
        }
        return true;
      }
      if (f) {
        if (d.kind === "robot") {
          sendTeleportRobot(d.teamColor, d.playerId, f, d.yaw);
        } else if (d.kind === "robot-rotate") {
          const yaw = Math.atan2(f[1] - d.pos[1], f[0] - d.pos[0]);
          sendTeleportRobot(d.teamColor, d.playerId, d.pos, yaw);
        } else {
          sendTeleportBall(f);
        }
      }
      return true;
    },
    [eventToField, sendTeleportRobot, sendTeleportBall]
  );

  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (maskEditRef.current || simEditRef.current) return;
      if (!canvasRef.current || !rendererRef.current) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const [fieldX, fieldY] = rendererRef.current.canvasToField([x, y]);
      const clickedPlayer = rendererRef.current.getPlayerAt(fieldX, fieldY);

      if (clickedPlayer !== null && clickedPlayer[0] === primaryTeam) {
        onSelectPlayer(clickedPlayer[1]);
      }
    },
    [onSelectPlayer, primaryTeam]
  );

  const handleMaskMouseDown = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!maskEditRef.current) return;
      const f = eventToField(event);
      if (!f) return;
      event.preventDefault();
      maskDragStartRef.current = f;
      setMaskDraft([f[0], f[1], f[0], f[1]]);
    },
    [eventToField]
  );

  const handleMaskMouseUp = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!maskEditRef.current || !maskDragStartRef.current) return;
      const end = eventToField(event) ?? maskDragStartRef.current;
      const [sx, sy] = maskDragStartRef.current;
      maskDragStartRef.current = null;
      setMaskDraft(null);
      // Ignore trivial drags (a stray click).
      if (Math.abs(end[0] - sx) > 50 && Math.abs(end[1] - sy) > 50) {
        commitMask([sx, sy, end[0], end[1]]);
      }
      setMaskEditMode(false);
    },
    [eventToField, commitMask, setMaskEditMode]
  );

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!canvasRef.current || !contRef.current || !rendererRef.current)
        return;

      const rect = canvasRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const fieldXY = rendererRef.current.canvasToField([x, y]);
      setMouseField(fieldXY);

      if (simEditRef.current) {
        if (simDragRef.current) {
          onSimDrag(fieldXY);
        } else {
          // Hover: show rotation controls around the robot under the cursor.
          const p = rendererRef.current.getPlayerAt(fieldXY[0], fieldXY[1]);
          setSimEditHover(
            p ? { teamColor: p[0], playerId: p[1] } : null
          );
        }
        setPlayerTooltip(null);
        return;
      }

      if (maskEditRef.current && maskDragStartRef.current) {
        const [sx, sy] = maskDragStartRef.current;
        setMaskDraft([sx, sy, fieldXY[0], fieldXY[1]]);
      }

      const contRect0 = contRef.current.getBoundingClientRect();
      mouseContRef.current = [
        event.clientX - contRect0.left,
        event.clientY - contRect0.top,
      ];

      const player = rendererRef.current.getPlayerAt(fieldXY[0], fieldXY[1]);
      if (player !== null) {
        const [color, playerId] = player;
        const contRect = contRef.current.getBoundingClientRect();
        const x = event.clientX - contRect.left + 10;
        const y = event.clientY - contRect.top + 10;
        setPlayerTooltip({
          position: [x, y],
          color,
          playerId,
        });
      } else {
        setPlayerTooltip(null);
      }
    },
    [onSimDrag]
  );
  const playerTooltipData =
    primaryTeam === TeamColor.Blue
      ? worldData?.blue_team.find((p) => p.id === playerTooltip?.playerId) ??
        null
      : worldData?.yellow_team.find((p) => p.id === playerTooltip?.playerId) ??
        null;

  const headingRef = useRef<number | null>(null);
  const handleContextMenu = (event: React.MouseEvent<HTMLElement>) => {
    if (!canvasRef.current || !contRef.current || !rendererRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    contextMenuPosRef.current = rendererRef.current.canvasToField([x, y]);
  };
  const handleTargetPosition = () => {
    sendCommand({
      type: "OverrideCommand",
      data: {
        team_color: primaryTeam,
        player_id: manualControlledPlayerIds[0],
        command: {
          type: "MoveTo",
          data: {
            position: contextMenuPosRef.current,
            arm_kick: false,
            yaw: headingRef.current ?? undefined,
            dribble_speed: 0,
          },
        },
      },
    });
    setManualTargets((prev) => ({
      ...prev,
      [manualTargetKey(primaryTeam, manualControlledPlayerIds[0])]:
        contextMenuPosRef.current,
    }));
  };

  const handleTargetHeading = () => {
    const defaultTeamId = 1; // TODO: Get from primary team selection
    // compute yaw
    const pos1 = selectedPlayerData?.position ?? [0, 0];
    const pos2 = contextMenuPosRef.current;
    const angle = Math.atan2(pos2[1] - pos1[1], pos2[0] - pos1[0]);
    headingRef.current = angle;
    sendCommand({
      type: "OverrideCommand",
      data: {
        team_color: primaryTeam,
        player_id: manualControlledPlayerIds[0],
        command: {
          type: "MoveTo",
          data: {
            position: selectedPlayerData?.position ?? [0, 0],
            arm_kick: false,
            yaw: angle,
            dribble_speed: 0,
          },
        },
      },
    });
  };

  return (
    <div
      ref={contRef}
      className="relative w-full h-full flex items-center justify-center overflow-hidden"
      style={{ padding: CONT_PADDING_PX }}
      onMouseEnter={() => (isOverFieldRef.current = true)}
      onMouseLeave={() => {
        isOverFieldRef.current = false;
        if (!simDragRef.current) setSimEditHover(null);
      }}
    >
      <Popover>
        <PopoverTrigger asChild>
          <Button className="absolute top-0 left-0">
            <Settings size={24} />
          </Button>
        </PopoverTrigger>

        <PopoverContent className="flex flex-col gap-3 w-80 max-h-[80vh] overflow-y-auto">
          <div className="flex flex-row items-center justify-between gap-4">
            <div className="text-sm">Position Display</div>
            <ToggleGroup
              type="multiple"
              value={
                positionDisplayMode === "both"
                  ? ["raw", "filtered"]
                  : [positionDisplayMode]
              }
              onValueChange={(val) =>
                val.length === 2
                  ? setPositionDisplayMode("both")
                  : val.length === 1
                  ? setPositionDisplayMode(val[0] as PositionDisplayMode)
                  : undefined
              }
              className="border border-gray-500 rounded-lg"
            >
              <ToggleGroupItem value="raw">Vision</ToggleGroupItem>
              <ToggleGroupItem value="filtered">Filtered</ToggleGroupItem>
            </ToggleGroup>
          </div>
          <div className="border-t border-border-dim pt-2">
            <DebugVisibilityControls />
          </div>
        </PopoverContent>
      </Popover>

      {/* Mask edit hint */}
      {maskEditMode ? (
        <div className="absolute top-9 left-1/2 -translate-x-1/2 z-30 bg-accent-cyan/90 text-black text-xs font-medium px-3 py-1 rounded shadow">
          Drag on the field to set the vision mask — Esc to cancel
        </div>
      ) : null}

      {/* Sim edit hint */}
      {simEditMode ? (
        <div className="absolute top-9 left-1/2 -translate-x-1/2 z-30 bg-accent-amber/90 text-black text-xs font-medium px-3 py-1 rounded shadow">
          Drag robots/ball to place — Shift-drag the ball to kick — Esc to exit
        </div>
      ) : null}

      {/* Pinned debug overlay (top-left, under the view-settings button) */}
      {(pinnedFieldKeys.length > 0 ||
        (selectedPlayerId !== null && pinnedPlayerKeys.length > 0)) && (
        <div className="absolute top-10 left-0 z-10 bg-black/45 backdrop-blur-sm rounded px-2 py-1.5 text-xs font-mono max-w-[16rem] pointer-events-auto">
          {pinnedFieldKeys.map((key) => {
            const val = formatFieldDebugValue(debugMap, key);
            return (
              <div key={key} className="flex items-center gap-2 group/pin py-px">
                <span className="text-text-dim truncate">
                  {prettyPrintSnakeCases(key.split(".").slice(-2).join("."))}
                </span>
                <span className="ml-auto text-text-bright tabular-nums">
                  {val ?? "—"}
                </span>
                <button
                  onClick={() =>
                    setPinnedFieldKeys((prev) => prev.filter((k) => k !== key))
                  }
                  title="Unpin"
                  className="opacity-0 group-hover/pin:opacity-100 text-text-muted hover:text-red-400"
                >
                  <X size={10} />
                </button>
              </div>
            );
          })}
          {selectedPlayerId !== null && pinnedPlayerKeys.length > 0 && (
            <>
              {pinnedFieldKeys.length > 0 && (
                <div className="border-t border-white/10 my-1" />
              )}
              <div className="text-text-muted">#{selectedPlayerId}</div>
              {pinnedPlayerKeys.map((subkey) => {
                const val = formatPlayerDebugValue(
                  debugMap,
                  selectedPlayerId,
                  primaryTeam,
                  subkey
                );
                return (
                  <div key={subkey} className="flex items-center gap-2 py-px">
                    <span className="text-text-dim truncate">
                      {prettyPrintSnakeCases(subkey)}
                    </span>
                    <span className="ml-auto text-text-bright tabular-nums">
                      {val ?? "—"}
                    </span>
                  </div>
                );
              })}
            </>
          )}
        </div>
      )}

      {playerTooltip ? (
        <div
          className="absolute z-10 bg-slate-950 bg-opacity-70 p-2 rounded"
          style={{
            left: playerTooltip.position[0],
            top: playerTooltip.position[1],
          }}
        >
          <div
            className={cn(
              "mb-2",
              selectedPlayerId === playerTooltip.playerId && "font-bold"
            )}
          >
            Player #{playerTooltip.playerId}
          </div>
          <div className="flex flex-row font-mono">
            <div className="w-full">
              X: {playerTooltipData?.position[0].toFixed(0)} mm
            </div>
          </div>
          <div className="flex flex-row font-mono">
            <div className="w-full">
              Y: {playerTooltipData?.position[1].toFixed(0)} mm
            </div>
          </div>
          <div className="flex flex-row font-mono">
            <div className="w-full">
              Yaw: {radiansToDegrees(playerTooltipData?.yaw ?? 0).toFixed(2)}{" "}
              deg
            </div>
          </div>
        </div>
      ) : null}

      {/* Target-player picker (opened by H when no player is selected) */}
      {targetPicker ? (
        <>
          <div
            className="absolute inset-0 z-20"
            onClick={() => setTargetPicker(null)}
            onContextMenu={(e) => {
              e.preventDefault();
              setTargetPicker(null);
            }}
          />
          <div
            className="absolute z-30 bg-bg-elevated border border-border-std shadow-lg p-1 flex flex-col min-w-28"
            style={{
              left: Math.min(targetPicker.at[0], (contWidth || 0) - 130),
              top: Math.min(targetPicker.at[1], (contHeight || 0) - 200),
            }}
          >
            <div className="px-2 py-1 text-[11px] uppercase tracking-wider text-text-dim">
              Target robot →
            </div>
            {ownPlayers.length === 0 ? (
              <div className="px-2 py-1 text-sm text-text-dim">No robots</div>
            ) : (
              [...ownPlayers]
                .sort((a, b) => a.id - b.id)
                .map((p) => (
                  <button
                    key={p.id}
                    className="text-left px-2 py-1 text-sm text-text-std hover:bg-bg-overlay"
                    onClick={() => {
                      sendTargetRef.current(p.id, targetPicker.field);
                      onSelectPlayer(p.id);
                      setLastShortcut({
                        label: `Target → #${p.id}`,
                        ts: Date.now(),
                      });
                      setTargetPicker(null);
                    }}
                  >
                    Player #{p.id}
                  </button>
                ))
            )}
          </div>
        </>
      ) : null}

      {/* Central status banner: score + game state + what's next */}
      {worldData && (
        <GameBanner
          gameState={worldData.game_state}
          sideAssignment={worldData.side_assignment}
        />
      )}

      {/* Right-edge game-state panel + GC quick actions + announcer feed */}
      <FieldAnnouncerOverlay />

      {/* Frame number (top-right) + replay transport (bottom, when replaying) */}
      <FrameCounter />
      <ReplayTransport />

      <div className="absolute bottom-0 right-0 bg-slate-950 bg-opacity-70 p-2 rounded">
        <div className="flex flex-row font-mono">
          <div className="w-20">X: {mouseField[0].toFixed(0)}</div>
          <span>mm</span>
        </div>
        <div className="flex flex-row font-mono">
          <div className="w-20">Y: {mouseField[1].toFixed(0)}</div>
          <span>mm</span>
        </div>
      </div>

      <ContextMenu>
        <ContextMenuTrigger onContextMenu={handleContextMenu}>
          <canvas
            ref={canvasRef}
            width={canvasWidth}
            height={canvasHeight}
            onClick={handleCanvasClick}
            onMouseMove={handleMouseMove}
            onMouseDown={(e) => {
              if (onSimMouseDown(e)) return;
              handleMaskMouseDown(e);
            }}
            onMouseUp={(e) => {
              if (onSimMouseUp(e)) return;
              handleMaskMouseUp(e);
            }}
            style={
              maskEditMode
                ? { cursor: "crosshair" }
                : simEditMode
                ? { cursor: "grab" }
                : undefined
            }
          />
        </ContextMenuTrigger>

        <ContextMenuContent>
          {manualControlledPlayerIds.length === 1 ? (
            <>
              <ContextMenuItem onClick={handleTargetPosition}>
                Set target position
              </ContextMenuItem>
              <ContextMenuItem onClick={handleTargetHeading}>
                Set target heading
              </ContextMenuItem>
            </>
          ) : null}

          {isSim ? (
            <ContextMenuItem
              onClick={() => setManualBallPlacementPosition(mouseField)}
            >
              Set ball placement position
            </ContextMenuItem>
          ) : null}
        </ContextMenuContent>
      </ContextMenu>
    </div>
  );
};

export default Field;

function useCanvasSize(
  worldData: WorldData | null,
  contWidth: number,
  contHeight: number
): { canvasWidth: number; canvasHeight: number } {
  const fieldSize = [
    (worldData?.field_geom?.field_length ?? DEFAULT_FIELD_SIZE[0]) +
      2 * CANVAS_PADDING,
    (worldData?.field_geom?.field_width ?? DEFAULT_FIELD_SIZE[1]) +
      2 * CANVAS_PADDING,
  ];
  const availableWidth = contWidth - 2 * CONT_PADDING_PX;
  const availableHeight = contHeight - 2 * CONT_PADDING_PX;
  const fieldAspectRatio = fieldSize[0] / fieldSize[1];
  const containerAspectRatio = availableWidth / availableHeight;

  let canvasWidth: number;
  let canvasHeight: number;
  if (containerAspectRatio > fieldAspectRatio) {
    // Container is wider than the field aspect ratio
    canvasHeight = availableHeight;
    canvasWidth = canvasHeight * fieldAspectRatio;
  } else {
    // Container is taller than the field aspect ratio
    canvasWidth = availableWidth;
    canvasHeight = canvasWidth / fieldAspectRatio;
  }
  canvasWidth = Math.floor(canvasWidth);
  canvasHeight = Math.floor(canvasHeight);
  return { canvasWidth, canvasHeight };
}
