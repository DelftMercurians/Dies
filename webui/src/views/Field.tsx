import React, { FC, useEffect, useRef, useCallback, useState } from "react";
import {
  useDebugData,
  useExecutorInfo,
  useSendCommand,
  useWorldState,
} from "../api";
import { Vector2, WorldData } from "../bindings";
import { useResizeObserver } from "@/lib/useResizeObserver";
import {
  CANVAS_PADDING,
  DEFAULT_FIELD_SIZE,
  FieldRenderer,
  PositionDisplayMode,
} from "./FieldRenderer";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Settings } from "lucide-react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from "@/components/ui/context-menu";

const CONT_PADDING_PX = 8;

interface FieldProps {
  selectedPlayerId: null | number;
  onSelectPlayer: (playerId: null | number) => void;
}

interface PlayerTooltip {
  position: [number, number];
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

  const manualControl = useExecutorInfo()?.manual_controlled_players ?? [];
  const world = useWorldState();
  const worldData = world.status === "connected" ? world.data : null;
  const sendCommand = useSendCommand();

  const debugMap = useDebugData();

  const [positionDisplayMode, setPositionDisplayMode] =
    useState<PositionDisplayMode>("filtered");

  const { width: contWidth = 0, height: contHeight = 0 } = useResizeObserver({
    ref: contRef,
  });
  const { canvasWidth, canvasHeight } = useCanvasSize(
    worldData,
    contWidth,
    contHeight
  );

  useEffect(() => {
    if (!canvasRef.current) return;

    if (!rendererRef.current) {
      rendererRef.current = new FieldRenderer(canvasRef.current);
    }

    if (debugMap) {
      rendererRef.current.setDebugData(debugMap);
    }
    rendererRef.current.setPositionDisplayMode(positionDisplayMode);
    rendererRef.current.setWorldData(worldData);
    rendererRef.current.render(selectedPlayerId, manualControl);
  }, [
    debugMap,
    worldData,
    canvasWidth,
    canvasHeight,
    manualControl,
    positionDisplayMode,
    selectedPlayerId,
  ]);

  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      if (!canvasRef.current || !rendererRef.current) return;

      const rect = canvasRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const [fieldX, fieldY] = rendererRef.current.canvasToField([x, y]);
      const clickedPlayer = rendererRef.current.getPlayerAt(fieldX, fieldY);

      if (clickedPlayer !== null) {
        onSelectPlayer(clickedPlayer);
      } else {
        onSelectPlayer(null);
      }
    },
    [onSelectPlayer]
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

      const playerId = rendererRef.current.getPlayerAt(fieldXY[0], fieldXY[1]);
      if (playerId !== null) {
        const contRect = contRef.current.getBoundingClientRect();
        const x = event.clientX - contRect.left + 10;
        const y = event.clientY - contRect.top + 10;
        setPlayerTooltip({
          position: [x, y],
          playerId,
        });
      } else {
        setPlayerTooltip(null);
      }
    },
    []
  );
  const playerTooltipData = worldData?.own_players.find(
    (p) => p.id === playerTooltip?.playerId
  );

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
        player_id: manualControl[0],
        command: {
          type: "MoveTo",
          data: {
            position: contextMenuPosRef.current,
            arm_kick: false,
            yaw: 0,
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
    >
      <Popover>
        <PopoverTrigger asChild>
          <Button className="absolute top-0 left-0">
            <Settings size={24} />
          </Button>
        </PopoverTrigger>

        <PopoverContent className="flex flex-col w-max">
          <div className="flex flex-row items-center gap-4">
            <div>Position Display Mode</div>
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
        </PopoverContent>
      </Popover>

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
            <div className="w-20">
              X: {playerTooltipData?.position[0].toFixed(0)}
            </div>
            <span>mm</span>
          </div>
          <div className="flex flex-row font-mono">
            <div className="w-20">
              Y: {playerTooltipData?.position[1].toFixed(0)}
            </div>
            <span>mm</span>
          </div>
        </div>
      ) : null}

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
            className="border-8 border-green-950 rounded"
            width={canvasWidth}
            height={canvasHeight}
            onClick={handleCanvasClick}
            onMouseMove={handleMouseMove}
          />
        </ContextMenuTrigger>

        <ContextMenuContent>
          {manualControl.length === 1 ? (
            <ContextMenuItem onClick={handleTargetPosition}>
              Set target position
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