import React, { FC, useEffect, useRef, useCallback, useState } from "react";
import { useExecutorInfo, useWorldState } from "../api";
import { WorldData } from "../bindings";
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

const CONT_PADDING_PX = 8;

interface FieldProps {
  selectedPlayerId: null | number;
  onSelectPlayer: (playerId: null | number) => void;
}

const Field: FC<FieldProps> = ({ selectedPlayerId, onSelectPlayer }) => {
  const contRef = useRef(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<FieldRenderer | null>(null);

  const manualControl = useExecutorInfo()?.manual_controlled_players ?? [];
  const world = useWorldState();
  const worldData = world.status === "connected" ? world.data : null;

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

    rendererRef.current.setPositionDisplayMode(positionDisplayMode);
    rendererRef.current.setWorldData(worldData);
    rendererRef.current.render(selectedPlayerId, manualControl);
  }, [
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

      const clickedObject = rendererRef.current.getClickedObject(x, y);

      if (clickedObject && clickedObject.type === "player") {
        onSelectPlayer(clickedObject.id ?? null);
      } else {
        onSelectPlayer(null);
      }
    },
    [onSelectPlayer]
  );

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

      <canvas
        ref={canvasRef}
        className="border-8 border-green-950 rounded"
        width={canvasWidth}
        height={canvasHeight}
        onClick={handleCanvasClick}
      />
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
