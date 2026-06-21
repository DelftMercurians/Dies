import React, { useMemo, useCallback, useRef, useEffect } from "react";
import { useAtom } from "jotai";
import { ChevronDown, ChevronRight } from "lucide-react";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { useDebugData } from "@/api";
import { prettyPrintSnakeCases, cn } from "@/lib/utils";
import {
  debugLayerVisibilityAtom,
  debugCategoryVisibilityAtom,
  buildLayerTree,
  isKeyVisible,
  categoryVisible,
  CATEGORIES,
  TreeNode,
} from "@/lib/debugLayers";

// ============================================================================
// Tri-state checkbox
// ============================================================================

type CheckState = "checked" | "unchecked" | "indeterminate";

const TriCheckbox: React.FC<{
  state: CheckState;
  onChange: () => void;
}> = ({ state, onChange }) => {
  const ref = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.indeterminate = state === "indeterminate";
    }
  }, [state]);

  return (
    <input
      ref={ref}
      type="checkbox"
      checked={state === "checked"}
      onChange={onChange}
      className="accent-blue-500 cursor-pointer shrink-0"
    />
  );
};

// ============================================================================
// Helpers
// ============================================================================

/** Collect all leaf fullPaths under a node (inclusive). */
function collectLeafPaths(node: TreeNode): string[] {
  if (node.children.length === 0) return [node.fullPath];
  const paths: string[] = [];
  for (const child of node.children) {
    paths.push(...collectLeafPaths(child));
  }
  return paths;
}

/** Collect all descendant fullPaths (nodes + leaves). */
function collectAllPaths(node: TreeNode): string[] {
  const paths = [node.fullPath];
  for (const child of node.children) {
    paths.push(...collectAllPaths(child));
  }
  return paths;
}

/**
 * Compute the check state for a node given the layer visibility map and
 * category visibility map. We check if all leaves under this node are visible.
 */
function getNodeCheckState(
  node: TreeNode,
  layerVis: Record<string, boolean>,
  categoryVis: Record<string, boolean>,
  allKeys: string[],
): CheckState {
  const leafPaths = collectLeafPaths(node);
  let visibleCount = 0;
  let totalCount = 0;
  for (const leafPath of leafPaths) {
    const matchingKeys = allKeys.filter(
      (k) => k === leafPath || k.startsWith(leafPath + "."),
    );
    if (matchingKeys.length === 0) {
      totalCount++;
      if (isKeyVisible(leafPath, layerVis, categoryVis)) visibleCount++;
    } else {
      for (const key of matchingKeys) {
        totalCount++;
        if (isKeyVisible(key, layerVis, categoryVis)) visibleCount++;
      }
    }
  }
  if (visibleCount === 0) return "unchecked";
  if (visibleCount === totalCount) return "checked";
  return "indeterminate";
}

// ============================================================================
// Tree Node component
// ============================================================================

const TreeNodeRow: React.FC<{
  node: TreeNode;
  layerVis: Record<string, boolean>;
  categoryVis: Record<string, boolean>;
  allKeys: string[];
  onToggle: (node: TreeNode) => void;
  openPaths: Set<string>;
  onOpenChange: (path: string, open: boolean) => void;
}> = ({ node, layerVis, categoryVis, allKeys, onToggle, openPaths, onOpenChange }) => {
  const checkState = getNodeCheckState(node, layerVis, categoryVis, allKeys);
  const isOpen = openPaths.has(node.fullPath);
  const hasChildren = node.children.length > 0;

  if (!hasChildren) {
    return (
      <div className="flex items-center gap-1.5 py-0.5">
        <TriCheckbox state={checkState} onChange={() => onToggle(node)} />
        <span className="text-sm text-text-std truncate">
          {prettyPrintSnakeCases(node.name)}
        </span>
        <span className="text-xs text-text-dim ml-auto shrink-0">
          {node.leafCount}
        </span>
      </div>
    );
  }

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={(open) => onOpenChange(node.fullPath, open)}
    >
      <div className="flex items-center gap-1.5 py-0.5">
        <TriCheckbox state={checkState} onChange={() => onToggle(node)} />
        <CollapsibleTrigger className="flex items-center gap-0.5 min-w-0">
          {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span className="text-sm font-medium text-text-std truncate">
            {prettyPrintSnakeCases(node.name)}
          </span>
        </CollapsibleTrigger>
        <span className="text-xs text-text-dim ml-auto shrink-0">
          {node.leafCount}
        </span>
      </div>
      <CollapsibleContent>
        <div className="ml-4 relative">
          <div className="w-px h-full bg-border-dim absolute -left-2 top-0" />
          {node.children.map((child) => (
            <TreeNodeRow
              key={child.fullPath}
              node={child}
              layerVis={layerVis}
              categoryVis={categoryVis}
              allKeys={allKeys}
              onToggle={onToggle}
              openPaths={openPaths}
              onOpenChange={onOpenChange}
            />
          ))}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
};

// ============================================================================
// Categories + Layers visibility controls
// ============================================================================

/**
 * Category chips + the layer-visibility tree. Lives in the field's view-settings
 * popover (moved out of the Debug panel, which now only lists values).
 */
const DebugVisibilityControls: React.FC = () => {
  const debugMap = useDebugData();
  const [layerVis, setLayerVis] = useAtom(debugLayerVisibilityAtom);
  const [categoryVis, setCategoryVis] = useAtom(debugCategoryVisibilityAtom);

  const allKeys = useMemo(
    () => (debugMap ? Object.keys(debugMap) : []),
    [debugMap],
  );

  const tree = useMemo(() => buildLayerTree(allKeys), [allKeys]);

  const [openPaths, setOpenPaths] = React.useState<Set<string>>(new Set());
  const handleOpenChange = useCallback((path: string, open: boolean) => {
    setOpenPaths((prev) => {
      const next = new Set(prev);
      if (open) next.add(path);
      else next.delete(path);
      return next;
    });
  }, []);

  const handleToggleNode = useCallback(
    (node: TreeNode) => {
      const state = getNodeCheckState(node, layerVis, categoryVis, allKeys);
      const newVisible = state !== "checked";

      setLayerVis((prev) => {
        const next = { ...prev };
        const allDescendants = collectAllPaths(node);
        for (const p of allDescendants) {
          delete next[p];
        }
        next[node.fullPath] = newVisible;
        return next;
      });
    },
    [layerVis, categoryVis, allKeys, setLayerVis],
  );

  const handleToggleCategory = useCallback(
    (catId: string) => {
      const cat = CATEGORIES.find((c) => c.id === catId);
      setCategoryVis((prev) => {
        const current = cat ? categoryVisible(cat, prev) : (prev[catId] ?? true);
        return { ...prev, [catId]: !current };
      });
    },
    [setCategoryVis],
  );

  const handleReset = useCallback(() => {
    setLayerVis({});
    setCategoryVis({});
  }, [setLayerVis, setCategoryVis]);

  return (
    <div className="flex flex-col min-h-0">
      {/* Categories */}
      <div className="flex items-center justify-between mb-1.5">
        <div className="text-xs text-text-dim">Categories</div>
        <Button
          variant="ghost"
          size="xs"
          className="h-5 px-1.5"
          onClick={handleReset}
        >
          Reset
        </Button>
      </div>
      <div className="flex flex-wrap gap-1.5 mb-2">
        {CATEGORIES.map((cat) => {
          const isOn = categoryVisible(cat, categoryVis);
          return (
            <button
              key={cat.id}
              onClick={() => handleToggleCategory(cat.id)}
              className={cn(
                "px-2.5 py-1 rounded-full text-xs font-medium transition-colors",
                isOn
                  ? "bg-blue-600 text-white"
                  : "bg-bg-base text-text-dim border border-border-dim",
              )}
            >
              {cat.label}
            </button>
          );
        })}
      </div>

      {/* Layers */}
      <div className="text-xs text-text-dim mb-1">Layers (visibility)</div>
      <div className="max-h-72 overflow-auto pr-1">
        {tree.length === 0 ? (
          <div className="text-sm text-text-dim">No debug data</div>
        ) : (
          tree.map((node) => (
            <TreeNodeRow
              key={node.fullPath}
              node={node}
              layerVis={layerVis}
              categoryVis={categoryVis}
              allKeys={allKeys}
              onToggle={handleToggleNode}
              openPaths={openPaths}
              onOpenChange={handleOpenChange}
            />
          ))
        )}
      </div>
    </div>
  );
};

export default DebugVisibilityControls;
