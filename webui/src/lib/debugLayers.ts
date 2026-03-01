import { atomWithStorage } from "jotai/utils";
import { DebugMap } from "../bindings";

// ============================================================================
// Atoms
// ============================================================================

/** Maps tree node paths to visibility overrides. Absent = visible (default all on). */
export const debugLayerVisibilityAtom = atomWithStorage<Record<string, boolean>>(
  "dies-debug-layers",
  {},
);

/** Maps category IDs to visibility overrides. Absent = visible. */
export const debugCategoryVisibilityAtom = atomWithStorage<
  Record<string, boolean>
>("dies-debug-categories", {});

// ============================================================================
// Categories
// ============================================================================

export interface DebugCategory {
  id: string;
  label: string;
  pattern: RegExp;
}

export const CATEGORIES: DebugCategory[] = [
  { id: "roles", label: "Roles", pattern: /\.role$/ },
  { id: "bt", label: "Behavior Trees", pattern: /\.bt(\..*)?$/ },
  { id: "strategy", label: "Strategy", pattern: /\.strategy(\..*)?$/ },
];

// ============================================================================
// Visibility logic
// ============================================================================

/**
 * Check if a debug key is visible given layer and category visibility maps.
 * Both tree and categories act as "off switches" (AND semantics).
 */
export function isKeyVisible(
  key: string,
  layerVisibility: Record<string, boolean>,
  categoryVisibility: Record<string, boolean>,
): boolean {
  // Walk up dot-separated segments — if any ancestor is explicitly false, hidden
  const parts = key.split(".");
  for (let i = 1; i <= parts.length; i++) {
    const prefix = parts.slice(0, i).join(".");
    if (layerVisibility[prefix] === false) {
      return false;
    }
  }

  // Check categories — if any matching category is false, hidden
  for (const cat of CATEGORIES) {
    if (cat.pattern.test(key) && categoryVisibility[cat.id] === false) {
      return false;
    }
  }

  return true;
}

/**
 * Filter a DebugMap, removing entries whose keys are hidden.
 */
export function filterDebugMap(
  debugMap: DebugMap,
  layerVisibility: Record<string, boolean>,
  categoryVisibility: Record<string, boolean>,
): DebugMap {
  const result: DebugMap = {};
  for (const key of Object.keys(debugMap)) {
    if (isKeyVisible(key, layerVisibility, categoryVisibility)) {
      result[key] = debugMap[key];
    }
  }
  return result;
}

// ============================================================================
// Tree building
// ============================================================================

export interface TreeNode {
  name: string;
  fullPath: string;
  children: TreeNode[];
  leafCount: number;
}

/**
 * Build a tree structure from dot-separated debug keys.
 */
export function buildLayerTree(keys: string[]): TreeNode[] {
  const root: Record<string, any> = {};

  for (const key of keys) {
    const parts = key.split(".");
    let current = root;
    for (const part of parts) {
      if (!current[part]) {
        current[part] = {};
      }
      current = current[part];
    }
  }

  function buildNodes(
    obj: Record<string, any>,
    parentPath: string,
  ): TreeNode[] {
    return Object.keys(obj)
      .sort()
      .map((name) => {
        const fullPath = parentPath ? `${parentPath}.${name}` : name;
        const children = buildNodes(obj[name], fullPath);
        const leafCount =
          children.length === 0
            ? 1
            : children.reduce((sum, c) => sum + c.leafCount, 0);
        return { name, fullPath, children, leafCount };
      });
  }

  return buildNodes(root, "");
}
