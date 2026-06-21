import { DockviewApi, DockviewGroupPanel } from "dockview";
import { PANEL_IDS } from "@/components/panels";

/**
 * Collapsible bottom drawer, implemented on top of Dockview by driving the
 * height of the drawer's groups to 0 and restoring it. The drawer is whichever
 * groups currently host the console + migrated-sidebar panels, so it keeps
 * working even if the user rearranges them.
 */

/** Height (px) below which the drawer counts as collapsed. */
const COLLAPSED_THRESHOLD = 24;
/** Min height the drawer snaps to when expanded with no remembered size. */
const DEFAULT_FRACTION = 0.3;
const MIN_EXPANDED = 80;

// Last expanded height, remembered across toggles (module scope survives
// re-renders; there is only ever one drawer).
let savedHeight = 0;

/** Unique groups that make up the bottom drawer. */
function drawerGroups(api: DockviewApi): DockviewGroupPanel[] {
  const candidates = [
    api.getPanel(PANEL_IDS.CONSOLE)?.group,
    api.getPanel(PANEL_IDS.GAME_CONTROLLER)?.group,
  ];
  const seen = new Set<string>();
  const out: DockviewGroupPanel[] = [];
  for (const g of candidates) {
    if (g && !seen.has(g.id)) {
      seen.add(g.id);
      out.push(g);
    }
  }
  return out;
}

export function isDrawerCollapsed(api: DockviewApi): boolean {
  const groups = drawerGroups(api);
  if (groups.length === 0) return false;
  return groups[0].api.height < COLLAPSED_THRESHOLD;
}

/** Collapse the drawer to 0px (button/Cmd+J) or restore its last height. */
export function toggleDrawer(api: DockviewApi): void {
  const groups = drawerGroups(api);
  if (groups.length === 0) return;

  if (isDrawerCollapsed(api)) {
    const target =
      savedHeight > MIN_EXPANDED
        ? savedHeight
        : Math.floor(api.height * DEFAULT_FRACTION);
    for (const g of groups) {
      g.api.setConstraints({ minimumHeight: MIN_EXPANDED });
      g.api.setSize({ height: target });
    }
  } else {
    savedHeight = groups[0].api.height;
    for (const g of groups) {
      // Allow a true 0-height collapse (default min would resist it).
      g.api.setConstraints({ minimumHeight: 0 });
      g.api.setSize({ height: 0 });
    }
  }
}
