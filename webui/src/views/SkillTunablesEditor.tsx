import React from "react";
import {
  ChevronRight,
  RotateCcw,
  Search,
  X,
  ChevronsDownUp,
  ChevronsUpDown,
} from "lucide-react";
import { useAtom } from "jotai";
import { atomWithStorage } from "jotai/utils";
import { Input } from "@/components/ui/input";
import { NumberInput } from "@/components/ui/number-input";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { cn, prettyPrintSnakeCases } from "@/lib/utils";
import { useExecutorInfo, useExecutorSettings } from "@/api";
import { ExecutorSettings, TunableSpec } from "@/bindings";

// UI-only view state, persisted to localStorage so iterating quickly (edit →
// reload) keeps the same tree expansion, filter and scroll position.
const expandedAtom = atomWithStorage<string[]>(
  "dies-skill-tunables-expanded",
  []
);
const searchAtom = atomWithStorage<string>("dies-skill-tunables-search", "");
const SCROLL_KEY = "dies-skill-tunables-scroll";

const INDENT = 12; // px per tree depth

// ---- Tree model -----------------------------------------------------------

interface TreeNode {
  name: string; // this segment
  path: string; // full dotted path
  children: Map<string, TreeNode>;
  spec?: TunableSpec; // set on leaf nodes
}

/** Build a tree from the dotted `spec.key` (`<module>.<...>.<NAME>`). */
function buildTree(specs: TunableSpec[]): TreeNode {
  const root: TreeNode = { name: "", path: "", children: new Map() };
  for (const spec of specs) {
    const parts = spec.key.split(".");
    let node = root;
    parts.forEach((part, i) => {
      const path = i === 0 ? part : `${node.path}.${part}`;
      let child = node.children.get(part);
      if (!child) {
        child = { name: part, path, children: new Map() };
        node.children.set(part, child);
      }
      node = child;
      if (i === parts.length - 1) node.spec = spec;
    });
  }
  return root;
}

/** Groups first, then alphabetical — stable ordering for the tree. */
function sortedChildren(node: TreeNode): TreeNode[] {
  return [...node.children.values()].sort((a, b) => {
    const aLeaf = a.children.size === 0;
    const bLeaf = b.children.size === 0;
    if (aLeaf !== bLeaf) return aLeaf ? 1 : -1;
    return a.name.localeCompare(b.name);
  });
}

function collectGroupPaths(node: TreeNode, out: string[] = []): string[] {
  for (const child of node.children.values()) {
    if (child.children.size > 0) {
      out.push(child.path);
      collectGroupPaths(child, out);
    }
  }
  return out;
}

function matchesQuery(spec: TunableSpec, q: string): boolean {
  const hay = `${spec.label} ${spec.key} ${spec.help ?? ""} ${
    spec.section ?? ""
  }`.toLowerCase();
  return hay.includes(q);
}

/** Count overridden leaves in a subtree (for the "N changed" badge). */
function countChanged(node: TreeNode, overrides: Record<string, number>): number {
  if (node.spec) return node.spec.key in overrides ? 1 : 0;
  let n = 0;
  for (const child of node.children.values())
    n += countChanged(child, overrides);
  return n;
}

// ---- Leaf row -------------------------------------------------------------

const TunableRow: React.FC<{
  spec: TunableSpec;
  depth: number;
  value: number;
  overridden: boolean;
  onChange: (value: number) => void;
  onReset: () => void;
}> = ({ spec, depth, value, overridden, onChange, onReset }) => {
  const name = (
    <span
      className={cn(
        "truncate text-xs flex-1 min-w-0",
        overridden ? "text-text-base" : "text-text-std"
      )}
    >
      {spec.label}
    </span>
  );
  return (
    <div
      className="flex items-center gap-1.5 pr-2 h-6 hover:bg-bg-muted/40 group/row"
      style={{ paddingLeft: depth * INDENT + 8 }}
    >
      <span
        className={cn(
          "w-1 h-1 rounded-full shrink-0",
          overridden ? "bg-accent-cyan" : "bg-transparent"
        )}
        title={overridden ? "overridden (differs from code default)" : undefined}
      />
      {spec.help ? (
        <SimpleTooltip
          title={spec.help}
          className="min-w-0 flex-1 cursor-help flex"
        >
          {name}
        </SimpleTooltip>
      ) : (
        name
      )}
      <NumberInput
        id={spec.key}
        value={value}
        onChange={onChange}
        inputSize="sm"
        className="w-16 text-right font-mono shrink-0"
      />
      <span className="text-[10px] text-text-muted w-7 shrink-0 truncate">
        {spec.unit ?? ""}
      </span>
      <button
        type="button"
        onClick={onReset}
        disabled={!overridden}
        title={`Reset to default (${spec.default})`}
        className={cn(
          "p-0.5 shrink-0 text-text-muted hover:text-text-base",
          "disabled:opacity-0 disabled:pointer-events-none",
          "opacity-0 group-hover/row:opacity-100"
        )}
      >
        <RotateCcw className="w-3 h-3" />
      </button>
    </div>
  );
};

// ---- Group header ---------------------------------------------------------

const GroupHeader: React.FC<{
  node: TreeNode;
  depth: number;
  open: boolean;
  changed: number;
  onToggle: () => void;
}> = ({ node, depth, open, changed, onToggle }) => (
  <button
    type="button"
    onClick={onToggle}
    className="flex items-center gap-1 w-full h-6 pr-2 hover:bg-bg-muted text-left"
    style={{ paddingLeft: depth * INDENT + 4 }}
  >
    <ChevronRight
      className={cn(
        "w-3.5 h-3.5 shrink-0 text-text-muted transition-transform",
        open && "rotate-90"
      )}
    />
    <span
      className={cn(
        "truncate text-xs font-medium",
        depth === 0
          ? "uppercase tracking-wide text-text-muted"
          : "text-text-std"
      )}
    >
      {prettyPrintSnakeCases(node.name)}
    </span>
    {changed > 0 && (
      <span className="ml-auto text-[10px] text-accent-cyan tabular-nums">
        {changed}
      </span>
    )}
  </button>
);

// ---- Editor ---------------------------------------------------------------

/**
 * Skill tunables — runtime knobs declared in the executor's skills with the
 * `tunables!` macro. Metadata (label/help/range/unit/section) is auto-discovered
 * from `ExecutorInfo.skill_tunable_specs`; the current overrides live in
 * `ExecutorSettings.skill_tunables` and are written through the normal settings
 * POST, so they persist and ride the baseline/revert bar like every other setting.
 *
 * The panel renders the knobs as a dense, searchable, collapsible tree keyed on
 * the dotted `spec.key`. Tree expansion, filter and scroll position are persisted
 * to localStorage so the view survives reloads while iterating.
 */
const SkillTunablesEditor: React.FC = () => {
  const info = useExecutorInfo();
  const { settings, updateSettings } = useExecutorSettings();

  const [expanded, setExpanded] = useAtom(expandedAtom);
  const [search, setSearch] = useAtom(searchAtom);

  const scrollRef = React.useRef<HTMLDivElement>(null);
  const restored = React.useRef(false);
  const scrollTimer = React.useRef<number | undefined>(undefined);

  const specs = info?.skill_tunable_specs ?? [];

  const tree = React.useMemo(() => buildTree(specs), [specs]);
  const expandedSet = React.useMemo(() => new Set(expanded), [expanded]);
  const q = search.trim().toLowerCase();
  const searching = q.length > 0;

  // Which leaf keys match the filter (empty search → all match).
  const matched = React.useMemo(() => {
    if (!q) return null;
    const set = new Set<string>();
    for (const spec of specs) if (matchesQuery(spec, q)) set.add(spec.key);
    return set;
  }, [specs, q]);

  // Restore scroll once, after the specs have loaded and the tree is on screen.
  React.useEffect(() => {
    if (restored.current || !scrollRef.current || specs.length === 0) return;
    const saved = Number(localStorage.getItem(SCROLL_KEY) || "0");
    if (saved > 0) scrollRef.current.scrollTop = saved;
    restored.current = true;
  }, [specs.length]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el || scrollTimer.current) return;
    scrollTimer.current = window.setTimeout(() => {
      scrollTimer.current = undefined;
      if (scrollRef.current)
        localStorage.setItem(SCROLL_KEY, String(scrollRef.current.scrollTop));
    }, 150);
  };

  if (!settings) {
    return <div className="p-4 text-sm text-text-muted">Loading…</div>;
  }
  if (specs.length === 0) {
    return (
      <div className="p-4 text-sm text-text-muted">
        No skill tunables declared (is an executor running?).
      </div>
    );
  }

  const overrides = settings.skill_tunables ?? {};
  const setOverrides = (next: Record<string, number>) =>
    updateSettings({
      ...(settings as ExecutorSettings),
      skill_tunables: next,
    });

  const toggle = (path: string) =>
    setExpanded(
      expandedSet.has(path)
        ? expanded.filter((p) => p !== path)
        : [...expanded, path]
    );

  const totalChanged = Object.keys(overrides).length;

  // Whether a subtree contains any filter match (drives visibility + auto-open).
  const nodeVisible = (node: TreeNode): boolean => {
    if (!matched) return true;
    if (node.spec) return matched.has(node.spec.key);
    for (const child of node.children.values())
      if (nodeVisible(child)) return true;
    return false;
  };

  const renderChildren = (node: TreeNode, depth: number): React.ReactNode =>
    sortedChildren(node).map((child) => {
      if (!nodeVisible(child)) return null;

      if (child.spec) {
        const overridden = child.spec.key in overrides;
        const value = overridden ? overrides[child.spec.key] : child.spec.default;
        return (
          <TunableRow
            key={child.path}
            spec={child.spec}
            depth={depth}
            value={value}
            overridden={overridden}
            onChange={(v) =>
              setOverrides({ ...overrides, [child.spec!.key]: v })
            }
            onReset={() => {
              const next = { ...overrides };
              delete next[child.spec!.key];
              setOverrides(next);
            }}
          />
        );
      }

      const open = searching || expandedSet.has(child.path);
      return (
        <div key={child.path}>
          <GroupHeader
            node={child}
            depth={depth}
            open={open}
            changed={countChanged(child, overrides)}
            onToggle={() => toggle(child.path)}
          />
          {open && renderChildren(child, depth + 1)}
        </div>
      );
    });

  const noMatches = searching && matched!.size === 0;

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar: search + expand/collapse all + changed count */}
      <div className="flex items-center gap-1 px-2 py-1 border-b border-bg-muted shrink-0">
        <div className="relative flex-1 min-w-0">
          <Search className="absolute left-1.5 top-1/2 -translate-y-1/2 w-3 h-3 text-text-muted pointer-events-none" />
          <Input
            inputSize="sm"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter knobs…"
            className="pl-6 pr-6 h-6"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch("")}
              title="Clear filter"
              className="absolute right-1 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-base"
            >
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
        <button
          type="button"
          onClick={() => setExpanded(collectGroupPaths(tree))}
          title="Expand all"
          className="p-1 text-text-muted hover:text-text-base shrink-0"
        >
          <ChevronsUpDown className="w-3.5 h-3.5" />
        </button>
        <button
          type="button"
          onClick={() => setExpanded([])}
          title="Collapse all"
          className="p-1 text-text-muted hover:text-text-base shrink-0"
        >
          <ChevronsDownUp className="w-3.5 h-3.5" />
        </button>
        <span
          className="text-[10px] text-text-muted tabular-nums w-14 text-right shrink-0"
          title="knobs overridden from their code default"
        >
          {totalChanged > 0 ? `${totalChanged} changed` : `${specs.length}`}
        </span>
      </div>

      <div ref={scrollRef} onScroll={onScroll} className="flex-1 overflow-auto">
        {noMatches ? (
          <div className="p-4 text-xs text-text-muted">
            No knobs match “{search}”.
          </div>
        ) : (
          renderChildren(tree, 0)
        )}
      </div>
    </div>
  );
};

export default SkillTunablesEditor;
