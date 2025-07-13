import { useDebugData, usePrimaryTeam } from "@/api";
import { TeamColor } from "@/bindings";
import React, { useMemo } from "react";

interface BehaviorTreeViewProps {
  selectedPlayerId: number | null;
  className?: string;
}

interface TreeNodeData {
  name: string;
  id: string;
  children_ids: string[];
  is_active: boolean;
  node_type: string;
  internal_state?: string;
  additional_info?: string;
}

interface TreeNode extends TreeNodeData {
  children: TreeNode[];
  x: number;
  y: number;
}

const nodeWidth = 240;
const nodeHeight = 80;
const horizontalGap = 30;
const verticalGap = 100;

const layoutTree = (root: TreeNode) => {
  const allNodes: TreeNode[] = [];

  let leafCount = 0;
  const assignLeafPositions = (node: TreeNode, depth: number) => {
    node.y = depth * verticalGap;
    if (node.children.length === 0) {
      node.x = leafCount * (nodeWidth + horizontalGap);
      leafCount++;
    }
    for (const child of node.children) {
      assignLeafPositions(child, depth + 1);
    }
  };

  const assignInternalPositions = (node: TreeNode): void => {
    if (node.children.length > 0) {
      for (const child of node.children) {
        assignInternalPositions(child);
      }
      const firstChild = node.children[0];
      const lastChild = node.children[node.children.length - 1];
      node.x = (firstChild.x + lastChild.x) / 2;
    }
  };

  assignLeafPositions(root, 0);
  assignInternalPositions(root);

  const collectNodes = (node: TreeNode) => {
    allNodes.push(node);
    node.children.forEach(collectNodes);
  };
  collectNodes(root);

  let minX = Infinity;
  allNodes.forEach((node) => {
    minX = Math.min(minX, node.x);
  });

  if (isFinite(minX)) {
    allNodes.forEach((node) => {
      node.x -= minX;
    });
  }

  let maxX = -Infinity,
    maxY = -Infinity;
  allNodes.forEach((node) => {
    maxX = Math.max(maxX, node.x + nodeWidth);
    maxY = Math.max(maxY, node.y + nodeHeight);
  });

  return { nodes: allNodes, width: maxX, height: maxY };
};

const BehaviorTreeView: React.FC<BehaviorTreeViewProps> = ({
  selectedPlayerId,
  className,
}) => {
  const debugData = useDebugData();
  const [selectedTeam] = usePrimaryTeam();
  const primaryTeam = selectedTeam === TeamColor.Blue ? "Blue" : "Yellow";

  const renderedTree = useMemo(() => {
    if (selectedPlayerId === null || !debugData) {
      return null;
    }

    const prefix = `team_${primaryTeam}.p${selectedPlayerId}.bt`;
    const nodeDataMap: Record<string, TreeNodeData> = {};
    for (const [key, value] of Object.entries(debugData)) {
      if (
        key.startsWith(prefix) &&
        value.type === "Shape" &&
        value.data.type === "TreeNode"
      ) {
        const nodeData = value.data.data;
        nodeDataMap[nodeData.id] = nodeData;
      }
    }

    if (Object.keys(nodeDataMap).length === 0) {
      return null;
    }

    const treeNodes: Record<string, TreeNode> = {};
    for (const id in nodeDataMap) {
      const node = nodeDataMap[id];
      treeNodes[id] = {
        ...node,
        children: [],
        x: 0,
        y: 0,
      };
    }

    const roots: TreeNode[] = [];
    const allChildrenIds = new Set<string>();
    Object.values(nodeDataMap).forEach((node) => {
      node.children_ids.forEach((childId) => allChildrenIds.add(childId));
    });

    for (const id in treeNodes) {
      const node = treeNodes[id];
      node.children = node.children_ids
        .map((childId) => treeNodes[childId])
        .filter(Boolean);
      if (!allChildrenIds.has(id)) {
        roots.push(node);
      }
    }

    if (roots.length === 0 && Object.keys(treeNodes).length > 0) {
      let rootId = Object.keys(treeNodes).sort(
        (a, b) => a.length - b.length
      )[0];
      if (rootId) {
        roots.push(treeNodes[rootId]);
      }
    }

    if (roots.length === 0) {
      return null;
    }

    const { nodes, width, height } = layoutTree(roots[0]);

    return { nodes, width, height };
  }, [debugData, selectedPlayerId]);

  if (!renderedTree) {
    return (
      <div className="p-4 text-slate-400">
        No behavior tree data available for this player.
      </div>
    );
  }

  return (
    <div className={`${className} p-4 overflow-auto`}>
      <svg width={renderedTree.width} height={renderedTree.height}>
        {renderedTree.nodes.map((node) => (
          <g key={node.id}>
            {node.children.map((child) => (
              <line
                key={`${node.id}-${child.id}`}
                x1={node.x + nodeWidth / 2}
                y1={node.y + nodeHeight}
                x2={child.x + nodeWidth / 2}
                y2={child.y}
                stroke="gray"
              />
            ))}
          </g>
        ))}
        {renderedTree.nodes.map((node) => (
          <g key={node.id} transform={`translate(${node.x}, ${node.y})`}>
            <rect
              width={nodeWidth}
              height={nodeHeight}
              rx={8}
              fill={node.is_active ? "#10b981" : "#374151"}
              stroke={node.is_active ? "#6ee7b7" : "#4b5563"}
              strokeWidth={2}
            />

            {/* Node type badge */}
            <rect
              x={4}
              y={4}
              width={60}
              height={16}
              rx={8}
              fill={node.is_active ? "#065f46" : "#1f2937"}
            />
            <text
              x={34}
              y={15}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="white"
              style={{ fontSize: "10px", fontWeight: "bold" }}
            >
              {node.node_type}
            </text>

            {/* Main node name */}
            <text
              x={nodeWidth / 2}
              y={28}
              textAnchor="middle"
              dominantBaseline="middle"
              fill="white"
              style={{ fontSize: "12px", fontWeight: "bold" }}
            >
              <tspan>
                {node.name.length > 28
                  ? `${node.name.substring(0, 25)}...`
                  : node.name}
              </tspan>
            </text>

            {/* Internal state */}
            {node.internal_state && (
              <text
                x={nodeWidth / 2}
                y={45}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={node.is_active ? "#d1fae5" : "#d1d5db"}
                style={{ fontSize: "10px" }}
              >
                <tspan>
                  {node.internal_state.length > 32
                    ? `${node.internal_state.substring(0, 29)}...`
                    : node.internal_state}
                </tspan>
              </text>
            )}

            {/* Additional info */}
            {node.additional_info && (
              <text
                x={nodeWidth / 2}
                y={62}
                textAnchor="middle"
                dominantBaseline="middle"
                fill={node.is_active ? "#a7f3d0" : "#9ca3af"}
                style={{ fontSize: "9px" }}
              >
                <tspan>
                  {node.additional_info.length > 35
                    ? `${node.additional_info.substring(0, 32)}...`
                    : node.additional_info}
                </tspan>
              </text>
            )}
          </g>
        ))}
      </svg>
    </div>
  );
};

export default BehaviorTreeView;
