import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { cn, formatDebugString, formatNumber, prettyPrintSnakeCases } from "@/lib/utils";
import { ChevronDown, ChevronRight } from "lucide-react";
import { FC, useState, useMemo } from "react";

interface HierarchicalListProps {
  data: [string, any][];
  className?: string;
  expandAll?: boolean;
}

const HierarchicalList: FC<HierarchicalListProps> = ({
  data,
  className,
  expandAll = false,
}) => {
  const [openKeys, setOpenKeys] = useState<string[]>(
    expandAll ? data.map(([key]) => key) : []
  );

  const groupedData = useMemo(() => {
    const grouped: Record<string, any> = {};
    data.forEach(([key, value]) => {
      const parts = key.split(".");
      let current = grouped;
      parts.forEach((part, index) => {
        if (!current[part]) {
          current[part] = index === parts.length - 1 ? value : {};
        }
        current = current[part];
      });
    });
    return grouped;
  }, [data]);

  const formatValue = (value: any): string => {
    if (typeof value === "number") {
      return formatNumber(value);
    }
    if (typeof value === "object") {
      if (Array.isArray(value) && value.every((v) => typeof v === "number")) {
        // 2-number arrays are treated as mm positions/vectors -> whole numbers.
        const isPosition = value.length === 2;
        return `[${value
          .map((v: number) => formatNumber(v, isPosition))
          .join(", ")}]`;
      }
      if (value === null) {
        return "null";
      }
      if (
        "type" in value &&
        "data" in value &&
        ["Line", "Circle", "Cross"].includes(value.type)
      ) {
        return `${value.type}(${formatValue(value.data).slice(1, -1)})`;
      }

      // recursively format object
      return `{${Object.entries(value)
        .map(([k, v]) => `${prettyPrintSnakeCases(k)}: ${formatValue(v)}`)
        .join(", ")}}`;
    }
    if (typeof value === "string") {
      return formatDebugString(value);
    }
    return JSON.stringify(value);
  };

  const renderGroup = (
    group: Record<string, any>,
    key = "",
    fullKey = "",
    depth = 0
  ) => {
    const isLeaf = typeof group.data !== "undefined";
    if (isLeaf) {
      return (
        <div key={key} className="flex flex-row items-stretch py-1 w-full">
          <div className="font-semibold mr-2 min-w-max">
            {prettyPrintSnakeCases(key)}:
          </div>
          <div className="w-full flex items-center overflow-x-auto">
            <span
              className="min-w-max font-mono whitespace-nowrap"
              title={rawTitle(group.data)}
            >
              {formatValue(group.data)}
            </span>
          </div>
        </div>
      );
    }

    const isOpen = openKeys.includes(fullKey);
    const handleOpenChange = (isOpen: boolean) => {
      if (isOpen) {
        setOpenKeys((keys) => [...keys, fullKey]);
      } else {
        setOpenKeys((keys) => keys.filter((k) => k !== fullKey));
      }
    };

    return (
      <div key={key} className="flex flex-col">
        <Collapsible
          open={isOpen}
          onOpenChange={handleOpenChange}
          className="w-full"
        >
          <CollapsibleTrigger className="flex items-center py-1 w-full">
            {isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
            <span className="font-semibold ml-1">
              {prettyPrintSnakeCases(key)}
            </span>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="ml-4 relative">
              <div className="w-4 h-full border-l border-gray-300 absolute -left-4"></div>
              {sortKeys(group).map(([subKey, subGroup]) =>
                renderGroup(subGroup, subKey, `${fullKey}.${subKey}`, depth + 1)
              )}
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    );
  };

  return (
    <div className={cn("p-2", className)}>
      {sortKeys(groupedData).map(([key, group]) =>
        renderGroup(group, key, key)
      )}
    </div>
  );
};

export default HierarchicalList;

/** Full-precision representation of a leaf value, shown on hover. */
const rawTitle = (value: any): string => {
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) return `[${value.join(", ")}]`;
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
};

const sortKeys = (group: Record<string, any>): [string, any][] => {
  return Object.entries(group).sort(([keyA, valueA], [keyB, valueB]) => {
    const isLeafA = typeof valueA.data !== "undefined";
    const isLeafB = typeof valueB.data !== "undefined";
    if (isLeafA !== isLeafB) {
      return isLeafA ? 1 : -1; // Non-leaf nodes first
    }
    return keyA.localeCompare(keyB); // Alphabetical order
  });
};
