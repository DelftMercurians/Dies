import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { cn, prettyPrintSnakeCases } from "@/lib/utils";
import { ChevronDown, ChevronRight } from "lucide-react";
import { FC, useState, useMemo } from "react";

interface HierarchicalListProps {
  data: [string, any][];
  className?: string;
}

const HierarchicalList: FC<HierarchicalListProps> = ({ data, className }) => {
  const [openKeys, setOpenKeys] = useState<string[]>([]);

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
      return value.toFixed(2);
    }
    if (typeof value === "object") {
      if (Array.isArray(value) && value.every((v) => typeof v === "number")) {
        return `[${value.map((v: number) => v.toFixed(2)).join(", ")}]`;
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
      return value;
    }
    return JSON.stringify(value);
  };

  const renderGroup = (group: Record<string, any>, key = "", depth = 0) => {
    const isLeaf = typeof group.data !== "undefined";
    if (isLeaf) {
      return (
        <div key={key} className="flex flex-row items-stretch py-1 w-full">
          <div className="font-semibold mr-2 min-w-max">
            {prettyPrintSnakeCases(key)}:
          </div>
          <div className="w-full flex items-center overflow-x-auto">
            <span className="min-w-max font-mono whitespace-nowrap">
              {formatValue(group.data)}
            </span>
          </div>
        </div>
      );
    }

    const isOpen = openKeys.includes(key);
    const handleOpenChange = (isOpen: boolean) => {
      if (isOpen) {
        setOpenKeys((keys) => [...keys, key]);
      } else {
        setOpenKeys((keys) => keys.filter((k) => k !== key));
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
                renderGroup(subGroup, subKey, depth + 1),
              )}
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    );
  };

  return (
    <div className={cn("p-2", className)}>
      {sortKeys(groupedData).map(([key, group]) => renderGroup(group, key))}
    </div>
  );
};

export default HierarchicalList;

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
