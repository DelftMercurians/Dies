import React, { useState, useEffect, useRef } from "react";
import {
  Terminal,
  X,
  ChevronDown,
  ChevronUp,
  Trash2,
  AlertCircle,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScriptError } from "@/bindings";
import { cn } from "@/lib/utils";

interface ScriptConsoleEntry {
  id: string;
  timestamp: Date;
  error: ScriptError;
}

interface ScriptConsoleProps {
  className?: string;
}

export const ScriptConsole: React.FC<ScriptConsoleProps> = ({ className }) => {
  const [entries, setEntries] = useState<ScriptConsoleEntry[]>([]);
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [hasNewErrors, setHasNewErrors] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // This will be called by the parent component when new script errors arrive
  const addError = (error: ScriptError) => {
    const entry: ScriptConsoleEntry = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: new Date(),
      error,
    };

    setEntries((prev) => [...prev, entry]);
    setHasNewErrors(true);

    // Auto-scroll to bottom
    setTimeout(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    }, 100);
  };

  // Clear all entries
  const clearConsole = () => {
    setEntries([]);
    setHasNewErrors(false);
  };

  // When console is expanded, clear the "new errors" indicator
  useEffect(() => {
    if (!isCollapsed) {
      setHasNewErrors(false);
    }
  }, [isCollapsed]);

  // Expose addError method to parent
  useEffect(() => {
    // We'll handle this via props or context in the main app
  }, []);

  const runtimeErrors = entries.filter(
    (entry) => entry.error.type === "Runtime"
  );
  const errorCount = runtimeErrors.length;

  if (errorCount === 0) {
    return null; // Don't show console if no errors
  }

  return (
    <div className={cn("bg-slate-900 border-t border-slate-700", className)}>
      {/* Header */}
      <div
        className="flex items-center justify-between p-2 cursor-pointer hover:bg-slate-800"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-orange-400" />
          <span className="text-sm font-medium">Script Console</span>
          <span className="text-xs bg-red-600 text-white px-2 py-1 rounded">
            {errorCount} error{errorCount !== 1 ? "s" : ""}
          </span>
          {hasNewErrors && (
            <AlertCircle className="h-3 w-3 text-red-400 animate-pulse" />
          )}
        </div>

        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              clearConsole();
            }}
            className="h-6 w-6 p-0"
          >
            <Trash2 className="h-3 w-3" />
          </Button>
          {isCollapsed ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </div>
      </div>

      {/* Console Content */}
      {!isCollapsed && (
        <div
          ref={scrollRef}
          className="max-h-60 overflow-y-auto bg-black p-2 text-xs font-mono"
        >
          {runtimeErrors.map((entry) => (
            <div key={entry.id} className="mb-3 border-l-2 border-red-500 pl-3">
              <div className="text-gray-400 mb-1">
                [{entry.timestamp.toLocaleTimeString()}] Runtime Error
              </div>

              <div className="text-red-400 mb-1">
                📁{" "}
                {entry.error.type === "Runtime"
                  ? entry.error.data.script_path
                  : ""}
              </div>

              {entry.error.type === "Runtime" && (
                <>
                  <div className="text-yellow-400 mb-1">
                    🔧 Function: {entry.error.data.function_name}
                  </div>

                  <div className="text-blue-400 mb-1">
                    👥 Team: {entry.error.data.team_color}
                    {entry.error.data.player_id &&
                      ` | Player: ${entry.error.data.player_id}`}
                  </div>

                  <div className="text-white bg-red-900/30 p-2 rounded border border-red-500/30">
                    {entry.error.data.message}
                  </div>
                </>
              )}
            </div>
          ))}

          {errorCount === 0 && (
            <div className="text-gray-500 text-center py-4">
              No script errors to display
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Create a ref type to expose methods to parent component
export interface ScriptConsoleRef {
  addError: (error: ScriptError) => void;
  clearConsole: () => void;
}

// Forward ref version for imperative access
export const ScriptConsoleWithRef = React.forwardRef<
  ScriptConsoleRef,
  ScriptConsoleProps
>(({ className }, ref) => {
  const [entries, setEntries] = useState<ScriptConsoleEntry[]>([]);
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [hasNewErrors, setHasNewErrors] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const addError = (error: ScriptError) => {
    const entry: ScriptConsoleEntry = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: new Date(),
      error,
    };

    setEntries((prev) => [...prev, entry]);
    setHasNewErrors(true);

    // Auto-scroll to bottom
    setTimeout(() => {
      if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      }
    }, 100);
  };

  const clearConsole = () => {
    setEntries([]);
    setHasNewErrors(false);
  };

  // Expose methods to parent via ref
  React.useImperativeHandle(ref, () => ({
    addError,
    clearConsole,
  }));

  // When console is expanded, clear the "new errors" indicator
  useEffect(() => {
    if (!isCollapsed) {
      setHasNewErrors(false);
    }
  }, [isCollapsed]);

  const runtimeErrors = entries.filter(
    (entry) => entry.error.type === "Runtime"
  );
  const errorCount = runtimeErrors.length;

  if (errorCount === 0) {
    return null; // Don't show console if no errors
  }

  return (
    <div className={cn("bg-slate-900 border-t border-slate-700", className)}>
      {/* Header */}
      <div
        className="flex items-center justify-between p-2 cursor-pointer hover:bg-slate-800"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-orange-400" />
          <span className="text-sm font-medium">Script Console</span>
          <span className="text-xs bg-red-600 text-white px-2 py-1 rounded">
            {errorCount} error{errorCount !== 1 ? "s" : ""}
          </span>
          {hasNewErrors && (
            <AlertCircle className="h-3 w-3 text-red-400 animate-pulse" />
          )}
        </div>

        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={(e) => {
              e.stopPropagation();
              clearConsole();
            }}
            className="h-6 w-6 p-0"
          >
            <Trash2 className="h-3 w-3" />
          </Button>
          {isCollapsed ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </div>
      </div>

      {/* Console Content */}
      {!isCollapsed && (
        <div
          ref={scrollRef}
          className="max-h-60 overflow-y-auto bg-black p-2 text-xs font-mono"
        >
          {runtimeErrors.map((entry) => (
            <div key={entry.id} className="mb-3 border-l-2 border-red-500 pl-3">
              <div className="text-gray-400 mb-1">
                [{entry.timestamp.toLocaleTimeString()}] Runtime Error
              </div>

              <div className="text-red-400 mb-1">
                📁{" "}
                {entry.error.type === "Runtime"
                  ? entry.error.data.script_path
                  : ""}
              </div>

              {entry.error.type === "Runtime" && (
                <>
                  <div className="text-yellow-400 mb-1">
                    🔧 Function: {entry.error.data.function_name}
                  </div>

                  <div className="text-blue-400 mb-1">
                    👥 Team: {entry.error.data.team_color}
                    {entry.error.data.player_id &&
                      ` | Player: ${entry.error.data.player_id}`}
                  </div>

                  <div className="text-white bg-red-900/30 p-2 rounded border border-red-500/30">
                    {entry.error.data.message}
                  </div>
                </>
              )}
            </div>
          ))}

          {errorCount === 0 && (
            <div className="text-gray-500 text-center py-4">
              No script errors to display
            </div>
          )}
        </div>
      )}
    </div>
  );
});
