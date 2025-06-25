import React, { useState, useEffect } from "react";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ChevronRight, Folder, File } from "lucide-react";

// Unix path utilities
const pathUtils = {
  /**
   * Normalize a Unix path by resolving . and .. components
   */
  normalize: (path: string): string => {
    if (!path || path === ".") return ".";

    // Split path into components, filter out empty strings
    const parts = path.split("/").filter((part) => part !== "");
    const resolved: string[] = [];

    for (const part of parts) {
      if (part === "." || part === "") {
        // Skip current directory references and empty parts
        continue;
      } else if (part === "..") {
        // Go up one directory if possible
        if (resolved.length > 0 && resolved[resolved.length - 1] !== "..") {
          resolved.pop();
        } else if (!path.startsWith("/")) {
          // Only add .. if we're not at an absolute path root
          resolved.push("..");
        }
      } else {
        resolved.push(part);
      }
    }

    // Handle absolute vs relative paths
    if (path.startsWith("/")) {
      return "/" + resolved.join("/");
    }

    return resolved.length === 0 ? "." : resolved.join("/");
  },

  /**
   * Get the parent directory of a given path
   */
  dirname: (path: string): string => {
    if (!path || path === "." || path === "/") return ".";

    const normalized = pathUtils.normalize(path);
    if (normalized === "." || normalized === "/") return ".";

    const lastSlash = normalized.lastIndexOf("/");
    if (lastSlash === -1) {
      // No slash found, we're in current directory
      return ".";
    }

    if (lastSlash === 0) {
      // Root directory
      return "/";
    }

    return normalized.substring(0, lastSlash) || ".";
  },

  /**
   * Join path components together
   */
  join: (...parts: string[]): string => {
    if (parts.length === 0) return ".";

    const joined = parts
      .filter((part) => part && part !== ".")
      .join("/")
      .replace(/\/+/g, "/"); // Remove multiple consecutive slashes

    return pathUtils.normalize(joined || ".");
  },

  /**
   * Check if a path is at the root level (can't go up further)
   */
  isAtRoot: (path: string): boolean => {
    const normalized = pathUtils.normalize(path);
    return normalized === "/";
  },
};

interface FileEntry {
  name: string;
  is_dir: boolean;
}

interface FilePathSelectorProps {
  value: string;
  onChange: (path: string | null) => void;
  startDir?: string;
  placeholder?: string;
}

export const FilePathSelector: React.FC<FilePathSelectorProps> = ({
  value,
  onChange,
  startDir = ".",
  placeholder = "Select file...",
}) => {
  const [open, setOpen] = useState(false);
  const [currentDir, setCurrentDir] = useState(startDir);
  const [entries, setEntries] = useState<FileEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<string[]>([]);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(`/api/list?dir=${encodeURIComponent(currentDir)}`)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to list directory");
        return res.json();
      })
      .then((data) => {
        setEntries(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [currentDir]);

  const handleDirClick = (name: string) => {
    const nextDir = pathUtils.join(currentDir, name);
    setHistory((h) => [...h, currentDir]);
    setCurrentDir(nextDir);
  };

  const handleBack = () => {
    if (history.length > 0) {
      const prev = history[history.length - 1];
      setCurrentDir(prev);
      setHistory((h) => h.slice(0, -1));
    }
  };

  const handleUp = () => {
    const parentDir = pathUtils.dirname(currentDir);
    if (parentDir !== currentDir) {
      setHistory((h) => [...h, currentDir]);
      setCurrentDir(parentDir);
    }
  };

  const handleFileClick = (name: string) => {
    const path = pathUtils.join(currentDir, name);
    onChange(path);
    setOpen(false);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button variant="outline" className="w-full justify-between">
          <span className="truncate">{value || placeholder}</span>
          <ChevronRight className="ml-2 h-4 w-4" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="p-0 w-80">
        <div className="p-2 border-b flex items-center gap-2">
          <Button
            size="sm"
            variant="ghost"
            onClick={handleBack}
            disabled={history.length === 0}
            title="Go back"
          >
            ←
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={handleUp}
            disabled={pathUtils.isAtRoot(currentDir)}
            title="Go up one directory"
          >
            ↑
          </Button>
          <span className="text-xs text-muted-foreground truncate flex-1">
            {currentDir}
          </span>

          <Button
            size="sm"
            variant="ghost"
            onClick={() => onChange(null)}
            title="Clear"
          >
            ×
          </Button>
        </div>
        <Separator />
        <div className="max-h-64 overflow-y-auto">
          {loading && <div className="p-4 text-center text-xs">Loading...</div>}
          {error && (
            <div className="p-4 text-center text-xs text-red-500">{error}</div>
          )}
          {!loading && !error && entries.length === 0 && (
            <div className="p-4 text-center text-xs text-muted-foreground">
              No files or directories
            </div>
          )}
          <ul>
            {entries
              .slice()
              .sort((a, b) => {
                if (a.is_dir !== b.is_dir) {
                  return a.is_dir ? -1 : 1; // Dirs first
                }
                return a.name.localeCompare(b.name); // Alphabetically
              })
              .map((entry) => (
                <li key={entry.name}>
                  <Button
                    variant="ghost"
                    className="w-full flex items-center justify-start gap-2 px-2 py-1 text-left"
                    onClick={() =>
                      entry.is_dir
                        ? handleDirClick(entry.name)
                        : handleFileClick(entry.name)
                    }
                  >
                    {entry.is_dir ? (
                      <Folder className="h-4 w-4 text-blue-500" />
                    ) : (
                      <File className="h-4 w-4" />
                    )}
                    <span className="truncate">{entry.name}</span>
                  </Button>
                </li>
              ))}
          </ul>
        </div>
      </PopoverContent>
    </Popover>
  );
};

export default FilePathSelector;
