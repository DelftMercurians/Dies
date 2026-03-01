import React, { useState } from "react";
import { Save, Trash2, RotateCcw, Layout } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { useAtom } from "jotai";
import {
  savedLayoutsAtom,
  currentLayoutNameAtom,
} from "@/components/DockviewWrapper";

/**
 * Layout selector dropdown with layout management.
 *
 * Features:
 * - Switch between saved layouts
 * - Save current layout with custom name
 * - Delete saved layouts
 * - Reset to default layout
 */

interface LayoutSelectorProps {
  onResetToDefault?: () => void;
}

const LayoutSelector: React.FC<LayoutSelectorProps> = ({ onResetToDefault }) => {
  const [savedLayouts, setSavedLayouts] = useAtom(savedLayoutsAtom);
  const [currentLayoutName, setCurrentLayoutName] = useAtom(
    currentLayoutNameAtom
  );
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showManageDialog, setShowManageDialog] = useState(false);
  const [newLayoutName, setNewLayoutName] = useState("");

  const layoutNames = Object.keys(savedLayouts);

  // Predefined layouts that can be selected
  const availableLayouts = [
    { value: "default", label: "Default", available: true },
    { value: "debugging", label: "Debugging", available: layoutNames.includes("debugging") },
    { value: "match", label: "Match", available: layoutNames.includes("match") },
    { value: "development", label: "Development", available: layoutNames.includes("development") },
  ];

  // Custom layouts (user-created)
  const customLayouts = layoutNames.filter(
    (name) => !["default", "debugging", "match", "development"].includes(name)
  );

  const handleLayoutChange = (value: string) => {
    if (value === "__save__") {
      setShowSaveDialog(true);
    } else if (value === "__manage__") {
      setShowManageDialog(true);
    } else if (value === "__reset__") {
      onResetToDefault?.();
    } else {
      setCurrentLayoutName(value);
      // Reload the page to apply the new layout
      window.location.reload();
    }
  };

  const handleSaveLayout = () => {
    if (!newLayoutName.trim()) return;

    // The layout is saved automatically via the onDidLayoutChange listener
    // We just need to update the current layout name
    setCurrentLayoutName(newLayoutName.trim());
    setShowSaveDialog(false);
    setNewLayoutName("");
  };

  const handleDeleteLayout = (name: string) => {
    if (name === "default") return;

    setSavedLayouts((prev) => {
      const { [name]: _, ...rest } = prev;
      return rest;
    });

    if (currentLayoutName === name) {
      setCurrentLayoutName("default");
    }
  };

  return (
    <>
      <Select value={currentLayoutName} onValueChange={handleLayoutChange}>
        <SelectTrigger className="h-5 w-[100px] text-sm font-medium uppercase tracking-wider border-border-muted bg-transparent">
          <SelectValue placeholder="Layout" />
        </SelectTrigger>
        <SelectContent>
          {/* Predefined layouts */}
          {availableLayouts.map((layout) => (
            <SelectItem
              key={layout.value}
              value={layout.value}
              disabled={!layout.available && layout.value !== "default"}
            >
              {layout.label}
            </SelectItem>
          ))}

          {/* Custom layouts */}
          {customLayouts.length > 0 && (
            <>
              <Separator className="my-1" />
              {customLayouts.map((name) => (
                <SelectItem key={name} value={name}>
                  {name}
                </SelectItem>
              ))}
            </>
          )}

          <Separator className="my-1" />

          {/* Actions */}
          <SelectItem value="__save__">
            <div className="flex items-center gap-2">
              <Save className="h-3 w-3" />
              Save As...
            </div>
          </SelectItem>
          <SelectItem value="__manage__">
            <div className="flex items-center gap-2">
              <Layout className="h-3 w-3" />
              Manage Layouts
            </div>
          </SelectItem>
          <SelectItem value="__reset__">
            <div className="flex items-center gap-2">
              <RotateCcw className="h-3 w-3" />
              Reset to Default
            </div>
          </SelectItem>
        </SelectContent>
      </Select>

      {/* Save Layout Dialog */}
      <Dialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
        <DialogContent className="sm:max-w-[400px]">
          <DialogHeader>
            <DialogTitle>Save Layout</DialogTitle>
            <DialogDescription>
              Save the current panel arrangement as a named layout.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid gap-2">
              <Label htmlFor="layout-name">Layout Name</Label>
              <Input
                id="layout-name"
                value={newLayoutName}
                onChange={(e) => setNewLayoutName(e.target.value)}
                placeholder="My Custom Layout"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleSaveLayout();
                  }
                }}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowSaveDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleSaveLayout} disabled={!newLayoutName.trim()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Manage Layouts Dialog */}
      <Dialog open={showManageDialog} onOpenChange={setShowManageDialog}>
        <DialogContent className="sm:max-w-[500px]">
          <DialogHeader>
            <DialogTitle>Manage Layouts</DialogTitle>
            <DialogDescription>
              View and delete saved layouts.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-2 py-4 max-h-[300px] overflow-y-auto">
            {layoutNames.length === 0 ? (
              <p className="text-text-dim text-center py-4">
                No saved layouts yet.
              </p>
            ) : (
              layoutNames.map((name) => (
                <div
                  key={name}
                  className="flex items-center justify-between p-2 bg-bg-elevated border border-border-subtle"
                >
                  <div className="flex items-center gap-2">
                    <Layout className="h-4 w-4 text-text-dim" />
                    <span className="font-medium uppercase tracking-wider">
                      {name}
                    </span>
                    {name === currentLayoutName && (
                      <span className="text-sm text-accent-cyan">(active)</span>
                    )}
                  </div>
                  {name !== "default" && (
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      onClick={() => handleDeleteLayout(name)}
                      className="text-text-dim hover:text-accent-red"
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  )}
                </div>
              ))
            )}
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowManageDialog(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default LayoutSelector;
