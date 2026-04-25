import React, { useState } from "react";
import { Plus } from "lucide-react";
import { DockviewApi } from "dockview";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { ALL_PANELS, PanelId } from "@/components/panels";

interface AddPanelMenuProps {
  getApi: () => DockviewApi | null;
}

export const openOrFocusPanel = (
  api: DockviewApi,
  id: PanelId,
  title: string
) => {
  const existing = api.getPanel(id);
  if (existing) {
    existing.api.setActive();
    return;
  }
  api.addPanel({
    id,
    component: id,
    title,
    floating: true,
  });
};

const AddPanelMenu: React.FC<AddPanelMenuProps> = ({ getApi }) => {
  const [open, setOpen] = useState(false);
  const api = getApi();

  // Recompute on every open so the list reflects current layout state
  const closedPanels = open
    ? ALL_PANELS.filter(({ id }) => !api?.getPanel(id))
    : [];

  const handleAdd = (id: PanelId, title: string) => {
    if (!api) return;
    openOrFocusPanel(api, id, title);
    setOpen(false);
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <SimpleTooltip title="Add Panel">
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            size="icon-sm"
            className="text-text-dim hover:text-text-std"
          >
            <Plus className="w-4 h-4" />
          </Button>
        </PopoverTrigger>
      </SimpleTooltip>
      <PopoverContent align="end" className="w-48 p-1">
        {closedPanels.length === 0 ? (
          <p className="px-2 py-1.5 text-sm text-text-dim">
            All panels are open.
          </p>
        ) : (
          <div className="flex flex-col">
            {closedPanels.map(({ id, title }) => (
              <button
                key={id}
                onClick={() => handleAdd(id, title)}
                className="text-left px-2 py-1.5 text-sm font-medium uppercase tracking-wider text-text-std hover:bg-bg-muted"
              >
                {title}
              </button>
            ))}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
};

export default AddPanelMenu;
