import React, {
  useRef,
  useEffect,
  useCallback,
  forwardRef,
  useImperativeHandle,
} from "react";
import {
  DockviewReact,
  DockviewApi,
  DockviewReadyEvent,
  IDockviewPanelProps,
} from "dockview";
import "dockview/dist/styles/dockview.css";
import { useAtom } from "jotai";
import { atomWithStorage } from "jotai/utils";

// ============================================================================
// Types
// ============================================================================

export interface PanelConfig {
  id: string;
  component: string;
  title: string;
  params?: Record<string, unknown>;
}

export interface LayoutConfig {
  name: string;
  data: object;
}

// Atom for storing layouts with localStorage persistence
export const savedLayoutsAtom = atomWithStorage<Record<string, object>>(
  "dies-saved-layouts",
  {}
);
export const currentLayoutNameAtom = atomWithStorage<string>(
  "dies-current-layout",
  "default"
);

// ============================================================================
// Panel Components Registry
// ============================================================================

interface PanelComponentsMap {
  [key: string]: React.FC<IDockviewPanelProps>;
}

export interface DockviewWrapperRef {
  api: DockviewApi | null;
  resetToDefault: () => void;
}

interface DockviewWrapperProps {
  components: PanelComponentsMap;
  onCreateDefaultLayout: (api: DockviewApi) => void;
  selectedPlayerId: number | null;
  onSelectPlayer: (id: number | null) => void;
}

// ============================================================================
// DockviewWrapper Component
// ============================================================================

const DockviewWrapper = forwardRef<DockviewWrapperRef, DockviewWrapperProps>(
  (
    { components, onCreateDefaultLayout, selectedPlayerId, onSelectPlayer },
    ref
  ) => {
    const apiRef = useRef<DockviewApi | null>(null);
    const [savedLayouts, setSavedLayouts] = useAtom(savedLayoutsAtom);
    const [currentLayoutName] = useAtom(currentLayoutNameAtom);

    // Expose API via ref
    useImperativeHandle(
      ref,
      () => ({
        api: apiRef.current,
        resetToDefault: () => {
          if (apiRef.current) {
            onCreateDefaultLayout(apiRef.current);
          }
        },
      }),
      [onCreateDefaultLayout]
    );

    // Wrap components to pass props
    const wrappedComponents = React.useMemo(() => {
      const wrapped: PanelComponentsMap = {};
      for (const [key, Component] of Object.entries(components)) {
        wrapped[key] = (props: IDockviewPanelProps) => (
          <Component
            {...props}
            params={{
              ...props.params,
              selectedPlayerId,
              onSelectPlayer,
            }}
          />
        );
      }
      return wrapped;
    }, [components, selectedPlayerId, onSelectPlayer]);

    // Handle Dockview ready
    const handleReady = useCallback(
      (event: DockviewReadyEvent) => {
        apiRef.current = event.api;

        // Try to restore saved layout
        const savedLayout = savedLayouts[currentLayoutName];
        if (savedLayout) {
          try {
            event.api.fromJSON(savedLayout as any);
            return;
          } catch (error) {
            console.error("Failed to restore layout:", error);
          }
        }

        // Fall back to default layout
        onCreateDefaultLayout(event.api);
      },
      [savedLayouts, currentLayoutName, onCreateDefaultLayout]
    );

    // Save layout on changes
    useEffect(() => {
      const api = apiRef.current;
      if (!api) return;

      const disposable = api.onDidLayoutChange(() => {
        try {
          const state = api.toJSON();
          setSavedLayouts((prev) => ({
            ...prev,
            [currentLayoutName]: state,
          }));
        } catch (error) {
          console.error("Failed to save layout:", error);
        }
      });

      return () => disposable.dispose();
    }, [currentLayoutName, setSavedLayouts]);

    // Save on window close
    useEffect(() => {
      const handleBeforeUnload = () => {
        if (apiRef.current) {
          try {
            const state = apiRef.current.toJSON();
            // Use synchronous localStorage to ensure save completes
            const currentLayouts = JSON.parse(
              localStorage.getItem("dies-saved-layouts") || "{}"
            );
            currentLayouts[currentLayoutName] = state;
            localStorage.setItem(
              "dies-saved-layouts",
              JSON.stringify(currentLayouts)
            );
          } catch (error) {
            console.error("Failed to save layout on unload:", error);
          }
        }
      };

      window.addEventListener("beforeunload", handleBeforeUnload);
      return () =>
        window.removeEventListener("beforeunload", handleBeforeUnload);
    }, [currentLayoutName]);

    return (
      <DockviewReact
        className="dockview-theme-dies"
        components={wrappedComponents}
        onReady={handleReady}
        disableFloatingGroups={false}
      />
    );
  }
);

DockviewWrapper.displayName = "DockviewWrapper";

export default DockviewWrapper;

// ============================================================================
// Layout Management Utilities
// ============================================================================

export const useLayoutManager = () => {
  const [savedLayouts, setSavedLayouts] = useAtom(savedLayoutsAtom);
  const [currentLayoutName, setCurrentLayoutName] = useAtom(
    currentLayoutNameAtom
  );

  const saveCurrentLayout = useCallback(
    (api: DockviewApi, name: string) => {
      try {
        const state = api.toJSON();
        setSavedLayouts((prev) => ({
          ...prev,
          [name]: state,
        }));
        setCurrentLayoutName(name);
      } catch (error) {
        console.error("Failed to save layout:", error);
      }
    },
    [setSavedLayouts, setCurrentLayoutName]
  );

  const loadLayout = useCallback(
    (api: DockviewApi, name: string) => {
      const layout = savedLayouts[name];
      if (layout) {
        try {
          api.fromJSON(layout as any);
          setCurrentLayoutName(name);
        } catch (error) {
          console.error("Failed to load layout:", error);
        }
      }
    },
    [savedLayouts, setCurrentLayoutName]
  );

  const deleteLayout = useCallback(
    (name: string) => {
      if (name === "default") return; // Don't delete default
      setSavedLayouts((prev) => {
        const { [name]: _, ...rest } = prev;
        return rest;
      });
      if (currentLayoutName === name) {
        setCurrentLayoutName("default");
      }
    },
    [currentLayoutName, setSavedLayouts, setCurrentLayoutName]
  );

  const getLayoutNames = useCallback(() => {
    return Object.keys(savedLayouts);
  }, [savedLayouts]);

  return {
    savedLayouts,
    currentLayoutName,
    setCurrentLayoutName,
    saveCurrentLayout,
    loadLayout,
    deleteLayout,
    getLayoutNames,
  };
};
