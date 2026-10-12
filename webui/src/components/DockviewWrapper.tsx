import {
  useRef,
  useEffect,
  useState,
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
import { getDefaultStore, useAtom } from "jotai";
import { atomWithStorage } from "jotai/utils";
import { DEFAULT_VIEW_ID, isBuiltinView, viewPreset } from "@/lib/views";

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

// Bumped whenever the built-in default layout changes shape. A stored version
// below this discards the persisted "default" layout so users pick up the new
// arrangement (e.g. the bottom drawer + console) instead of their stale one.
const LAYOUT_SCHEMA_VERSION = 9;

// Atom for storing layouts with localStorage persistence
// `getOnInit`: read localStorage synchronously on first access. Without it the
// atoms hold their defaults until mounted, so Dockview's ready handler (which
// runs during the first render pass) would never see the persisted view.
export const savedLayoutsAtom = atomWithStorage<Record<string, object>>(
  "dies-saved-layouts",
  {},
  undefined,
  { getOnInit: true }
);
export const currentLayoutNameAtom = atomWithStorage<string>(
  "dies-current-layout",
  "default",
  undefined,
  { getOnInit: true }
);
export const layoutSchemaVersionAtom = atomWithStorage<number>(
  "dies-layout-version",
  0,
  undefined,
  { getOnInit: true }
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
}

/** Load a view's built-in preset (or the default view's, for custom views). */
export function applyViewPreset(api: DockviewApi, viewId: string) {
  const preset = viewPreset(viewId) ?? viewPreset(DEFAULT_VIEW_ID)!;
  api.clear();
  api.fromJSON(preset.layout as any);
}

/**
 * Switch to another view in place (no reload). The current arrangement is saved
 * under the current view first; the target view restores its saved arrangement,
 * or its preset if it has none.
 */
export function switchView(api: DockviewApi, viewId: string) {
  const store = getDefaultStore();
  const current = store.get(currentLayoutNameAtom);
  if (current === viewId) return;
  try {
    store.set(savedLayoutsAtom, { ...store.get(savedLayoutsAtom), [current]: api.toJSON() });
  } catch (error) {
    console.error("Failed to save layout:", error);
  }
  store.set(currentLayoutNameAtom, viewId);
  const saved = store.get(savedLayoutsAtom)[viewId];
  if (saved) {
    try {
      api.clear();
      api.fromJSON(saved as any);
      return;
    } catch (error) {
      console.error("Failed to restore layout:", error);
    }
  }
  applyViewPreset(api, viewId);
}

// ============================================================================
// DockviewWrapper Component
// ============================================================================

const DockviewWrapper = forwardRef<DockviewWrapperRef, DockviewWrapperProps>(
  ({ components }, ref) => {
    const apiRef = useRef<DockviewApi | null>(null);
    const [apiReady, setApiReady] = useState(false);
    const [savedLayouts, setSavedLayouts] = useAtom(savedLayoutsAtom);
    const [currentLayoutName] = useAtom(currentLayoutNameAtom);
    const [layoutVersion, setLayoutVersion] = useAtom(layoutSchemaVersionAtom);

    // Expose API via ref. `api` is a getter so it always reflects the current
    // value — the DockviewApi only exists after `handleReady`, well after this
    // handle is first created.
    useImperativeHandle(
      ref,
      () => ({
        get api() {
          return apiRef.current;
        },
        resetToDefault: () => {
          if (apiRef.current) {
            applyViewPreset(apiRef.current, getDefaultStore().get(currentLayoutNameAtom));
          }
        },
      }),
      []
    );

    // Handle Dockview ready
    const handleReady = useCallback(
      (event: DockviewReadyEvent) => {
        apiRef.current = event.api;
        setApiReady(true);

        // One-time migration: if persisted built-in views predate the current
        // layout schema, drop them and rebuild from the presets.
        if (layoutVersion < LAYOUT_SCHEMA_VERSION) {
          setLayoutVersion(LAYOUT_SCHEMA_VERSION);
          setSavedLayouts((prev) =>
            Object.fromEntries(Object.entries(prev).filter(([name]) => !isBuiltinView(name)))
          );
          if (isBuiltinView(currentLayoutName)) {
            applyViewPreset(event.api, currentLayoutName);
            return;
          }
        }

        // Try to restore saved layout
        const savedLayout = savedLayouts[currentLayoutName];
        if (savedLayout) {
          try {
            event.api.fromJSON(savedLayout as any);
            setLayoutVersion(LAYOUT_SCHEMA_VERSION);
            return;
          } catch (error) {
            console.error("Failed to restore layout:", error);
          }
        }

        // Fall back to the view's preset
        setLayoutVersion(LAYOUT_SCHEMA_VERSION);
        applyViewPreset(event.api, currentLayoutName);
      },
      [savedLayouts, setSavedLayouts, currentLayoutName, layoutVersion, setLayoutVersion]
    );

    // Save layout on changes
    useEffect(() => {
      const api = apiRef.current;
      if (!api) return;

      const disposable = api.onDidLayoutChange(() => {
        try {
          const state = api.toJSON();
          // Read the view name at event time: `switchView` changes it right
          // before loading the next layout, ahead of this effect re-running.
          const name = getDefaultStore().get(currentLayoutNameAtom);
          setSavedLayouts((prev) => ({
            ...prev,
            [name]: state,
          }));
        } catch (error) {
          console.error("Failed to save layout:", error);
        }
      });

      return () => disposable.dispose();
    }, [apiReady, setSavedLayouts]);

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
            currentLayouts[getDefaultStore().get(currentLayoutNameAtom)] = state;
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
        components={components}
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
