/**
 * Interactive notebook plots: replays the engine's recorded `Plot.*` calls with
 * Observable Plot, adding hover tips, crosshairs, a default grid for line plots,
 * zoom/pan/axis ranges (local state), PNG copy/save and a fullscreen view.
 */
import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import * as Plot from "@observablehq/plot";
import { Copy, Download, Maximize2, RotateCcw, SlidersHorizontal, X } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------- spec handling

type Call = { __plot: string; args: unknown[] };

const isCall = (x: unknown): x is Call =>
  !!x && typeof x === "object" && typeof (x as Call).__plot === "string" && Array.isArray((x as Call).args);

const isPlainObject = (x: unknown): x is Record<string, unknown> =>
  !!x && typeof x === "object" && !Array.isArray(x) && !isCall(x);

/** Rebuild real Plot calls from the recorded `{__plot, args}` tree. */
function revive(x: unknown): unknown {
  if (Array.isArray(x)) return x.map(revive);
  if (isCall(x)) {
    const fn = x.__plot.split(".").reduce<any>((acc, k) => (acc == null ? acc : acc[k]), Plot as any);
    if (typeof fn !== "function") throw new Error(`Plot.${x.__plot} is not a function`);
    return fn(...x.args.map(revive));
  }
  if (isPlainObject(x)) return Object.fromEntries(Object.entries(x).map(([k, v]) => [k, revive(v)]));
  return x;
}

const LINE_MARKS = /^(line|lineX|lineY|area|areaX|areaY)$/;
const POINT_MARKS = /^(dot|dotX|dotY)$/;
const TIP_MARKS = /^(line|lineX|lineY|area|areaX|areaY|dot|dotX|dotY|barX|barY|cell|rect|rectX|rectY|tickX|tickY)$/;

export interface Domains {
  x?: [number, number];
  y?: [number, number];
}

const STYLE = {
  background: "transparent",
  color: "#c9d1d9",
  fontFamily: "JetBrains Mono, monospace",
  fontSize: "11px",
  overflow: "visible",
};

/**
 * Turn any recorded spec into a `Plot.plot` spec and add the interactive
 * defaults. Only fills in what the user didn't set.
 */
function prepare(spec: unknown, width: number, height: number | undefined, domains: Domains, timeX: boolean, timeY: boolean) {
  const base: Call = isCall(spec) && spec.__plot === "plot" ? spec : { __plot: "plot", args: [{ marks: [spec] }] };
  const opts = { ...((base.args[0] as Record<string, unknown>) ?? {}) };
  const marks = (Array.isArray(opts.marks) ? opts.marks : []) as unknown[];
  let hasLine = false;
  const extra: Call[] = [];
  const zoomed = !!(domains.x || domains.y);

  const newMarks = marks.map((m) => {
    if (!isCall(m)) return m;
    if (LINE_MARKS.test(m.__plot)) hasLine = true;
    const [data, o, ...rest] = m.args;
    if (o !== undefined && !isPlainObject(o)) return m; // transform-wrapped options: leave alone
    const mo: Record<string, unknown> = { ...((o as Record<string, unknown>) ?? {}) };
    if (TIP_MARKS.test(m.__plot) && mo.tip === undefined) {
      mo.tip = LINE_MARKS.test(m.__plot) ? "x" : POINT_MARKS.test(m.__plot) ? "xy" : true;
    }
    if (zoomed && mo.clip === undefined) mo.clip = "frame";
    const channels = Object.fromEntries(
      ["x", "y", "fx", "fy"].filter((k) => typeof mo[k] === "string").map((k) => [k, mo[k]])
    );
    if (Array.isArray(data) && channels.x && channels.y) {
      if (LINE_MARKS.test(m.__plot)) extra.push({ __plot: "crosshairX", args: [data, channels] });
      else if (POINT_MARKS.test(m.__plot)) extra.push({ __plot: "crosshair", args: [data, channels] });
    }
    return { __plot: m.__plot, args: [data, mo, ...rest] };
  });

  if (hasLine && opts.grid === undefined) opts.grid = true;
  const axis = (key: "x" | "y", d: [number, number] | undefined, time: boolean) => {
    if (!d) return;
    const domain = time ? d.map((v) => new Date(v)) : d;
    opts[key] = { ...((opts[key] as object) ?? {}), domain };
  };
  axis("x", domains.x, timeX);
  axis("y", domains.y, timeY);
  opts.marks = [...newMarks, ...extra];
  opts.width = width;
  if (height !== undefined) opts.height = height;
  opts.style = { ...STYLE, ...((opts.style as object) ?? {}) };
  return revive({ __plot: "plot", args: [opts] }) as (SVGSVGElement | HTMLElement) & {
    scale: (name: string) => { type: string; domain: unknown[]; range?: number[] } | undefined;
  };
}

// ---------------------------------------------------------------- raster export

const PNG_BG = "#12141a";

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not rasterize plot"));
    img.src = src;
  });
}

/** Rasterize a rendered plot (SVGs + HTML legend text) to a PNG at 2x. */
async function plotToPng(root: HTMLElement): Promise<Blob> {
  const box = root.getBoundingClientRect();
  const scale = 2;
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(box.width * scale);
  canvas.height = Math.ceil(box.height * scale);
  const ctx = canvas.getContext("2d")!;
  ctx.scale(scale, scale);
  ctx.fillStyle = PNG_BG;
  ctx.fillRect(0, 0, box.width, box.height);

  // Every SVG (plot frames and legend swatches) at its on-screen position.
  for (const svg of Array.from(root.querySelectorAll("svg"))) {
    if (svg.parentElement?.closest("svg")) continue;
    const r = svg.getBoundingClientRect();
    if (!r.width || !r.height) continue;
    const clone = svg.cloneNode(true) as SVGSVGElement;
    const cs = getComputedStyle(svg);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    clone.setAttribute("width", String(r.width));
    clone.setAttribute("height", String(r.height));
    clone.style.color = cs.color;
    clone.style.fontFamily = cs.fontFamily;
    clone.style.fontSize = cs.fontSize;
    // Hover-only marks (tips, crosshairs) must not end up in the image.
    clone.querySelectorAll("[aria-label=tip], [aria-label^=crosshair]").forEach((n) => n.remove());
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)], { type: "image/svg+xml" }));
    try {
      const img = await loadImage(url);
      ctx.drawImage(img, r.left - box.left, r.top - box.top, r.width, r.height);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  // HTML text outside SVGs (legend labels, titles) at its on-screen position.
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n = walker.nextNode(); n; n = walker.nextNode()) {
    const text = n.textContent?.trim();
    const el = n.parentElement;
    if (!text || !el || el.closest("svg")) continue;
    const range = document.createRange();
    range.selectNodeContents(n);
    const r = range.getBoundingClientRect();
    const cs = getComputedStyle(el);
    ctx.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    ctx.fillStyle = cs.color;
    ctx.textBaseline = "middle";
    ctx.fillText(text, r.left - box.left, r.top - box.top + r.height / 2);
  }

  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encoding failed"))), "image/png")
  );
}

async function copyPng(root: HTMLElement) {
  try {
    if (!navigator.clipboard || typeof ClipboardItem === "undefined") {
      throw new Error("clipboard needs https or localhost — use Save instead");
    }
    await navigator.clipboard.write([new ClipboardItem({ "image/png": plotToPng(root) })]);
    toast.success("Plot copied");
  } catch (e: any) {
    toast.error(`Copy failed: ${e?.message ?? e}`);
  }
}

async function savePng(root: HTMLElement) {
  try {
    const blob = await plotToPng(root);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `plot-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  } catch (e: any) {
    toast.error(`Save failed: ${e?.message ?? e}`);
  }
}

// ---------------------------------------------------------------- the plot

interface ScaleInfo {
  type: string;
  domain: [number, number];
  range: [number, number];
}

const ZOOMABLE = new Set(["linear", "utc", "time", "pow", "sqrt"]);

function continuousScale(plot: ReturnType<typeof prepare>, name: "x" | "y"): ScaleInfo | null {
  const s = plot.scale(name);
  if (!s || !ZOOMABLE.has(s.type) || !s.range || s.domain.length !== 2) return null;
  const d = s.domain.map((v) => (v instanceof Date ? v.getTime() : Number(v))) as [number, number];
  return { type: s.type, domain: d, range: s.range as [number, number] };
}

const invert = (s: ScaleInfo, px: number) =>
  s.domain[0] + ((px - s.range[0]) / (s.range[1] - s.range[0])) * (s.domain[1] - s.domain[0]);

interface Scales {
  x: ScaleInfo | null;
  y: ScaleInfo | null;
}

/** Renders the plot and handles zoom/pan. */
const PlotCanvas: React.FC<{
  spec: unknown;
  height?: number;
  domains: Domains;
  setDomains: (d: Domains) => void;
  /** Zoom with a plain wheel (fullscreen) instead of Ctrl/⌘+wheel. */
  wheelZoom: boolean;
  onScales: (s: Scales) => void;
  rootRef: React.RefObject<HTMLDivElement>;
}> = ({ spec, height, domains, setDomains, wheelZoom, onScales, rootRef }) => {
  const [width, setWidth] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const scales = useRef<Scales>({ x: null, y: null });
  const timeAxes = useRef({ x: false, y: false });
  const domainsRef = useRef(domains);
  domainsRef.current = domains;

  useLayoutEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = Math.floor(entries[0].contentRect.width);
      if (w > 0) setWidth((prev) => (Math.abs(prev - w) > 4 ? w : prev));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [rootRef]);

  useEffect(() => {
    const el = rootRef.current;
    if (!el || width <= 0) return;
    let frame = 0;
    frame = requestAnimationFrame(() => {
      try {
        const plot = prepare(spec, width, height, domains, timeAxes.current.x, timeAxes.current.y);
        el.replaceChildren(plot);
        // Zoom/pan only for unfaceted continuous axes.
        const faceted = !!(plot.scale("fx") || plot.scale("fy"));
        const x = faceted ? null : continuousScale(plot, "x");
        const y = faceted ? null : continuousScale(plot, "y");
        timeAxes.current = { x: x?.type === "utc" || x?.type === "time", y: y?.type === "utc" || y?.type === "time" };
        scales.current = { x, y };
        onScales(scales.current);
        setError(null);
      } catch (e: any) {
        el.replaceChildren();
        setError(String(e?.message ?? e));
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [spec, width, height, domains, rootRef, onScales]);

  // Wheel zoom around the cursor, drag to pan, double-click to reset.
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const local = (e: MouseEvent) => {
      const svg = el.querySelector("svg:not(svg svg)") ?? el.querySelector("svg");
      const r = (svg ?? el).getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top] as const;
    };
    const onWheel = (e: WheelEvent) => {
      const { x, y } = scales.current;
      if (!x && !y) return;
      if (!wheelZoom && !(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const [px, py] = local(e);
      const k = Math.exp(e.deltaY * (e.ctrlKey && !e.metaKey ? 0.01 : 0.002));
      const zoom = (s: ScaleInfo, p: number): [number, number] => {
        const c = invert(s, p);
        return [c - (c - s.domain[0]) * k, c - (c - s.domain[1]) * k];
      };
      const next: Domains = { ...domainsRef.current };
      if (x && !e.altKey) next.x = zoom(x, px);
      if (y && !e.shiftKey) next.y = zoom(y, py);
      setDomains(next);
    };
    let drag: { px: number; py: number; s: Scales } | null = null;
    const onDown = (e: MouseEvent) => {
      if (e.button !== 0 || (!scales.current.x && !scales.current.y)) return;
      const [px, py] = local(e);
      drag = { px, py, s: scales.current };
      el.style.cursor = "grabbing";
      e.preventDefault();
    };
    const onMove = (e: MouseEvent) => {
      if (!drag) return;
      const [px, py] = local(e);
      const pan = (s: ScaleInfo, d: number): [number, number] => {
        const shift = (d / (s.range[1] - s.range[0])) * (s.domain[1] - s.domain[0]);
        return [s.domain[0] - shift, s.domain[1] - shift];
      };
      const next: Domains = { ...domainsRef.current };
      if (drag.s.x) next.x = pan(drag.s.x, px - drag.px);
      if (drag.s.y) next.y = pan(drag.s.y, py - drag.py);
      setDomains(next);
    };
    const onUp = () => {
      drag = null;
      el.style.cursor = "";
    };
    const onDbl = () => setDomains({});
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    el.addEventListener("dblclick", onDbl);
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      el.removeEventListener("dblclick", onDbl);
    };
  }, [rootRef, setDomains, wheelZoom]);

  return error ? <pre className="text-accent-red text-xs whitespace-pre-wrap">{error}</pre> : null;
};

// ---------------------------------------------------------------- chrome

const ToolButton: React.FC<{ title: string; onClick: () => void; children: React.ReactNode; active?: boolean }> = ({
  title,
  onClick,
  children,
  active,
}) => (
  <button
    title={title}
    onClick={onClick}
    className={cn(
      "h-6 w-6 flex items-center justify-center rounded-sm text-text-dim hover:text-text-bright hover:bg-bg-overlay transition-colors",
      active && "text-accent-cyan"
    )}
  >
    {children}
  </button>
);

const fmt = (v: number) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-2 && v !== 0) ? v.toExponential(2) : String(+v.toPrecision(5)));

const AxisEditor: React.FC<{ scales: Scales; domains: Domains; setDomains: (d: Domains) => void }> = ({
  scales,
  domains,
  setDomains,
}) => {
  const row = (key: "x" | "y") => {
    const s = scales[key];
    if (!s || s.type !== "linear") return null;
    const d = domains[key] ?? s.domain;
    const set = (i: 0 | 1, v: string) => {
      const n = Number(v);
      if (!Number.isFinite(n)) return;
      const next: [number, number] = [...d] as [number, number];
      next[i] = n;
      setDomains({ ...domains, [key]: next });
    };
    return (
      <div className="flex items-center gap-1">
        <span className="w-3 text-text-dim">{key}</span>
        {[0, 1].map((i) => (
          <input
            key={`${key}${i}${d[i]}`}
            defaultValue={fmt(d[i])}
            onKeyDown={(e) => e.key === "Enter" && set(i as 0 | 1, e.currentTarget.value)}
            onBlur={(e) => set(i as 0 | 1, e.currentTarget.value)}
            className="w-20 bg-bg-base border border-border-muted focus:border-border-std outline-none px-1 py-0.5 tabular-nums"
          />
        ))}
      </div>
    );
  };
  return (
    <div className="absolute right-1 top-8 z-10 flex flex-col gap-1 p-2 text-xs bg-bg-elevated/95 border border-border-std shadow-lg backdrop-blur-sm">
      {row("x")}
      {row("y")}
      <div className="text-text-muted pt-0.5">⌘/Ctrl+wheel zoom · drag pan · double-click reset</div>
    </div>
  );
};

/** One plot surface (inline or fullscreen) with its hover toolbar. */
const PlotSurface: React.FC<{
  spec: unknown;
  height?: number;
  domains: Domains;
  setDomains: (d: Domains) => void;
  fullscreen: boolean;
  onExpand?: () => void;
  onClose?: () => void;
}> = ({ spec, height, domains, setDomains, fullscreen, onExpand, onClose }) => {
  const rootRef = useRef<HTMLDivElement>(null);
  const [scales, setScales] = useState<Scales>({ x: null, y: null });
  const [axesOpen, setAxesOpen] = useState(false);
  const zoomable = !!(scales.x || scales.y);
  const zoomed = !!(domains.x || domains.y);

  return (
    <div className="group/plot relative">
      <div ref={rootRef} className="notebook-plot w-full select-none" />
      <PlotCanvas
        spec={spec}
        height={height}
        domains={domains}
        setDomains={setDomains}
        wheelZoom={fullscreen}
        onScales={setScales}
        rootRef={rootRef}
      />
      <div
        className={cn(
          "absolute right-1 top-1 flex items-center gap-0.5 p-0.5 rounded-sm bg-bg-elevated/80 backdrop-blur-sm border border-border-subtle shadow-sm transition-opacity duration-150",
          fullscreen ? "opacity-100" : "opacity-0 group-hover/plot:opacity-100"
        )}
      >
        {zoomable && (
          <ToolButton title="Axis ranges" onClick={() => setAxesOpen((v) => !v)} active={axesOpen}>
            <SlidersHorizontal className="h-3.5 w-3.5" />
          </ToolButton>
        )}
        {zoomed && (
          <ToolButton title="Reset zoom" onClick={() => setDomains({})}>
            <RotateCcw className="h-3.5 w-3.5" />
          </ToolButton>
        )}
        <ToolButton title="Copy as PNG" onClick={() => rootRef.current && copyPng(rootRef.current)}>
          <Copy className="h-3.5 w-3.5" />
        </ToolButton>
        <ToolButton title="Save PNG" onClick={() => rootRef.current && savePng(rootRef.current)}>
          <Download className="h-3.5 w-3.5" />
        </ToolButton>
        {fullscreen ? (
          <ToolButton title="Close (Esc)" onClick={() => onClose?.()}>
            <X className="h-3.5 w-3.5" />
          </ToolButton>
        ) : (
          <ToolButton title="Expand" onClick={() => onExpand?.()}>
            <Maximize2 className="h-3.5 w-3.5" />
          </ToolButton>
        )}
      </div>
      {axesOpen && zoomable && <AxisEditor scales={scales} domains={domains} setDomains={setDomains} />}
    </div>
  );
};

/** Fullscreen modal: ~90% of the viewport, darkened overlay, scale/fade in. */
const PlotModal: React.FC<{
  spec: unknown;
  domains: Domains;
  setDomains: (d: Domains) => void;
  onClose: () => void;
}> = ({ spec, domains, setDomains, onClose }) => {
  const [shown, setShown] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number>();

  const close = useCallback(() => {
    setShown(false);
    setTimeout(onClose, 180);
  }, [onClose]);

  useEffect(() => {
    const id = requestAnimationFrame(() => setShown(true));
    // Capture phase: Esc closes the modal and goes no further.
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        e.preventDefault();
        close();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => {
      cancelAnimationFrame(id);
      window.removeEventListener("keydown", onKey, true);
    };
  }, [close]);

  useLayoutEffect(() => {
    const el = bodyRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setHeight(Math.max(200, Math.floor(el.clientHeight) - 8)));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return createPortal(
    <div
      className={cn(
        "fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-[2px] transition-opacity duration-200 ease-out",
        shown ? "opacity-100" : "opacity-0"
      )}
      onMouseDown={(e) => e.target === e.currentTarget && close()}
    >
      <div
        className={cn(
          "w-[90vw] h-[90vh] flex flex-col bg-bg-surface border border-border-std shadow-2xl transition-all duration-200 ease-out",
          shown ? "opacity-100 scale-100" : "opacity-0 scale-[0.96]"
        )}
      >
        <div ref={bodyRef} className="flex-1 min-h-0 p-4 overflow-hidden">
          {height !== undefined && (
            <PlotSurface
              spec={spec}
              height={height}
              domains={domains}
              setDomains={setDomains}
              fullscreen
              onClose={close}
            />
          )}
        </div>
      </div>
    </div>,
    document.body
  );
};

/** A notebook plot output: inline surface + optional fullscreen view, sharing zoom. */
export const PlotOutput: React.FC<{ spec: unknown }> = ({ spec }) => {
  const [domains, setDomains] = useState<Domains>({});
  const [expanded, setExpanded] = useState(false);
  return (
    <>
      <PlotSurface
        spec={spec}
        domains={domains}
        setDomains={setDomains}
        fullscreen={false}
        onExpand={() => setExpanded(true)}
      />
      {expanded && (
        <PlotModal spec={spec} domains={domains} setDomains={setDomains} onClose={() => setExpanded(false)} />
      )}
    </>
  );
};
