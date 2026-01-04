# Dies UI Redesign Specification

> **Status:** Design Document  
> **Version:** 1.0  
> **Last Updated:** January 2026

This document specifies the comprehensive UI overhaul for Dies, transforming the interface from a generic shadcn-based design into a distinct, information-dense, mission-control aesthetic.

---

## Table of Contents

1. [Design Philosophy](#design-philosophy)
2. [Current State Analysis](#current-state-analysis)
3. [Visual Language](#visual-language)
4. [Typography System](#typography-system)
5. [Color System](#color-system)
6. [Component Design](#component-design)
7. [Layout Architecture](#layout-architecture)
8. [Toolbar Design](#toolbar-design)
9. [Panel Inventory](#panel-inventory)
10. [Field View](#field-view)
11. [Settings Modal](#settings-modal)
12. [Debug Visualization](#debug-visualization)
13. [Libraries & Technologies](#libraries--technologies)
14. [Future Considerations](#future-considerations)

---

## Design Philosophy

### Vision

The new Dies UI draws inspiration from:

- **Star Wars control panels** — Dense, geometric, purposeful interfaces with minimal decoration
- **High-frequency trading terminals** — Information density, real-time data streams, precision
- **Game engine editors (Unity, Godot)** — Central viewport with configurable peripheral panels

### Core Principles

1. **Density over whitespace** — Every pixel should earn its place. Compact spacing, small fonts, packed information.

2. **Function over decoration** — No rounded corners, no shadows for aesthetics. Visual elements serve communication purposes.

3. **Geometric precision** — Sharp edges, angular elements, grid-aligned layouts. The interface should feel engineered.

4. **Focused color** — Predominantly monochromatic with high-contrast accent colors reserved for status, alerts, and selection.

5. **Configurability** — Users can arrange panels to match their workflow. Layouts persist and can be named/switched.

6. **Glanceability** — Critical state information visible at all times without interaction.

---

## Current State Analysis

### Current Technology Stack

| Component       | Technology                   | Status                          |
| --------------- | ---------------------------- | ------------------------------- |
| Framework       | React 18 + Vite              | Keep                            |
| Styling         | Tailwind CSS                 | Keep, reconfigure               |
| Components      | Radix UI primitives + shadcn | Keep Radix, redesign components |
| State           | Jotai + React Query          | Keep                            |
| Layout          | react-resizable-panels       | Replace with Dockview           |
| Field Rendering | Canvas2D                     | Keep, extend                    |
| Code Editor     | Monaco                       | Keep                            |
| Charts          | Recharts                     | Keep                            |

### Current Layout Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TOOLBAR (56px, verbose)                                                     │
│ [Logo] [Sim/Live] [Play/Pause/Stop] [Team Settings btn] [Swap btns] [Team]  │
├────────────┬─────────────────────────────────────────────┬──────────────────┤
│ Game       │                                             │                  │
│ Controller │              FIELD                          │    Player        │
│ Panel      │           (canvas)                          │    Sidebar       │
│            ├─────────────────────────────────────────────┤                  │
│ Team       │  Settings Tabs (collapsed by default)       │                  │
│ Overview   │  [Controller] [Tracker] [Skill]             │                  │
├────────────┴─────────────────────────────────────────────┴──────────────────┤
│ STATUS BAR                                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Current Pain Points

- **Visual identity:** Generic shadcn appearance, rounded corners, conventional styling
- **Toolbar verbosity:** Text labels on buttons, duplicate controls, excessive horizontal space
- **Fixed layout:** Panels can resize but not rearrange, no tabs, no floating
- **Limited debug shapes:** Only Cross, Circle, Line supported
- **No debug organization:** All shapes render unconditionally, no layer system
- **No scenario management:** Cannot save/restore robot positions
- **No log playback UI:** Backend supports it, no frontend interface
- **No text log viewer:** Logs captured but not displayed

---

## Visual Language

### Geometry

| Property      | Current            | New               |
| ------------- | ------------------ | ----------------- |
| Border radius | 0.5rem (8px)       | 0 (sharp corners) |
| Borders       | 1px, subtle        | 1px, defined      |
| Shadows       | Subtle elevation   | None (flat)       |
| Padding       | Generous (12-16px) | Compact (4-8px)   |
| Gaps          | 16-24px            | 8-12px            |

### Angular Accents

For key UI elements (active tabs, selected items, panel headers), optional beveled corners using CSS `clip-path`:

```
Standard rectangle:     Beveled corner accent:
┌─────────────────┐     ┌─────────────────┬┐
│                 │     │                 ││
│                 │     │                 ││
└─────────────────┘     └─────────────────┴┘
```

Bevel angle: 45°, bevel size: 4-6px. Applied sparingly to avoid visual noise.

### Information Density

- Default font size: 11-12px (body), 9-10px (labels)
- Line height: 1.2-1.3 (tighter than typical)
- Component height: 24-28px (inputs, buttons)
- Panel header height: 24px
- Tab height: 24px

### Visual Rhythm

Consistent 4px grid system:

- Micro spacing: 4px
- Small spacing: 8px
- Medium spacing: 12px
- Large spacing: 16px

---

## Typography System

### Font

**JetBrains Mono** — single font family for the entire interface.

| Fallback Stack                                                            |
| ------------------------------------------------------------------------- |
| `"JetBrains Mono", "IBM Plex Mono", "Fira Code", ui-monospace, monospace` |

### Rationale

- Excellent legibility at small sizes (9-11px)
- Designed for code/data display with distinctive character shapes
- Consistent technical aesthetic across all UI elements
- Single font = faster load, simpler CSS, cohesive feel
- Fits the "mission control terminal" vibe perfectly

### Type Scale

| Use Case                            | Size    | Weight | Style      |
| ----------------------------------- | ------- | ------ | ---------- |
| **Large displays** (timers, status) | 14-16px | 600    | uppercase  |
| **Panel titles**                    | 11-12px | 600    | uppercase  |
| **Body / Data values**              | 11px    | 400    | normal     |
| **Labels / Secondary**              | 10px    | 400    | normal     |
| **Small labels / Hints**            | 9px     | 400    | --text-dim |

### Text Styling

- Uppercase for: panel titles, status indicators, category labels
- Sentence case for: body text, descriptions, values
- Letter-spacing: +0.05em for uppercase text, normal otherwise
- Line height: 1.3 (body), 1.1 (headings)

---

## Color System

### Base Palette

```
Background Hierarchy (darkest to lightest):
┌──────────────────────────────────────────────────────────────┐
│ --bg-void       #08090a   Deepest background, unused areas   │
│ --bg-base       #0c0d0f   Application background             │
│ --bg-surface    #12141a   Panel backgrounds                  │
│ --bg-elevated   #1a1d24   Cards, dialogs, popovers           │
│ --bg-overlay    #22262f   Hover states, selection background │
└──────────────────────────────────────────────────────────────┘

Border & Divider Hierarchy:
┌──────────────────────────────────────────────────────────────┐
│ --border-subtle #1e2228   Subtle internal divisions          │
│ --border-muted  #2a2f38   Standard borders                   │
│ --border-std    #3a4250   Prominent borders, focus rings     │
└──────────────────────────────────────────────────────────────┘

Text Hierarchy:
┌──────────────────────────────────────────────────────────────┐
│ --text-muted    #5a6270   Disabled, tertiary text            │
│ --text-dim      #7a8290   Secondary text, labels             │
│ --text-std      #a0a8b4   Primary text                       │
│ --text-bright   #d0d4dc   Emphasized text, values            │
│ --text-max      #f0f2f5   Maximum contrast, alerts           │
└──────────────────────────────────────────────────────────────┘
```

### Accent Colors

Reserved for status indication, interaction feedback, and semantic meaning:

```
Functional Accents:
┌──────────────────────────────────────────────────────────────┐
│ --accent-cyan    #00d4e5   Primary accent, selection, focus  │
│ --accent-green   #00c853   Success, running, active          │
│ --accent-amber   #ffab00   Warning, attention, pending       │
│ --accent-red     #ff3d3d   Error, critical, stopped          │
│ --accent-blue    #2979ff   Info, links, secondary action     │
└──────────────────────────────────────────────────────────────┘

Team Colors (preserved):
┌──────────────────────────────────────────────────────────────┐
│ --team-blue      #2563eb   Blue team primary                 │
│ --team-blue-dim  #1e40af   Blue team dimmed                  │
│ --team-yellow    #eab308   Yellow team primary               │
│ --team-yellow-dim #a16207  Yellow team dimmed                │
└──────────────────────────────────────────────────────────────┘
```

### Color Usage Rules

1. **Accents are earned:** Color only appears when something requires attention
2. **Status over decoration:** Color indicates state (running/stopped/error)
3. **Team colors are sacred:** Blue and yellow reserved exclusively for team identification
4. **One accent at a time:** Avoid multiple accent colors in close proximity

---

## Component Design

### Buttons

```
Standard Button (28px height):
┌─────────────────┐
│    LABEL        │  <- Uppercase, 10px font
└─────────────────┘
Border: 1px --border-muted
Background: transparent or --bg-overlay on hover
Text: --text-std

Icon-Only Button (24x24):
┌────┐
│ ▶  │
└────┘
Same border treatment, centered icon
```

**Variants:**

- **Ghost:** No border, appears on hover
- **Outline:** 1px border, transparent fill
- **Solid:** --bg-overlay fill, used for primary actions
- **Danger:** --accent-red border and text on hover

### Inputs

```
Text Input (24px height):
┌─────────────────────────────────────┐
│ placeholder or value                │
└─────────────────────────────────────┘
Border: 1px --border-muted, --border-std on focus
Background: --bg-base
Font: JetBrains Mono, 11px
Padding: 4px 8px
```

### Tabs

```
Tab Bar:
┌────────────────────────────────────────────────────┐
│ [OVERVIEW]  [LOGS]  [CONSOLE]                      │
│ ═══════════                                        │ <- Active indicator (2px line)
└────────────────────────────────────────────────────┘
Height: 24px
Tab font: 10px uppercase
Active: --accent-cyan underline, --text-bright
Inactive: --text-dim
```

### Panels (Dockview)

```
Panel Frame:
┌─ PANEL TITLE ──────────────────────────────────[×]─┐
│                                                    │
│              Panel content area                    │
│                                                    │
└────────────────────────────────────────────────────┘
Header: 24px, --bg-elevated
Title: 10px uppercase, --text-dim
Close button: 12px icon, appears on hover
Border: 1px --border-subtle
```

### Toggle Groups

```
Mode Toggle:
┌────┬────┐
│ A  │ B  │
└────┴────┘
Active segment: --bg-overlay, --text-bright
Inactive: transparent, --text-muted
Border: 1px --border-muted around group
No internal borders
```

### Dropdowns/Selects

```
Select (24px height):
┌───────────────────────────────────┬───┐
│ Selected Option                   │ ▼ │
└───────────────────────────────────┴───┘

Dropdown Menu:
┌───────────────────────────────────────┐
│ Option 1                              │
├───────────────────────────────────────┤
│ Option 2                          [✓] │
├───────────────────────────────────────┤
│ Option 3                              │
└───────────────────────────────────────┘
Background: --bg-elevated
Border: 1px --border-std
Item height: 24px
Hover: --bg-overlay
Selected: checkmark or --accent-cyan indicator
```

---

## Layout Architecture

### Dockview-Based Workspace

The application uses Dockview for a fully configurable panel layout:

```
┌─ TOOLBAR ───────────────────────────────────────────────────────────────────┐
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         DOCKVIEW WORKSPACE                                  │
│                                                                             │
│  ┌──────────────┐  ┌────────────────────────────────────┐  ┌─────────────┐ │
│  │              │  │                                    │  │             │ │
│  │   Panel A    │  │           Panel B                  │  │  Panel C    │ │
│  │   (tabs)     │  │           (tabs)                   │  │  (tabs)     │ │
│  │              │  │                                    │  │             │ │
│  │              │  ├────────────────────────────────────┤  │             │ │
│  │              │  │           Panel D                  │  │             │ │
│  │              │  │           (tabs)                   │  │             │ │
│  └──────────────┘  └────────────────────────────────────┘  └─────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layout Features

| Feature             | Description                                  |
| ------------------- | -------------------------------------------- |
| **Drag-to-dock**    | Drag panel headers to rearrange              |
| **Tabbed groups**   | Multiple panels in same area as tabs         |
| **Floating panels** | Detach panels as floating windows within app |
| **Popout windows**  | Detach panels to separate browser windows    |
| **Resize**          | Drag borders to resize panel areas           |
| **Collapse**        | Minimize panels to bar                       |
| **Serialization**   | Save/restore complete layout state           |

### Named Layouts

Users can save and switch between named layouts:

| Layout          | Description                             | Use Case             |
| --------------- | --------------------------------------- | -------------------- |
| **Default**     | Balanced three-column with field center | General development  |
| **Debugging**   | Large field + logs + timeline           | Investigating issues |
| **Match**       | Maximum field, minimal chrome           | Competition          |
| **Development** | Code editor prominent                   | Writing strategies   |

Layouts stored in localStorage, selectable from toolbar dropdown.

### Default Layout

```
┌─ TOOLBAR ───────────────────────────────────────────────────────────────────┐
├────────────────┬────────────────────────────────────┬───────────────────────┤
│                │                                    │                       │
│  GAME          │                                    │   PLAYER              │
│  CONTROLLER    │            FIELD                   │   INSPECTOR           │
│                │                                    │                       │
│ ─ ─ ─ ─ ─ ─ ─ ─│                                    │                       │
│                │                                    │                       │
│  TEAM          ├────────────────────────────────────┤                       │
│  OVERVIEW      │  [LOGS] [TIMELINE] [CONSOLE]       │                       │
│                │                                    │                       │
└────────────────┴────────────────────────────────────┴───────────────────────┘
```

---

## Toolbar Design

### Toolbar Specification

Height: 32px  
Background: --bg-surface  
Border: 1px --border-subtle bottom

### Toolbar Layout

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ ◆ │ [SIM│LIV] │ [▶ ⏸ ⏹] │ BLU→ ●YEL │ ════════════════════ │ [Default ▼] │ ⚙ │ ● 16ms │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
 │        │          │           │                │               │       │      │
 │        │          │           │                │               │       │      └─ Status cluster
 │        │          │           │                │               │       └─ Settings button
 │        │          │           │                │               └─ Layout selector
 │        │          │           │                └─ Spacer
 │        │          │           └─ Team/Side indicator
 │        │          └─ Executor controls
 │        └─ Mode toggle
 └─ Logo
```

### Toolbar Elements

#### Logo (24x24)

- Minimal brand mark
- Click: no action (just branding)

#### Mode Toggle

```
┌─────┬─────┐
│ SIM │ LIV │
└─────┴─────┘
```

- Two-segment toggle
- SIM: Simulation mode
- LIV: Live mode (disabled if unavailable)
- Active segment: --accent-green background
- Width: ~64px

#### Executor Controls

```
┌───┬───┬───┐
│ ▶ │ ⏸ │ ⏹ │
└───┴───┴───┘
```

- Icon buttons, 24x24 each
- Play: --accent-green when running
- Pause: --accent-amber when paused
- Stop: --accent-red on hover
- Width: ~80px

#### Team/Side Indicator

```
┌─────────────────┐
│ BLU→     ●YEL   │
└─────────────────┘
```

- Shows primary team and side assignment in one compact element
- "BLU→" or "YEL→" indicates which team attacks positive X (right side)
- "●" prefix indicates primary team (the one we're focused on)
- Click: cycles primary team
- Ctrl+Click or right-click: opens quick side-swap menu
- Colors: team color for text
- Width: ~100px

**States:**
| Primary | Attacking +X | Display |
|---------|--------------|---------|
| Blue | Blue | `●BLU→  YEL` |
| Blue | Yellow | `●BLU  →YEL` |
| Yellow | Blue | `BLU→  ●YEL` |
| Yellow | Yellow | `BLU  →●YEL` |

#### Layout Selector

```
┌──────────────┬───┐
│ Default      │ ▼ │
└──────────────┴───┘
```

- Dropdown showing current layout name
- Options: saved layouts + "Manage Layouts..."
- Width: ~120px

#### Settings Button

```
┌───┐
│ ⚙ │
└───┘
```

- Single icon button
- Opens Settings modal
- 24x24

#### Status Cluster

```
┌───────────────┐
│ ● RUNNING 16ms│
└───────────────┘
```

- Connection indicator (● green/red)
- Status text: RUNNING / STOPPED / ERROR
- Latency: dt in milliseconds
- Status color: --accent-green (running), --text-muted (stopped), --accent-red (error)
- Width: ~100px

### Toolbar Total Width Breakdown

| Element           | Width         |
| ----------------- | ------------- |
| Logo              | 32px          |
| Gap               | 8px           |
| Mode toggle       | 64px          |
| Gap               | 8px           |
| Executor controls | 80px          |
| Gap               | 8px           |
| Team indicator    | 100px         |
| Spacer            | flex          |
| Layout selector   | 120px         |
| Gap               | 8px           |
| Settings          | 32px          |
| Gap               | 8px           |
| Status            | 100px         |
| **Total fixed**   | ~568px + flex |

Minimum window width: ~800px for toolbar to render properly.

---

## Panel Inventory

### Primary Panels

#### Field Panel

- **Purpose:** Main visualization of robots, ball, field, debug shapes
- **Content:** Canvas-based renderer with interactive elements
- **Always visible:** Typically yes, as central viewport
- **Special:** Contains floating debug layer controls (see Field View section)

#### Game Controller Panel

- **Purpose:** Send game controller commands
- **Content:** Button groups for Halt, Stop, Start, Kickoffs, Penalties, Free Kicks, Ball Placement
- **Layout:** Vertical list of command categories

#### Team Overview Panel

- **Purpose:** Team roster and status at a glance
- **Content:** List of players with ID, status indicator, quick stats
- **Interaction:** Click player to select in Field and open in Player Inspector

#### Player Inspector Panel

- **Purpose:** Detailed view of selected player
- **Content:**
  - Position, velocity, heading (numeric)
  - Current skill and parameters
  - Relevant debug values for this player
  - Manual control toggle and keyboard control
- **Context-sensitive:** Shows "No player selected" when none selected

#### Scenarios Panel

- **Purpose:** Save/load simulation states
- **Content:**
  - "Save Current" button
  - List of saved scenarios (name, timestamp)
  - Each scenario: Load, Delete actions
  - Import/Export buttons
- **Simulation only:** Disabled or hidden in live mode

#### Log Console Panel

- **Purpose:** View real-time text logs
- **Content:**
  - Scrolling log entries with timestamp, level, target, message
  - Filter by level (ERROR, WARN, INFO, DEBUG, TRACE)
  - Filter by target/module
  - Search box
  - Clear button
- **Styling:** Monospace, color-coded by level

#### Timeline Panel

- **Purpose:** Control playback when viewing logs
- **Content:**
  - Scrubber bar with current position
  - Play/Pause/Step buttons
  - Speed selector (0.25x, 0.5x, 1x, 2x, 4x)
  - Current time / Total time display
  - Event markers (game state changes)
- **Mode-specific:** Only shown in Playback mode

#### Console Panel (Debug REPL)

- **Purpose:** Command-line interaction for debugging
- **Content:**
  - Input line at bottom
  - Output/history above
  - Command history (up/down arrows)
- **Future feature:** Initially simple, can expand to Rhai scripting

#### Time Series Panel

- **Purpose:** Real-time graphs of numeric values
- **Content:**
  - Selectable metrics
  - Scrolling chart
  - Multiple series support
- **Uses:** Recharts or similar

#### Basestation Panel

- **Purpose:** Hardware status for live mode
- **Content:**
  - Connected robots list
  - Battery levels
  - Connection quality
  - Firmware version
- **Live only:** Hidden or disabled in simulation

### Panel Summary Table

| Panel            | Always Available | Default Position | Priority  |
| ---------------- | ---------------- | ---------------- | --------- |
| Field            | Yes              | Center           | Essential |
| Game Controller  | Yes              | Left             | High      |
| Team Overview    | Yes              | Left             | High      |
| Player Inspector | Yes              | Right            | High      |
| Scenarios        | Simulation only  | Left bottom      | Medium    |
| Log Console      | Yes              | Bottom           | Medium    |
| Timeline         | Playback mode    | Bottom           | Medium    |
| Console          | Yes              | Bottom           | Low       |
| Time Series      | Yes              | Right/Bottom     | Low       |
| Basestation      | Live only        | Left             | Medium    |

---

## Field View

### Field Canvas

The Field panel contains a canvas rendering the soccer field, robots, ball, and debug visualizations.

### Field Toolbar

A minimal toolbar inside the Field panel header:

```
┌─ FIELD ────────────────────────────────────────────────────────────────────────────┐
│                                                                    [⚙] [🔍] [⛶]   │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│                            (canvas content)                                        │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

- **⚙ Config dropdown:** Position display mode (raw/filtered/both), other render options
- **🔍 Zoom controls:** Zoom in/out/reset (or use scroll wheel)
- **⛶ Fullscreen:** Maximize field to fill workspace

### Debug Layers Overlay

Floating collapsible panel within the Field view:

```
┌─ LAYERS ──────────[−]─┐
│ [✓] Player Targets    │
│ [✓] Ball Prediction   │
│ [ ] Potential Fields  │
│ [✓] Strategy Shapes   │
│ [ ] Skill Internals   │
│ ──────────────────────│
│ [All] [None]          │
└───────────────────────┘
```

**Behavior:**

- Positioned in corner of field (user-draggable within field bounds)
- Collapsible to just header bar: `┌─ LAYERS ─[+]─┐`
- Opacity: slightly transparent (~90%) to not fully obscure field
- Toggle all / Toggle none quick buttons
- Each layer maps to debug key prefixes

### Interaction Modes

The Field supports different interaction modes:

| Mode        | Activation         | Behavior                              |
| ----------- | ------------------ | ------------------------------------- |
| **View**    | Default            | Click to select robots, scroll to pan |
| **Edit**    | Hold Alt or toggle | Drag robots/ball to reposition        |
| **Measure** | Hold Shift         | Click-drag to measure distances       |

### Mouse Interactions (View Mode)

- **Click robot:** Select, opens in Player Inspector
- **Click ball:** Select ball
- **Click empty:** Deselect
- **Scroll wheel:** Zoom in/out
- **Right-click:** Context menu (quick actions)

### Context Menu

```
┌─────────────────────────────┐
│ Move Robot Here             │
│ Place Ball Here             │
├─────────────────────────────┤
│ Copy Position               │
│ ────────────────────────────│
│ Set Ball Placement Target   │
└─────────────────────────────┘
```

---

## Settings Modal

### Modal Design

Large, nearly full-screen modal with sidebar navigation (Obsidian-style):

```
┌─ SETTINGS ──────────────────────────────────────────────────────────────────────────────[×]─┐
│                                                                                             │
│  ┌─────────────────────┐  ┌───────────────────────────────────────────────────────────────┐│
│  │                     │  │                                                               ││
│  │  General            │  │                                                               ││
│  │  ─────────────────  │  │                     Settings Content Area                     ││
│  │  Team Configuration │  │                                                               ││
│  │  Player Mapping     │  │                     (varies by selected section)              ││
│  │  ─────────────────  │  │                                                               ││
│  │  Controller         │  │                                                               ││
│  │  Tracker            │  │                                                               ││
│  │  Skills             │  │                                                               ││
│  │  ─────────────────  │  │                                                               ││
│  │  Appearance         │  │                                                               ││
│  │  Keybindings        │  │                                                               ││
│  │                     │  │                                                               ││
│  └─────────────────────┘  └───────────────────────────────────────────────────────────────┘│
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Modal Specifications

- **Size:** 90% viewport width, 85% viewport height, max 1200x800
- **Background:** --bg-elevated
- **Overlay:** Semi-transparent black (#000 @ 60%)
- **Animation:** Fade in, slight scale (0.98 → 1.0)

### Settings Sections

#### General

- Application preferences
- Default layout selection
- Auto-save interval

#### Team Configuration

- Blue team: Active toggle, Strategy path selector
- Yellow team: Active toggle, Strategy path selector
- Side assignment toggle
- Quick swap buttons (colors, sides)

#### Player Mapping

- Table showing Logical ID → Physical ID mapping per team
- Auto-assign button
- Reset to default button

#### Controller Settings

- PID gains
- Velocity limits
- Acceleration limits
- (Existing controller_settings parameters)

#### Tracker Settings

- Filtering parameters
- Prediction settings
- (Existing tracker_settings parameters)

#### Skills Settings

- Skill-specific parameters
- (Existing skill_settings parameters)

#### Appearance

- Theme selection (for future light/dark variants)
- Font size multiplier
- Debug shape opacity
- Field zoom default

#### Keybindings

- List of keyboard shortcuts
- Customizable bindings (future)

### Settings Interaction

- **Sidebar:** Click section to navigate
- **Changes:** Applied immediately (live preview where applicable)
- **Close:** Click ×, press Escape, or click overlay
- **Persistence:** Settings saved to dies-settings.json via backend

---

## Debug Visualization

### Extended Shape Types

New debug shape types beyond the existing Cross, Circle, Line:

| Shape           | Description                         | Use Cases                            |
| --------------- | ----------------------------------- | ------------------------------------ |
| **Trajectory**  | Multi-point path with style options | Ball prediction, planned robot paths |
| **Arrow**       | Directed line with arrowhead        | Velocity vectors, force indicators   |
| **VectorField** | Grid of arrows                      | Potential fields, flow visualization |
| **Heatmap**     | Grid of colored cells               | Scoring zones, threat assessment     |
| **Polygon**     | Arbitrary closed shape              | Zones, regions                       |
| **Text**        | Positioned text label               | Debug values on field                |

### Shape Styling

Each shape type supports:

- Color from DebugColor enum
- Opacity levels
- Layer assignment

### Debug Layers

Layers are inferred from debug key prefixes or explicitly set:

| Layer             | Key Pattern     | Description                  |
| ----------------- | --------------- | ---------------------------- |
| `player.targets`  | `p{id}.target*` | Player target positions      |
| `player.paths`    | `p{id}.path*`   | Player trajectories          |
| `ball.prediction` | `ball.pred*`    | Ball prediction              |
| `strategy`        | `strategy.*`    | Strategy-level shapes        |
| `skills`          | `skill.*`       | Skill-internal visualization |
| `potential`       | `potential.*`   | Potential fields             |
| `custom`          | Default         | Uncategorized shapes         |

Layer visibility persists across sessions.

---

## Libraries & Technologies

### Layout System

**Dockview** (`dockview` pnpm package)

- Zero-dependency layout manager
- React, Vue, and vanilla TypeScript support
- Tabs, groups, floating, popout windows
- Full serialization/deserialization
- Excellent theming support

### Existing Stack (Retained)

| Library       | Purpose                 |
| ------------- | ----------------------- |
| React 18      | UI framework            |
| Vite          | Build tool              |
| Tailwind CSS  | Utility styling         |
| Radix UI      | Accessible primitives   |
| Jotai         | Atomic state management |
| React Query   | Server state            |
| Monaco Editor | Code editing            |
| Recharts      | Charting                |
| Canvas2D      | Field rendering         |

### New Dependencies

| Library                    | Purpose             |
| -------------------------- | ------------------- |
| Dockview                   | Panel layout system |
| @fontsource/jetbrains-mono | UI font (all text)  |

### Styling Approach

- Tailwind CSS with custom theme configuration
- CSS custom properties for design tokens
- Component-level CSS for complex styling
- No CSS-in-JS (keep current approach)

### Potential Future Additions

| Library       | Purpose         | When                                 |
| ------------- | --------------- | ------------------------------------ |
| PixiJS        | WebGL rendering | If Canvas2D performance insufficient |
| cmdk          | Command palette | When command palette is implemented  |
| Framer Motion | Animations      | For enhanced transitions             |

---

## Future Considerations

### Command Palette

A Cmd+K activated overlay for quick access to all actions. Deferred from initial implementation but architecture should allow easy addition.

```
┌─────────────────────────────────────────────────────────────────┐
│ > _                                                             │
├─────────────────────────────────────────────────────────────────┤
│ > Open Panel: Team Overview                                     │
│ > Load Layout: Debugging                                        │
│ > Toggle Debug Layer: Potential Fields                          │
│ > Load Scenario: Penalty Setup                                  │
│ > Team Configuration...                                         │
└─────────────────────────────────────────────────────────────────┘
```

### Simulation Rewind

Ability to step backward in simulation time by restoring snapshots. Requires simulator state serialization at intervals.

### Strategy Debugging Integration

Integration with native debuggers (DAP protocol) for line-by-line strategy debugging. Complex feature requiring IPC coordination.

### Multi-Monitor Support

Dockview's popout window feature enables true multi-monitor workflows. Consider optimizing for this use case.

### Theming

Initial implementation is dark-only. Architecture supports future light theme via CSS custom properties swap.

---

## Appendix: Design Mockups

### Toolbar Mockup (ASCII)

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║ ◆ │ SIM│LIV │ ▶ ⏸ ⏹ │ ●BLU→  YEL │ ══════════════════════════ │ Default ▼ │ ⚙ │ ● 16ms ║
╚═══════════════════════════════════════════════════════════════════════════════════════════╝
```

### Default Layout Mockup (ASCII)

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║ ◆ │ SIM│LIV │ ▶ ⏸ ⏹ │ ●BLU→  YEL │ ══════════════════════════ │ Default ▼ │ ⚙ │ ● 16ms ║
╠═══════════════╦═══════════════════════════════════════════════════╦═══════════════════════╣
║ GAME CTRL     ║                                                   ║ PLAYER INSPECTOR      ║
║ ──────────────║                                                   ║ ─────────────────────║
║ [Halt] [Stop] ║                                                   ║ Player 3 (Blue)       ║
║ [Start]       ║                                                   ║                       ║
║               ║              F I E L D                            ║ Position: 1200, 450   ║
║ Kickoffs:     ║                                             ┌───┐ ║ Velocity: 500, 120    ║
║ [Blue] [Yel]  ║                                             │LAY│ ║ Heading: 45°          ║
║               ║                                             │ERS│ ║                       ║
║ ══════════════║                                             └───┘ ║ Skill: GoTo           ║
║ TEAM OVERVIEW ║                                                   ║ Target: 1500, 600     ║
║ ──────────────║                                                   ║                       ║
║ ● P0  GK      ║═══════════════════════════════════════════════════║                       ║
║ ● P1  DEF     ║ [LOGS] [CONSOLE]                                  ║ [Manual Control]      ║
║ ● P3  MID  ◀──║ 12:45:32 INFO  Executor started                   ║                       ║
║ ○ P5  FWD     ║ 12:45:33 DEBUG GoTo target reached                ║                       ║
║               ║ 12:45:34 WARN  Ball position uncertain            ║                       ║
╚═══════════════╩═══════════════════════════════════════════════════╩═══════════════════════╝
```

### Settings Modal Mockup (ASCII)

```
╔═══════════════════════════════════════════════════════════════════════════════════════════╗
║ SETTINGS                                                                              [×] ║
╠═══════════════════╦═══════════════════════════════════════════════════════════════════════╣
║                   ║                                                                       ║
║ General           ║  TEAM CONFIGURATION                                                   ║
║ ─────────────────║                                                                       ║
║ Team Config    ◀──║  Blue Team                                                           ║
║ Player Mapping    ║  ┌──────────────────────────────────────────────────────────────────┐║
║ ─────────────────║  │ [✓] Active    Strategy: [test-strategy           ▼] [Browse]     │║
║ Controller        ║  └──────────────────────────────────────────────────────────────────┘║
║ Tracker           ║                                                                       ║
║ Skills            ║  Yellow Team                                                          ║
║ ─────────────────║  ┌──────────────────────────────────────────────────────────────────┐║
║ Appearance        ║  │ [ ] Active    Strategy: [none                   ▼] [Browse]     │║
║ Keybindings       ║  └──────────────────────────────────────────────────────────────────┘║
║                   ║                                                                       ║
║                   ║  Side Assignment                                                      ║
║                   ║  ┌────────────────────────────────────────┐                          ║
║                   ║  │ [Blue +X] [Yellow +X]                  │                          ║
║                   ║  └────────────────────────────────────────┘                          ║
║                   ║                                                                       ║
║                   ║  Quick Actions                                                        ║
║                   ║  [Swap Colors] [Swap Sides]                                          ║
║                   ║                                                                       ║
╚═══════════════════╩═══════════════════════════════════════════════════════════════════════╝
```

---

## Implementation Plan

This section outlines a phased approach to implementing the UI redesign. Each phase builds upon the previous one, allowing for incremental testing and validation.

### Current State Summary

**Existing Stack:**

- React 18 + Vite (keep)
- Tailwind CSS with shadcn-style components (reconfigure)
- react-resizable-panels (replace with Dockview)
- Radix UI primitives (keep, restyle)
- Jotai + React Query (keep)
- Canvas2D Field rendering (keep, extend)

**Current Layout:**

- 56px toolbar with text labels, rounded toggle groups
- ResizablePanelGroup with fixed panel positions
- Left: Game Controller + Team Overview with tabs
- Center: Field canvas
- Right: Player Sidebar
- Bottom: Status bar

**Current Styling Issues:**

- Generic shadcn aesthetic with `--radius: 0.5rem`
- Slate color palette, not dark enough
- System/sans-serif fonts, not monospace
- Generous padding (12-16px)
- Conventional component styling

---

### Phase 1: Design Foundation + Tailwind v4 Migration

**Goal:** Migrate to Tailwind CSS v4 and establish the new visual language.

Tailwind v4 introduces a CSS-first configuration approach, eliminating the need for `tailwind.config.js` and simplifying the build setup. For Vite projects, it uses a dedicated plugin instead of PostCSS.

**Tasks:**

#### 1. Migrate to Tailwind CSS v4

**1a. Run the automatic upgrade tool (requires Node.js 20+):**

```bash
cd webui
pnpm dlx @tailwindcss/upgrade
```

This tool will:

- Update dependencies to v4
- Migrate `tailwind.config.js` to CSS-based configuration
- Update template files for renamed/removed utilities

**1b. Install Tailwind v4 Vite plugin and remove PostCSS:**

```bash
pnpm install tailwindcss@latest @tailwindcss/vite@latest
pnpm uninstall postcss autoprefixer
```

**1c. Update `vite.config.ts` to use the Vite plugin:**

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
});
```

**1d. Remove PostCSS configuration:**

- Delete `postcss.config.js`

**1e. Update `index.css` to use new import syntax:**

```css
@import "tailwindcss";
```

#### 2. Install JetBrains Mono font

```bash
pnpm install @fontsource/jetbrains-mono
```

Import in `main.tsx`:

```tsx
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/600.css";
```

#### 3. Define the new theme using `@theme` directive

Replace the old CSS variables and Tailwind config with a CSS-first theme in `index.css`:

```css
@import "tailwindcss";

@theme {
  /* Typography */
  --font-mono: "JetBrains Mono", "IBM Plex Mono", "Fira Code", ui-monospace,
    monospace;

  /* Background Hierarchy */
  --color-bg-void: #08090a;
  --color-bg-base: #0c0d0f;
  --color-bg-surface: #12141a;
  --color-bg-elevated: #1a1d24;
  --color-bg-overlay: #22262f;

  /* Border Hierarchy */
  --color-border-subtle: #1e2228;
  --color-border-muted: #2a2f38;
  --color-border-std: #3a4250;

  /* Text Hierarchy */
  --color-text-muted: #5a6270;
  --color-text-dim: #7a8290;
  --color-text-std: #a0a8b4;
  --color-text-bright: #d0d4dc;
  --color-text-max: #f0f2f5;

  /* Accent Colors */
  --color-accent-cyan: #00d4e5;
  --color-accent-green: #00c853;
  --color-accent-amber: #ffab00;
  --color-accent-red: #ff3d3d;
  --color-accent-blue: #2979ff;

  /* Team Colors */
  --color-team-blue: #2563eb;
  --color-team-blue-dim: #1e40af;
  --color-team-yellow: #eab308;
  --color-team-yellow-dim: #a16207;

  /* Spacing (4px grid) */
  --spacing-1: 4px;
  --spacing-2: 8px;
  --spacing-3: 12px;
  --spacing-4: 16px;

  /* Border Radius (sharp corners) */
  --radius: 0px;
  --radius-sm: 0px;
  --radius-md: 0px;
  --radius-lg: 0px;
}
```

#### 4. Update base styles

Add base layer styles for global defaults:

```css
@layer base {
  * {
    border-color: var(--color-border-muted);
  }

  html {
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.3;
  }

  body {
    background-color: var(--color-bg-base);
    color: var(--color-text-std);
  }
}
```

#### 5. Update scrollbar styling

```css
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: var(--color-bg-surface);
}

::-webkit-scrollbar-thumb {
  background: var(--color-border-std);
}

::-webkit-scrollbar-thumb:hover {
  background: var(--color-text-muted);
}
```

#### 6. Update component usage for v4 syntax

In Tailwind v4, the `theme()` function is replaced with CSS variables. Update components to use the new approach:

```html
<!-- Old v3 with theme() or hsl(var(--x)) -->
<div class="bg-background text-foreground">
  <!-- New v4 with CSS variables -->
  <div class="bg-bg-base text-text-std"></div>
</div>
```

The `--color-*` prefix in `@theme` automatically creates utility classes like `bg-bg-base`, `text-text-std`, `border-border-muted`, etc.

For custom utilities not auto-generated:

```css
@layer utilities {
  .text-uppercase-label {
    text-transform: uppercase;
    font-size: 10px;
    letter-spacing: 0.05em;
  }
}
```

**Affected Files:**

- `webui/package.json` (update tailwindcss, add @tailwindcss/vite, add font, remove postcss/autoprefixer)
- `webui/vite.config.ts` (add Tailwind Vite plugin)
- `webui/src/index.css` (complete overhaul with @theme directive)
- `webui/src/main.tsx` (import JetBrains Mono font)
- `webui/postcss.config.js` (delete)
- `webui/tailwind.config.js` (delete after migration — theme moves to CSS)

**Testing:**

- Build completes without PostCSS errors: `pnpm run build`
- Dev server starts successfully: `pnpm run dev`
- Visual inspection: entire app should use JetBrains Mono
- All corners should be sharp (no rounded edges)
- Background should be darker (#0c0d0f base)
- Text should use new hierarchy (brighter on dark)
- Verify Tailwind utilities still work (inspect elements in browser)
- No functional changes expected

---

### Phase 2: Component Restyling

**Goal:** Update all UI components to match the new design language.

**Tasks:**

1. **Restyle Button component**

   - Remove rounded corners
   - Update size variants (24-28px height)
   - Implement ghost, outline, solid, danger variants
   - Update color states for hover/active
   - Add uppercase text styling option

2. **Restyle Input/Select components**

   - 24px height, compact padding (4px 8px)
   - Sharp borders, dark background
   - Focus ring using --accent-cyan

3. **Restyle Tabs component**

   - 24px height tabs
   - Uppercase 10px font
   - Active indicator: 2px --accent-cyan underline
   - Remove background styling

4. **Restyle Toggle/ToggleGroup**

   - Sharp corners
   - Segment styling without internal borders
   - Active state: --bg-overlay background

5. **Restyle Dialog/Popover**

   - Sharp corners
   - --bg-elevated background
   - --border-std borders

6. **Restyle Card component**

   - Remove shadows
   - Sharp corners
   - Minimal padding
   - Subtle borders

7. **Update Badge, Separator, Switch, Slider**
   - Consistent sharp styling
   - Proper color usage

**Affected Files:**

- `webui/src/components/ui/button.tsx`
- `webui/src/components/ui/input.tsx`
- `webui/src/components/ui/tabs.tsx`
- `webui/src/components/ui/toggle.tsx`
- `webui/src/components/ui/toggle-group.tsx`
- `webui/src/components/ui/select.tsx`
- `webui/src/components/ui/dialog.tsx`
- `webui/src/components/ui/popover.tsx`
- `webui/src/components/ui/card.tsx`
- `webui/src/components/ui/badge.tsx`
- `webui/src/components/ui/separator.tsx`
- `webui/src/components/ui/switch.tsx`
- `webui/src/components/ui/slider.tsx`
- `webui/src/components/ui/context-menu.tsx`
- `webui/src/components/ui/tooltip.tsx`

**Testing:**

- Visual inspection of all components
- Interactive states working (hover, focus, active)
- Component sizes match spec (24-28px heights)
- Color contrast readable
- No functionality regressions

---

### Phase 3: Toolbar Redesign

**Goal:** Implement the new compact 32px toolbar with icon-based controls.

**Tasks:**

1. **Create new Toolbar component structure**

   - 32px height container
   - Flex layout with proper spacing
   - --bg-surface background with bottom border

2. **Implement Logo section**

   - 24x24 minimal brand mark
   - Update or simplify existing logo

3. **Implement Mode Toggle (SIM/LIV)**

   - Compact two-segment toggle (~64px)
   - --accent-green for active segment
   - Disabled state for unavailable live mode

4. **Implement Executor Controls**

   - Icon-only buttons (Play, Pause, Stop)
   - 24x24 each in a grouped container
   - Color states (green running, amber paused, red stop hover)

5. **Implement Team/Side Indicator**

   - Compact combined display (~100px)
   - Shows primary team and attack direction
   - Click to cycle, right-click for quick menu
   - Replace TeamSettingsDialog trigger + TeamSwapControls + PrimaryTeamSelector

6. **Implement Layout Selector dropdown**

   - Placeholder for Phase 4 (shows "Default")
   - ~120px width

7. **Implement Settings button**

   - 24x24 icon button
   - Placeholder trigger for Phase 5 modal

8. **Implement Status Cluster**
   - Connection indicator dot
   - Status text (RUNNING/STOPPED/ERROR)
   - Latency display

**Affected Files:**

- `webui/src/App.tsx` (replace toolbar section)
- `webui/src/components/Toolbar.tsx` (new file)
- `webui/src/components/ModeToggle.tsx` (new file)
- `webui/src/components/ExecutorControls.tsx` (new file)
- `webui/src/components/TeamIndicator.tsx` (new file)
- `webui/src/components/StatusCluster.tsx` (new file)
- Remove/refactor: `TeamSwapControls.tsx`, `PrimaryTeamSelector.tsx`

**Testing:**

- Toolbar renders at 32px height
- All controls functional (mode switch, play/pause/stop)
- Team indicator shows correct state
- Status updates in real-time
- Responsive layout doesn't break at 800px minimum

---

### Phase 4: Dockview Layout System

**Goal:** Replace react-resizable-panels with Dockview for full panel flexibility.

**Tasks:**

1. **Install and configure Dockview**

   - Add `dockview` package
   - Remove `react-resizable-panels`
   - Create Dockview theme CSS matching design system

2. **Create DockviewWrapper component**

   - Initialize Dockview with custom theme
   - Handle panel registration
   - Manage layout serialization/deserialization

3. **Create Panel wrapper components**

   - Standard panel header (24px, title, close button)
   - Panel content area with proper styling
   - Hover effects for close button

4. **Migrate existing panels to Dockview**

   - FieldPanel (central viewport)
   - GameControllerPanel (left)
   - TeamOverviewPanel (left, with tabs for Team/Basestation)
   - PlayerInspectorPanel (right, rename from PlayerSidebar)
   - SettingsPanel (bottom, for Controller/Tracker/Skill settings)

5. **Implement default layout**

   - Three-column layout matching spec
   - Field as central focal point
   - Proper default sizes

6. **Implement layout persistence**

   - Save layout to localStorage on change
   - Restore on app load
   - Reset to default option

7. **Implement Layout Selector**
   - Connect to toolbar dropdown
   - Save current layout with name
   - Switch between saved layouts
   - "Manage Layouts..." option

**Affected Files:**

- `webui/package.json` (add dockview, remove react-resizable-panels)
- `webui/src/App.tsx` (replace ResizablePanelGroup with Dockview)
- `webui/src/components/DockviewWrapper.tsx` (new file)
- `webui/src/components/DockPanel.tsx` (new file)
- `webui/src/components/LayoutSelector.tsx` (new file)
- `webui/src/index.css` (add Dockview theme styles)
- `webui/src/views/*.tsx` (adapt for panel system)

**Testing:**

- Panels render correctly in Dockview
- Drag-to-dock works (rearrange panels)
- Tabs work within panel groups
- Resize by dragging borders works
- Layout persists across page reloads
- Layout selector shows saved layouts
- "Default" layout can be restored

---

### Phase 5: Settings Modal & New Panels

**Goal:** Implement the Obsidian-style settings modal and additional panels.

**Tasks:**

1. **Create Settings Modal component**

   - Large modal (90% viewport, max 1200x800)
   - Sidebar navigation
   - Content area for each section
   - Fade + scale animation

2. **Implement Settings sections**

   - General: application preferences
   - Team Configuration: active teams, strategy paths, side assignment
   - Player Mapping: logical-to-physical ID table
   - Controller: existing controller_settings
   - Tracker: existing tracker_settings
   - Skills: existing skill_settings
   - Appearance: theme, font size, debug opacity
   - Keybindings: shortcut list

3. **Migrate existing settings UI**

   - Move SettingsEditor content to modal sections
   - Move TeamSettingsDialog content to Team Configuration
   - Maintain API integration for saving settings

4. **Create Log Console Panel**

   - Scrolling log entries with timestamp, level, target, message
   - Color-coded by level (ERROR red, WARN amber, INFO cyan, DEBUG dim)
   - Filter controls (by level, by target, search)
   - Clear button
   - Monospace styling

5. **Create Scenarios Panel (simulation only)**

   - Save Current button
   - List of saved scenarios
   - Load/Delete actions per scenario
   - Import/Export functionality
   - Integrate with backend scenario API

6. **Create Console Panel (Debug REPL)**
   - Input line at bottom
   - Output/history above
   - Command history (up/down keys)
   - Basic command execution

**Affected Files:**

- `webui/src/components/SettingsModal.tsx` (new file)
- `webui/src/components/SettingsSidebar.tsx` (new file)
- `webui/src/components/settings/*.tsx` (new section components)
- `webui/src/views/LogConsole.tsx` (new file)
- `webui/src/views/ScenariosPanel.tsx` (new file)
- `webui/src/views/ConsolePanel.tsx` (new file)
- `webui/src/App.tsx` (add Settings modal trigger)
- Remove: `webui/src/components/TeamSettingsDialog.tsx`

**Testing:**

- Settings modal opens/closes correctly
- All settings sections accessible via sidebar
- Settings changes apply and persist
- Log console shows real-time logs
- Log filtering works
- Scenarios panel (sim mode only) can save/load
- Console panel accepts input

---

### Phase 6: Field Enhancements & Polish

**Goal:** Enhance field visualization, add debug layers, and final polish.

**Tasks:**

1. **Implement Debug Layers overlay**

   - Floating collapsible panel within Field
   - Checkbox list of debug layers
   - Toggle All / Toggle None buttons
   - Persist visibility state
   - Position draggable within field bounds

2. **Update Field toolbar**

   - Move settings popover to header icons
   - Add zoom controls (in/out/reset)
   - Add fullscreen toggle

3. **Extend debug shape rendering**

   - Add Trajectory shape type
   - Add Arrow shape type
   - Add Polygon shape type
   - Add Text label shape type
   - Implement layer filtering in renderer

4. **Implement debug layer categorization**

   - Parse key prefixes (player.targets, ball.prediction, etc.)
   - Assign shapes to layers automatically
   - Filter rendering based on layer visibility

5. **Polish Field interactions**

   - Improve tooltip styling
   - Update context menu styling
   - Coordinate display in new style

6. **Final visual polish**

   - Consistent spacing throughout
   - Micro-interaction feedback
   - Loading/error states
   - Empty states for panels
   - Status bar update to match design

7. **Performance optimization**
   - Review render cycles
   - Optimize Dockview configuration
   - Canvas rendering efficiency

**Affected Files:**

- `webui/src/views/Field.tsx` (add layers overlay, update toolbar)
- `webui/src/views/FieldRenderer.ts` (extend shape types, add layer filtering)
- `webui/src/components/DebugLayersOverlay.tsx` (new file)
- `webui/src/components/FieldToolbar.tsx` (new file)
- `webui/src/App.tsx` (status bar update)
- Various component files (polish pass)

**Testing:**

- Debug layers overlay toggles correctly
- Layer visibility affects shape rendering
- New shape types render correctly
- Zoom controls work
- Fullscreen toggle works
- Overall visual consistency check
- Performance acceptable (smooth 60fps canvas)

---

### Implementation Notes

**Parallel Work:**

- Phase 1-2 can be done by one person while another researches Dockview integration
- Phase 5 panels (Log Console, Scenarios, Console) can be developed in parallel

**Risk Areas:**

- Dockview integration complexity (Phase 4) — allocate extra time
- Settings migration (Phase 5) — ensure no settings loss
- Debug layer performance with many shapes (Phase 6)

**Testing Strategy:**

- Each phase should end with a working application
- Visual regression testing recommended after Phase 2
- Integration testing after Phase 4 (layout persistence)
- End-to-end testing after Phase 6

**Rollback Points:**

- After Phase 2: can revert to old layout if Dockview fails
- After Phase 4: full layout system working, can defer Phase 5-6

---

### Phase Dependencies

```
Phase 1: Design Foundation
    ↓
Phase 2: Component Restyling
    ↓
Phase 3: Toolbar Redesign ─────────┐
    ↓                              │
Phase 4: Dockview Layout ←─────────┘
    ↓
Phase 5: Settings Modal & Panels
    ↓
Phase 6: Field Enhancements & Polish
```

---

_End of Design Document_
