import * as React from "react";
import * as TabsPrimitive from "@radix-ui/react-tabs";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/**
 * Tabs component following the Dies mission control aesthetic.
 *
 * Design specs:
 * - Tab height: 24px
 * - Uppercase 10px font
 * - Active indicator: 2px accent-cyan underline
 * - No background styling, minimal
 */

const tabsVariants = cva("", {
  variants: {
    size: {
      default: "",
      sm: "",
      xs: "",
    },
  },
  defaultVariants: {
    size: "default",
  },
});

const tabsListVariants = cva(
  "inline-flex items-end border-b border-border-subtle gap-0",
  {
    variants: {
      size: {
        default: "h-6",
        sm: "h-5",
        xs: "h-4",
      },
    },
    defaultVariants: {
      size: "default",
    },
  }
);

const tabsTriggerVariants = cva(
  // Base: uppercase label style, sharp, minimal
  [
    "inline-flex items-center justify-center",
    "font-semibold uppercase tracking-wider",
    "transition-colors",
    "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan",
    "disabled:pointer-events-none disabled:opacity-50",
    // Text colors
    "text-text-muted hover:text-text-dim",
    // Active state: cyan underline, bright text
    "data-[state=active]:text-text-bright",
    "data-[state=active]:border-b-2 data-[state=active]:border-accent-cyan",
    "data-[state=active]:-mb-px",
    // Relative for the underline positioning
    "relative",
  ],
  {
    variants: {
      size: {
        default: "px-3 h-6 text-[10px]",
        sm: "px-2 h-5 text-[9px]",
        xs: "px-1.5 h-4 text-[8px]",
      },
    },
    defaultVariants: {
      size: "default",
    },
  }
);

export interface TabsProps
  extends React.ComponentPropsWithoutRef<typeof TabsPrimitive.Root>,
    VariantProps<typeof tabsVariants> {}

const TabsContext = React.createContext<VariantProps<typeof tabsVariants>>({
  size: "default",
});

const Tabs = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Root>,
  TabsProps
>(({ className, size = "default", ...props }, ref) => (
  <TabsContext.Provider value={{ size }}>
    <TabsPrimitive.Root
      ref={ref}
      className={cn(tabsVariants({ size, className }))}
      {...props}
    />
  </TabsContext.Provider>
));
Tabs.displayName = TabsPrimitive.Root.displayName;

const TabsList = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.List>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.List>
>(({ className, ...props }, ref) => {
  const { size } = React.useContext(TabsContext);
  return (
    <TabsPrimitive.List
      ref={ref}
      className={cn(tabsListVariants({ size }), className)}
      {...props}
    />
  );
});
TabsList.displayName = TabsPrimitive.List.displayName;

const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Trigger>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Trigger>
>(({ className, ...props }, ref) => {
  const { size } = React.useContext(TabsContext);
  return (
    <TabsPrimitive.Trigger
      ref={ref}
      className={cn(tabsTriggerVariants({ size }), className)}
      {...props}
    />
  );
});
TabsTrigger.displayName = TabsPrimitive.Trigger.displayName;

const TabsContent = React.forwardRef<
  React.ElementRef<typeof TabsPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TabsPrimitive.Content>
>(({ className, ...props }, ref) => (
  <TabsPrimitive.Content
    ref={ref}
    className={cn(
      "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan",
      className
    )}
    {...props}
  />
));
TabsContent.displayName = TabsPrimitive.Content.displayName;

export { Tabs, TabsList, TabsTrigger, TabsContent };
