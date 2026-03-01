import * as React from "react";
import * as ToggleGroupPrimitive from "@radix-ui/react-toggle-group";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/**
 * Toggle Group component following the Dies mission control aesthetic.
 *
 * Design specs:
 * - Sharp corners
 * - No internal borders between items
 * - Single border around the group
 * - Active segment: bg-overlay background
 */

const toggleGroupVariants = cva(
  // Base: flex container with border around the whole group
  "inline-flex items-center border border-border-muted",
  {
    variants: {
      variant: {
        default: "bg-transparent",
        outline: "bg-bg-surface",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

const toggleGroupItemVariants = cva(
  // Base styles
  [
    "inline-flex items-center justify-center",
    "font-medium transition-colors",
    "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:z-10",
    "disabled:pointer-events-none disabled:opacity-50",
    // Default state
    "text-text-muted hover:text-text-std hover:bg-bg-overlay",
    // Active state
    "data-[state=on]:bg-bg-overlay data-[state=on]:text-text-bright",
    // First and last item styling (no double borders)
    "first:border-l-0 last:border-r-0",
    // Border between items
    "border-r border-border-subtle last:border-r-0",
  ],
  {
    variants: {
      size: {
        default: "h-6 px-3 text-sm",
        sm: "h-5 px-2 text-sm",
        lg: "h-7 px-4",
      },
    },
    defaultVariants: {
      size: "default",
    },
  }
);

const ToggleGroupContext = React.createContext<{
  size?: "default" | "sm" | "lg";
  variant?: "default" | "outline";
}>({
  size: "default",
  variant: "default",
});

const ToggleGroup = React.forwardRef<
  React.ElementRef<typeof ToggleGroupPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ToggleGroupPrimitive.Root> &
    VariantProps<typeof toggleGroupVariants> & {
      size?: "default" | "sm" | "lg";
    }
>(({ className, variant = "default", size = "default", children, ...props }, ref) => (
  <ToggleGroupPrimitive.Root
    ref={ref}
    className={cn(toggleGroupVariants({ variant }), className)}
    {...props}
  >
    <ToggleGroupContext.Provider value={{ variant: variant ?? undefined, size }}>
      {children}
    </ToggleGroupContext.Provider>
  </ToggleGroupPrimitive.Root>
));

ToggleGroup.displayName = ToggleGroupPrimitive.Root.displayName;

const ToggleGroupItem = React.forwardRef<
  React.ElementRef<typeof ToggleGroupPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof ToggleGroupPrimitive.Item> &
    VariantProps<typeof toggleGroupItemVariants>
>(({ className, children, size, ...props }, ref) => {
  const context = React.useContext(ToggleGroupContext);

  return (
    <ToggleGroupPrimitive.Item
      ref={ref}
      className={cn(
        toggleGroupItemVariants({
          size: size || context.size,
        }),
        className
      )}
      {...props}
    >
      {children}
    </ToggleGroupPrimitive.Item>
  );
});

ToggleGroupItem.displayName = ToggleGroupPrimitive.Item.displayName;

export { ToggleGroup, ToggleGroupItem };
