import * as React from "react";
import * as TogglePrimitive from "@radix-ui/react-toggle";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/**
 * Toggle component following the Dies mission control aesthetic.
 *
 * Design specs:
 * - Sharp corners
 * - Compact sizing
 * - Active state: bg-overlay background, bright text
 */
const toggleVariants = cva(
  // Base styles
  [
    "inline-flex items-center justify-center",
    "font-medium transition-colors",
    "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan",
    "disabled:pointer-events-none disabled:opacity-50",
    // Default state
    "text-text-dim hover:text-text-std hover:bg-bg-overlay",
    // Active/pressed state
    "data-[state=on]:bg-bg-overlay data-[state=on]:text-text-bright",
  ],
  {
    variants: {
      variant: {
        default: "bg-transparent",
        outline:
          "border border-border-muted bg-transparent hover:border-border-std data-[state=on]:border-border-std",
      },
      size: {
        default: "h-6 px-2 text-sm",
        sm: "h-5 px-1.5 text-sm",
        lg: "h-7 px-3",
        icon: "h-6 w-6",
        "icon-sm": "h-5 w-5",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
);

const Toggle = React.forwardRef<
  React.ElementRef<typeof TogglePrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof TogglePrimitive.Root> &
    VariantProps<typeof toggleVariants>
>(({ className, variant, size, ...props }, ref) => (
  <TogglePrimitive.Root
    ref={ref}
    className={cn(toggleVariants({ variant, size, className }))}
    {...props}
  />
));

Toggle.displayName = TogglePrimitive.Root.displayName;

export { Toggle, toggleVariants };
