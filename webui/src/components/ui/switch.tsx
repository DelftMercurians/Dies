import * as React from "react";
import * as SwitchPrimitives from "@radix-ui/react-switch";

import { cn } from "@/lib/utils";

/**
 * Switch component following the Dies mission control aesthetic.
 *
 * Design specs:
 * - Sharp corners (squared off, not rounded)
 * - Compact sizing
 * - Cyan accent when checked
 */

const Switch = React.forwardRef<
  React.ElementRef<typeof SwitchPrimitives.Root>,
  React.ComponentPropsWithoutRef<typeof SwitchPrimitives.Root>
>(({ className, ...props }, ref) => (
  <SwitchPrimitives.Root
    className={cn(
      // Base: inline flex, sharp corners
      "peer inline-flex h-4 w-7 shrink-0 cursor-pointer items-center",
      // Border and background
      "border border-border-muted",
      "transition-colors",
      // Focus state
      "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan",
      // Disabled state
      "disabled:cursor-not-allowed disabled:opacity-50",
      // States
      "data-[state=checked]:bg-accent-cyan/30 data-[state=checked]:border-accent-cyan",
      "data-[state=unchecked]:bg-bg-surface",
      className
    )}
    {...props}
    ref={ref}
  >
    <SwitchPrimitives.Thumb
      className={cn(
        // Sharp corners, compact thumb
        "pointer-events-none block h-3 w-3",
        "transition-transform",
        // States
        "data-[state=checked]:translate-x-3 data-[state=unchecked]:translate-x-0.5",
        "data-[state=checked]:bg-accent-cyan data-[state=unchecked]:bg-text-dim"
      )}
    />
  </SwitchPrimitives.Root>
));
Switch.displayName = SwitchPrimitives.Root.displayName;

export { Switch };
