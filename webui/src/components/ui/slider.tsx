import * as React from "react";
import * as SliderPrimitive from "@radix-ui/react-slider";

import { cn } from "@/lib/utils";

/**
 * Slider component following the Dies mission control aesthetic.
 *
 * Design specs:
 * - Sharp corners
 * - Compact track height
 * - Cyan accent for range and thumb
 */

const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn(
      "relative flex w-full touch-none select-none items-center",
      className
    )}
    {...props}
  >
    <SliderPrimitive.Track
      className={cn(
        // Sharp corners, compact height
        "relative h-1 w-full grow overflow-hidden",
        "bg-bg-overlay border border-border-subtle"
      )}
    >
      <SliderPrimitive.Range className="absolute h-full bg-accent-cyan" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb
      className={cn(
        // Sharp corners, compact thumb
        "block h-3 w-3",
        "border border-accent-cyan bg-accent-cyan",
        "transition-colors",
        // Focus state
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-accent-cyan focus-visible:ring-offset-1 focus-visible:ring-offset-bg-base",
        // Disabled state
        "disabled:pointer-events-none disabled:opacity-50",
        // Hover
        "hover:bg-accent-cyan/80"
      )}
    />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;

export { Slider };
