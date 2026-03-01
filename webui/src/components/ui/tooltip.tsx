import * as React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";

import { cn } from "@/lib/utils";

/**
 * Tooltip component following the Dies mission control aesthetic.
 *
 * Design specs:
 * - Sharp corners
 * - bg-elevated background, border-std borders
 * - Compact padding
 * - Fast delay (200ms)
 */

const TooltipProvider = TooltipPrimitive.Provider;

const Tooltip: React.FC<React.ComponentProps<typeof TooltipPrimitive.Root>> = ({
  ...props
}) => <TooltipPrimitive.Root delayDuration={200} {...props} />;

const TooltipTrigger = TooltipPrimitive.Trigger;

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Portal>
    <TooltipPrimitive.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        "z-50 overflow-hidden",
        // Sharp corners, elevated bg, std border
        "border border-border-std bg-bg-elevated px-2 py-1 text-sm text-text-std",
        // Animations
        "animate-in fade-in-0 zoom-in-95",
        "data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95",
        "data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2",
        "data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",
        className
      )}
      {...props}
    />
  </TooltipPrimitive.Portal>
));
TooltipContent.displayName = TooltipPrimitive.Content.displayName;

const SimpleTooltip: React.FC<
  React.ComponentProps<typeof TooltipPrimitive.Root> & {
    title: string;
    className?: string;
  }
> = ({ children, className, ...props }) => (
  <Tooltip {...props}>
    <TooltipTrigger asChild>
      <div className={className}>{children}</div>
    </TooltipTrigger>
    <TooltipContent>{props.title}</TooltipContent>
  </Tooltip>
);

export {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  TooltipProvider,
  SimpleTooltip,
};
