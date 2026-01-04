import * as React from "react";
import * as LabelPrimitive from "@radix-ui/react-label";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/**
 * Label component following the Dies mission control aesthetic.
 *
 * Design specs:
 * - Compact font size (10-11px)
 * - Text hierarchy colors
 * - Optional uppercase variant
 */
const labelVariants = cva(
  "text-[11px] font-medium leading-none text-text-dim peer-disabled:cursor-not-allowed peer-disabled:opacity-70",
  {
    variants: {
      variant: {
        default: "",
        uppercase: "text-[10px] uppercase tracking-wider",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

const Label = React.forwardRef<
  React.ElementRef<typeof LabelPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof LabelPrimitive.Root> &
    VariantProps<typeof labelVariants>
>(({ className, variant, ...props }, ref) => (
  <LabelPrimitive.Root
    ref={ref}
    className={cn(labelVariants({ variant }), className)}
    {...props}
  />
));
Label.displayName = LabelPrimitive.Root.displayName;

export { Label };
