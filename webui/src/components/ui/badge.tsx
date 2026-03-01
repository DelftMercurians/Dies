import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/**
 * Badge component following the Dies mission control aesthetic.
 *
 * Design specs:
 * - Sharp corners (no rounded-full)
 * - Compact sizing
 * - Color variants for status indication
 */
const badgeVariants = cva(
  // Base styles - sharp corners, compact
  "inline-flex items-center border px-1.5 py-0.5 text-sm font-semibold uppercase tracking-wide transition-colors focus:outline-none focus:ring-1 focus:ring-accent-cyan",
  {
    variants: {
      variant: {
        // Default: neutral, subdued
        default:
          "border-border-muted bg-bg-overlay text-text-std",
        // Status variants
        success:
          "border-accent-green/50 bg-accent-green/20 text-accent-green",
        warning:
          "border-accent-amber/50 bg-accent-amber/20 text-accent-amber",
        destructive:
          "border-accent-red/50 bg-accent-red/20 text-accent-red",
        info:
          "border-accent-cyan/50 bg-accent-cyan/20 text-accent-cyan",
        // Outline: just border
        outline:
          "border-border-muted bg-transparent text-text-dim",
        // Team colors
        "team-blue":
          "border-team-blue/50 bg-team-blue/20 text-team-blue",
        "team-yellow":
          "border-team-yellow/50 bg-team-yellow/20 text-team-yellow",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
