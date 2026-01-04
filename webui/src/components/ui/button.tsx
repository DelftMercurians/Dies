import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

/**
 * Button component following the Dies mission control aesthetic.
 * 
 * Design specs:
 * - Sharp corners (no border-radius)
 * - Compact heights: default 28px, sm 24px, xs 20px
 * - Icon buttons: 24x24 or 20x20
 * - Uppercase text option for labels
 */
const buttonVariants = cva(
  // Base styles: sharp corners, compact, functional
  "inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50",
  {
    variants: {
      variant: {
        // Default: subtle solid background
        default:
          "bg-bg-overlay text-text-bright hover:bg-bg-elevated hover:text-text-max border border-border-muted",
        // Primary: emphasized action
        primary:
          "bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/50 hover:bg-accent-cyan/30 hover:border-accent-cyan",
        // Destructive: danger action
        destructive:
          "bg-transparent text-text-std border border-border-muted hover:bg-accent-red/20 hover:text-accent-red hover:border-accent-red",
        // Outline: bordered, transparent fill
        outline:
          "border border-border-muted bg-transparent text-text-std hover:bg-bg-overlay hover:text-text-bright",
        // Ghost: no border, appears on hover
        ghost:
          "bg-transparent text-text-dim hover:bg-bg-overlay hover:text-text-std",
        // Link: text with underline
        link: "text-accent-cyan underline-offset-4 hover:underline",
        // Success: positive action
        success:
          "bg-accent-green/20 text-accent-green border border-accent-green/50 hover:bg-accent-green/30 hover:border-accent-green",
      },
      size: {
        // Default: 28px height
        default: "h-7 px-3 py-1 text-[11px]",
        // Small: 24px height
        sm: "h-6 px-2 py-0.5 text-[10px]",
        // Extra small: 20px height
        xs: "h-5 px-1.5 py-0.5 text-[9px]",
        // Large: 32px height
        lg: "h-8 px-4 py-1.5 text-[11px]",
        // Icon: square buttons
        icon: "h-6 w-6 p-0",
        "icon-sm": "h-5 w-5 p-0",
        "icon-xs": "h-4 w-4 p-0",
      },
      uppercase: {
        true: "uppercase tracking-wide",
        false: "",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
      uppercase: false,
    },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, uppercase, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, uppercase, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

export { Button, buttonVariants };
