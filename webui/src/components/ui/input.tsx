import * as React from "react";

import { cn } from "@/lib/utils";

/**
 * Input component following the Dies mission control aesthetic.
 *
 * Design specs:
 * - Default height: 24px
 * - Compact padding: 4px 8px
 * - Sharp borders
 * - Dark background (bg-base)
 * - Focus ring using accent-cyan
 */
export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  inputSize?: "default" | "sm" | "xs";
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, type, inputSize = "default", ...props }, ref) => {
    const sizeClasses = {
      default: "h-6 px-2 py-1",
      sm: "h-5 px-1.5 py-0.5 text-sm",
      xs: "h-4 px-1 py-0.5 text-sm",
    };

    return (
      <input
        type={type}
        className={cn(
          // Base styles
          "flex w-full border bg-bg-base text-text-std",
          "border-border-muted",
          // Placeholder
          "placeholder:text-text-muted",
          // Focus state - cyan ring
          "focus:outline-none focus:border-accent-cyan focus:ring-1 focus:ring-accent-cyan/50",
          // Disabled state
          "disabled:cursor-not-allowed disabled:opacity-50 disabled:bg-bg-surface",
          // File input
          "file:border-0 file:bg-transparent file:text-sm file:font-medium file:text-text-dim",
          // Size
          sizeClasses[inputSize],
          className
        )}
        ref={ref}
        {...props}
      />
    );
  }
);
Input.displayName = "Input";

export { Input };
