import { type ClassValue, clsx } from "clsx";
import React, { Ref } from "react";
import { twMerge } from "tailwind-merge";
import { useResizeObserver } from "./useResizeObserver";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export const prettyPrintSnakeCases = (s: string): string =>
  s
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

export const useIsOverflow = (
  ref: React.MutableRefObject<HTMLElement | null>,
  orientation: "horizontal" | "vertical" = "horizontal"
) => {
  const [isOverflow, setIsOverflow] = React.useState(false);
  useResizeObserver({
    ref,
    onResize: () => {
      if (ref.current) {
        if (orientation === "horizontal") {
          const childrenTotalWidth = Array.from(ref.current.children).reduce(
            (acc, child) => acc + child.clientWidth,
            0
          );
          const hasOverflow = childrenTotalWidth > ref.current.clientWidth;
          setIsOverflow(hasOverflow);
        } else {
          const childrenTotalHeight = Array.from(ref.current.children).reduce(
            (acc, child) => acc + child.clientHeight,
            0
          );
          const hasOverflow = childrenTotalHeight > ref.current.clientHeight;
          setIsOverflow(hasOverflow);
        }
      }
    },
  });
  return isOverflow;
};
