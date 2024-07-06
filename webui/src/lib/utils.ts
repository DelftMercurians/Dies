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

export const areChildrenWithinBounds = (
  parentRef: React.MutableRefObject<HTMLElement | null>
) => {
  const checkChildrenBounds = () => {
    if (parentRef.current) {
      const parentBounds = parentRef.current.getBoundingClientRect();
      const children = Array.from(parentRef.current.children);

      for (const child of children) {
        const childBounds = child.getBoundingClientRect();

        if (
          childBounds.left < parentBounds.left ||
          childBounds.right > parentBounds.right ||
          childBounds.top < parentBounds.top ||
          childBounds.bottom > parentBounds.bottom
        ) {
          return false;
        }
      }
    }

    return true;
  };

  return checkChildrenBounds();
};

export const useIsOverflow = (
  ref: React.MutableRefObject<HTMLElement | null>,
  orientation: "horizontal" | "vertical" = "horizontal"
) => {
  const [isOverflow, setIsOverflow] = React.useState(false);
  useResizeObserver({
    ref,
    onResize: () => {
      if (ref.current) {
        const hasOverflow = !areChildrenWithinBounds(ref);
        console.log(hasOverflow);
        setIsOverflow(hasOverflow);
      }
    },
  });
  return isOverflow;
};
