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

export const radiansToDegrees = (radians: number): number => {
  return (radians * 180) / Math.PI;
};

import { useEffect, useRef } from 'react';

export const useWarningSound = (triggerValue: boolean) => {
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    // Create AudioContext only once
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    if (triggerValue) {
      const playWarningSound = () => {
        const context = audioContextRef.current;
        if (!context) {
          return;
        }
        
        const oscillator = context.createOscillator();
        const gainNode = context.createGain();

        oscillator.type = 'sine';
        oscillator.frequency.setValueAtTime(440, context.currentTime); // A4 note
        oscillator.frequency.exponentialRampToValueAtTime(880, context.currentTime + 0.5); // Ramp to A5

        gainNode.gain.setValueAtTime(0.5, context.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, context.currentTime + 0.5);

        oscillator.connect(gainNode);
        gainNode.connect(context.destination);

        oscillator.start();
        oscillator.stop(context.currentTime + 0.5);
      };

      playWarningSound();
    }
  }, [triggerValue]);
};
