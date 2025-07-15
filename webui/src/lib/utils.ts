import { type ClassValue, clsx } from "clsx";
import React, { Ref } from "react";
import { twMerge } from "tailwind-merge";
import { useResizeObserver } from "./useResizeObserver";
import { TeamColor } from "../bindings";

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
  orientation: "horizontal" | "vertical" = "horizontal",
) => {
  const [isOverflow, setIsOverflow] = React.useState(false);
  useResizeObserver({
    ref,
    onResize: () => {
      if (ref.current) {
        if (orientation === "horizontal") {
          const childrenTotalWidth = Array.from(ref.current.children).reduce(
            (acc, child) => acc + child.clientWidth,
            0,
          );
          const hasOverflow = childrenTotalWidth > ref.current.clientWidth;
          setIsOverflow(hasOverflow);
        } else {
          const childrenTotalHeight = Array.from(ref.current.children).reduce(
            (acc, child) => acc + child.clientHeight,
            0,
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

import { useEffect, useRef } from "react";

export const useWarningSound = (triggerValue: boolean) => {
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    // Create AudioContext only once
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext ||
        (window as any).webkitAudioContext)();
    }

    if (triggerValue) {
      const playWarningSound = () => {
        const context = audioContextRef.current;
        if (!context) {
          return;
        }

        const oscillator = context.createOscillator();
        const gainNode = context.createGain();

        oscillator.type = "sine";
        oscillator.frequency.setValueAtTime(440, context.currentTime); // A4 note
        oscillator.frequency.exponentialRampToValueAtTime(
          880,
          context.currentTime + 0.5,
        ); // Ramp to A5

        gainNode.gain.setValueAtTime(0.5, context.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(
          0.01,
          context.currentTime + 0.5,
        );

        oscillator.connect(gainNode);
        gainNode.connect(context.destination);

        oscillator.start();
        oscillator.stop(context.currentTime + 0.5);
      };

      playWarningSound();
    }
  }, [triggerValue]);
};

// Robot count sound effects
export const useRobotCountSounds = () => {
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    // Create AudioContext only once
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext ||
        (window as any).webkitAudioContext)();
    }
  }, []);

  const playTooManyRobotsSound = () => {
    const context = audioContextRef.current;
    if (!context) return;

    // Low priority sound - descending tone
    const oscillator = context.createOscillator();
    const gainNode = context.createGain();

    oscillator.type = "triangle";
    oscillator.frequency.setValueAtTime(660, context.currentTime); // E5 note
    oscillator.frequency.exponentialRampToValueAtTime(
      330,
      context.currentTime + 0.7,
    ); // Descending to E4

    gainNode.gain.setValueAtTime(0.3, context.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(
      0.01,
      context.currentTime + 0.7,
    );

    oscillator.connect(gainNode);
    gainNode.connect(context.destination);

    oscillator.start();
    oscillator.stop(context.currentTime + 0.7);
  };

  const playTooFewRobotsSound = () => {
    const context = audioContextRef.current;
    if (!context) return;

    // High priority sound - rapid beeping
    const playBeep = (startTime: number, frequency: number) => {
      const oscillator = context.createOscillator();
      const gainNode = context.createGain();

      oscillator.type = "square";
      oscillator.frequency.setValueAtTime(frequency, startTime);

      gainNode.gain.setValueAtTime(0.4, startTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, startTime + 0.1);

      oscillator.connect(gainNode);
      gainNode.connect(context.destination);

      oscillator.start(startTime);
      oscillator.stop(startTime + 0.1);
    };

    // Play three rapid beeps with increasing frequency (higher priority)
    playBeep(context.currentTime, 800);
    playBeep(context.currentTime + 0.15, 1000);
    playBeep(context.currentTime + 0.3, 1200);
  };

  return { playTooManyRobotsSound, playTooFewRobotsSound };
};

// Hook to monitor robot count for the primary team and play sound alerts
export const useRobotCountAlerts = (
  worldState: any,
  primaryTeam: TeamColor
) => {
  const { playTooManyRobotsSound, playTooFewRobotsSound } = useRobotCountSounds();
  const prevRobotCountRef = useRef<number | null>(null);
  const lastSoundTimeRef = useRef<number>(0);

  useEffect(() => {
    if (worldState.status !== "connected" || !worldState.data) {
      return;
    }

    const data = worldState.data;
    const yellowCards = primaryTeam === TeamColor.Blue 
      ? data.game_state.blue_team_yellow_cards 
      : data.game_state.yellow_team_yellow_cards;
    
    const primaryTeamPlayers = primaryTeam === TeamColor.Blue 
      ? data.blue_team 
      : data.yellow_team;

    const currentRobotCount = primaryTeamPlayers.length;
    const allowedRobotCount = Math.max(0, 6 - yellowCards);

    // Only check if we have a previous count to compare against
    if (prevRobotCountRef.current !== null) {
      const now = Date.now();
      const timeSinceLastSound = now - lastSoundTimeRef.current;
      
      // Debounce sounds - only play if at least 2 seconds have passed
      if (timeSinceLastSound >= 2000) {
        if (currentRobotCount > allowedRobotCount && prevRobotCountRef.current <= allowedRobotCount) {
          // Robot count just exceeded allowed count
          playTooManyRobotsSound();
          lastSoundTimeRef.current = now;
        } else if (currentRobotCount < allowedRobotCount && prevRobotCountRef.current >= allowedRobotCount) {
          // Robot count just went below allowed count (higher priority)
          playTooFewRobotsSound();
          lastSoundTimeRef.current = now;
        }
      }
    }

    prevRobotCountRef.current = currentRobotCount;
  }, [worldState, primaryTeam, playTooManyRobotsSound, playTooFewRobotsSound]);
};
