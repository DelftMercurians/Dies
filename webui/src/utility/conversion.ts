//TODO: FIX PINIA

import { XY, XYZ } from "../types";
import { useAppStore } from "../store/app";

const appStore = useAppStore();

/**
 * Convert from server length to canvas length.
 */
const convertLength = (
  length: number,
  width: number,
  fieldW: number
): number => {
  return Math.ceil(length * (width / fieldW));
};

/**
 * Convert from server coordinates to canvas coordinates.
 *
 * The server's coordinate system has its origin at the center of the field,
 */
const convertCoords = (
  coords: XY | XYZ,
  width: number,
  fieldW: number,
  height: number,
  fieldH: number
): XY => {
  const [x, y] = coords;

  return [
    (x + fieldW / 2) * (width / fieldW) + appStore.PADDING,
    (-y + fieldH / 2) * (height / fieldH) + appStore.PADDING,
  ];
};

export { convertLength, convertCoords };
