const KNOWN_OBJECT_HEIGHTS: { [key: string]: number } = {
  person: 1.7,
  chair: 0.9,
  'dining table': 0.75,
  'potted plant': 0.5,
  bottle: 0.25,
  cup: 0.12,
  laptop: 0.02,
  'cell phone': 0.15,
  book: 0.25,
  clock: 0.3,
  vase: 0.3,
  door: 2.0,
  refrigerator: 1.7,
  oven: 0.9,
  sink: 0.9,
  bed: 0.6,
  couch: 0.8,
  toilet: 0.75,
  tv: 0.6,
  backpack: 0.4,
  handbag: 0.3,
  suitcase: 0.6,
  bench: 0.5,
};

const FOCAL_LENGTH_REFERENCE = 600;

export const estimateDistance = (
  objectClass: string,
  boundingBoxHeight: number,
  imageHeight: number
): number => {
  const knownHeight = KNOWN_OBJECT_HEIGHTS[objectClass.toLowerCase()] || 1.0;

  const pixelHeight = boundingBoxHeight;

  const distance = (knownHeight * FOCAL_LENGTH_REFERENCE) / pixelHeight;

  return Math.max(0.5, Math.min(10, distance));
};

export const getDirection = (
  boxCenterX: number,
  imageWidth: number
): string => {
  const normalizedX = boxCenterX / imageWidth;

  if (normalizedX < 0.35) {
    return 'left';
  } else if (normalizedX > 0.65) {
    return 'right';
  } else {
    return 'center';
  }
};

export const formatDistance = (distance: number): string => {
  if (distance < 1) {
    return `${Math.round(distance * 100)} centimeters`;
  } else {
    return `${distance.toFixed(1)} meters`;
  }
};
