const KNOWN_OBJECT_HEIGHTS: { [key: string]: number } = {
  // People & Body Parts
  person: 1.7,
  
  // Furniture - Seating
  chair: 0.9,
  couch: 0.8,
  bench: 0.5,
  
  // Furniture - Tables
  'dining table': 0.75,
  desk: 0.75,
  'coffee table': 0.45,
  
  // Furniture - Storage
  bed: 0.6,
  bookshelf: 1.8,
  cabinet: 1.5,
  dresser: 1.2,
  nightstand: 0.6,
  
  // Kitchen & Appliances
  refrigerator: 1.7,
  oven: 0.9,
  microwave: 0.4,
  sink: 0.9,
  dishwasher: 0.9,
  stove: 0.9,
  toaster: 0.2,
  
  // Bathroom
  toilet: 0.75,
  bathtub: 0.5,
  shower: 2.0,
  
  // Electronics
  tv: 0.6,
  laptop: 0.02,
  monitor: 0.5,
  keyboard: 0.03,
  mouse: 0.04,
  'cell phone': 0.15,
  remote: 0.15,
  
  // Doors & Openings
  door: 2.0,
  window: 1.5,
  
  // Containers & Items
  bottle: 0.25,
  cup: 0.12,
  bowl: 0.08,
  plate: 0.02,
  glass: 0.15,
  vase: 0.3,
  
  // Plants & Decor
  'potted plant': 0.5,
  plant: 0.4,
  clock: 0.3,
  painting: 0.6,
  mirror: 1.2,
  
  // Bags & Luggage
  backpack: 0.4,
  handbag: 0.3,
  suitcase: 0.6,
  purse: 0.25,
  
  // Office & School
  book: 0.25,
  pencil: 0.15,
  scissors: 0.2,
  stapler: 0.08,
  printer: 0.4,
  
  // Sports & Recreation
  ball: 0.22,
  'sports ball': 0.22,
  bicycle: 1.1,
  skateboard: 0.15,
  
  // Animals
  dog: 0.5,
  cat: 0.3,
  bird: 0.15,
  
  // Outdoor & Misc
  umbrella: 0.9,
  tie: 0.4,
  hat: 0.15,
  shoe: 0.12,
  'fire hydrant': 0.8,
  'stop sign': 2.2,
  'parking meter': 1.2,
  trash: 0.6,
  'trash can': 0.9,
  bin: 0.6,
  basket: 0.4,
  box: 0.4,
  bag: 0.3,
  pillow: 0.15,
  blanket: 0.1,
  towel: 0.8,
  curtain: 2.0,
  rug: 0.02,
  mat: 0.02,
  
  // Kitchen Items
  fork: 0.18,
  knife: 0.2,
  spoon: 0.15,
  spatula: 0.25,
  pan: 0.08,
  pot: 0.15,
  kettle: 0.25,
  
  // Lighting
  lamp: 0.5,
  'table lamp': 0.4,
  'floor lamp': 1.5,
  'ceiling light': 0.3,
  
  // Safety & Emergency
  'fire extinguisher': 0.6,
  'first aid kit': 0.3,
  smoke: 0.2,
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
