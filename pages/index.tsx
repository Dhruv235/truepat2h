import { useEffect, useRef, useState, useCallback } from 'react';

interface Detection {
  class: string;
  score: number;
  bbox: [number, number, number, number];
}

interface ProductDetection extends Detection {
  distance: number;
  position: string;
  id?: string;
}

interface TrackedDetection extends ProductDetection {
  frameCount: number;
  lastSeen: number;
  smoothedDistance: number;
  smoothedBbox: [number, number, number, number];
}

// Classes to filter out (human-related and non-grocery items)
const FILTERED_CLASSES = ['person', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'];

// Grocery-focused detection heights (in meters)
const KNOWN_HEIGHTS: { [key: string]: number } = {
  // Produce - Fruits
  apple: 0.08,
  banana: 0.2,
  orange: 0.09,

  // Produce - Vegetables
  broccoli: 0.15,
  carrot: 0.2,

  // Packaged Foods
  sandwich: 0.08,
  cake: 0.15,
  'hot dog': 0.15,
  pizza: 0.03,
  donut: 0.08,

  // Beverages
  bottle: 0.25,
  wine: 0.3,
  cup: 0.12,

  // Utensils & Containers
  bowl: 0.08,
  fork: 0.18,
  knife: 0.2,
  spoon: 0.15,

  // Shopping Items
  backpack: 0.4,
  handbag: 0.3,
  suitcase: 0.6,

  // Other common items
  'cell phone': 0.15,
  book: 0.25,
  laptop: 0.02,
  mouse: 0.04,
  keyboard: 0.03,
  chair: 0.9,
  'dining table': 0.75,
  car: 1.5,
  tie: 0.4,
  frisbee: 0.02,
  skis: 1.7,
  snowboard: 1.5,
  'sports ball': 0.22,
  kite: 0.8,
  'baseball bat': 0.85,
  'baseball glove': 0.25,
  skateboard: 0.15,
  surfboard: 2.1,
  'tennis racket': 0.7,
};

export default function ShelfSight() {
  const [mode, setMode] = useState<'intro' | 'active'>('intro');
  const [detections, setDetections] = useState<ProductDetection[]>([]);
  const [currentAnnouncement, setCurrentAnnouncement] = useState<string>('');
  const [tapAnimation, setTapAnimation] = useState(false);
  const [cameraStatus, setCameraStatus] = useState<string>('Not started');
  const [model, setModel] = useState<any>(null);
  const [error, setError] = useState<string>('');
  const [showPermissionHelp, setShowPermissionHelp] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const tapCountRef = useRef(0);
  const tapTimerRef = useRef<NodeJS.Timeout>();
  const animationRef = useRef<number>();
  const detectionCountRef = useRef(0);
  const trackedDetectionsRef = useRef<Map<string, TrackedDetection>>(new Map());
  const lastStableDetectionsRef = useRef<ProductDetection[]>([]);
  const announcedObjectsRef = useRef<Set<string>>(new Set());

  // Calculate Intersection over Union (IoU) for bounding boxes
  const calculateIoU = useCallback((bbox1: [number, number, number, number], bbox2: [number, number, number, number]): number => {
    const [x1, y1, w1, h1] = bbox1;
    const [x2, y2, w2, h2] = bbox2;

    const xLeft = Math.max(x1, x2);
    const yTop = Math.max(y1, y2);
    const xRight = Math.min(x1 + w1, x2 + w2);
    const yBottom = Math.min(y1 + h1, y2 + h2);

    if (xRight < xLeft || yBottom < yTop) return 0;

    const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
    const bbox1Area = w1 * h1;
    const bbox2Area = w2 * h2;
    const unionArea = bbox1Area + bbox2Area - intersectionArea;

    return intersectionArea / unionArea;
  }, []);

  // Remove overlapping detections, keeping the one with highest confidence
  const removeOverlappingDetections = useCallback((detections: ProductDetection[]): ProductDetection[] => {
    const filtered: ProductDetection[] = [];
    const sorted = [...detections].sort((a, b) => b.score - a.score);

    for (const detection of sorted) {
      let isOverlapping = false;

      for (const kept of filtered) {
        const iou = calculateIoU(detection.bbox, kept.bbox);
        if (iou > 0.3) {
          isOverlapping = true;
          break;
        }
      }

      if (!isOverlapping) {
        filtered.push(detection);
      }
    }

    return filtered;
  }, [calculateIoU]);

  const speak = useCallback((text: string, interrupt: boolean = false) => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;

    if (interrupt) {
      window.speechSynthesis.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.95;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;

    utterance.onstart = () => {
      setCurrentAnnouncement(text);
      console.log('🔊 Speaking:', text);
    };

    utterance.onend = () => {
      setCurrentAnnouncement('');
    };

    window.speechSynthesis.speak(utterance);
  }, []);

  useEffect(() => {
    if (mode === 'intro') {
      setTimeout(() => {
        speak('Welcome to ShelfSight, your voice-guided grocery shopping assistant. This app helps you identify products on grocery store shelves. To begin, double tap anywhere on the screen to activate your camera. You can double tap anywhere, anytime.', true);
      }, 500);
    }
  }, [mode, speak]);

  // Load model
  useEffect(() => {
    if (mode === 'active' && !model) {
      setCameraStatus('Loading AI model...');
      speak('Loading product detection system. This may take a moment.', true);

      const loadModel = async () => {
        try {
          console.log('Loading TensorFlow...');
          const tf = await import('@tensorflow/tfjs');
          await tf.ready();
          console.log('TensorFlow ready, loading COCO-SSD...');
          const cocoSsd = await import('@tensorflow-models/coco-ssd');
          const loadedModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
          console.log('✅ Model loaded successfully');
          setModel(loadedModel);
          setCameraStatus('Model loaded, ready for camera');
          speak('Detection system ready. Now activating your camera.', true);
        } catch (err: any) {
          console.error('❌ Model load error:', err);
          setError(`Model error: ${err.message}`);
          setCameraStatus('Model failed to load');
          speak('Error loading detection system. Please refresh the page.', true);
        }
      };

      loadModel();
    }
  }, [mode, model, speak]);

  const startCamera = useCallback(async () => {
    console.log('Starting camera...');
    setCameraStatus('Requesting camera access...');
    setError('');
    setShowPermissionHelp(false);

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Camera API not available - please use HTTPS or localhost');
      }

      const constraints = {
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      };

      console.log('Requesting camera...');
      setCameraStatus('Waiting for permission...');

      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      console.log('✅ Camera stream obtained');
      streamRef.current = stream;
      setCameraStatus('Camera stream obtained');

      if (videoRef.current) {
        videoRef.current.srcObject = stream;

        videoRef.current.onloadedmetadata = async () => {
          console.log('✅ Video metadata loaded');
          setCameraStatus('Video metadata loaded');

          try {
            await videoRef.current!.play();
            console.log('✅ Video playing');
            setCameraStatus('✅ Camera active! Detecting objects...');
            speak('Camera active. Point at products to identify them. I will focus on the nearest items within reach and announce them automatically. Hold your phone steady for best results.', true);
            setIsDetecting(true);
          } catch (playError: any) {
            console.error('❌ Play error:', playError);
            setError(`Play error: ${playError.message}`);
            setCameraStatus('Failed to play video');
          }
        };
      }
    } catch (err: any) {
      console.error('❌ Camera error:', err);

      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setError('Camera permission denied');
        setCameraStatus('Permission denied');
        setShowPermissionHelp(true);
        speak('Camera permission denied. Please allow camera access in your browser settings and try again.', true);
      } else if (err.name === 'NotFoundError') {
        setError('No camera found');
        setCameraStatus('No camera found');
        speak('No camera found on this device.', true);
      } else {
        setError(`Camera error: ${err.message}`);
        setCameraStatus('Camera error');
        speak('Camera error occurred.', true);
      }
    }
  }, [speak]);

  useEffect(() => {
    if (mode === 'active' && model) {
      startCamera();
    }

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      // Clear announced objects when camera stops
      announcedObjectsRef.current.clear();
      trackedDetectionsRef.current.clear();
    };
  }, [mode, model, startCamera]);

  // Detection loop
  const detectObjects = useCallback(async () => {
    if (!model || !videoRef.current || !canvasRef.current || !isDetecting) {
      animationRef.current = requestAnimationFrame(detectObjects);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx || video.readyState !== 4 || video.paused) {
      animationRef.current = requestAnimationFrame(detectObjects);
      return;
    }

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      console.log(`Canvas size set to ${canvas.width}x${canvas.height}`);
    }

    // Draw current video frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      const predictions = await model.detect(canvas);
      detectionCountRef.current++;

      if (detectionCountRef.current % 30 === 0) {
        console.log(`Detection #${detectionCountRef.current}: Found ${predictions.length} objects`);
      }

      // Filter and map raw predictions
      // Higher threshold for better accuracy (50% confidence minimum)
      let rawDetections: ProductDetection[] = predictions
        .filter((p: Detection) => {
          const className = p.class.toLowerCase();
          return p.score > 0.5 && !FILTERED_CLASSES.includes(className);
        })
        .map((prediction: Detection) => {
          const [x, y, width, height] = prediction.bbox;
          const knownHeight = KNOWN_HEIGHTS[prediction.class.toLowerCase()] || 0.3;
          const distance = (knownHeight * 600) / height;
          const centerX = x + width / 2;

          return {
            class: prediction.class,
            score: prediction.score,
            bbox: prediction.bbox,
            distance: Math.max(0.3, Math.min(10, distance)),
            position: centerX < canvas.width * 0.35 ? 'left' : centerX > canvas.width * 0.65 ? 'right' : 'center'
          };
        });

      // Remove overlapping detections
      rawDetections = removeOverlappingDetections(rawDetections);

      // Update tracking with exponential moving average
      const currentTime = Date.now();
      const trackedMap = trackedDetectionsRef.current;
      const updatedIds = new Set<string>();

      // Smoothing factor (0 = no smoothing, 1 = full smoothing)
      const SMOOTHING_ALPHA = 0.3;
      const MIN_FRAMES_FOR_STABILITY = 5; // Increased from 3 to 5 for more accuracy
      const MAX_TRACKING_AGE_MS = 600; // Slightly longer tracking window

      // Match new detections with tracked ones
      for (const detection of rawDetections) {
        let bestMatch: TrackedDetection | null = null;
        let bestIoU = 0;

        // Find best matching tracked detection
        for (const [id, tracked] of trackedMap.entries()) {
          if (tracked.class === detection.class) {
            const iou = calculateIoU(detection.bbox, tracked.smoothedBbox);
            if (iou > 0.2 && iou > bestIoU) {
              bestMatch = tracked;
              bestIoU = iou;
            }
          }
        }

        if (bestMatch) {
          // Update existing tracked detection with smoothing
          // Preserve the original ID
          const id = bestMatch.id || `${bestMatch.class}_${Math.floor(bestMatch.smoothedBbox[0])}`;

          bestMatch.frameCount++;
          bestMatch.lastSeen = currentTime;
          bestMatch.score = detection.score;
          bestMatch.id = id; // Ensure ID is preserved

          // Apply exponential moving average for smooth transitions
          bestMatch.smoothedDistance = SMOOTHING_ALPHA * detection.distance + (1 - SMOOTHING_ALPHA) * bestMatch.smoothedDistance;

          bestMatch.smoothedBbox = [
            SMOOTHING_ALPHA * detection.bbox[0] + (1 - SMOOTHING_ALPHA) * bestMatch.smoothedBbox[0],
            SMOOTHING_ALPHA * detection.bbox[1] + (1 - SMOOTHING_ALPHA) * bestMatch.smoothedBbox[1],
            SMOOTHING_ALPHA * detection.bbox[2] + (1 - SMOOTHING_ALPHA) * bestMatch.smoothedBbox[2],
            SMOOTHING_ALPHA * detection.bbox[3] + (1 - SMOOTHING_ALPHA) * bestMatch.smoothedBbox[3],
          ];

          bestMatch.bbox = bestMatch.smoothedBbox;
          bestMatch.distance = bestMatch.smoothedDistance;
          bestMatch.position = detection.position;

          updatedIds.add(id);
          trackedMap.set(id, bestMatch);
        } else {
          // Create new tracked detection
          const id = `${detection.class}_${Math.floor(detection.bbox[0])}_${currentTime}`;
          const tracked: TrackedDetection = {
            ...detection,
            id,
            frameCount: 1,
            lastSeen: currentTime,
            smoothedDistance: detection.distance,
            smoothedBbox: [...detection.bbox] as [number, number, number, number]
          };
          updatedIds.add(id);
          trackedMap.set(id, tracked);
        }
      }

      // Remove stale tracked detections and clean up announced objects
      for (const [id, tracked] of Array.from(trackedMap.entries())) {
        if (!updatedIds.has(id) && currentTime - tracked.lastSeen > MAX_TRACKING_AGE_MS) {
          trackedMap.delete(id);
          // Remove from announced set when object is no longer tracked
          announcedObjectsRef.current.delete(id);
        }
      }

      // Only show stable detections (seen for multiple frames)
      // Limit to closest 5 objects within reasonable distance (< 4 meters)
      const MAX_DISPLAY_OBJECTS = 5;
      const MAX_DETECTION_DISTANCE = 4.0;

      const stableDetections: ProductDetection[] = Array.from(trackedMap.values())
        .filter(tracked =>
          tracked.frameCount >= MIN_FRAMES_FOR_STABILITY &&
          currentTime - tracked.lastSeen <= MAX_TRACKING_AGE_MS &&
          tracked.smoothedDistance <= MAX_DETECTION_DISTANCE
        )
        .sort((a, b) => a.smoothedDistance - b.smoothedDistance)
        .slice(0, MAX_DISPLAY_OBJECTS);

      // Update display
      setDetections(stableDetections);
      lastStableDetectionsRef.current = stableDetections;

      // Redraw video frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Draw bounding boxes with glowing effect
      stableDetections.forEach((det: ProductDetection, index: number) => {
        const [x, y, width, height] = det.bbox;

        // Glow effect
        ctx.shadowColor = index === 0 ? '#00ff00' : '#ffff00';
        ctx.shadowBlur = 15;

        // Different colors for closest item
        const isClosest = index === 0;
        ctx.strokeStyle = isClosest ? '#00ff00' : '#ffff00';
        ctx.lineWidth = isClosest ? 5 : 3;

        // Draw rectangle
        ctx.strokeRect(x, y, width, height);

        // Reset shadow
        ctx.shadowBlur = 0;

        // Label background
        const label = `${det.class} (${det.distance.toFixed(1)}m)`;
        ctx.font = 'bold 18px Arial';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
        ctx.fillRect(x, y - 35, textWidth + 20, 35);

        // Label text
        ctx.fillStyle = isClosest ? '#00ff00' : '#ffff00';
        ctx.fillText(label, x + 10, y - 10);

        // Distance indicator
        if (isClosest && det.distance < 1.0) {
          ctx.font = 'bold 14px Arial';
          ctx.fillStyle = '#00ff00';
          ctx.fillText('CLOSEST', x + 10, y + height + 25);
        }
      });

      // Announce closest item (only if stable)
      if (stableDetections.length > 0) {
        announceDetection(stableDetections[0], false);
      }
    } catch (err) {
      console.error('❌ Detection error:', err);
    }

    animationRef.current = requestAnimationFrame(detectObjects);
  }, [model, isDetecting, calculateIoU, removeOverlappingDetections]);

  useEffect(() => {
    if (mode === 'active' && model && isDetecting) {
      console.log('Starting detection loop...');
      animationRef.current = requestAnimationFrame(detectObjects);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [mode, model, isDetecting, detectObjects]);

  const announceDetection = useCallback((detection: ProductDetection, forceAnnounce: boolean = false) => {
    // Use the tracked ID if available
    const objectId = detection.id;

    if (!objectId) {
      console.log('⚠️ No ID for detection, skipping announcement');
      return;
    }

    // Only announce items within 3 meters
    if (!forceAnnounce && detection.distance > 3.0) {
      return;
    }

    // Check if this object has already been announced
    if (!forceAnnounce && announcedObjectsRef.current.has(objectId)) {
      return;
    }

    console.log(`📢 Announcing: ${detection.class} at ${detection.distance.toFixed(1)}m (confidence: ${(detection.score * 100).toFixed(0)}%) [ID: ${objectId}]`);

    // Mark this object as announced
    announcedObjectsRef.current.add(objectId);

    let announcement = '';

    if (detection.distance < 0.5) {
      announcement = `${detection.class}, very close, within reach`;
    } else if (detection.distance < 1.0) {
      announcement = `${detection.class}, ${Math.round(detection.distance * 100)} centimeters away, on your ${detection.position}`;
    } else if (detection.distance < 2.0) {
      announcement = `${detection.class}, ${detection.distance.toFixed(1)} meters away, on your ${detection.position}`;
    } else {
      announcement = `${detection.class}, about ${Math.round(detection.distance)} meters away`;
    }

    speak(announcement, false);
  }, [speak]);

  const handleDoubleTap = () => {
    tapCountRef.current += 1;
    setTapAnimation(true);

    if (tapTimerRef.current) {
      clearTimeout(tapTimerRef.current);
    }

    if (tapCountRef.current === 2) {
      if (mode === 'intro') {
        setMode('active');
      }
      tapCountRef.current = 0;
    } else {
      tapTimerRef.current = setTimeout(() => {
        tapCountRef.current = 0;
        setTapAnimation(false);
      }, 300);
    }

    setTimeout(() => setTapAnimation(false), 300);
  };

  if (mode === 'intro') {
    return (
      <div
        onClick={handleDoubleTap}
        style={{
          width: '100vw',
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'linear-gradient(135deg, #1e40af 0%, #7c3aed 50%, #db2777 100%)',
          padding: '20px',
          cursor: 'pointer',
        }}
      >
        <h1 style={{ fontSize: '4rem', color: '#fff', marginBottom: '30px', fontWeight: 'bold', textShadow: '0 4px 6px rgba(0,0,0,0.3)' }}>
          ShelfSight
        </h1>
        <p style={{ fontSize: '1.8rem', color: '#fff', textAlign: 'center', maxWidth: '800px', lineHeight: '1.6', marginBottom: '20px' }}>
          Voice-Guided Grocery Shopping Assistant
        </p>
        <p style={{ fontSize: '1.2rem', color: '#e0e7ff', textAlign: 'center', maxWidth: '700px', lineHeight: '1.5' }}>
          Identify products on shelves with real-time voice guidance
        </p>
        <div style={{
          marginTop: '60px',
          padding: '40px 60px',
          background: 'rgba(255, 255, 255, 0.2)',
          backdropFilter: 'blur(10px)',
          borderRadius: '20px',
          border: '3px solid rgba(255, 255, 255, 0.3)',
          boxShadow: '0 8px 32px 0 rgba(0, 0, 0, 0.37)'
        }}>
          <p style={{ fontSize: '2.2rem', color: '#fff', fontWeight: 'bold', textAlign: 'center', marginBottom: '15px' }}>
            Double Tap to Start
          </p>
          <p style={{ fontSize: '1.1rem', color: '#e0e7ff', textAlign: 'center' }}>
            Tap anywhere on the screen twice
          </p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', background: '#000' }}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          display: 'none'
        }}
      />

      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          objectFit: 'cover',
        }}
      />

      {/* Status Display */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        right: '20px',
        background: 'rgba(0, 0, 0, 0.85)',
        padding: '15px 20px',
        borderRadius: '15px',
        color: '#fff',
        zIndex: 10,
        border: '2px solid rgba(0, 255, 0, 0.3)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '1.3rem', margin: 0 }}>ShelfSight</h2>
          <div style={{
            padding: '6px 12px',
            background: isDetecting ? '#10b981' : '#f59e0b',
            borderRadius: '20px',
            fontSize: '0.85rem',
            fontWeight: 'bold'
          }}>
            {isDetecting ? 'DETECTING' : 'LOADING'}
          </div>
        </div>
      </div>

      {/* Permission Help */}
      {showPermissionHelp && (
        <div style={{
          position: 'absolute',
          top: '100px',
          left: '20px',
          right: '20px',
          background: 'rgba(239, 68, 68, 0.95)',
          padding: '25px',
          borderRadius: '15px',
          color: '#fff',
          zIndex: 10,
          border: '2px solid #fca5a5'
        }}>
          <h3 style={{ fontSize: '1.3rem', marginBottom: '15px', fontWeight: 'bold' }}>
            Camera Permission Needed
          </h3>
          <p style={{ fontSize: '1rem', marginBottom: '15px', lineHeight: 1.6 }}>
            Click the camera icon in your browser's address bar and select "Allow"
          </p>
          <button
            onClick={startCamera}
            style={{
              width: '100%',
              padding: '15px',
              fontSize: '1.1rem',
              fontWeight: 'bold',
              background: '#fff',
              color: '#ef4444',
              border: 'none',
              borderRadius: '10px',
              cursor: 'pointer',
            }}
          >
            Try Again
          </button>
        </div>
      )}

      {/* Current Announcement */}
      {currentAnnouncement && !showPermissionHelp && (
        <div style={{
          position: 'absolute',
          top: '100px',
          left: '20px',
          right: '20px',
          background: 'rgba(99, 102, 241, 0.95)',
          padding: '20px',
          borderRadius: '15px',
          color: '#fff',
          zIndex: 10,
          border: '3px solid #818cf8',
          animation: 'pulse 2s ease-in-out infinite'
        }}>
          <p style={{ fontSize: '0.9rem', fontWeight: 'bold', marginBottom: '8px', opacity: 0.9 }}>
            Speaking:
          </p>
          <p style={{ fontSize: '1.4rem', fontWeight: 'bold', margin: 0, lineHeight: 1.4 }}>
            {currentAnnouncement}
          </p>
        </div>
      )}

      {/* Detection Results */}
      <div style={{
        position: 'absolute',
        bottom: '20px',
        left: '20px',
        right: '20px',
        background: 'rgba(0, 0, 0, 0.85)',
        padding: '20px',
        borderRadius: '15px',
        zIndex: 10,
        border: '2px solid rgba(0, 255, 0, 0.3)'
      }}>
        {detections.length > 0 ? (
          <div>
            <p style={{ fontSize: '1.1rem', color: '#10b981', fontWeight: 'bold', marginBottom: '15px' }}>
              {detections.length} Nearest Item{detections.length > 1 ? 's' : ''} (within 4m)
            </p>
            {detections.map((det, idx) => (
              <div key={det.id || idx} style={{
                background: idx === 0 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(234, 179, 8, 0.2)',
                padding: '12px',
                borderRadius: '10px',
                marginBottom: '8px',
                border: idx === 0 ? '2px solid #10b981' : '1px solid rgba(234, 179, 8, 0.5)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <p style={{ fontSize: '1.2rem', color: '#fff', fontWeight: 'bold', margin: 0 }}>
                    {idx === 0 ? 'CLOSEST: ' : ''}{det.class}
                  </p>
                  <span style={{
                    fontSize: '0.8rem',
                    color: idx === 0 ? '#10b981' : '#eab308',
                    fontWeight: 'bold',
                    background: 'rgba(0,0,0,0.5)',
                    padding: '4px 8px',
                    borderRadius: '5px'
                  }}>
                    {(det.score * 100).toFixed(0)}%
                  </span>
                </div>
                <p style={{ fontSize: '0.95rem', color: '#cbd5e1', margin: '5px 0 0 0' }}>
                  {det.distance < 1 ? `${Math.round(det.distance * 100)} cm` : `${det.distance.toFixed(1)} m`} away • {det.position} side
                </p>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: '1.2rem', color: '#fff', marginBottom: '8px' }}>
              {isDetecting ? 'Scanning for Products...' : 'Loading...'}
            </p>
            <p style={{ fontSize: '0.95rem', color: '#cbd5e1' }}>
              {isDetecting ? 'Point camera at grocery items' : 'Please wait'}
            </p>
          </div>
        )}
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.9; transform: scale(1.02); }
        }
      `}</style>
    </div>
  );
}
