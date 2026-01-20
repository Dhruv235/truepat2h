import { useEffect, useRef, useState, useCallback } from 'react';

interface Detection {
  class: string;
  score: number;
  bbox: [number, number, number, number];
}

interface ProductDetection extends Detection {
  distance: number;
  position: string;
}

const KNOWN_HEIGHTS: { [key: string]: number } = {
  person: 1.7,
  bottle: 0.25,
  cup: 0.12,
  bowl: 0.08,
  'cell phone': 0.15,
  book: 0.25,
  laptop: 0.02,
  mouse: 0.04,
  keyboard: 0.03,
  apple: 0.08,
  banana: 0.2,
  orange: 0.09,
  chair: 0.9,
  'dining table': 0.75,
  car: 1.5,
  dog: 0.5,
  cat: 0.3,
  backpack: 0.4,
  handbag: 0.3,
  tie: 0.4,
  suitcase: 0.6,
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
  wine: 0.3,
  fork: 0.18,
  knife: 0.2,
  spoon: 0.15,
  sandwich: 0.08,
  cake: 0.15,
  carrot: 0.2,
  'hot dog': 0.15,
  pizza: 0.03,
  donut: 0.08,
  broccoli: 0.15,
};

export default function ShelfSight() {
  const [mode, setMode] = useState<'intro' | 'active'>('intro');
  const [detections, setDetections] = useState<ProductDetection[]>([]);
  const [lastAnnouncement, setLastAnnouncement] = useState<{ [key: string]: number }>({});
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

  const speak = useCallback((text: string, interrupt: boolean = false) => {
    if (typeof window === 'undefined' || !window.speechSynthesis) return;

    if (interrupt) {
      window.speechSynthesis.cancel();
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
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
        speak('Welcome to ShelfSight. Double tap to start camera.', true);
      }, 500);
    }
  }, [mode, speak]);

  // Load model
  useEffect(() => {
    if (mode === 'active' && !model) {
      setCameraStatus('Loading AI model...');
      speak('Loading detection model...', true);
      
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
          speak('Detection model loaded. Ready to scan.', true);
        } catch (err: any) {
          console.error('❌ Model load error:', err);
          setError(`Model error: ${err.message}`);
          setCameraStatus('Model failed to load');
          speak('Error loading detection model.', true);
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
            speak('Camera active. Point at objects to identify them.', true);
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
        speak('Camera permission denied. Please grant access.', true);
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

      const detectedObjects: ProductDetection[] = predictions
        .filter((p: Detection) => p.score > 0.4)
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
        })
        .sort((a, b) => a.distance - b.distance);

      setDetections(detectedObjects);

      // Redraw video frame
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Draw bounding boxes with glowing effect
      detectedObjects.forEach((det: ProductDetection, index: number) => {
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

      // Announce closest item
      if (detectedObjects.length > 0) {
        announceDetection(detectedObjects[0], false);
      }
    } catch (err) {
      console.error('❌ Detection error:', err);
    }

    animationRef.current = requestAnimationFrame(detectObjects);
  }, [model, isDetecting]);

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
    const now = Date.now();
    const key = detection.class;
    const lastTime = lastAnnouncement[key] || 0;
    
    // Announce every 3 seconds for same object
    if (!forceAnnounce && now - lastTime < 3000) {
      return;
    }
    
    console.log(`📢 Announcing: ${detection.class} at ${detection.distance.toFixed(1)}m`);
    setLastAnnouncement(prev => ({ ...prev, [key]: now }));
    
    let announcement = '';
    
    if (detection.distance < 0.5) {
      announcement = `${detection.class} very close, in your hand`;
    } else if (detection.distance < 1.0) {
      announcement = `${detection.class}, ${Math.round(detection.distance * 100)} centimeters away`;
    } else if (detection.distance < 2.0) {
      announcement = `${detection.class}, ${detection.distance.toFixed(1)} meters, ${detection.position}`;
    } else {
      announcement = `${detection.class} detected, far away`;
    }
    
    speak(announcement, false);
  }, [lastAnnouncement, speak]);

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
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)',
          padding: '20px',
          cursor: 'pointer',
        }}
      >
        <h1 style={{ fontSize: '4rem', color: '#fff', marginBottom: '30px' }}>
          🛒 ShelfSight
        </h1>
        <p style={{ fontSize: '1.5rem', color: '#fff', textAlign: 'center', maxWidth: '800px' }}>
          Your Voice-Guided Grocery Shopping Assistant
        </p>
        <div style={{
          marginTop: '60px',
          padding: '30px 50px',
          background: 'rgba(255, 255, 255, 0.15)',
          backdropFilter: 'blur(10px)',
          borderRadius: '20px',
        }}>
          <p style={{ fontSize: '2rem', color: '#fff', fontWeight: 'bold' }}>
            👆 Double Tap to Start 👆
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
          <h2 style={{ fontSize: '1.3rem', margin: 0 }}>🛒 ShelfSight</h2>
          <div style={{
            padding: '6px 12px',
            background: isDetecting ? '#10b981' : '#f59e0b',
            borderRadius: '20px',
            fontSize: '0.85rem',
            fontWeight: 'bold'
          }}>
            {isDetecting ? '✅ DETECTING' : '⏳ LOADING'}
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
            📷 Camera Permission Needed
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
            🔄 Try Again
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
            🔊 Speaking:
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
              ✅ {detections.length} Object{detections.length > 1 ? 's' : ''} Detected
            </p>
            {detections.slice(0, 5).map((det, idx) => (
              <div key={idx} style={{
                background: idx === 0 ? 'rgba(16, 185, 129, 0.3)' : 'rgba(234, 179, 8, 0.2)',
                padding: '12px',
                borderRadius: '10px',
                marginBottom: '8px',
                border: idx === 0 ? '2px solid #10b981' : '1px solid rgba(234, 179, 8, 0.5)'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <p style={{ fontSize: '1.2rem', color: '#fff', fontWeight: 'bold', margin: 0 }}>
                    {idx === 0 ? '🎯 ' : ''}{det.class}
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
                  {det.distance < 1 ? `${Math.round(det.distance * 100)} cm` : `${det.distance.toFixed(1)} m`} • {det.position}
                  {idx === 0 && det.distance < 1.0 && ' • CLOSEST'}
                </p>
              </div>
            ))}
          </div>
        ) : (
          <div style={{ textAlign: 'center' }}>
            <p style={{ fontSize: '1.2rem', color: '#fff', marginBottom: '8px' }}>
              {isDetecting ? '🔍 Scanning...' : '⏳ Loading...'}
            </p>
            <p style={{ fontSize: '0.95rem', color: '#cbd5e1' }}>
              {isDetecting ? 'Point camera at objects' : 'Please wait'}
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