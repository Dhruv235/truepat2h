import { useEffect, useRef, useState, useCallback } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import { speak, stopSpeaking } from '../utils/voice';
import { estimateDistance, getDirection, formatDistance } from '../utils/distance';

interface Detection {
  class: string;
  score: number;
  bbox: [number, number, number, number];
  distance: number;
  direction: string;
}

interface SmoothedBBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export default function Home() {
  const [showIntro, setShowIntro] = useState(true);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [lastAnnouncement, setLastAnnouncement] = useState<{ [key: string]: number }>({});
  const [tapAnimation, setTapAnimation] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number>();
  const tapCountRef = useRef(0);
  const tapTimerRef = useRef<NodeJS.Timeout>();
  
  // Smoothing buffer for bounding boxes
  const smoothingBuffer = useRef<Map<string, SmoothedBBox[]>>(new Map());
  const SMOOTHING_FRAMES = 5;
  const SMOOTHING_ALPHA = 0.3;

  useEffect(() => {
    if (showIntro) {
      const introMessage =
        'Welcome to TruePath, your A R navigation assistant. ' +
        'This app will help you navigate indoor spaces by detecting objects and measuring distances. ' +
        'Double tap anywhere on the screen to start the camera and begin detection.';

      setTimeout(() => {
        speak(introMessage, true);
      }, 500);
    }
  }, [showIntro]);

  useEffect(() => {
    if (!showIntro) {
      loadModel();
      startCamera();
    }

    return () => {
      stopCamera();
      stopSpeaking();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [showIntro]);

  const loadModel = async () => {
    try {
      const loadedModel = await cocoSsd.load({
        base: 'lite_mobilenet_v2'
      });
      setModel(loadedModel);
      speak('Object detection model loaded successfully');
    } catch (error) {
      console.error('Error loading model:', error);
      speak('Error loading detection model');
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        speak('Camera started. Begin scanning your environment.');
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      speak('Unable to access camera. Please grant camera permissions.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  const smoothBoundingBox = (key: string, bbox: [number, number, number, number]): SmoothedBBox => {
    const [x, y, width, height] = bbox;
    const currentBox = { x, y, width, height };
    
    if (!smoothingBuffer.current.has(key)) {
      smoothingBuffer.current.set(key, []);
    }
    
    const buffer = smoothingBuffer.current.get(key)!;
    buffer.push(currentBox);
    
    if (buffer.length > SMOOTHING_FRAMES) {
      buffer.shift();
    }
    
    if (buffer.length === 1) {
      return currentBox;
    }
    
    const smoothed = buffer.reduce((acc, box, idx, arr) => {
      const weight = idx === arr.length - 1 ? SMOOTHING_ALPHA : (1 - SMOOTHING_ALPHA) / (arr.length - 1);
      return {
        x: acc.x + box.x * weight,
        y: acc.y + box.y * weight,
        width: acc.width + box.width * weight,
        height: acc.height + box.height * weight
      };
    }, { x: 0, y: 0, width: 0, height: 0 });
    
    return smoothed;
  };

  const detectObjects = useCallback(async () => {
    if (!model || !videoRef.current || !canvasRef.current) {
      animationRef.current = requestAnimationFrame(detectObjects);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx || video.readyState !== 4) {
      animationRef.current = requestAnimationFrame(detectObjects);
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const predictions = await model.detect(canvas);

    const detectedObjects: Detection[] = predictions
      .filter(p => p.score > 0.6)
      .map(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const distance = estimateDistance(
          prediction.class,
          height,
          canvas.height
        );
        const centerX = x + width / 2;
        const direction = getDirection(centerX, canvas.width);

        return {
          class: prediction.class,
          score: prediction.score,
          bbox: prediction.bbox,
          distance,
          direction
        };
      });

    setDetections(detectedObjects);

    const now = Date.now();
    detectedObjects.forEach(detection => {
      const key = `${detection.class}-${detection.direction}`;
      const lastTime = lastAnnouncement[key] || 0;

      if (now - lastTime > 3000) {
        const message = `${detection.class} detected, ${detection.direction}, ${formatDistance(detection.distance)} away`;
        speak(message);
        setLastAnnouncement(prev => ({ ...prev, [key]: now }));
      }
    });

    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 3;
    ctx.font = 'bold 18px Arial';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
    ctx.shadowBlur = 4;

    detectedObjects.forEach(detection => {
      const key = `${detection.class}-${Math.floor(detection.bbox[0] / 50)}`;
      const smoothed = smoothBoundingBox(key, detection.bbox);
      
      const gradient = ctx.createLinearGradient(
        smoothed.x, 
        smoothed.y, 
        smoothed.x + smoothed.width, 
        smoothed.y + smoothed.height
      );
      gradient.addColorStop(0, '#00ff00');
      gradient.addColorStop(1, '#00cc00');
      
      ctx.strokeStyle = gradient;
      ctx.strokeRect(smoothed.x, smoothed.y, smoothed.width, smoothed.height);

      const label = `${detection.class} ${formatDistance(detection.distance)}`;
      ctx.font = 'bold 18px Arial';
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(smoothed.x, smoothed.y - 30, textWidth + 16, 30);

      ctx.fillStyle = '#00ff00';
      ctx.fillText(label, smoothed.x + 8, smoothed.y - 9);
      
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
    });

    animationRef.current = requestAnimationFrame(detectObjects);
  }, [model, lastAnnouncement]);

  useEffect(() => {
    if (!showIntro && model && videoRef.current) {
      videoRef.current.addEventListener('loadeddata', () => {
        animationRef.current = requestAnimationFrame(detectObjects);
      });
    }
  }, [showIntro, model, detectObjects]);

  const handleDoubleTap = () => {
    tapCountRef.current += 1;
    setTapAnimation(true);

    if (tapTimerRef.current) {
      clearTimeout(tapTimerRef.current);
    }

    if (tapCountRef.current === 2) {
      if (showIntro) {
        stopSpeaking();
        setShowIntro(false);
        speak('Starting camera');
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

  if (showIntro) {
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
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 50% 50%, rgba(255,255,255,0.1) 0%, transparent 50%)',
            animation: 'pulse 3s ease-in-out infinite'
          }}
        />

        <style>{`
          @keyframes pulse {
            0%, 100% { opacity: 0.3; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.05); }
          }
          @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-20px); }
          }
          @keyframes ripple {
            0% { transform: scale(0.8); opacity: 1; }
            100% { transform: scale(2); opacity: 0; }
          }
          @keyframes tap-feedback {
            0% { transform: scale(1); }
            50% { transform: scale(0.95); }
            100% { transform: scale(1); }
          }
        `}</style>

        {tapAnimation && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              width: '200px',
              height: '200px',
              marginLeft: '-100px',
              marginTop: '-100px',
              borderRadius: '50%',
              border: '3px solid rgba(255, 255, 255, 0.8)',
              animation: 'ripple 0.6s ease-out'
            }}
          />
        )}

        <div style={{
          animation: 'float 3s ease-in-out infinite',
          textAlign: 'center',
          zIndex: 1
        }}>
          <h1 style={{
            fontSize: '4rem',
            fontWeight: 'bold',
            marginBottom: '30px',
            color: '#fff',
            textShadow: '0 4px 20px rgba(0,0,0,0.3)',
            letterSpacing: '2px'
          }}>
            TruePath
          </h1>

          <div style={{
            width: '80px',
            height: '4px',
            background: 'linear-gradient(90deg, transparent, #fff, transparent)',
            margin: '20px auto',
            borderRadius: '2px'
          }} />

          <p style={{
            fontSize: '1.8rem',
            textAlign: 'center',
            lineHeight: '1.8',
            maxWidth: '900px',
            color: '#fff',
            marginBottom: '20px',
            textShadow: '0 2px 10px rgba(0,0,0,0.2)'
          }}>
            Your AR Navigation Assistant
          </p>

          <p style={{
            fontSize: '1.3rem',
            textAlign: 'center',
            lineHeight: '1.8',
            maxWidth: '800px',
            color: 'rgba(255, 255, 255, 0.9)',
            marginTop: '20px',
            textShadow: '0 2px 8px rgba(0,0,0,0.2)'
          }}>
            Navigate indoor spaces with real-time object detection and distance measurement
          </p>
        </div>

        <div style={{
          marginTop: '60px',
          padding: '30px 50px',
          background: 'rgba(255, 255, 255, 0.15)',
          backdropFilter: 'blur(10px)',
          borderRadius: '20px',
          border: '2px solid rgba(255, 255, 255, 0.3)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.2)',
          animation: tapAnimation ? 'tap-feedback 0.3s ease-out' : 'none',
          zIndex: 1
        }}>
          <p style={{
            fontSize: '2rem',
            textAlign: 'center',
            color: '#fff',
            fontWeight: '700',
            margin: 0,
            textShadow: '0 2px 10px rgba(0,0,0,0.3)'
          }}>
            👆 Double Tap to Start 👆
          </p>
        </div>

        <div style={{
          position: 'absolute',
          bottom: '30px',
          display: 'flex',
          gap: '15px'
        }}>
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                background: 'rgba(255, 255, 255, 0.5)',
                animation: `pulse 2s ease-in-out infinite ${i * 0.3}s`
              }}
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      position: 'relative',
      overflow: 'hidden',
      background: '#000'
    }}>
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
          objectFit: 'cover'
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
          objectFit: 'cover'
        }}
      />

      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        right: '20px',
        background: 'rgba(0, 0, 0, 0.7)',
        padding: '15px',
        borderRadius: '12px',
        color: '#fff',
        backdropFilter: 'blur(5px)'
      }}>
        <h2 style={{ fontSize: '1.5rem', marginBottom: '10px' }}>TruePath Active</h2>
        <p style={{ fontSize: '1rem', color: '#cbd5e1' }}>
          {detections.length} object{detections.length !== 1 ? 's' : ''} detected
        </p>
      </div>

      {detections.length > 0 && (
        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          right: '20px',
          background: 'rgba(0, 0, 0, 0.8)',
          padding: '15px',
          borderRadius: '12px',
          maxHeight: '200px',
          overflowY: 'auto',
          backdropFilter: 'blur(5px)'
        }}>
          {detections.map((detection, index) => (
            <div key={index} style={{
              marginBottom: '10px',
              padding: '10px',
              background: 'rgba(56, 189, 248, 0.2)',
              borderRadius: '8px',
              borderLeft: '4px solid #38bdf8'
            }}>
              <p style={{ fontSize: '1.1rem', fontWeight: 'bold', color: '#fff' }}>
                {detection.class}
              </p>
              <p style={{ fontSize: '0.9rem', color: '#cbd5e1' }}>
                Direction: {detection.direction} | Distance: {formatDistance(detection.distance)}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
