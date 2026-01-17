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

export default function Home() {
  const [showIntro, setShowIntro] = useState(true);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [lastAnnouncement, setLastAnnouncement] = useState<{ [key: string]: number }>({});

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number>();
  const tapCountRef = useRef(0);
  const tapTimerRef = useRef<NodeJS.Timeout>();

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
    ctx.font = '18px Arial';
    ctx.fillStyle = '#00ff00';

    detectedObjects.forEach(detection => {
      const [x, y, width, height] = detection.bbox;

      ctx.strokeRect(x, y, width, height);

      const label = `${detection.class} ${formatDistance(detection.distance)}`;
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x, y - 25, textWidth + 10, 25);

      ctx.fillStyle = '#00ff00';
      ctx.fillText(label, x + 5, y - 7);
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
      }, 300);
    }
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
          background: 'linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%)',
          padding: '20px',
          cursor: 'pointer'
        }}
      >
        <h1 style={{
          fontSize: '3rem',
          fontWeight: 'bold',
          marginBottom: '20px',
          textAlign: 'center',
          color: '#fff'
        }}>
          TruePath
        </h1>
        <p style={{
          fontSize: '1.5rem',
          textAlign: 'center',
          lineHeight: '2',
          maxWidth: '800px',
          color: '#cbd5e1'
        }}>
          Welcome to TruePath, your AR navigation assistant.
        </p>
        <p style={{
          fontSize: '1.3rem',
          textAlign: 'center',
          lineHeight: '1.8',
          maxWidth: '800px',
          marginTop: '20px',
          color: '#94a3b8'
        }}>
          This app will help you navigate indoor spaces by detecting objects and measuring distances.
        </p>
        <p style={{
          fontSize: '1.5rem',
          textAlign: 'center',
          marginTop: '40px',
          padding: '20px',
          background: 'rgba(56, 189, 248, 0.2)',
          borderRadius: '12px',
          color: '#7dd3fc',
          fontWeight: '600'
        }}>
          Double tap anywhere on the screen to start
        </p>
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
        color: '#fff'
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
          overflowY: 'auto'
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
