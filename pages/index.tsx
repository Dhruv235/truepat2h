import { useEffect, useRef, useState, useCallback } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs';
import { speak, stopSpeaking } from '../utils/voice';
import { estimateDistance, getDetailedPosition, formatDistance } from '../utils/distance';
import { extractTextFromImage, extractProductInfo, matchProduct } from '../utils/ocr';
import { createSpeechRecognition } from '../utils/speechRecognition';

interface ProductDetection {
  class: string;
  score: number;
  bbox: [number, number, number, number];
  distance: number;
  position: string;
  ocrText?: string;
  productInfo?: {
    brand?: string;
    productName?: string;
    size?: string;
  };
  matchResult?: {
    match: boolean;
    confidence: number;
    matchedFields: string[];
  };
}

interface CartItem {
  id: string;
  productName: string;
  brand?: string;
  size?: string;
  detectedAt: number;
}

type AppMode = 'intro' | 'search' | 'cartCheck';

interface SmoothedBBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export default function Home() {
  const [mode, setMode] = useState<AppMode>('intro');
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [detections, setDetections] = useState<ProductDetection[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [isSearching, setIsSearching] = useState(false);
  const [cartItems, setCartItems] = useState<CartItem[]>([]);
  const [lastAnnouncement, setLastAnnouncement] = useState<{ [key: string]: number }>({});
  const [tapAnimation, setTapAnimation] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [ocrProcessing, setOcrProcessing] = useState(false);
  const [lastReadText, setLastReadText] = useState<string>('');
  const [readItemMode, setReadItemMode] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number>();
  const tapCountRef = useRef(0);
  const tapTimerRef = useRef<NodeJS.Timeout>();
  const speechRecognitionRef = useRef(createSpeechRecognition());
  const frameSkipRef = useRef(0);
  const lastReadTimeRef = useRef<number>(0);
  const readItemCooldownRef = useRef<number>(0);
  
  // Smoothing buffer for bounding boxes
  const smoothingBuffer = useRef<Map<string, SmoothedBBox[]>>(new Map());
  const SMOOTHING_FRAMES = 5;
  const SMOOTHING_ALPHA = 0.3;
  const FRAME_SKIP = 3; // Process every 3rd frame for performance

  useEffect(() => {
    if (mode === 'intro') {
      const introMessage =
        'Welcome to ShelfSight. ' +
        'Hold any item close to the camera and I will read it to you. ' +
        'Double tap to start.';
      setTimeout(() => {
        speak(introMessage, true);
      }, 500);
    }
  }, [mode]);

  useEffect(() => {
    if (mode !== 'intro') {
      loadModel();
      startCamera();
    }

    return () => {
      stopCamera();
      stopSpeaking();
      speechRecognitionRef.current.stopListening();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [mode]);

  const loadModel = async () => {
    try {
      const loadedModel = await cocoSsd.load({
        base: 'lite_mobilenet_v2'
      });
      setModel(loadedModel);
      speak('Detection model loaded. Ready to scan shelves.');
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
        speak('Camera started. Hold an item close to the camera and I will read it to you.');
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

  const readItemInHand = async (
    detection: ProductDetection,
    canvas: HTMLCanvasElement
  ): Promise<void> => {
    const now = Date.now();
    // Cooldown to prevent reading the same item repeatedly
    if (now - lastReadTimeRef.current < 2000) {
      return;
    }

    // Extract region of interest for OCR - use the entire detected bounding box
    const [x, y, width, height] = detection.bbox;
    const roiX = Math.max(0, Math.floor(x));
    const roiY = Math.max(0, Math.floor(y));
    const roiWidth = Math.min(canvas.width - roiX, Math.floor(width));
    const roiHeight = Math.min(canvas.height - roiY, Math.floor(height));

    // Require minimum size for OCR
    if (roiWidth < 80 || roiHeight < 80) {
      return;
    }

    try {
      setOcrProcessing(true);
      const roiCanvas = document.createElement('canvas');
      roiCanvas.width = roiWidth;
      roiCanvas.height = roiHeight;
      const roiCtx = roiCanvas.getContext('2d');
      
      if (roiCtx) {
        roiCtx.drawImage(
          canvas,
          roiX, roiY, roiWidth, roiHeight,
          0, 0, roiWidth, roiHeight
        );
        
        const ocrResult = await extractTextFromImage(roiCanvas);
        const productInfo = extractProductInfo(ocrResult.text);
        
        lastReadTimeRef.current = now;
        
        // Format the announcement - prioritize readable text
        let announcement = '';
        
        if (ocrResult.text.trim().length > 0) {
          // If we have product info, use it
          if (productInfo.brand && productInfo.productName) {
            announcement = `${productInfo.brand} ${productInfo.productName}`;
            if (productInfo.size) {
              announcement += `, ${productInfo.size}`;
            }
          } else if (ocrResult.text.trim().length < 150) {
            // If text is reasonably short, read it directly
            announcement = ocrResult.text.trim();
          } else {
            // If text is long, read first meaningful part
            const words = ocrResult.text.trim().split(/\s+/).slice(0, 25);
            announcement = words.join(' ');
          }
          
          if (announcement) {
            setLastReadText(announcement);
            speak(announcement, false);
          } else {
            // Fallback to object class
            speak(`Detected ${detection.class}`, false);
          }
        } else {
          // No text found, just announce the object type
          speak(`Detected ${detection.class}`, false);
        }
        
        setOcrProcessing(false);
      }
    } catch (error) {
      console.error('Error reading item:', error);
      setOcrProcessing(false);
      // Fallback announcement
      speak(`Detected ${detection.class}`, false);
    }
  };

  const processProductDetection = async (
    detection: ProductDetection,
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D
  ): Promise<ProductDetection> => {
    if (!searchQuery || mode !== 'search') {
      return detection;
    }

    // Extract region of interest for OCR
    const [x, y, width, height] = detection.bbox;
    const roiX = Math.max(0, Math.floor(x));
    const roiY = Math.max(0, Math.floor(y));
    const roiWidth = Math.min(canvas.width - roiX, Math.floor(width));
    const roiHeight = Math.min(canvas.height - roiY, Math.floor(height));

    if (roiWidth < 50 || roiHeight < 50) {
      return detection;
    }

    try {
      const roiCanvas = document.createElement('canvas');
      roiCanvas.width = roiWidth;
      roiCanvas.height = roiHeight;
      const roiCtx = roiCanvas.getContext('2d');
      
      if (roiCtx) {
        roiCtx.drawImage(
          canvas,
          roiX, roiY, roiWidth, roiHeight,
          0, 0, roiWidth, roiHeight
        );
        
        const ocrResult = await extractTextFromImage(roiCanvas);
        const productInfo = extractProductInfo(ocrResult.text);
        const matchResult = matchProduct(searchQuery, ocrResult.text, productInfo.brand);
        
        return {
          ...detection,
          ocrText: ocrResult.text,
          productInfo,
          matchResult
        };
      }
    } catch (error) {
      console.error('OCR processing error:', error);
    }

    return detection;
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

    // Frame skipping for performance
    frameSkipRef.current++;
    if (frameSkipRef.current % FRAME_SKIP !== 0) {
      animationRef.current = requestAnimationFrame(detectObjects);
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const predictions = await model.detect(canvas);

    // ONLY detect objects that are close (in hand) - filter by distance first
    const detectedObjects: ProductDetection[] = predictions
      .filter(p => p.score > 0.4) // Lower threshold to catch more items
      .map(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const distance = estimateDistance(
          prediction.class,
          height,
          canvas.height
        );
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        const position = getDetailedPosition(centerX, centerY, canvas.width, canvas.height);

        return {
          class: prediction.class,
          score: prediction.score,
          bbox: prediction.bbox,
          distance,
          position
        };
      });

    // ONLY process items that are close (in hand) - distance < 1.2 meters
    const itemsInHand = detectedObjects.filter(det => det.distance < 1.2);
    
    // Update detections to only show items in hand
    setDetections(itemsInHand);

    // Read the item in hand automatically
    if (itemsInHand.length > 0 && !ocrProcessing) {
      // Find the largest/closest object (most likely the main item being held)
      const itemInHand = itemsInHand.reduce((prev, current) => {
        const prevSize = prev.bbox[2] * prev.bbox[3];
        const currentSize = current.bbox[2] * current.bbox[3];
        // Prefer larger objects, but if similar size, prefer closer
        if (Math.abs(prevSize - currentSize) < prevSize * 0.2) {
          return current.distance < prev.distance ? current : prev;
        }
        return currentSize > prevSize ? current : prev;
      });
      
      // Read if it's been at least 2 seconds since last read
      const now = Date.now();
      if (now - lastReadTimeRef.current > 2000) {
        readItemInHand(itemInHand, canvas);
      }
    }

    // Draw bounding boxes ONLY for items in hand
    if (itemsInHand.length > 0) {
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 4;
      ctx.font = 'bold 18px Arial';
      ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
      ctx.shadowBlur = 4;

      itemsInHand.forEach(detection => {
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

        const label = detection.class;
        ctx.font = 'bold 18px Arial';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(smoothed.x, smoothed.y - 30, textWidth + 20, 30);

        ctx.fillStyle = '#00ff00';
        ctx.fillText(label, smoothed.x + 10, smoothed.y - 8);
        
        ctx.shadowColor = 'transparent';
        ctx.shadowBlur = 0;
      });
    }

    animationRef.current = requestAnimationFrame(detectObjects);
  }, [model, ocrProcessing]);

  useEffect(() => {
    if (mode !== 'intro' && model && videoRef.current) {
      videoRef.current.addEventListener('loadeddata', () => {
        animationRef.current = requestAnimationFrame(detectObjects);
      });
    }
  }, [mode, model, detectObjects]);

  const handleDoubleTap = () => {
    tapCountRef.current += 1;
    setTapAnimation(true);

    if (tapTimerRef.current) {
      clearTimeout(tapTimerRef.current);
    }

    if (tapCountRef.current === 2) {
      if (mode === 'intro') {
        stopSpeaking();
        setMode('search');
        speak('Search mode activated. Say "search" followed by the product name to find items, or "cart check" to verify your cart.');
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

  const handleVoiceCommand = (transcript: string) => {
    const command = transcript.toLowerCase().trim();
    
    if (command.startsWith('search')) {
      const query = command.replace(/^search\s+/, '');
      if (query) {
        setSearchQuery(query);
        setIsSearching(true);
        setMode('search');
        setReadItemMode(false);
        speak(`Searching for ${query}. Point your camera at the shelf.`);
      } else {
        startVoiceSearch();
      }
    } else if (command.includes('cart check') || command.includes('check cart')) {
      setMode('cartCheck');
      setSearchQuery('');
      setReadItemMode(false);
      speak('Cart check mode. Point your camera at your shopping cart to verify items.');
    } else if (command.includes('read item') || command.includes('read') || command.includes('what is this')) {
      setReadItemMode(true);
      setMode('search');
      speak('Read item mode activated. Hold an item close to the camera and I will read it to you.');
    } else if (command.includes('clear') || command.includes('reset')) {
      setSearchQuery('');
      setIsSearching(false);
      setReadItemMode(false);
      speak('Search cleared.');
    } else if (command) {
      // Treat as search query
      setSearchQuery(command);
      setIsSearching(true);
      setMode('search');
      setReadItemMode(false);
      speak(`Searching for ${command}. Point your camera at the shelf.`);
    }
  };

  const startVoiceSearch = () => {
    if (!speechRecognitionRef.current.isSupported()) {
      speak('Voice input is not supported in this browser. Please type your search query.');
      return;
    }

    setIsListening(true);
    speak('Listening for your search query...');
    
    speechRecognitionRef.current.startListening(
      (result) => {
        setIsListening(false);
        handleVoiceCommand(result);
      },
      (error) => {
        setIsListening(false);
        speak(`Error with voice input: ${error}`);
      }
    );
  };

  const addToCart = (detection: ProductDetection) => {
    const cartItem: CartItem = {
      id: `${Date.now()}-${Math.random()}`,
      productName: detection.productInfo?.productName || detection.class,
      brand: detection.productInfo?.brand,
      size: detection.productInfo?.size,
      detectedAt: Date.now()
    };
    
    setCartItems(prev => [...prev, cartItem]);
    speak(`Added ${cartItem.productName} to cart.`);
  };

  const performCartCheck = async () => {
    if (!canvasRef.current || cartItems.length === 0) {
      speak('No items in cart to check.');
      return;
    }

    speak('Scanning cart items...');
    setOcrProcessing(true);

    try {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Process current frame for cart items
      const ocrResult = await extractTextFromImage(canvas);
      const productInfo = extractProductInfo(ocrResult.text);

      // Check each cart item
      const issues: string[] = [];
      cartItems.forEach(item => {
        const match = matchProduct(
          `${item.brand || ''} ${item.productName}`,
          ocrResult.text,
          productInfo.brand
        );

        if (!match.match) {
          issues.push(`${item.productName} not found in cart`);
        }
      });

      if (issues.length === 0) {
        speak('Cart check complete. All items verified.');
      } else {
        speak(`Cart check found issues: ${issues.join('. ')}`);
      }
    } catch (error) {
      console.error('Cart check error:', error);
      speak('Error during cart check.');
    } finally {
      setOcrProcessing(false);
    }
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
            ShelfSight
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
            Your Accessible Grocery Shopping Assistant
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
            Find products on shelves and verify your cart with confidence
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

      {/* Top Control Bar */}
      <div style={{
        position: 'absolute',
        top: '20px',
        left: '20px',
        right: '20px',
        background: 'rgba(0, 0, 0, 0.8)',
        padding: '15px',
        borderRadius: '12px',
        color: '#fff',
        backdropFilter: 'blur(5px)',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexWrap: 'wrap',
        gap: '10px'
      }}>
        <div>
          <h2 style={{ fontSize: '1.3rem', marginBottom: '5px' }}>ShelfSight</h2>
          <p style={{ fontSize: '0.9rem', color: '#cbd5e1' }}>
            {ocrProcessing ? 'Reading item...' : 'Hold an item close to read it'}
          </p>
        </div>
      </div>

      {/* Last Read Text Display */}
      {lastReadText && (
        <div style={{
          position: 'absolute',
          top: '100px',
          left: '20px',
          right: '20px',
          background: 'rgba(156, 39, 176, 0.9)',
          padding: '15px',
          borderRadius: '12px',
          color: '#fff',
          backdropFilter: 'blur(5px)',
          border: '2px solid rgba(255, 255, 255, 0.3)',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)'
        }}>
          <p style={{ fontSize: '0.9rem', fontWeight: 'bold', marginBottom: '5px' }}>
            📖 Last Read:
          </p>
          <p style={{ fontSize: '1rem', lineHeight: '1.4' }}>
            {lastReadText}
          </p>
        </div>
      )}

      {/* Status indicator when no item detected */}
      {detections.length === 0 && !ocrProcessing && (
        <div style={{
          position: 'absolute',
          bottom: '20px',
          left: '20px',
          right: '20px',
          background: 'rgba(0, 0, 0, 0.7)',
          padding: '20px',
          borderRadius: '12px',
          textAlign: 'center',
          backdropFilter: 'blur(5px)'
        }}>
          <p style={{ fontSize: '1.1rem', color: '#fff', marginBottom: '10px' }}>
            👋 Hold an item close to the camera
          </p>
          <p style={{ fontSize: '0.9rem', color: '#cbd5e1' }}>
            I will automatically read what you're holding
          </p>
        </div>
      )}
    </div>
  );
}
