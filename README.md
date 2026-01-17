# TruePath - AR Navigation Assistant

TruePath is a smartphone-based indoor navigation system that uses real-time object detection, distance estimation, and voice guidance to help blind and low-vision users navigate indoor spaces independently.

## Features

### ✓ Real-Time Object Detection
- Uses TensorFlow.js with COCO-SSD model for accurate object detection
- Detects doors, walls, furniture, people, and common obstacles
- Confidence threshold of 60% ensures reliable detection

### ✓ Accurate Distance Measurement
- Focal length-based distance estimation using known object heights
- Provides measurements in meters and centimeters
- Calibrated for common indoor objects (doors, chairs, tables, people, etc.)

### ✓ Voice-Over Accessibility
- Automatic voice announcements when app opens
- Continuous audio feedback for detected objects
- Announces object type, direction, and distance
- Smart announcement throttling (3-second minimum between repeated announcements)

### ✓ Clear Camera View
- Full-screen camera with no background blur or obstruction
- Real-time visual overlays showing detected objects
- Bounding boxes and labels for visual reference

### ✓ Double-Tap Navigation
- Simple double-tap gesture to start camera from intro screen
- No complex button pressing required
- Fully accessible interaction model

## How It Works

1. **Welcome Screen**: App opens with a voice-over introduction explaining TruePath
2. **Double-Tap to Start**: User double-taps anywhere on screen to activate camera
3. **Real-Time Detection**: Camera continuously scans environment and detects objects
4. **Voice Announcements**: System speaks detected objects with direction and distance
5. **Visual Feedback**: On-screen overlays show detected objects for sighted helpers

## Technical Stack

- **Frontend**: Next.js 14 with TypeScript
- **Object Detection**: TensorFlow.js + COCO-SSD (lite_mobilenet_v2)
- **Voice Output**: Web Speech API
- **Camera**: WebRTC getUserMedia API
- **Distance Estimation**: Focal length-based algorithm with known object heights

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

**Important**: Camera access requires HTTPS in production. For local development, localhost is allowed.

### Build for Production

```bash
npm run build
npm start
```

## Browser Compatibility

- Chrome/Edge: Full support
- Safari: Full support (iOS 11+)
- Firefox: Full support
- Requires camera permissions

## Distance Estimation Algorithm

TruePath uses focal length-based distance estimation:

```
Distance = (Known Object Height × Focal Length) / Pixel Height
```

Known heights for common objects:
- Person: 1.7m
- Door: 2.0m
- Chair: 0.9m
- Table: 0.75m
- And many more...

## Accessibility Features

1. **No Visual Interaction Required**: All navigation via double-tap and voice
2. **Immediate Voice Feedback**: App announces its state and actions
3. **Continuous Announcements**: Objects announced as soon as detected
4. **High Contrast UI**: Visual elements use high-contrast colors for low-vision users
5. **Large Touch Targets**: Full-screen interaction areas

## Future Enhancements

- ARCore/ARKit integration for improved depth sensing
- Custom YOLO-tiny model trained on indoor navigation
- Haptic feedback patterns for direction guidance
- Offline detection model
- Path planning and route guidance
- Integration with building floor plans

## Team

- Dhruv Jadhav - Research & Backend Development
- Abhay Gowda - Text-to-Speech & Frontend Controls
- Subinay Kalya - AI Vision System & Model Integration
- Harshith Gande - Frontend Interface & UX Design

## License

Developed for Jackson TSA Pre-Regional Event 2024-2025