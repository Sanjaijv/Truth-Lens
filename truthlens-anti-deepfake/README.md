# TruthLens – Anti-Deepfake Video Call Guard 🛡️

**Real-time Deepfake Detection for Google Meet, Zoom, and WebRTC.**

## The Problem
Deepfakes aren't just for viral videos anymore. Sophisticated attackers are now using real-time face-swapping AI to impersonate executives and family members during live video calls. Traditional forensic tools are too slow; we need a shield that works **live**.

## How It Works (The "Liveness" Engine)
TruthLens runs entirely in your browser using **Sensor Fusion**. Instead of relying on a single fallible metric, we combine three distinct biological signals:

1.  **Geometric Stability (40%)**: Deepfakes often "jitter" or warp. We track 468 facial landmarks to detect if the nose or chin moves inconsistent with the head's 3D rotation.
2.  **Blink Patterns (35%)**: Real humans blink naturally (10-20 times/min). Deepfakes often stare (frozen) or flutter (glitchy). We use Eye Aspect Ratio (EAR) to score naturalness.
3.  **Pulse Detection (25%)**: Real faces flush with invisible blood flow (rPPG). We analyze pixel-level color changes in the cheeks to detect a heartbeat signal that generative AI fails to replicate.

## Why TensorFlow.js?
**Privacy & Performance.**
By running `tfjs` directly in the Chrome Extension:
- **Zero Latency**: No round-trip to a server. Analysis happens at 15 FPS locally.
- **Privacy First**: Your video feed **never** leaves your laptop. We analyze landmarks locally and discard the frames immediately.

## Optional Escalation: UNITE Protocol
If TruthLens detects a high risk (>80%) for a sustained period (3+ seconds), it offers an optional escalation:
- Captures a encrypted, short 3-second forensic clip.
- Sends it to our secure Cloud Forensic Engine for deep analysis.
- Returns a verified advisory label.
*This feature is strictly opt-in and triggered only by extreme anomalies.*

## Running the Project
1.  **Build**: Run the build script to bundle the TF.js modules (requires Webpack/Vite).
2.  **Load**: Go to `chrome://extensions`, enable Developer Mode, and "Load Unpacked".
3.  **Test**: Open a video call or YouTube video. The overlay will appear automatically.

---
*Built for the [Event Name] Hackathon.*
