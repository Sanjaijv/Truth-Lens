console.log("TruthLens: Content Script Loaded (State Machine Mode)");
if (window.hasTruthLensContentScript) throw new Error("TruthLens already loaded");
window.hasTruthLensContentScript = true;

import { TruthLensOverlay } from './overlay.js';
import './styles.css';

class AnalysisStateMachine {
    constructor(video) {
        this.video = video;
        this.state = 'IDLE'; // IDLE, ROI_VISIBLE, ANALYZING, COMPLETED
        this.overlay = new TruthLensOverlay();
        this.canvas = null;
        this.ctx = null;
        this.worker = null;

        // Loop Control
        this.animationId = null;
        this.lastFrameTime = 0;
        this.FPS = 30; // High FPS for UI smoothness

        // ROI Data
        this.roiBox = null; // {x, y, w, h} (Video Coordinates)

        // Analysis Data
        this.framesProcessed = 0;
        this.totalRiskScore = 0;
        this.SCAN_DURATION = 90; // Frames (~3 sec)

        this.init();
    }

    async init() {
        console.log("TruthLens: Init started.");

        // 1. Setup Overlay (Sidebar)
        this.overlay.createOverlay();
        this.overlay.onClose = () => this.stop("User Cancelled");

        // 2. Setup Canvas
        this.setupCanvas();

        // 3. Setup Worker via Iframe Proxy (Bypass SOP)
        this.setupWorkerProxy();
    }

    setupWorkerProxy() {
        try {
            const frameUrl = chrome.runtime.getURL('dist/frame.html');
            console.log(`TruthLens: Creating Worker Proxy Frame at ${frameUrl}`);

            this.proxyFrame = document.createElement('iframe');
            this.proxyFrame.src = frameUrl;
            this.proxyFrame.style.display = 'none';
            this.proxyFrame.id = 'truthlens-worker-frame';
            document.body.appendChild(this.proxyFrame);

            // Listener for Frame Messages
            window.addEventListener('message', (event) => {
                // Ensure message is from our extension
                // logic: chrome-extension://<id>
                if (!event.data || event.data.source !== 'truthlens-worker') return;

                if (event.data.type === 'frame-ready') {
                    this.workerReady = true;
                    this.overlay.log("Worker Proxy Ready.");
                } else {
                    this.handleWorkerMessage({ data: event.data });
                }
            });

            // Handshake after load
            this.proxyFrame.onload = () => {
                this.overlay.log("Proxy Frame Loaded. Initializing Worker...");
                // Send INIT to frame
                this.proxyFrame.contentWindow.postMessage({
                    type: 'INIT_WORKER'
                }, '*'); // Target any origin (frame is chrome-ext)
            };

        } catch (e) {
            this.overlay.log(`Worker Proxy Failed: ${e.message}`);
            console.error("Worker Proxy Failed:", e);
        }

        // 4. Start Continuous Render Loop (Video Mirroring)
        // Wait for worker ready? No, start loop but check state
        this.startRenderLoop();

        this.overlay.log("System Initialized. Sidebar Open.");
    }

    setupCanvas() {
        if (!this.overlay.element) return;
        this.canvas = this.overlay.element.querySelector('.truthlens-overlay-canvas');
        if (!this.canvas) {
            console.error("Canvas element not found in overlay!");
            return;
        }
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

        // Match Video Dimensions exactly
        this.canvas.width = this.video.videoWidth || 640;
        this.canvas.height = this.video.videoHeight || 480;

        // Offscreen canvas for raw capture (to avoid scaling artifacts)
        this.captureCanvas = document.createElement('canvas');
        this.captureCanvas.width = this.canvas.width;
        this.captureCanvas.height = this.canvas.height;
        this.captureCtx = this.captureCanvas.getContext('2d', { willReadFrequently: true });

        console.log(`Canvas setup: ${this.canvas.width}x${this.canvas.height}`);
        this.overlay.log(`Canvas: ${this.canvas.width}x${this.canvas.height}`);
    }

    startRenderLoop() {
        if (!this.video || this.video.paused || this.video.ended) {
            // If video paused, we might still want to run loop to clear/draw UI, 
            // but strictly speaking we only need to analyze playing video.
            // We'll keep checking.
        }

        const loop = (now, metadata) => {
            this.renderFrame();
            // Use rVFC if available for tight sync, else fallback to rAF
            if (this.video.requestVideoFrameCallback) {
                this.video.requestVideoFrameCallback(loop);
            } else {
                requestAnimationFrame(() => loop(performance.now(), null));
            }
        };

        if (this.video.requestVideoFrameCallback) {
            this.video.requestVideoFrameCallback(loop);
        } else {
            requestAnimationFrame(() => loop(performance.now(), null));
        }
    }

    renderFrame() {
        if (!this.ctx || !this.canvas || !this.video) return;

        // 1. Clear & Background
        const cW = this.canvas.width;
        const cH = this.canvas.height;
        this.ctx.fillStyle = '#0a0a0a'; // Dark, not pitch black
        this.ctx.fillRect(0, 0, cW, cH);

        // --- VISUAL HEARTBEAT ---
        // Draw a pulsing circle in top-left to prove loop is running
        const pulse = Math.sin(performance.now() / 200) * 0.5 + 0.5;
        this.ctx.fillStyle = `rgba(255, 0, 0, ${pulse})`;
        this.ctx.beginPath();
        this.ctx.arc(20, 20, 10, 0, Math.PI * 2);
        this.ctx.fill();

        // Print Debug info ON CANVAS
        this.ctx.fillStyle = 'white';
        this.ctx.font = '12px monospace';
        this.ctx.fillText(`State: ${this.state}`, 40, 25);
        this.ctx.fillText(`VidReady: ${this.video.readyState}`, 40, 40);
        this.ctx.fillText(`Size: ${this.video.videoWidth}x${this.video.videoHeight}`, 40, 55);
        // ------------------------

        // Heartbeat Debug (Throttled log)
        if (Math.random() < 0.005) {
            console.log(`[TruthLens] Loop Alive. ReadyState=${this.video.readyState}`);
        }

        // 2. Draw Source Video (Mirror)
        if (this.video.readyState >= 1) {
            const vW = this.video.videoWidth;
            const vH = this.video.videoHeight;
            const scale = Math.min(cW / vW, cH / vH);
            const drawW = vW * scale;
            const drawH = vH * scale;
            const drawX = (cW - drawW) / 2;
            const drawY = (cH - drawH) / 2;

            try {
                this.ctx.drawImage(this.video, drawX, drawY, drawW, drawH);
            } catch (drawErr) {
                this.ctx.fillStyle = 'red';
                this.ctx.fillText(`Draw Error: ${drawErr.name}`, 10, 100);
            }

            // 3. Draw Red Box (if exists)
            if (this.roiBox) {
                const boxX = (this.roiBox.xMin * scale) + drawX;
                const boxY = (this.roiBox.yMin * scale) + drawY;
                const boxW = this.roiBox.width * scale;
                const boxH = this.roiBox.height * scale;

                this.ctx.strokeStyle = '#ef4444';
                this.ctx.lineWidth = 3;
                this.ctx.strokeRect(boxX, boxY, boxW, boxH);
            }

            // 4. Send Frame to Worker
            this.feedWorker(vW, vH);
        } else {
            this.ctx.fillStyle = 'yellow';
            this.ctx.fillText("Waiting for Video Data...", cW / 2 - 50, cH / 2);
        }
    }

    async feedWorker(vW, vH) {
        if (this.state === 'COMPLETED') return;
        if (!this.proxyFrame) return;

        const now = performance.now();
        if (now - this.lastFrameTime < (1000 / 10)) return; // Cap at 10 FPS
        this.lastFrameTime = now;

        try {
            // Resize capture canvas if needed
            if (this.captureCanvas.width !== vW || this.captureCanvas.height !== vH) {
                this.captureCanvas.width = vW;
                this.captureCanvas.height = vH;
            }

            // Draw to capture canvas
            this.captureCtx.drawImage(this.video, 0, 0, vW, vH);

            // Extract Pixel Data
            // This is where CORS security error might happen
            const imageData = this.captureCtx.getImageData(0, 0, vW, vH);

            // Transfer buffer to worker
            const message = {
                type: 'FRAME',
                pixels: imageData.data.buffer,
                width: vW,
                height: vH,
                timestamp: now
            };

            if (this.proxyFrame && this.proxyFrame.contentWindow) {
                this.proxyFrame.contentWindow.postMessage(message, '*', [imageData.data.buffer]);
            }

        } catch (err) {
            // Check for CORS SecurityError
            if (err.name === 'SecurityError') {
                if (Math.random() < 0.05) this.overlay.log("CORS: Video tainting detected.");

                // Try to mitigate
                // if (!this.video.crossOrigin) { 
                //    destructive hack removed
                // }
                this.overlay.updateStatus("Error: Protected Content (CORS)", 'fake');
            } else {
                if (Math.random() < 0.05) this.overlay.log(`Capture Error: ${err.message}`);
                console.error("Frame capture error:", err);
            }
        }
    }

    handleWorkerMessage(e) {
        const { type, data } = e.data;
        if (type === 'INIT_COMPLETE') {
            this.overlay.log("Worker Ready. Model Loaded.");
        } else if (type === 'PROGRESS') {
            this.overlay.updateStatus(data.status);
            this.overlay.log(data.status);
        } else if (type === 'ERROR') {
            this.overlay.log(`Worker Critical: ${data.error}`);
            this.overlay.showError(data.error);
        } else if (type === 'RESULT') {
            // this.overlay.log(`Result: found=${data.found}`); // Too spammy? Maybe just errors or state changes.
            this.processResult(data);
        }
    }

    transition(newState) {
        this.overlay.log(`State: ${this.state} -> ${newState}`);
        this.state = newState;

        // UI Updates based on state
        if (newState === 'ROI_VISIBLE') {
            this.overlay.updateStatus("Target Acquired. Analyzing...");
            // Auto-start analysis
            this.transition('ANALYZING');
        } else if (newState === 'ANALYZING') {
            // Keep Going
        } else if (newState === 'COMPLETED') {
            this.showResults();
        }
    }

    // StopLoop is just for legacy cleanup, mostly handled by cancelAnimationFrame now
    stopLoop() {
        // No-op
    }

    drawRedBox() {
        // Deprecated: Logic moved to renderFrame() which runs every tick.
        // We update state here if needed, but drawing happens in loop.
    }

    resetROI() {
        this.roiBox = null;
        this.clearCanvas();
    }

    clearCanvas() {
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    showResults() {
        const validCount = this.validFrames || 0;

        // If we didn't get enough data points, we can't give a verdict
        if (validCount < (this.SCAN_DURATION * 0.2)) {
            this.overlay.showVerdict('No Face Detected', 0, () => this.resetSession());
            return;
        }

        const avgRisk = this.totalRiskScore / validCount;
        const finalScore = Math.round((1 - avgRisk) * 100);
        let label = 'Safe';
        if (avgRisk > 0.6) label = 'Fake';
        else if (avgRisk > 0.3) label = 'Suspicious';

        this.overlay.showVerdict(label, finalScore, () => this.resetSession());
    }

    processResult(data) {
        if (this.state === 'COMPLETED') return;

        if (!data.found && this.state !== 'IDLE') {
            if (this.state === 'ANALYZING') {
                this.missedFrames = (this.missedFrames || 0) + 1;
                // If lost for too long, clear roiBox so it vanishes from renderLoop
                if (this.missedFrames > 15) {
                    this.roiBox = null;
                }
            }
            return;
        }

        if (data.found) {
            this.missedFrames = 0;
            this.roiBox = data.box; // Updated for renderLoop

            // State Logic
            if (this.state === 'IDLE') {
                this.transition('ROI_VISIBLE');
            } else if (this.state === 'ANALYZING') {
                this.framesProcessed++;
                this.totalRiskScore += (data.risk ? data.risk.score : 0);
                this.lastRiskScore = (data.risk ? data.risk.score : 0);

                const pct = Math.round((this.framesProcessed / this.SCAN_DURATION) * 100);
                this.overlay.updateStatus(`Analyzing frame ${this.framesProcessed}... ${pct}%`);

                if (this.framesProcessed >= this.SCAN_DURATION) {
                    this.transition('COMPLETED');
                }
            }
        }
    }

    resetSession() {
        this.framesProcessed = 0;
        this.totalRiskScore = 0;
        this.validFrames = 0;
        this.transition('IDLE');
    }

    stop(reason) {
        console.log("Analysis Stopped:", reason);
        if (this.animationId) cancelAnimationFrame(this.animationId);
        this.stopLoop(); // Stop interval if any (from old code)
        if (this.worker) this.worker.terminate();

        this.resetROI();
        this.overlay.remove();

        // Allow GC
        currentSession = null;
    }
}

// Singleton-like session
let currentSession = null;



chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log("Content Script Received Message:", request);
    if (request.action === "ANALYZE_VIDEO") {

        // 1. Immediate Visual Feedback (Vital check)
        if (currentSession) {
            currentSession.stop("Restarting");
        }

        // Helper to find video
        const findVideo = () => {
            const videos = Array.from(document.querySelectorAll('video'))
                .filter(v => v.videoWidth > 100 && v.videoHeight > 100);

            // Sort: Playing videos first, then by size
            videos.sort((a, b) => {
                const aPlaying = !a.paused && !a.ended && a.readyState > 2;
                const bPlaying = !b.paused && !b.ended && b.readyState > 2;
                if (aPlaying && !bPlaying) return -1;
                if (!aPlaying && bPlaying) return 1;
                return (b.videoWidth * b.videoHeight) - (a.videoWidth * a.videoHeight);
            });

            return videos[0];
        };

        // Retry logic (5 attempts over 2.5 seconds)
        let attempts = 0;
        const maxAttempts = 10;

        const attemptFind = () => {
            const bestVideo = findVideo();
            if (bestVideo) {
                // If video found but not fully ready, wait for it?
                if (bestVideo.readyState === 0) {
                    console.log("TruthLens: Video found but not ready. Waiting...");
                    bestVideo.addEventListener('loadeddata', () => {
                        if (!currentSession) currentSession = new AnalysisStateMachine(bestVideo);
                    }, { once: true });
                    // Or just start, and the render loop handles waiting (which it does)
                    // The render loop checks readyState.
                    currentSession = new AnalysisStateMachine(bestVideo);
                } else {
                    currentSession = new AnalysisStateMachine(bestVideo);
                }
            } else if (attempts < maxAttempts) {
                attempts++;
                console.log(`TruthLens: No video found. Retrying (${attempts}/${maxAttempts})...`);
                setTimeout(attemptFind, 500);
            } else {
                console.warn("No suitable video found after retries.");
                const errOverlay = new TruthLensOverlay();
                errOverlay.showError("TruthLens: No playable video found on this page.");
            }
        };

        attemptFind();
    }
});

