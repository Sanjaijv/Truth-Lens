class TruthLensOverlay {
  constructor() {
    this.element = null;
    this.mirrorVideo = null;
    this.canvas = null;
    this.onClose = null;
  }
  createOverlay(targetParent) {
    if (this.element)
      this.remove();
    this.element = document.createElement("div");
    this.element.className = "truthlens-modal-overlay";
    this.element.innerHTML = `
            <div class="truthlens-modal-header">
                <span class="truthlens-modal-title">TruthLens</span>
                <button class="truthlens-modal-close-btn" id="tl-modal-close">Exit</button>
            </div>

            <div class="truthlens-video-wrapper">
                <canvas class="truthlens-overlay-canvas" style="pointer-events: none;"></canvas>
            </div>

            <div class="truthlens-modal-footer">
                <div class="truthlens-modal-status analyzing" id="tl-modal-status">
                    <span class="truthlens-indicator"></span>
                    <span id="tl-status-text">Initializing...</span>
                </div>
            </div>
            
            <div id="tl-debug-console" class="truthlens-debug-console"></div>
        `;
    document.body.appendChild(this.element);
    this.canvas = this.element.querySelector(".truthlens-overlay-canvas");
    this.element.querySelector("#tl-modal-close").onclick = () => {
      if (this.onClose)
        this.onClose();
      this.remove();
    };
  }
  updateStatus(text, statusClass = "analyzing") {
    if (!this.element)
      return;
    const statusEl = this.element.querySelector("#tl-modal-status");
    const textEl = this.element.querySelector("#tl-status-text");
    statusEl.classList.remove("analyzing", "safe", "fake", "suspicious");
    statusEl.classList.add(statusClass);
    textEl.textContent = text;
  }
  log(msg) {
    if (!this.element)
      return;
    const consoleEl = this.element.querySelector("#tl-debug-console");
    if (consoleEl) {
      const line = document.createElement("div");
      let formattedMsg = msg.replace(/(CORS|tainting|SecurityError|Protected Content)/gi, '<span class="tl-log-cors">$1</span>').replace(/(Error|Failed|Exception)/gi, '<span class="tl-log-error">$1</span>').replace(/(Warning|Suspicious)/gi, '<span class="tl-log-warn">$1</span>').replace(/(Info|Loaded|Ready)/gi, '<span class="tl-log-info">$1</span>');
      line.innerHTML = `> ${formattedMsg}`;
      consoleEl.appendChild(line);
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }
  }
  showVerdict(label, score, onRescan) {
    if (!this.element)
      return;
    let statusClass = "safe";
    if (label === "Fake")
      statusClass = "fake";
    else if (label === "Suspicious")
      statusClass = "suspicious";
    this.updateStatus(`Analysis Complete: ${label} (${score}% Confidence)`, statusClass);
  }
  showError(message) {
    if (!this.element)
      return;
    const footer = this.element.querySelector(".truthlens-modal-footer");
    if (!footer)
      return;
    let title = "Analysis Error";
    let desc = message;
    const isCORS = message.includes("CORS") || message.includes("Protected") || message.includes("SecurityError") || message.includes("Tainted");
    if (isCORS) {
      title = "Protected Content (CORS)";
      desc = "This video is protected by browser security policies and cannot be analyzed directly. Try opening the video in a new tab or use a different source.";
    }
    footer.innerHTML = `
            <div class="truthlens-error-alert">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="truthlens-error-icon"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                <div class="truthlens-error-content">
                    <span class="truthlens-error-title">${title}</span>
                    <span class="truthlens-error-desc">${desc}</span>
                </div>
            </div>
        `;
    this.log(`Error: ${message}`);
  }
  remove() {
    if (this.element) {
      this.element.remove();
      this.element = null;
      this.canvas = null;
    }
  }
}
const styles = "";
console.log("TruthLens: Content Script Loaded (State Machine Mode)");
if (window.hasTruthLensContentScript)
  throw new Error("TruthLens already loaded");
window.hasTruthLensContentScript = true;
class AnalysisStateMachine {
  constructor(video) {
    this.video = video;
    this.state = "IDLE";
    this.overlay = new TruthLensOverlay();
    this.canvas = null;
    this.ctx = null;
    this.worker = null;
    this.animationId = null;
    this.lastFrameTime = 0;
    this.FPS = 30;
    this.roiBox = null;
    this.framesProcessed = 0;
    this.totalRiskScore = 0;
    this.SCAN_DURATION = 90;
    this.init();
  }
  async init() {
    console.log("TruthLens: Init started.");
    this.overlay.createOverlay();
    this.overlay.onClose = () => this.stop("User Cancelled");
    this.setupCanvas();
    this.setupWorkerProxy();
  }
  setupWorkerProxy() {
    try {
      const frameUrl = chrome.runtime.getURL("dist/frame.html");
      console.log(`TruthLens: Creating Worker Proxy Frame at ${frameUrl}`);
      this.proxyFrame = document.createElement("iframe");
      this.proxyFrame.src = frameUrl;
      this.proxyFrame.style.display = "none";
      this.proxyFrame.id = "truthlens-worker-frame";
      document.body.appendChild(this.proxyFrame);
      window.addEventListener("message", (event) => {
        if (!event.data || event.data.source !== "truthlens-worker")
          return;
        if (event.data.type === "frame-ready") {
          this.workerReady = true;
          this.overlay.log("Worker Proxy Ready.");
        } else {
          this.handleWorkerMessage({ data: event.data });
        }
      });
      this.proxyFrame.onload = () => {
        this.overlay.log("Proxy Frame Loaded. Initializing Worker...");
        this.proxyFrame.contentWindow.postMessage({
          type: "INIT_WORKER"
        }, "*");
      };
    } catch (e) {
      this.overlay.log(`Worker Proxy Failed: ${e.message}`);
      console.error("Worker Proxy Failed:", e);
    }
    this.startRenderLoop();
    this.overlay.log("System Initialized. Sidebar Open.");
  }
  setupCanvas() {
    if (!this.overlay.element)
      return;
    this.canvas = this.overlay.element.querySelector(".truthlens-overlay-canvas");
    if (!this.canvas) {
      console.error("Canvas element not found in overlay!");
      return;
    }
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.canvas.width = this.video.videoWidth || 640;
    this.canvas.height = this.video.videoHeight || 480;
    this.captureCanvas = document.createElement("canvas");
    this.captureCanvas.width = this.canvas.width;
    this.captureCanvas.height = this.canvas.height;
    this.captureCtx = this.captureCanvas.getContext("2d", { willReadFrequently: true });
    console.log(`Canvas setup: ${this.canvas.width}x${this.canvas.height}`);
    this.overlay.log(`Canvas: ${this.canvas.width}x${this.canvas.height}`);
  }
  startRenderLoop() {
    if (!this.video || this.video.paused || this.video.ended)
      ;
    const loop = (now, metadata) => {
      this.renderFrame();
      if (this.video.requestVideoFrameCallback) {
        this.video.requestVideoFrameCallback(loop);
      } else {
        requestAnimationFrame(() => loop(performance.now()));
      }
    };
    if (this.video.requestVideoFrameCallback) {
      this.video.requestVideoFrameCallback(loop);
    } else {
      requestAnimationFrame(() => loop(performance.now()));
    }
  }
  renderFrame() {
    if (!this.ctx || !this.canvas || !this.video)
      return;
    const cW = this.canvas.width;
    const cH = this.canvas.height;
    this.ctx.fillStyle = "#0a0a0a";
    this.ctx.fillRect(0, 0, cW, cH);
    const pulse = Math.sin(performance.now() / 200) * 0.5 + 0.5;
    this.ctx.fillStyle = `rgba(255, 0, 0, ${pulse})`;
    this.ctx.beginPath();
    this.ctx.arc(20, 20, 10, 0, Math.PI * 2);
    this.ctx.fill();
    this.ctx.fillStyle = "white";
    this.ctx.font = "12px monospace";
    this.ctx.fillText(`State: ${this.state}`, 40, 25);
    this.ctx.fillText(`VidReady: ${this.video.readyState}`, 40, 40);
    this.ctx.fillText(`Size: ${this.video.videoWidth}x${this.video.videoHeight}`, 40, 55);
    if (Math.random() < 5e-3) {
      console.log(`[TruthLens] Loop Alive. ReadyState=${this.video.readyState}`);
    }
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
        this.ctx.fillStyle = "red";
        this.ctx.fillText(`Draw Error: ${drawErr.name}`, 10, 100);
      }
      if (this.roiBox) {
        const boxX = this.roiBox.xMin * scale + drawX;
        const boxY = this.roiBox.yMin * scale + drawY;
        const boxW = this.roiBox.width * scale;
        const boxH = this.roiBox.height * scale;
        this.ctx.strokeStyle = "#ef4444";
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(boxX, boxY, boxW, boxH);
      }
      this.feedWorker(vW, vH);
    } else {
      this.ctx.fillStyle = "yellow";
      this.ctx.fillText("Waiting for Video Data...", cW / 2 - 50, cH / 2);
    }
  }
  async feedWorker(vW, vH) {
    if (this.state === "COMPLETED")
      return;
    if (!this.proxyFrame)
      return;
    const now = performance.now();
    if (now - this.lastFrameTime < 1e3 / 10)
      return;
    this.lastFrameTime = now;
    try {
      if (this.captureCanvas.width !== vW || this.captureCanvas.height !== vH) {
        this.captureCanvas.width = vW;
        this.captureCanvas.height = vH;
      }
      this.captureCtx.drawImage(this.video, 0, 0, vW, vH);
      const imageData = this.captureCtx.getImageData(0, 0, vW, vH);
      const message = {
        type: "FRAME",
        pixels: imageData.data.buffer,
        width: vW,
        height: vH,
        timestamp: now
      };
      if (this.proxyFrame && this.proxyFrame.contentWindow) {
        this.proxyFrame.contentWindow.postMessage(message, "*", [imageData.data.buffer]);
      }
    } catch (err) {
      if (err.name === "SecurityError") {
        if (Math.random() < 0.05)
          this.overlay.log("CORS: Video tainting detected.");
        this.overlay.updateStatus("Error: Protected Content (CORS)", "fake");
      } else {
        if (Math.random() < 0.05)
          this.overlay.log(`Capture Error: ${err.message}`);
        console.error("Frame capture error:", err);
      }
    }
  }
  handleWorkerMessage(e) {
    const { type, data } = e.data;
    if (type === "INIT_COMPLETE") {
      this.overlay.log("Worker Ready. Model Loaded.");
    } else if (type === "PROGRESS") {
      this.overlay.updateStatus(data.status);
      this.overlay.log(data.status);
    } else if (type === "ERROR") {
      this.overlay.log(`Worker Critical: ${data.error}`);
      this.overlay.showError(data.error);
    } else if (type === "RESULT") {
      this.processResult(data);
    }
  }
  transition(newState) {
    this.overlay.log(`State: ${this.state} -> ${newState}`);
    this.state = newState;
    if (newState === "ROI_VISIBLE") {
      this.overlay.updateStatus("Target Acquired. Analyzing...");
      this.transition("ANALYZING");
    } else if (newState === "ANALYZING")
      ;
    else if (newState === "COMPLETED") {
      this.showResults();
    }
  }
  // StopLoop is just for legacy cleanup, mostly handled by cancelAnimationFrame now
  stopLoop() {
  }
  drawRedBox() {
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
    if (validCount < this.SCAN_DURATION * 0.2) {
      this.overlay.showVerdict("No Face Detected", 0, () => this.resetSession());
      return;
    }
    const avgRisk = this.totalRiskScore / validCount;
    const finalScore = Math.round((1 - avgRisk) * 100);
    let label = "Safe";
    if (avgRisk > 0.6)
      label = "Fake";
    else if (avgRisk > 0.3)
      label = "Suspicious";
    this.overlay.showVerdict(label, finalScore, () => this.resetSession());
  }
  processResult(data) {
    if (this.state === "COMPLETED")
      return;
    if (!data.found && this.state !== "IDLE") {
      if (this.state === "ANALYZING") {
        this.missedFrames = (this.missedFrames || 0) + 1;
        if (this.missedFrames > 15) {
          this.roiBox = null;
        }
      }
      return;
    }
    if (data.found) {
      this.missedFrames = 0;
      this.roiBox = data.box;
      if (this.state === "IDLE") {
        this.transition("ROI_VISIBLE");
      } else if (this.state === "ANALYZING") {
        this.framesProcessed++;
        this.totalRiskScore += data.risk ? data.risk.score : 0;
        this.lastRiskScore = data.risk ? data.risk.score : 0;
        const pct = Math.round(this.framesProcessed / this.SCAN_DURATION * 100);
        this.overlay.updateStatus(`Analyzing frame ${this.framesProcessed}... ${pct}%`);
        if (this.framesProcessed >= this.SCAN_DURATION) {
          this.transition("COMPLETED");
        }
      }
    }
  }
  resetSession() {
    this.framesProcessed = 0;
    this.totalRiskScore = 0;
    this.validFrames = 0;
    this.transition("IDLE");
  }
  stop(reason) {
    console.log("Analysis Stopped:", reason);
    if (this.animationId)
      cancelAnimationFrame(this.animationId);
    this.stopLoop();
    if (this.worker)
      this.worker.terminate();
    this.resetROI();
    this.overlay.remove();
    currentSession = null;
  }
}
let currentSession = null;
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("Content Script Received Message:", request);
  if (request.action === "ANALYZE_VIDEO") {
    if (currentSession) {
      currentSession.stop("Restarting");
    }
    const findVideo = () => {
      const videos = Array.from(document.querySelectorAll("video")).filter((v) => v.videoWidth > 100 && v.videoHeight > 100);
      videos.sort((a, b) => {
        const aPlaying = !a.paused && !a.ended && a.readyState > 2;
        const bPlaying = !b.paused && !b.ended && b.readyState > 2;
        if (aPlaying && !bPlaying)
          return -1;
        if (!aPlaying && bPlaying)
          return 1;
        return b.videoWidth * b.videoHeight - a.videoWidth * a.videoHeight;
      });
      return videos[0];
    };
    let attempts = 0;
    const maxAttempts = 10;
    const attemptFind = () => {
      const bestVideo = findVideo();
      if (bestVideo) {
        if (bestVideo.readyState === 0) {
          console.log("TruthLens: Video found but not ready. Waiting...");
          bestVideo.addEventListener("loadeddata", () => {
            if (!currentSession)
              currentSession = new AnalysisStateMachine(bestVideo);
          }, { once: true });
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
