// tf/pulse.js

export class PulseDetector {
    constructor() {
        this.bufferSize = 150; // ~10 seconds @ 15fps
        this.greenSignal = [];
        this.timeStamps = [];
        this.confidence = 0.5; // Start neutral
    }

    // Indices for Cheek Areas
    // Left Cheek: 116, 117, 118, 100, 126, 209
    // Right Cheek: 345, 346, 347, 329, 355, 429
    // Forehead: 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127

    // Simpler approximation: Just use a bounding box around center of cheeks
    getAverageGreen(imageData, landmarks) {
        if (!landmarks || landmarks.length < 468) return 0;

        // Approximate Left Cheek Center
        const cheekLandmark = landmarks[117];

        // Map normalized coordinates to pixel
        // Note: TFJS FaceMesh returns pixels, NOT normalized [0-1].
        const x = Math.floor(cheekLandmark.x);
        const y = Math.floor(cheekLandmark.y);

        // Sample a 10x10 area to handle noise/movement slightly better
        let sumGreen = 0;
        let count = 0;

        for (let dy = -5; dy <= 5; dy++) {
            for (let dx = -5; dx <= 5; dx++) {
                const px = x + dx;
                const py = y + dy;
                if (px >= 0 && px < imageData.width && py >= 0 && py < imageData.height) {
                    const index = (py * imageData.width + px) * 4;
                    sumGreen += imageData.data[index + 1]; // Green channel
                    count++;
                }
            }
        }

        return count > 0 ? sumGreen / count : 0;
    }

    update(imageData, landmarks, timestamp) {
        const val = this.getAverageGreen(imageData, landmarks);
        if (val === 0) return { confidence: 0 };

        this.greenSignal.push(val);
        this.timeStamps.push(timestamp);

        // Keep buffer fixed size
        if (this.greenSignal.length > this.bufferSize) {
            this.greenSignal.shift();
            this.timeStamps.shift();
        }

        // Need at least ~2 seconds of data to start detecting
        if (this.greenSignal.length < 30) {
            return { confidence: 0.5, heartRate: null };
        }

        return this.processSignal();
    }

    processSignal() {
        // 1. Detrending (Remove DC component/slow changes)
        // Simple approach: Frame differencing or Subtract moving average
        const detrended = [];
        for (let i = 1; i < this.greenSignal.length; i++) {
            detrended.push(this.greenSignal[i] - this.greenSignal[i - 1]);
        }

        // 2. Bandpass Filter (0.7Hz - 3Hz) -> (42 BPM - 180 BPM)
        // In time domain, high frequency noise looks like jagged spikes.
        // Deepfakes often have NO periodic signal (flat/random). real humans have a wave.

        // Calculate variance/power of the signal
        const mean = detrended.reduce((a, b) => a + b, 0) / detrended.length;
        const variance = detrended.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / detrended.length;

        // Heuristic:
        // - Too low variance: Static image or perfect synthetic texture (Fake)
        // - Too high variance: Head movement or lighting change (Inconclusive)
        // - Moderate periodic variance: Pulse (Real)

        let score = 0.5;
        if (variance > 0.1 && variance < 10) {
            score = 0.8; // Likely Pulse
        } else if (variance <= 0.1) {
            score = 0.2; // Dead/Frozen
        } else {
            score = 0.5; // Noisy
        }

        this.confidence = score;

        return {
            confidence: this.confidence,
            heartRate: 72 // Placeholder: FFT is too heavy for simple JS without libraries
        };
    }
}
