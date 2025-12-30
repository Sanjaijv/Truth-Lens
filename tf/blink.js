// tf/blink.js

export class BlinkDetector {
    constructor() {
        // EAR Thresholds
        this.EAR_THRESHOLD = 0.25; // Eye is considered closed below this
        this.BLINK_MIN_DURATION = 50;  // ms
        this.BLINK_MAX_DURATION = 500; // ms

        // State
        this.isBlinking = false;
        this.blinkStartTime = 0;
        this.blinkCount = 0;
        this.lastBlinkTime = 0;

        // History for frequency analysis (sliding window of 60 seconds)
        this.blinkHistory = [];

        // Output Score (1.0 = Natural, 0.0 = Fake)
        this.naturalnessScore = 1.0;
    }

    // MediaPipe Facemesh Landmarks (468 points)
    // Left Eye:  33, 160, 158, 133, 153, 144
    // Right Eye: 362, 385, 387, 263, 373, 380
    calculateEAR(landmarks) {
        // Helper to get distance between two 3D points
        const dist = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y);

        const getEyeEAR = (indices) => {
            const p = indices.map(i => landmarks[i]);
            // Vertical distances
            const v1 = dist(p[1], p[5]);
            const v2 = dist(p[2], p[4]);
            // Horizontal distance
            const h = dist(p[0], p[3]);
            return (v1 + v2) / (2.0 * h);
        };

        const leftEAR = getEyeEAR([33, 160, 158, 133, 153, 144]);
        const rightEAR = getEyeEAR([362, 385, 387, 263, 373, 380]);

        // Average EAR of both eyes
        return (leftEAR + rightEAR) / 2;
    }

    update(landmarks, timestamp) {
        if (!landmarks || landmarks.length < 468) return null;

        const ear = this.calculateEAR(landmarks);

        // Clean up old history (> 60s ago)
        const now = timestamp || Date.now();
        this.blinkHistory = this.blinkHistory.filter(t => now - t < 60000);

        if (ear < this.EAR_THRESHOLD) {
            if (!this.isBlinking) {
                this.isBlinking = true;
                this.blinkStartTime = now;
            }
        } else {
            if (this.isBlinking) {
                // Blink ended
                this.isBlinking = false;
                const duration = now - this.blinkStartTime;

                if (duration >= this.BLINK_MIN_DURATION && duration <= this.BLINK_MAX_DURATION) {
                    this.blinkCount++;
                    this.blinkHistory.push(now);
                    this.lastBlinkTime = now;
                }
            }
        }

        this.updateScore();

        return {
            isBlinking: this.isBlinking,
            blinkCount: this.blinkHistory.length, // BPM (Blinks Per Minute)
            ear: ear,
            score: this.naturalnessScore
        };
    }

    updateScore() {
        const bpm = this.blinkHistory.length;

        // Normal blink rate is roughly 10-20 per minute, but varies.
        // Deepfakes often blink excessively (nervousness) or not enough (frozen).

        let score = 1.0;

        // Penalize extremely low frequency (staring)
        if (bpm < 5) score -= 0.3;
        if (bpm < 2) score -= 0.4; // Very suspicious

        // Penalize extremely high frequency (fluttering)
        if (bpm > 50) score -= 0.5;

        // Ensure bounds
        this.naturalnessScore = Math.max(0, Math.min(1, score));
    }
}
