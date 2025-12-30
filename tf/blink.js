
export class BlinkDetector {
    constructor() {
        this.EAR_THRESHOLD = 0.25;
        this.BLINK_MIN_DURATION = 50;
        this.BLINK_MAX_DURATION = 500;

        this.isBlinking = false;
        this.blinkStartTime = 0;
        this.blinkCount = 0;
        this.lastBlinkTime = 0;

        this.blinkHistory = [];

        this.naturalnessScore = 1.0;
    }

    calculateEAR(landmarks) {
        const dist = (p1, p2) => Math.hypot(p1.x - p2.x, p1.y - p2.y);

        const getEyeEAR = (indices) => {
            const p = indices.map(i => landmarks[i]);
            const v1 = dist(p[1], p[5]);
            const v2 = dist(p[2], p[4]);
            const h = dist(p[0], p[3]);
            return (v1 + v2) / (2.0 * h);
        };

        const leftEAR = getEyeEAR([33, 160, 158, 133, 153, 144]);
        const rightEAR = getEyeEAR([362, 385, 387, 263, 373, 380]);

        return (leftEAR + rightEAR) / 2;
    }

    update(landmarks, timestamp) {
        if (!landmarks || landmarks.length < 468) return null;

        const ear = this.calculateEAR(landmarks);

        const now = timestamp || Date.now();
        this.blinkHistory = this.blinkHistory.filter(t => now - t < 60000);

        if (ear < this.EAR_THRESHOLD) {
            if (!this.isBlinking) {
                this.isBlinking = true;
                this.blinkStartTime = now;
            }
        } else {
            if (this.isBlinking) {
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
            blinkCount: this.blinkHistory.length,
            ear: ear,
            score: this.naturalnessScore
        };
    }

    updateScore() {
        const bpm = this.blinkHistory.length;

        let score = 1.0;

        if (bpm < 5) score -= 0.3;
        if (bpm < 2) score -= 0.4;

        if (bpm > 50) score -= 0.5;

        this.naturalnessScore = Math.max(0, Math.min(1, score));
    }
}
