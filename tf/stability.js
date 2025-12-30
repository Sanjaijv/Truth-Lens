
export class StabilityDetector {
    constructor() {
        this.prevLandmarks = null;
        this.frameCounter = 0;

        this.stabilityScore = 1.0;
        this.history = [];
        this.HISTORY_SIZE = 30;
    }

    update(landmarks) {
        if (!landmarks || landmarks.length < 468) return { score: 1.0 };

        let frameScore = 1.0;

        const pLeft = landmarks[133];
        const pRight = landmarks[362];
        const currentIOD = Math.hypot(pLeft.x - pRight.x, pLeft.y - pRight.y);

        if (this.prevLandmarks) {
            const prevPLeft = this.prevLandmarks[133];
            const prevPRight = this.prevLandmarks[362];
            const prevIOD = Math.hypot(prevPLeft.x - prevPRight.x, prevPLeft.y - prevPRight.y);


            const nose = landmarks[1];
            const prevNose = this.prevLandmarks[1];

            const scaleChange = Math.abs(currentIOD - prevIOD) / prevIOD;
            const noseMove = Math.hypot(nose.x - prevNose.x, nose.y - prevNose.y) / prevIOD;

            if (noseMove > 0.05 && scaleChange < 0.01) {
                frameScore -= 0.2;
            }

            if (noseMove > 0.2) {
                frameScore -= 0.5;
            }
        }

        this.prevLandmarks = landmarks;

        this.history.push(frameScore);
        if (this.history.length > this.HISTORY_SIZE) this.history.shift();

        const avg = this.history.reduce((a, b) => a + b, 0) / this.history.length;
        this.stabilityScore = Math.max(0, Math.min(1, avg));

        return {
            score: this.stabilityScore,
            iod: currentIOD
        };
    }
}
