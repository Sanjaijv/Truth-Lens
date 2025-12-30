// tf/stability.js

export class StabilityDetector {
    constructor() {
        this.prevLandmarks = null;
        this.frameCounter = 0;

        // Stabilized Score
        this.stabilityScore = 1.0;
        this.history = [];
        this.HISTORY_SIZE = 30; // 2 seconds @ 15fps
    }

    // Indices for rigid points (Nose bridge, Eye corners)
    // Nose Tip: 1
    // Left Eye Inner: 133
    // Right Eye Inner: 362
    // Chin: 152

    update(landmarks) {
        if (!landmarks || landmarks.length < 468) return { score: 1.0 };

        let frameScore = 1.0;

        // 1. Calculate Inter-Ocular Distance (IOD)
        // This should be constant unless the person moves closer/further (scale change).
        // Deepfakes sometimes warp this distance rapidly frame-to-frame without Z-depth change.
        const pLeft = landmarks[133];
        const pRight = landmarks[362];
        const currentIOD = Math.hypot(pLeft.x - pRight.x, pLeft.y - pRight.y);

        if (this.prevLandmarks) {
            const prevPLeft = this.prevLandmarks[133];
            const prevPRight = this.prevLandmarks[362];
            const prevIOD = Math.hypot(prevPLeft.x - prevPRight.x, prevPLeft.y - prevPRight.y);

            // 2. Local Feature Jitter (Relative to scale/IOD)
            // Even if head moves, the nose tip relative to eyes shouldn't jitter randomly.

            // Normalize current nose tip by IOD
            const nose = landmarks[1];
            const prevNose = this.prevLandmarks[1];

            // This is a simplified check. Ideally we'd use a full pose estimator matrix.
            // Here we just check if "Face Size" changed > 5% but "Nose Position" changed > 20% (Warping)

            const scaleChange = Math.abs(currentIOD - prevIOD) / prevIOD;
            const noseMove = Math.hypot(nose.x - prevNose.x, nose.y - prevNose.y) / prevIOD;

            // Detection Logic:
            // Real face: Moving head causes Scale + Nose move together.
            // Deepfake: Sometimes texture jitters while outline is stable, or vice versa.

            // Heavy Jitter Penalty (Unexpected high frequency movement)
            if (noseMove > 0.05 && scaleChange < 0.01) {
                // Nose moved 5% of face width, but face didn't get closer/further? Warping.
                frameScore -= 0.2;
            }

            // Extreme Frame-to-Frame drift (Glitch)
            if (noseMove > 0.2) {
                frameScore -= 0.5;
            }
        }

        this.prevLandmarks = landmarks;

        // Smoothing
        this.history.push(frameScore);
        if (this.history.length > this.HISTORY_SIZE) this.history.shift();

        // Average score
        const avg = this.history.reduce((a, b) => a + b, 0) / this.history.length;
        this.stabilityScore = Math.max(0, Math.min(1, avg));

        return {
            score: this.stabilityScore,
            iod: currentIOD
        };
    }
}
