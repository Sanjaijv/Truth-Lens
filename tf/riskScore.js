// tf/riskScore.js

export class RiskScorer {
    constructor() {
        this.currentRisk = 0;
        this.alpha = 0.1; // Smoothing factor for EMA (Exponential Moving Average)
    }

    calculate(signals) {
        // Signals expected:
        // signals.stability (0.0 - 1.0) [1 = Stable]
        // signals.blink     (0.0 - 1.0) [1 = Natural]
        // signals.pulse     (0.0 - 1.0) [1 = Live Pulse]

        // Weights
        const W_STABILITY = 0.40;
        const W_BLINK = 0.35;
        const W_PULSE = 0.25;

        // Calculate Authenticity Score (1.0 = Real, 0.0 = Fake)
        // Default to neutral (0.5) if signal missing
        const s = signals.stability !== undefined ? signals.stability : 0.5;
        const b = signals.blink !== undefined ? signals.blink : 0.5;
        const p = signals.pulse !== undefined ? signals.pulse : 0.5;

        const authenticity = (s * W_STABILITY) + (b * W_BLINK) + (p * W_PULSE);

        // Risk is inverse of Authenticity
        const rawRisk = 1.0 - authenticity;

        // Smooth output to prevent UI flickering
        this.currentRisk = (this.alpha * rawRisk) + ((1 - this.alpha) * this.currentRisk);

        return {
            score: this.currentRisk,
            label: this.getLabel(this.currentRisk)
        };
    }

    getLabel(score) {
        if (score < 0.35) return "SAFE";
        if (score < 0.75) return "SUSPICIOUS";
        return "LIKELY DEEPFAKE";
    }
}
