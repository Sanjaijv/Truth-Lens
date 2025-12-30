
export class RiskScorer {
    constructor() {
        this.currentRisk = 0;
        this.alpha = 0.1;
    }

    calculate(signals) {
        const W_STABILITY = 0.40;
        const W_BLINK = 0.35;
        const W_PULSE = 0.25;

        const s = signals.stability !== undefined ? signals.stability : 0.5;
        const b = signals.blink !== undefined ? signals.blink : 0.5;
        const p = signals.pulse !== undefined ? signals.pulse : 0.5;

        const authenticity = (s * W_STABILITY) + (b * W_BLINK) + (p * W_PULSE);

        const rawRisk = 1.0 - authenticity;

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
