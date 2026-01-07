// src/controllers/analyzeController.js
import aiEscalationService from "../services/aiEscalationService.js";
import ScanResult from "../models/ScanResult.js";

export const analyzeVideo = async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No video file uploaded" });
        }

        const videoPath = req.file.path;
        const scanType = req.body.scanType || "quick";

        console.log(`Analyzing video: ${videoPath}, type: ${scanType}`);

        // Escalate to AI Service
        const aiResult = await aiEscalationService.escalateScan(videoPath, scanType);

        // Save Result
        const scanResult = new ScanResult({
            videoPath,
            status: aiResult.status,
            verdict: aiResult.verdict,
            scores: aiResult.scores,
            artifacts: aiResult.artifacts
        });
        // await scanResult.save();

        res.json({
            message: "Analysis complete",
            data: scanResult
        });

    } catch (error) {
        console.error("Analysis error:", error);
        res.status(500).json({ error: "Analysis failed", details: error.message });
    }
};
