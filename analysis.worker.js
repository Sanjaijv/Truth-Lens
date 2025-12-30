import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';
import { detectFace, loadFaceMeshModel } from './tf/faceMesh.js';
import { BlinkDetector } from './tf/blink.js';
import { PulseDetector } from './tf/pulse.js';
import { StabilityDetector } from './tf/stability.js';
import { RiskScorer } from './tf/riskScore.js';


console.log("Worker: Loaded (Prod Mode)");

// Global Error Handler
self.onerror = function (err) {
    console.error("Worker Global Error:", err);
    self.postMessage({
        type: 'ERROR',
        error: `Worker Global Error: ${err.message || JSON.stringify(err)}`
    });
};

let isInitialized = false;
let blink, pulse, stability, risk;

async function init(rootUrl) {
    if (isInitialized) return;

    try {
        console.log("Worker: Init called with rootUrl:", rootUrl);
        self.postMessage({ type: 'PROGRESS', status: 'Starting AI Engine...' });

        // Load TF.js backend
        try {
            self.postMessage({ type: 'PROGRESS', status: 'Initializing GPU...' });

            // Check if TF is loaded
            if (!tf) throw new Error("TensorFlow.js not loaded");

            // Strictly WebGL
            await tf.setBackend('webgl');
            await tf.ready();
            console.log("Worker: TF Backend ready:", tf.getBackend());
        } catch (backendErr) {
            console.error("Worker: WebGL Init Failed", backendErr);
            self.postMessage({ type: 'ERROR', error: "GPU Access Denied" });
            return;
        }

        // Load Model
        self.postMessage({ type: 'PROGRESS', status: 'Loading Face Model...' });
        await loadFaceMeshModel(rootUrl);

        self.postMessage({ type: 'PROGRESS', status: 'Compiling Shaders...' });

        // Init Logic Modules
        blink = new BlinkDetector();
        pulse = new PulseDetector();
        stability = new StabilityDetector();
        risk = new RiskScorer();

        isInitialized = true;
        console.log("Worker: Initialization Complete");
        self.postMessage({ type: 'INIT_COMPLETE' });
    } catch (err) {
        console.error("Worker: Init Failed", err);
        self.postMessage({ type: 'ERROR', error: `Init Failed: ${err.message}` });
    }
}

// Handle Messages
self.onmessage = async (e) => {
    const msg = e.data;

    try {
        if (msg.type === 'INIT') {
            await init(msg.rootUrl);
        } else if (msg.type === 'FRAME') {
            if (!isInitialized) return;
            await processFrame(msg.pixels, msg.width, msg.height, msg.timestamp);
        }
    } catch (handlerErr) {
        console.error("Worker: Message Handler Error", handlerErr);
    }
};

async function processFrame(pixels, width, height, timestamp) {
    try {
        // Reconstruct ImageData from the received ArrayBuffer
        const data = new Uint8ClampedArray(pixels);
        const imageData = new ImageData(data, width, height);

        console.log("Worker: Processing frame", width, "x", height);

        // 1. Detect Face (TFJS accepts ImageData)
        const faces = await detectFace(imageData);

        // Default result
        let result = {
            found: false,
            timestamp: timestamp
        };

        if (faces && faces.length > 0) {
            const face = faces[0];
            const keypoints = face.keypoints;

            console.log("Worker: Face Found!");

            // Run Benchmarks
            const blinkRes = blink.update(keypoints, timestamp);
            const stabilityRes = stability.update(keypoints);
            const pulseRes = pulse.update(imageData, keypoints, timestamp);

            // Calculate Risk
            const riskRes = risk.calculate({
                blink: (blinkRes && blinkRes.score) || 0.5,
                pulse: (pulseRes && pulseRes.confidence) || 0.5,
                stability: (stabilityRes && stabilityRes.score) || 0.5
            });

            result = {
                found: true,
                timestamp: timestamp,
                blink: blinkRes,
                pulse: pulseRes,
                stability: stabilityRes,
                risk: riskRes,
                box: face.box // Return local box coords within the ROI if needed
            };
        } else {
            console.log("Worker: No face detected in this frame");
        }

        // Send Result
        self.postMessage({ type: 'RESULT', data: result });

    } catch (err) {
        console.error("Worker Error:", err);
    } finally {
        // Cleanup
    }

}
