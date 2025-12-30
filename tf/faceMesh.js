// tf/faceMesh.js
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import '@tensorflow/tfjs-backend-webgl';

let detector;
let modelLoadingPromise;

export async function loadFaceMeshModel(rootUrl) {
    if (detector) return detector;
    if (modelLoadingPromise) return modelLoadingPromise;

    console.log("Loading FaceMesh model...");

    modelLoadingPromise = (async () => {
        // Ensure backend is ready
        await import('@tensorflow/tfjs-backend-webgl');
        // await tf.ready(); // Dynamically if needed, but the import usually handles it.

        const model = faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh;

        const detectorConfig = {
            runtime: 'tfjs',
            refineLandmarks: false,
            maxFaces: 1,
            minDetectionConfidence: 0.3 // Lower threshold for anime/harder faces
        };

        console.log("Creating detector with config:", detectorConfig);
        detector = await faceLandmarksDetection.createDetector(model, detectorConfig);
        console.log("FaceMesh model loaded.");
        return detector;
    })();

    return modelLoadingPromise;
}

export async function detectFace(input) {
    if (!detector) {
        await loadFaceMeshModel();
    }

    try {
        const faces = await detector.estimateFaces(input, {
            flipHorizontal: false, // Mirror mode not usually needed for incoming video
            staticImageMode: false
        });
        return faces; // Returns array of { keypoints, box, etc. }
    } catch (error) {
        console.error("Face detection error:", error);
        return [];
    }
}
