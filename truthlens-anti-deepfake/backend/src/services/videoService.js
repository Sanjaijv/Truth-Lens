import path from 'path';
import fs from 'fs';

const AI_MODEL_SERVICE_URL = process.env.AI_MODEL_SERVICE_URL || 'http://localhost:5000';

export const analyzeVideo = async (videoPath, scanType) => {
    try {
        console.log(`Reading video from: ${videoPath}`);
        const fileContent = fs.readFileSync(videoPath);
        const fileName = path.basename(videoPath);

        // In Node 20+, File and Blob are global.
        const videoFile = new File([fileContent], fileName, { type: 'video/mp4' });

        const formData = new FormData();
        formData.append('video', videoFile);
        formData.append('scanType', scanType);

        console.log(`Sending ${fileName} to AI Model Service at ${AI_MODEL_SERVICE_URL}/analyze-video`);

        const response = await fetch(`${AI_MODEL_SERVICE_URL}/analyze-video`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            console.error(`AI service returned ${response.status}: ${errorText}`);
            throw new Error(`AI service responded with ${response.status}: ${errorText}`);
        }

        const mlResults = await response.json();
        console.log("AI analysis completed successfully");
        return mlResults;

    } catch (error) {
        console.error('Error in videoService:', error.message);
        throw new Error(`Failed to get analysis: ${error.message}`);
    }
};