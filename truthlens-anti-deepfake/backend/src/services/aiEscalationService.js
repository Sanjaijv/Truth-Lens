// src/services/aiEscalationService.js
import axios from 'axios';
import fs from 'fs'; // Import fs to read the video file

class AIEscalationService {
    async escalateScan(videoPath, scanType = 'quick') {
        try {
            const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:5000'; // Correct port is 5000
            const endpoint = '/analyze-video'; // Your Python service has one endpoint for analysis

            console.log(`Escalating ${scanType} scan for ${videoPath} to ${aiServiceUrl}${endpoint}`);

            // Read the video file into a buffer
            const videoBuffer = fs.readFileSync(videoPath);

            // Create FormData to send the file and scanType
            const formData = new FormData();
            formData.append('video', new Blob([videoBuffer]), videoPath.split('/').pop()); // Append video as Blob
            formData.append('scanType', scanType);

            const response = await axios.post(`${aiServiceUrl}${endpoint}`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            return response.data;

        } catch (error) {
            console.error('AI Escalation failed:', error.message);
            // Check if the error is an Axios error and extract more details
            if (error.response) {
                console.error('AI Service Response Data:', error.response.data);
                console.error('AI Service Response Status:', error.response.status);
                console.error('AI Service Response Headers:', error.response.headers);
                throw new Error(`AI Service error: ${error.response.status} - ${JSON.stringify(error.response.data)}`);
            } else if (error.request) {
                // The request was made but no response was received
                console.error('AI Service Request made but no response received:', error.request);
                throw new Error('AI Service is unreachable. Please ensure it is running.');
            } else {
                // Something else happened in setting up the request
                throw new Error(`Error setting up AI Service request: ${error.message}`);
            }
        }
    }
}

export default new AIEscalationService();
