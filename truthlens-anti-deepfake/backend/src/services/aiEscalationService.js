// src/services/aiEscalationService.js
import axios from 'axios';

class AIEscalationService {
    async escalateScan(videoPath, scanType = 'quick') {
        try {
            // Placeholder URL, assuming AI service is running on port 8000
            const aiServiceUrl = process.env.AI_SERVICE_URL || 'http://localhost:8000';
            const endpoint = scanType === 'forensic' ? '/forensic-scan' : '/quick-scan';

            console.log(`Escalating ${scanType} scan for ${videoPath} to ${aiServiceUrl}${endpoint}`);

            // In a real implementation:
            // const response = await axios.post(`${aiServiceUrl}${endpoint}`, { video_path: videoPath });
            // return response.data;

            return {
                status: 'success',
                scanId: Date.now(),
                verdict: 'Pending'
            };
        } catch (error) {
            console.error('AI Escalation failed:', error);
            throw new Error('AI Service unavailable');
        }
    }
}

export default new AIEscalationService();
