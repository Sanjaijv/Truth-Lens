const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';

export interface AnalysisResult {
  filename: string;
  scanType: string;
  aiLikelihood: number;
  physicsMarkers: Array<{
    name: string;
    score: number;
    status: 'pass' | 'fail';
    description: string;
  }>;
}

export async function analyzeVideo(file: File, scanType: string): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('scanType', scanType);

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => 'No error body');
    throw new Error(`Failed to analyze video: ${response.status} ${response.statusText} - ${errorBody}`);
  }

  return response.json();
}