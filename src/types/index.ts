export type MRIClass = 'glioma' | 'meningioma' | 'notumor' | 'pituitary';

export interface ClassProbability {
  className: MRIClass;
  displayName: string;
  probability: number;
  percentage: string;
  description: string;
}

export interface PredictionResult {
  id: string;
  timestamp: string;
  fileName: string;
  imageUrl: string;
  prediction: MRIClass;
  displayName: string;
  confidence: number;
  display_confidence: string;
  low_confidence: boolean;
  model_name: string;
  inference_time_ms: number;
  gradcam_available: boolean;
  gradcam_heatmap_url?: string;
  gradcam_overlay_url?: string;
  probabilities: ClassProbability[];
  explanation?: MedicalExplanation;
  technicalDetails?: {
    inputDimensions: string;
    normalizedRange: string;
    gradcamLayer: string;
    activationFunction: string;
  };
}

export interface MedicalExplanation {
  summary: string;
  patternAnalysis: string;
  uncertainties: string;
  recommendedNextSteps: string;
  clinicalDisclaimer: string;
  source: 'gemini' | 'template';
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  fileName: string;
  imageUrl: string;
  prediction: MRIClass;
  displayName: string;
  confidence: number;
  display_confidence: string;
  low_confidence: boolean;
  inference_time_ms: number;
}

export interface SampleMRI {
  id: string;
  name: string;
  expectedClass: MRIClass;
  displayName: string;
  url: string;
  sequence: string;
  description: string;
}

export interface ModelMetrics {
  name: string;
  architecture: string;
  classes: string[];
  inputShape: number[];
  numParameters: number;
  trainableParameters: number;
  datasetSize: {
    total: number;
    training: number;
    testing: number;
  };
  epochs: number;
  finalValidationAccuracy: number;
  finalValidationLoss: number;
  epochHistory: {
    epoch: number;
    trainAcc: number;
    valAcc: number;
    trainLoss: number;
    valLoss: number;
  }[];
}
