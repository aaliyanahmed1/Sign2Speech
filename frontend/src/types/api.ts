export interface Gesture {
  class: string;
  confidence: number;
  timestamp: number;
}

export interface UploadResponse {
  job_id: string;
  gestures: string[];
  sentence: string;
  error?: string;
}

export interface SpeakResponse {
  audio_url: string;
}

export interface AnalyticsResponse {
  total_sessions: number;
  gestures_detected: number;
  avg_confidence: number;
  sentences_spoken: number;
  gesture_frequency: Record<string, number>;
  session_history: SessionEntry[];
  recent_detections: DetectionEntry[];
}

export interface SessionEntry {
  id: string;
  date: string;
  duration_minutes: number;
  sentences_count: number;
  gestures_count: number;
}

export interface DetectionEntry {
  gesture: string;
  confidence: number;
  timestamp: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  uptime_seconds: number;
  version: string;
}

export interface GestureClass {
  id: number;
  name: string;
  category: string;
}

export interface ClassesResponse {
  classes: GestureClass[];
}

export interface SessionHistoryEntry {
  timestamp: string;
  sentence: string;
  confidence: number;
  track_id: number | null;
  sign_sequence: string[];
}

export interface SessionHistoryResponse {
  session_id: string;
  sentences: SessionHistoryEntry[];
  statistics: {
    total_sentences: number;
    total_audio_files: number;
    session_duration: number;
    average_sentence_length: number;
  };
}

// ---- SignSpeak specific types ----

export interface SignSpeakHistoryEntry {
  id: string;
  sentence: string;
  signs: string[];
  confidence: number;
  timestamp: string;
  audioUrl?: string;
  pinned: boolean;
}

export interface CalibrationState {
  left: boolean;
  right: boolean;
  threshold: number;
}

export interface SignSpeakSettings {
  voiceSpeed: number;
  voicePitch: number;
  autoSpeak: boolean;
  vibrationEnabled: boolean;
  handCalibration: CalibrationState;
  confidenceThreshold: number;
}
