const API_BASE = import.meta.env.VITE_API_BASE_URL || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:8001');
const WS_BASE = import.meta.env.VITE_WS_BASE_URL || (typeof window !== 'undefined' ? (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host : 'ws://localhost:8001');

class ApiClient {
  private baseUrl: string;
  private wsUrl: string;

  constructor() {
    this.baseUrl = API_BASE;
    this.wsUrl = WS_BASE;
  }

  async uploadImage(file: File, signal?: AbortSignal) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/upload`, {
      method: 'POST',
      body: formData,
      signal,
    });

    if (!response.ok) {
      throw new Error(`Upload failed: ${response.statusText}`);
    }

    return response.json();
  }

  async speak(sentence: string, voiceId?: string, signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/api/speak`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentence, voice_id: voiceId }),
      signal,
    });

    if (!response.ok) {
      throw new Error(`Speak failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getAnalytics(signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/api/analytics`, {
      signal,
    });

    if (!response.ok) {
      throw new Error(`Analytics fetch failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getClasses(signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/api/classes`, {
      signal,
    });

    if (!response.ok) {
      throw new Error(`Classes fetch failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getHealth(signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/api/health`, {
      signal,
    });

    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }

    return response.json();
  }

  async getSessionHistory(signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/api/session-history`, {
      signal,
    });

    if (!response.ok) {
      throw new Error(`Session history fetch failed: ${response.statusText}`);
    }

    return response.json();
  }

  createWebSocket(path: string = '/api/stream'): WebSocket {
    return new WebSocket(`${this.wsUrl}${path}`);
  }

  getApiBaseUrl(): string {
    return this.baseUrl;
  }

  getWsBaseUrl(): string {
    return this.wsUrl;
  }

  // ---- SignSpeak specific ----

  async saveSessionEntry(entry: {
    sentence: string;
    signs: string[];
    confidence: number;
  }, signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/api/session/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(entry),
      signal,
    });
    if (!response.ok) throw new Error(`Save failed: ${response.statusText}`);
    return response.json();
  }

  async deleteSessionEntry(id: string, signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/api/session/${id}`, {
      method: 'DELETE',
      signal,
    });
    if (!response.ok) throw new Error(`Delete failed: ${response.statusText}`);
    return response.json();
  }

  async clearAllSessions(signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/api/session/clear`, {
      method: 'DELETE',
      signal,
    });
    if (!response.ok) throw new Error(`Clear failed: ${response.statusText}`);
    return response.json();
  }

  async getSignSpeakHistory(signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/api/signspeak/history`, { signal });
    if (!response.ok) throw new Error(`History fetch failed: ${response.statusText}`);
    return response.json();
  }

  async uploadCalibration(side: 'left' | 'right', file: Blob, signal?: AbortSignal) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('side', side);
    const response = await fetch(`${this.baseUrl}/api/calibrate`, {
      method: 'POST',
      body: formData,
      signal,
    });
    if (!response.ok) throw new Error(`Calibration failed: ${response.statusText}`);
    return response.json();
  }

  async translate(sentence: string, useOllama: boolean = true, voiceId?: string, signal?: AbortSignal) {
    const response = await fetch(`${this.baseUrl}/api/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sentence, use_ollama: useOllama, voice_id: voiceId }),
      signal,
    });
    if (!response.ok) throw new Error(`Translation failed: ${response.statusText}`);
    return response.json();
  }
}

export const apiClient = new ApiClient();
