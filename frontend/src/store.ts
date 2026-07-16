import { create } from 'zustand';
import type { Gesture, SignSpeakHistoryEntry } from './types/api';

interface SignSpeakState {
  currentGesture: Gesture | null;
  gestureHistory: Gesture[];
  sentence: string;
  confidenceThreshold: number;
  useOllama: boolean;
  isStreaming: boolean;
  cameraOn: boolean;
  // History page
  historyEntries: SignSpeakHistoryEntry[];
  pinnedIds: Set<string>;
  // Settings
  voiceSpeed: number;
  voicePitch: number;
  voiceId: string;
  autoSpeak: boolean;
  vibrationEnabled: boolean;
  handCalibration: { left: boolean; right: boolean };
  facingMode: 'user' | 'environment';
  bandwidthMode: 'high' | 'standard' | 'low';
  setFacingMode: (mode: 'user' | 'environment') => void;
  setBandwidthMode: (mode: 'high' | 'standard' | 'low') => void;
  // Ui
  currentResult: SignSpeakHistoryEntry | null;
  isAuthenticated: boolean;
  user: { email: string; name: string } | null;
  loginUser: (email: string, name: string) => void;
  logoutUser: () => void;
  setCurrentGesture: (gesture: Gesture | null) => void;
  addGestureToHistory: (gesture: Gesture) => void;
  setSentence: (sentence: string) => void;
  appendToSentence: (word: string) => void;
  setConfidenceThreshold: (val: number) => void;
  setUseOllama: (val: boolean) => void;
  setIsStreaming: (val: boolean) => void;
  setCameraOn: (val: boolean) => void;
  clearSession: () => void;
  // History actions
  setHistoryEntries: (entries: SignSpeakHistoryEntry[]) => void;
  addHistoryEntry: (entry: SignSpeakHistoryEntry) => void;
  removeHistoryEntry: (id: string) => void;
  clearAllHistory: () => void;
  togglePin: (id: string) => void;
  // Settings
  setVoiceSpeed: (val: number) => void;
  setVoicePitch: (val: number) => void;
  setVoiceId: (val: string) => void;
  setAutoSpeak: (val: boolean) => void;
  setVibrationEnabled: (val: boolean) => void;
  setHandCalibration: (side: 'left' | 'right', val: boolean) => void;
  // Result
  setCurrentResult: (entry: SignSpeakHistoryEntry | null) => void;
}

export const useAppStore = create<SignSpeakState>((set) => ({
  currentGesture: null,
  gestureHistory: [],
  sentence: '',
  confidenceThreshold: 0.5,
  useOllama: false,
  isStreaming: false,
  cameraOn: false,
  historyEntries: [],
  pinnedIds: new Set(),
  voiceSpeed: 1.0,
  voicePitch: 1.0,
  voiceId: 'ur-PK-AsmaNeural',
  autoSpeak: false,
  vibrationEnabled: true,
  handCalibration: { left: false, right: false },
  facingMode: 'user',
  bandwidthMode: 'standard',
  currentResult: null,
  isAuthenticated: false,
  user: null,
  loginUser: (email, name) => set({ isAuthenticated: true, user: { email, name } }),
  logoutUser: () => set({ isAuthenticated: false, user: null }),

  setCurrentGesture: (gesture) => set({ currentGesture: gesture }),

  addGestureToHistory: (gesture) =>
    set((state) => ({
      gestureHistory: [gesture, ...state.gestureHistory].slice(0, 10),
    })),

  setSentence: (sentence) => set({ sentence }),

  appendToSentence: (word) =>
    set((state) => {
      if (!state.sentence) {
        return { sentence: word };
      }
      const words = state.sentence.trim().split(/\s+/);
      const lastWord = words[words.length - 1];
      if (lastWord && lastWord.toLowerCase() === word.trim().toLowerCase()) {
        return {};
      }
      return { sentence: state.sentence + ' ' + word };
    }),

  setConfidenceThreshold: (val) => set({ confidenceThreshold: val }),
  setUseOllama: (val) => set({ useOllama: val }),
  setIsStreaming: (val) => set({ isStreaming: val }),
  setCameraOn: (val) => set({ cameraOn: val }),

  clearSession: () =>
    set({ currentGesture: null, gestureHistory: [], sentence: '' }),

  setHistoryEntries: (entries) => set({ historyEntries: entries }),
  addHistoryEntry: (entry) =>
    set((state) => ({
      historyEntries: [entry, ...state.historyEntries],
    })),
  removeHistoryEntry: (id) =>
    set((state) => ({
      historyEntries: state.historyEntries.filter((e) => e.id !== id),
    })),
  clearAllHistory: () => set({ historyEntries: [], pinnedIds: new Set() }),
  togglePin: (id) =>
    set((state) => {
      const next = new Set(state.pinnedIds);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { pinnedIds: next };
    }),

  setVoiceSpeed: (val) => set({ voiceSpeed: val }),
  setVoicePitch: (val) => set({ voicePitch: val }),
  setVoiceId: (val) => set({ voiceId: val }),
  setAutoSpeak: (val) => set({ autoSpeak: val }),
  setVibrationEnabled: (val) => set({ vibrationEnabled: val }),
  setHandCalibration: (side, val) =>
    set((state) => ({
      handCalibration: { ...state.handCalibration, [side]: val },
    })),
  setFacingMode: (mode) => set({ facingMode: mode }),
  setBandwidthMode: (mode) => set({ bandwidthMode: mode }),
  setCurrentResult: (entry) => set({ currentResult: entry }),
}));
