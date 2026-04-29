import { create } from 'zustand';

interface Gesture {
    class: string;
    confidence: number;
    timestamp: number;
}

interface AppState {
    currentGesture: Gesture | null;
    gestureHistory: Gesture[];
    sentence: string;
    confidenceThreshold: number;
    useOllama: boolean;
    isStreaming: boolean;
    setCurrentGesture: (gesture: Gesture | null) => void;
    addGestureToHistory: (gesture: Gesture) => void;
    setSentence: (sentence: string) => void;
    setConfidenceThreshold: (val: number) => void;
    setUseOllama: (val: boolean) => void;
    setIsStreaming: (val: boolean) => void;
    clearSession: () => void;
}

export const useAppStore = create<AppState>((set) => ({
    currentGesture: null,
    gestureHistory: [],
    sentence: '',
    confidenceThreshold: 0.5,
    useOllama: false,
    isStreaming: false,

    setCurrentGesture: (gesture) => set({ currentGesture: gesture }),

    addGestureToHistory: (gesture) => set((state) => ({
        gestureHistory: [gesture, ...state.gestureHistory].slice(0, 10)
    })),

    setSentence: (sentence) => set({ sentence }),

    setConfidenceThreshold: (val) => set({ confidenceThreshold: val }),

    setUseOllama: (val) => set({ useOllama: val }),

    setIsStreaming: (val) => set({ isStreaming: val }),

    clearSession: () => set({ currentGesture: null, gestureHistory: [], sentence: '' })
}));
