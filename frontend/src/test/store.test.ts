import { describe, it, expect } from 'vitest';
import { useAppStore } from '../store';

describe('Zustand Store', () => {
  it('has correct initial state', () => {
    const state = useAppStore.getState();
    expect(state.currentGesture).toBeNull();
    expect(state.gestureHistory).toEqual([]);
    expect(state.sentence).toBe('');
    expect(state.confidenceThreshold).toBe(0.5);
    expect(state.isStreaming).toBe(false);
  });

  it('clears session', () => {
    const { clearSession, setCurrentGesture, addGestureToHistory } = useAppStore.getState();
    addGestureToHistory({ class: 'test', confidence: 0.9, timestamp: Date.now() });
    setCurrentGesture({ class: 'test', confidence: 0.9, timestamp: Date.now() });
    useAppStore.getState().setSentence('hello world');

    clearSession();
    const state = useAppStore.getState();
    expect(state.currentGesture).toBeNull();
    expect(state.gestureHistory).toEqual([]);
    expect(state.sentence).toBe('');
  });

  it('appends to sentence', () => {
    const { setSentence, appendToSentence } = useAppStore.getState();
    setSentence('');
    appendToSentence('hello');
    expect(useAppStore.getState().sentence).toBe('hello');
    appendToSentence('world');
    expect(useAppStore.getState().sentence).toBe('hello world');
  });
});
