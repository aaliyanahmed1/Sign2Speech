import { useCallback, useRef, useState } from 'react';
import { useAppStore } from '../store';
import { apiClient } from '../lib/api';

interface UseWebSocketOptions {
  onDetection?: (data: { gesture: string; confidence: number }) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    onDetection,
    onConnect,
    onDisconnect,
    onError,
    reconnectInterval = 3000,
    maxReconnectAttempts = 10,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempts = useRef(0);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [connected, setConnected] = useState(false);

  const { setCurrentGesture, addGestureToHistory, appendToSentence } =
    useAppStore();

  const cleanup = useCallback(() => {
    if (reconnectTimer.current) {
      clearTimeout(reconnectTimer.current);
      reconnectTimer.current = null;
    }
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
  }, []);

  const connect = useCallback(() => {
    cleanup();

    try {
      const ws = apiClient.createWebSocket();
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        reconnectAttempts.current = 0;
        onConnect?.();
      };

      ws.onmessage = (evt) => {
        try {
          const data = JSON.parse(evt.data);

          if (data.type === 'detection') {
            const gesture = {
              class: data.gesture,
              confidence: data.confidence,
              timestamp: Date.now(),
            };

            setCurrentGesture(gesture);
            addGestureToHistory(gesture);
            appendToSentence(data.gesture);
            onDetection?.(data);
          }
        } catch {
          console.warn('Failed to parse WebSocket message');
        }
      };

      ws.onclose = () => {
        setConnected(false);
        onDisconnect?.();

        if (reconnectAttempts.current < maxReconnectAttempts) {
          reconnectTimer.current = setTimeout(() => {
            reconnectAttempts.current++;
            connect();
          }, reconnectInterval * Math.min(reconnectAttempts.current + 1, 5));
        }
      };

      ws.onerror = (error) => {
        onError?.(error);
      };
    } catch (err) {
      console.error('WebSocket connection failed:', err);
    }
  }, [
    cleanup,
    onConnect,
    onDisconnect,
    onError,
    onDetection,
    setCurrentGesture,
    addGestureToHistory,
    appendToSentence,
    reconnectInterval,
    maxReconnectAttempts,
  ]);

  const disconnect = useCallback(() => {
    reconnectAttempts.current = maxReconnectAttempts;
    cleanup();
  }, [cleanup, maxReconnectAttempts]);

  const sendMessage = useCallback((data: unknown) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { connected, connect, disconnect, sendMessage };
}
