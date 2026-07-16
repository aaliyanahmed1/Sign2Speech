import { useEffect, useRef, useState } from 'react';
import { Play, Square, Settings, Volume2, Trash2, Camera } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppStore } from '../store';
import { useWebSocket } from '../hooks/useWebSocket';
import { useToast } from '../components/Toast';
import { apiClient } from '../lib/api';

export default function LiveDetection() {
  const {
    currentGesture,
    gestureHistory,
    sentence,
    confidenceThreshold,
    useOllama,
    isStreaming,
    setIsStreaming,
    setConfidenceThreshold,
    setUseOllama,
    setSentence,
    clearSession,
    voiceId,
  } = useAppStore();

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const { addToast } = useToast();

  const { connected, connect, disconnect } = useWebSocket({
    onConnect: () => {},
    onDisconnect: () => {},
  });

  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsStreaming(true);
      connect();
    } catch {
      addToast('Camera access denied. Please allow camera permissions.', 'error');
    }
  };

  const stopStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    disconnect();
    setIsStreaming(false);
  };

  const toggleStream = () => {
    if (isStreaming) stopStream();
    else startStream();
  };

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      disconnect();
    };
  }, [disconnect]);

  const handleSpeak = async () => {
    if (!sentence) return;
    try {
      const data = await apiClient.speak(sentence, voiceId);
      if (data.audio_url) {
        const audio = new Audio(`${apiClient.getApiBaseUrl()}${data.audio_url}`);
        audio.play().catch(() => {});
      }
      addToast('Speaking...', 'success');
    } catch {
      const utterance = new SpeechSynthesisUtterance(sentence);
      window.speechSynthesis.speak(utterance);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-64px)] overflow-hidden bg-background">
      {/* Status bar */}
      <div className="h-12 border-b border-gray-800 glass-panel flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-3">
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              connected
                ? 'bg-[#00E5CC] animate-pulse shadow-[0_0_8px_#00E5CC]'
                : 'bg-red-500'
            }`}
          />
          <span className="font-mono text-xs tracking-tight text-gray-400">
            {connected ? 'CONNECTED / DETECTING' : isStreaming ? 'CONNECTING...' : 'IDLE'}
          </span>
        </div>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-1.5 hover:text-[#00E5CC] transition-colors rounded hover:bg-white/5"
          aria-label="Settings"
        >
          <Settings size={18} />
        </button>
      </div>

      <div className="flex flex-1 overflow-hidden relative">
        {/* Video feed */}
        <div className="w-full md:w-[60%] border-b md:border-b-0 md:border-r border-gray-800 p-4 flex flex-col items-center justify-center relative bg-black">
          {!isStreaming ? (
            <div className="text-center">
              <Camera size={56} className="mx-auto mb-4 text-gray-600" />
              <p className="text-gray-500 text-sm mb-4">Grant camera access to begin</p>
              <button
                onClick={toggleStream}
                className="bg-[#00E5CC] text-black font-bold px-7 py-2.5 rounded-lg hover:bg-[#00E5CC]/80 transition-colors flex items-center gap-2 mx-auto text-sm"
              >
                <Play size={16} /> Start Camera Feed
              </button>
            </div>
          ) : (
            <div className="relative w-full max-w-2xl aspect-video rounded-xl overflow-hidden border border-[#6C3FC8]/50 shadow-[0_0_24px_rgba(108,63,200,0.15)]">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="w-full h-full object-cover"
              />
              <button
                onClick={toggleStream}
                className="absolute top-3 right-3 bg-red-500/80 text-white px-3 py-1.5 rounded-md flex items-center gap-1.5 hover:bg-red-500 backdrop-blur text-xs font-bold"
              >
                <Square size={14} /> Stop
              </button>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="w-full md:w-[40%] flex flex-col p-4 overflow-y-auto glass-panel">
          {/* Current gesture */}
          <div className="mb-6">
            <h3 className="text-gray-500 font-syne text-xs uppercase tracking-wider mb-2">
              Live Detection
            </h3>
            <AnimatePresence mode="popLayout">
              {currentGesture ? (
                <motion.div
                  key={currentGesture.timestamp}
                  initial={{ scale: 0.9, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 1.05, opacity: 0 }}
                  className="bg-[#00E5CC]/10 border border-[#00E5CC]/60 rounded-xl p-5 text-center shadow-[0_0_20px_rgba(0,229,204,0.1)]"
                >
                  <span className="text-3xl font-mono font-bold text-[#00E5CC]">
                    {currentGesture.class}
                  </span>
                </motion.div>
              ) : (
                <div className="border border-dashed border-gray-800 rounded-xl p-5 text-center text-gray-600">
                  <span className="text-sm font-mono">Waiting for sign...</span>
                </div>
              )}
            </AnimatePresence>
            <div className="mt-3 h-1.5 w-full bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-[#6C3FC8] to-[#00E5CC] transition-all duration-300"
                style={{ width: `${(currentGesture?.confidence || 0) * 100}%` }}
              />
            </div>
          </div>

          {/* Detection log */}
          <div className="flex-1 min-h-[120px] flex flex-col border border-gray-800 rounded-xl p-3 bg-black/30 mb-4 font-mono text-xs max-h-[260px]">
            <h3 className="text-gray-600 mb-2 border-b border-gray-800/60 pb-1.5">Detection Log</h3>
            <div className="flex-1 overflow-y-auto space-y-1.5">
              <AnimatePresence>
                {gestureHistory.map((g, i) => (
                  <motion.div
                    key={g.timestamp + i}
                    initial={{ x: 12, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    className="flex justify-between items-center text-gray-400"
                  >
                    <span className="text-[#6C3FC8]">[{new Date(g.timestamp).toLocaleTimeString()}]</span>
                    <span className="text-gray-300">{g.class}</span>
                    <span className="text-[#00E5CC]">{(g.confidence * 100).toFixed(0)}%</span>
                  </motion.div>
                ))}
              </AnimatePresence>
              {gestureHistory.length === 0 && (
                <div className="text-gray-600 text-center py-4">No detections yet</div>
              )}
            </div>
          </div>

          {/* Sentence output */}
          <div className="border border-gray-700 bg-[#0A0A0F] rounded-xl p-3 shadow-lg shrink-0">
            <h3 className="font-syne text-xs text-gray-500 mb-2">Assembled Sentence</h3>
            <textarea
              readOnly
              value={sentence}
              placeholder="Output will appear here..."
              className="w-full h-20 bg-transparent resize-none focus:outline-none text-gray-300 text-sm border-none"
            />
            <div className="flex gap-2 mt-2">
              <button
                onClick={() => { clearSession(); setSentence(''); }}
                className="p-2 border border-red-500/40 text-red-500 rounded-lg hover:bg-red-500/10 flex-1 flex justify-center items-center gap-1.5 text-xs font-bold transition-colors"
              >
                <Trash2 size={14} /> Clear
              </button>
              <button
                onClick={handleSpeak}
                className="p-2 bg-[#00E5CC] text-black rounded-lg flex-1 flex justify-center items-center gap-1.5 text-xs font-bold hover:bg-[#00E5CC]/80 transition-colors"
              >
                <Volume2 size={14} /> Speak
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Settings drawer */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ x: 320 }}
            animate={{ x: 0 }}
            exit={{ x: 320 }}
            transition={{ type: 'spring', damping: 25, stiffness: 250 }}
            className="absolute top-12 right-0 bottom-0 w-80 bg-[#0A0A0F] border-l border-gray-800 p-6 shadow-[-16px_0_30px_rgba(0,0,0,0.4)] z-20 overflow-y-auto"
          >
            <h2 className="text-lg font-syne font-bold mb-6 border-b border-gray-800 pb-3">
              Detection Settings
            </h2>
            <div className="space-y-5">
              <div>
                <label className="block text-xs text-gray-400 mb-1.5">Confidence Threshold</label>
                <input
                  type="range"
                  min={30}
                  max={90}
                  value={confidenceThreshold * 100}
                  onChange={(e) => setConfidenceThreshold(Number(e.target.value) / 100)}
                  className="w-full accent-[#00E5CC]"
                />
                <div className="flex justify-between text-xs text-gray-600 mt-0.5">
                  <span>0.30</span>
                  <span className="text-[#00E5CC] font-mono">{confidenceThreshold.toFixed(2)}</span>
                  <span>0.90</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <label className="text-xs text-gray-400">Use LLM Phrase Engine</label>
                <button
                  onClick={() => setUseOllama(!useOllama)}
                  className={`w-10 h-5 rounded-full transition-colors relative ${
                    useOllama ? 'bg-[#00E5CC]' : 'bg-gray-700'
                  }`}
                >
                  <span
                    className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                      useOllama ? 'left-5.5 translate-x-0' : 'left-0.5'
                    }`}
                    style={{ left: useOllama ? '22px' : '2px' }}
                  />
                </button>
              </div>

              <div className="pt-2 border-t border-gray-800 text-xs text-gray-600 space-y-1">
                <p>WebSocket: {connected ? 'Connected' : 'Disconnected'}</p>
                <p>Gestures detected: {gestureHistory.length}</p>
              </div>
            </div>

            <button
              onClick={() => setShowSettings(false)}
              className="mt-6 w-full border border-gray-700 py-2 rounded-lg text-xs text-gray-400 hover:bg-white/5 transition-colors"
            >
              Close
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
