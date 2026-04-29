import { useEffect, useRef, useState } from 'react';
import { Play, Square, Settings, Volume2, Trash2, Camera } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAppStore } from '../store';

const WS_URL = 'ws://localhost:8000/api/stream';

export default function LiveDetection() {
    const {
        currentGesture, gestureHistory, sentence,
        isStreaming, setIsStreaming, setSentence,
        setCurrentGesture, addGestureToHistory, clearSession
    } = useAppStore();

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const wsRef = useRef<WebSocket | null>(null);

    const [connected, setConnected] = useState(false);
    const [showSettings, setShowSettings] = useState(false);

    // Start webcam and generic loop simulation for connection
    const toggleStream = async () => {
        if (isStreaming) {
            stopStream();
        } else {
            await startStream();
        }
    };

    const startStream = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
            setIsStreaming(true);
            connectWebSocket();
        } catch (err) {
            console.error("Camera access denied", err);
            alert("Camera access was denied. Cannot start streaming.");
        }
    };

    const stopStream = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
            tracks.forEach(track => track.stop());
        }
        if (wsRef.current) wsRef.current.close();
        setIsStreaming(false);
        setConnected(false);
    };

    const connectWebSocket = () => {
        wsRef.current = new WebSocket(WS_URL);

        wsRef.current.onopen = () => {
            setConnected(true);
            console.log('WS Connected');
        };

        wsRef.current.onmessage = (evt) => {
            const data = JSON.parse(evt.data);
            if (data.type === 'detection') {
                const gesture = {
                    class: data.gesture,
                    confidence: data.confidence,
                    timestamp: Date.now()
                };
                setCurrentGesture(gesture);
                addGestureToHistory(gesture);

                // Mock assemble
                setSentence((prev: string) => prev ? prev + " " + data.gesture : data.gesture);
            }
        };

        wsRef.current.onclose = () => setConnected(false);
    };

    useEffect(() => {
        return () => {
            stopStream(); // Cleanup on unmount
        };
    }, []);

    const handleSpeak = async () => {
        if (!sentence) return;
        try {
            const res = await fetch('http://localhost:8000/api/speak', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentence })
            });
            const data = await res.json();
            console.log("Audio generated", data);
            alert("Speaking (Backend synthesized): " + sentence);
        } catch (err) {
            console.error("Failed to speak", err);
            // Fallback
            const utterance = new SpeechSynthesisUtterance(sentence);
            window.speechSynthesis.speak(utterance);
        }
    };

    return (
        <div className="flex flex-col h-[calc(100vh-64px)] overflow-hidden bg-background">
            {/* Status Bar */}
            <div className="h-14 border-b border-gray-800 glass-panel flex items-center justify-between px-6 shrink-0">
                <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                        <span className={`w-3 h-3 rounded-full ${connected ? 'bg-[#00E5CC] animate-pulse shadow-[0_0_10px_#00E5CC]' : 'bg-red-500'}`} />
                        <span className="font-mono text-sm tracking-tight text-gray-300">
                            {connected ? 'CONNECTED / DETECTING' : 'IDLE'}
                        </span>
                    </div>
                </div>
                <button onClick={() => setShowSettings(!showSettings)} className="p-2 hover:text-[#00E5CC] transition-colors rounded hover:bg-white/5">
                    <Settings size={20} />
                </button>
            </div>

            <div className="flex flex-1 overflow-hidden relative">
                {/* Main Video View (60%) */}
                <div className="w-[60%] border-r border-gray-800 p-6 flex flex-col items-center justify-center relative bg-black">
                    {!isStreaming ? (
                        <div className="text-center">
                            <Camera size={64} className="mx-auto mb-4 text-gray-600" />
                            <button
                                onClick={toggleStream}
                                className="bg-[#00E5CC] text-black font-bold px-8 py-3 rounded hover:opacity-90 transition-opacity flex items-center gap-2"
                            >
                                <Play size={18} /> Start Camera Feed
                            </button>
                        </div>
                    ) : (
                        <div className="relative w-full max-w-3xl aspect-video rounded-xl overflow-hidden border border-[#6C3FC8] shadow-[0_0_30px_rgba(108,63,200,0.2)]">
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                muted
                                className="w-full h-full object-cover"
                            />
                            <canvas
                                ref={canvasRef}
                                className="absolute inset-0 w-full h-full pointer-events-none"
                            />
                            <button
                                onClick={toggleStream}
                                className="absolute top-4 right-4 bg-red-500/80 text-white px-4 py-2 rounded flex items-center gap-2 hover:bg-red-500 backdrop-blur"
                            >
                                <Square size={16} /> Stop
                            </button>
                        </div>
                    )}
                </div>

                {/* Sidebar Panel (40%) */}
                <div className="w-[40%] flex flex-col p-6 overflow-y-auto glass-panel">

                    {/* Current Gesture Badge */}
                    <div className="mb-8">
                        <h3 className="text-gray-400 font-syne text-sm uppercase tracking-wider mb-3">Live Detection</h3>
                        <div className="relative">
                            <AnimatePresence mode="popLayout">
                                {currentGesture ? (
                                    <motion.div
                                        key={currentGesture.timestamp}
                                        initial={{ scale: 0.8, opacity: 0 }}
                                        animate={{ scale: 1, opacity: 1 }}
                                        exit={{ scale: 1.1, opacity: 0 }}
                                        className="bg-[#00E5CC]/10 border border-[#00E5CC] rounded-xl p-6 text-center text-[#00E5CC] shadow-[0_0_25px_rgba(0,229,204,0.15)]"
                                    >
                                        <span className="text-4xl font-mono font-bold">{currentGesture.class}</span>
                                    </motion.div>
                                ) : (
                                    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6 text-center text-gray-600 border-dashed">
                                        <span className="text-xl font-mono">Waiting for sign...</span>
                                    </div>
                                )}
                            </AnimatePresence>
                        </div>
                        {/* Confidence Bar */}
                        <div className="mt-4 h-2 w-full bg-gray-800 rounded-full overflow-hidden">
                            <div
                                className="h-full bg-gradient-to-r from-[#6C3FC8] to-[#00E5CC] transition-all duration-300"
                                style={{ width: `${(currentGesture?.confidence || 0) * 100}%` }}
                            />
                        </div>
                    </div>

                    {/* Rolling History */}
                    <div className="flex-1 min-h-[150px] flex flex-col border border-gray-800 rounded-xl p-4 bg-black/40 mb-6 font-mono text-sm max-h-[300px]">
                        <h3 className="text-gray-500 mb-2 border-b border-gray-800 pb-2">Detection Log</h3>
                        <div className="flex-1 overflow-y-auto pr-2 space-y-2">
                            <AnimatePresence>
                                {gestureHistory.map((g, i) => (
                                    <motion.div
                                        key={g.timestamp + i}
                                        initial={{ x: 20, opacity: 0 }}
                                        animate={{ x: 0, opacity: 1 }}
                                        className="flex justify-between items-center text-gray-300"
                                    >
                                        <span className="text-[#6C3FC8] font-bold">[{new Date(g.timestamp).toLocaleTimeString()}]</span>
                                        <span>{g.class}</span>
                                        <span className="text-[#00E5CC]">{(g.confidence * 100).toFixed(0)}%</span>
                                    </motion.div>
                                ))}
                            </AnimatePresence>
                        </div>
                    </div>

                    {/* Output Display */}
                    <div className="border border-gray-700 bg-[#0A0A0F] rounded-xl p-4 shadow-xl shrink-0">
                        <h3 className="font-syne text-[#ECECEC] mb-3">Assembled Sentence</h3>
                        <textarea
                            readOnly
                            value={sentence}
                            placeholder="Output will appear here..."
                            className="w-full h-24 bg-transparent resize-none focus:outline-none text-gray-300 border-none px-2"
                        />
                        <div className="flex justify-between mt-3 gap-2">
                            <button onClick={clearSession} className="p-2 border border-red-500/50 text-red-500 rounded hover:bg-red-500/10 flex-1 flex justify-center items-center gap-2 transition-colors">
                                <Trash2 size={18} /> Clear
                            </button>
                            <button onClick={handleSpeak} className="p-2 bg-[#00E5CC] text-black rounded flex-1 flex justify-center items-center gap-2 font-bold hover:bg-[#00E5CC]/80 transition-colors">
                                <Volume2 size={18} /> Speak
                            </button>
                        </div>
                    </div>
                </div>
            </div>

            {/* Settings Drawer */}
            <AnimatePresence>
                {showSettings && (
                    <motion.div
                        initial={{ x: 300 }}
                        animate={{ x: 0 }}
                        exit={{ x: 300 }}
                        className="absolute top-14 right-0 bottom-0 w-80 bg-[#0A0A0F] border-l border-gray-800 p-6 shadow-[-20px_0_30px_rgba(0,0,0,0.5)] z-20 overflow-y-auto"
                    >
                        <h2 className="text-xl font-syne mb-6 border-b border-gray-800 pb-4">Detection Settings</h2>

                        <div className="space-y-6">
                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Camera Source</label>
                                <select className="w-full bg-gray-900 border border-gray-700 p-2 rounded text-sm outline-none focus:border-[#00E5CC]">
                                    <option>FaceTime HD Camera</option>
                                    <option>External Webcam</option>
                                </select>
                            </div>

                            <div>
                                <label className="block text-sm text-gray-400 mb-2">Confidence Threshold</label>
                                <input type="range" min="30" max="90" className="w-full accent-[#00E5CC]" />
                                <div className="flex justify-between text-xs text-gray-500 mt-1">
                                    <span>0.3</span><span>0.9</span>
                                </div>
                            </div>

                            <div className="flex items-center justify-between">
                                <label className="text-sm text-gray-400">Use LLM Phrase Engine</label>
                                <input type="checkbox" className="toggle-checkbox" />
                            </div>
                        </div>

                        <button onClick={() => setShowSettings(false)} className="mt-8 w-full border border-gray-700 py-2 rounded text-sm hover:bg-white/5">
                            Close
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
