import { useEffect, useRef, useCallback, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAppStore } from '../store';
import { useWebSocket } from '../hooks/useWebSocket';
import { useToast } from '../components/Toast';
import { apiClient } from '../lib/api';
import type { SignSpeakHistoryEntry } from '../types/api';
import { motion, AnimatePresence } from 'framer-motion';

// Complete visual tutorials database for the 23 classes
const SIGN_TUTORIALS: Record<string, { title: string; steps: string[]; illustration: string }> = {
  school: {
    title: 'School',
    steps: ['Place your dominant hand flat, palm down.', 'Clap it against your other flat hand at a right angle twice.', 'Mimic a teacher clapping for attention.'],
    illustration: '👏'
  },
  sorry: {
    title: 'Sorry',
    steps: ['Form an "A" hand shape (closed fist, thumb on the side).', 'Place it over your chest/heart area.', 'Rub your fist in a circle on your chest clockwise 2-3 times.'],
    illustration: '✊'
  },
  help: {
    title: 'Help',
    steps: ['Place your closed dominant hand with thumb up (like a thumbs up) on top of your flat non-dominant hand palm.', 'Lift both hands together upward in a single smooth motion.'],
    illustration: '👍'
  },
  easy: {
    title: 'Easy',
    steps: ['Hold both hands in front of you, palms facing up, fingers curved.', 'Brush the fingertips of your dominant hand upward against the back of the other fingertips twice.'],
    illustration: '🪶'
  },
  work: {
    title: 'Work',
    steps: ['Form closed fists with both hands.', 'Tap the wrist/heel of your dominant hand on the back of your non-dominant wrist twice.'],
    illustration: '🔨'
  },
  age: {
    title: 'Age',
    steps: ['Place your dominant hand in a fist near your chin, like holding a beard.', 'Pull the hand downward a few inches while squeezing your fist tighter.'],
    illustration: '🧔'
  },
  effort: {
    title: 'Effort',
    steps: ['Form "E" hand shapes with both hands (fingers curled, resting on thumb).', 'Push both hands forward in a circular motion, symbolizing pushing hard.'],
    illustration: '💪'
  },
  respect: {
    title: 'Respect',
    steps: ['Form "R" hand shapes with both hands (index and middle fingers crossed).', 'Move both hands from near your forehead forward and downward towards the person.'],
    illustration: '🫡'
  },
  near: {
    title: 'Near',
    steps: ['Hold your non-dominant hand bent slightly in front of your chest.', 'Bring your dominant hand from behind it, moving it close to the back of the non-dominant palm.'],
    illustration: '🤏'
  },
  home: {
    title: 'Home',
    steps: ['Form a flat "O" hand shape (fingertips touching thumb).', 'Touch the fingertips to your cheek near your mouth, then touch near your ear.'],
    illustration: '🏠'
  },
  friend: {
    title: 'Friend',
    steps: ['Hook your right index finger over your left index finger.', 'Reverse the hook, hooking your left index finger over your right index finger.'],
    illustration: '🤝'
  },
  washroom: {
    title: 'Washroom',
    steps: ['Form a "T" hand shape (fist with thumb tucked under index finger).', 'Shake your wrist back and forth gently several times.'],
    illustration: '🚽'
  },
  preset: {
    title: 'Preset',
    steps: ['Form both hands flat, palms facing you.', 'Push both hands forward together, representing locking a set position in place.'],
    illustration: '⚙️'
  },
  pass: {
    title: 'Pass',
    steps: ['Hold both hands in fists, thumbs up, palms facing each other.', 'Move your dominant fist forward past the stationary non-dominant fist.'],
    illustration: '🏁'
  },
  fail: {
    title: 'Fail',
    steps: ['Place your non-dominant hand flat, palm facing up.', 'Slide your flat dominant hand (palm facing up) off the edge of the non-dominant hand quickly.'],
    illustration: '❌'
  },
  village: {
    title: 'Village',
    steps: ['Form a roof shape with your hands (fingertips touching).', 'Tap your fingertips together several times while moving your hands side-to-side, representing houses.'],
    illustration: '🏡'
  },
  eating: {
    title: 'Eating',
    steps: ['Form a flat "O" hand shape with your dominant hand (fingertips touching thumb).', 'Bring the fingertips to your mouth twice in a feeding motion.'],
    illustration: '🍞'
  },
  eatig: {
    title: 'Eating',
    steps: ['Form a flat "O" hand shape with your dominant hand (fingertips touching thumb).', 'Bring the fingertips to your mouth twice in a feeding motion.'],
    illustration: '🍞'
  },
  drinking: {
    title: 'Drinking',
    steps: ['Form a "C" hand shape with your dominant hand, as if holding a cup.', 'Tilt the hand toward your mouth, mimicking drinking from a glass.'],
    illustration: '🥛'
  },
  teacher: {
    title: 'Teacher',
    steps: ['Form flattened "O" shapes with both hands near your temples.', 'Move them forward twice, then move both hands flat downwards to indicate a person.'],
    illustration: '👨‍🏫'
  },
  dress: {
    title: 'Dress',
    steps: ['Place both hands flat on your chest (thumbs touching chest).', 'Sweep both hands downward along your body, flaring outwards.'],
    illustration: '👗'
  },
  message: {
    title: 'Message',
    steps: ['Pinch your thumb and index fingers together on both hands.', 'Touch them together, then pull them apart while opening and closing the pinch.'],
    illustration: '✉️'
  },
  good: {
    title: 'Good',
    steps: ['Place the fingers of your flat right hand to your lips.', 'Bring the hand down, resting the back of it in the palm of your flat left hand.'],
    illustration: '✨'
  }
};

export default function SignSpeakTranslate() {
  const navigate = useNavigate();
  const { addToast } = useToast();
  const {
    gestureHistory, sentence,
    cameraOn,
    setIsStreaming, setCameraOn,
    voiceSpeed, voicePitch, vibrationEnabled,
    setCurrentResult, addHistoryEntry,
    useOllama,
    voiceId,
    facingMode, setFacingMode,
    bandwidthMode,
    currentGesture,
    clearSession,
    isStreaming,
  } = useAppStore();

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);

  // App mode segment selector
  const [translateMode, setTranslateMode] = useState<'sign2speech' | 'speech2sign'>('sign2speech');

  // Speech-to-Sign (mic translation) state
  const [isListening, setIsListening] = useState(false);
  const [recognizedText, setRecognizedText] = useState('');
  const [matchedSigns, setMatchedSigns] = useState<string[]>([]);
  const recognitionRef = useRef<any>(null);
  const [typedText, setTypedText] = useState('');



  const { connect, disconnect, connected, sendMessage } = useWebSocket({
    onDetection: () => {
      if (vibrationEnabled && 'vibrate' in navigator) {
        navigator.vibrate(50);
      }


    },
  });

  const startCamera = useCallback(async (modeOverride?: 'user' | 'environment') => {
    try {
      const activeMode = modeOverride || facingMode;
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: activeMode },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setCameraOn(true);
      setIsStreaming(true);
      connect();
      addToast(`Camera started using ${activeMode === 'user' ? 'front' : 'rear'} lens.`, 'success');
    } catch {
      addToast('Camera access denied. Please allow camera permissions.', 'error');
    }
  }, [connect, setCameraOn, setIsStreaming, addToast, facingMode]);

  const handleSwapCamera = useCallback(() => {
    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(30);
    const newMode = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(newMode);
    if (cameraOn) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
      }
      startCamera(newMode);
    } else {
      addToast(`Camera mode set to ${newMode === 'user' ? 'front' : 'rear'}. Tap to start.`, 'info');
    }
  }, [cameraOn, facingMode, startCamera, vibrationEnabled, addToast]);

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    disconnect();
    setIsStreaming(false);
    setCameraOn(false);
  }, [disconnect, setIsStreaming, setCameraOn]);

  useEffect(() => {
    return () => { stopCamera(); };
  }, [stopCamera]);

  // Frame sender loop for Sign-to-Speech & Practice Arena
  useEffect(() => {
    if (!cameraOn || !connected || translateMode !== 'sign2speech') return;

    const intervalTime = bandwidthMode === 'high' ? 200 : bandwidthMode === 'standard' ? 250 : 333;

    const intervalId = setInterval(() => {
      if (videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        if (context && video.readyState >= 2 && video.videoWidth > 0) {
          context.drawImage(video, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL('image/jpeg', 0.6);
          sendMessage({
            type: 'frame',
            image: dataUrl,
          });
        }
      }
    }, intervalTime);

    return () => clearInterval(intervalId);
  }, [cameraOn, connected, sendMessage, translateMode, bandwidthMode]);

  // Stop camera when switching to Speech-to-Sign
  useEffect(() => {
    if (translateMode === 'speech2sign' && cameraOn) {
      stopCamera();
    }
  }, [translateMode, cameraOn, stopCamera]);



  const handleClearStream = () => {
    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(15);
    clearSession();
    addToast('Translation stream cleared.', 'info');
  };

  const handleToggleCamera = () => {
    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(15);
    if (cameraOn) {
      stopCamera();
    } else {
      startCamera();
    }
  };

  const handleSpeak = async (text?: string) => {
    const t = text || sentence || "Hello, how can I help you today?";
    if (!t.trim()) {
      addToast('No translation text available to speak.', 'error');
      return;
    }
    try {
      const data = await apiClient.speak(t, voiceId);
      if (data.audio_url) {
        const audio = new Audio(`${apiClient.getApiBaseUrl()}${data.audio_url}`);
        audio.playbackRate = voiceSpeed;
        await audio.play();
      } else {
        const utterance = new SpeechSynthesisUtterance(t);
        utterance.rate = voiceSpeed;
        utterance.pitch = voicePitch;
        window.speechSynthesis.speak(utterance);
      }
      addToast('Speaking...', 'success');
    } catch {
      const utterance = new SpeechSynthesisUtterance(t);
      utterance.rate = voiceSpeed;
      utterance.pitch = voicePitch;
      window.speechSynthesis.speak(utterance);
    }
  };

  const handleSpeakAndSave = async () => {
    let textToSpeak = sentence || "Hello, how can I help you today?";
    
    if (useOllama && sentence) {
      addToast('Refining sentence using Ollama local AI...', 'info');
      try {
        const response = await apiClient.translate(sentence, useOllama, voiceId);
        if (response.translated) {
          textToSpeak = response.translated;
        }
      } catch (err) {
        console.warn('Ollama refinement fallback', err);
      }
    }

    const newResult: SignSpeakHistoryEntry = {
      id: crypto.randomUUID(),
      sentence: textToSpeak,
      signs: gestureHistory.map(g => g.class),
      confidence: gestureHistory.length > 0 ? (gestureHistory.reduce((sum, g) => sum + g.confidence, 0) / gestureHistory.length) : 0.9,
      timestamp: new Date().toISOString(),
      audioUrl: '',
      pinned: false
    };

    try {
      const speakResponse = await apiClient.speak(textToSpeak, voiceId);
      if (speakResponse.audioUrl || speakResponse.audio_url) {
        newResult.audioUrl = speakResponse.audioUrl || speakResponse.audio_url || '';
      }
    } catch (err) {
      console.warn('Speak generation fail', err);
    }

    addHistoryEntry(newResult);
    setCurrentResult(newResult);

    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate([30, 30]);

    if (newResult.audioUrl) {
      const audio = new Audio(`${apiClient.getApiBaseUrl()}${newResult.audioUrl}`);
      audio.playbackRate = voiceSpeed;
      audio.play().catch(() => {});
    } else {
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.rate = voiceSpeed;
      utterance.pitch = voicePitch;
      window.speechSynthesis.speak(utterance);
    }

    addToast('Speak and Archive session saved.', 'success');
    navigate('/live/result');
  };

  const handleToggleVoiceListening = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      addToast('Web Speech API is not supported in this browser.', 'error');
      return;
    }

    if (isListening) {
      if (recognitionRef.current) {
        recognitionRef.current.stop();
      }
      setIsListening(false);
      return;
    }

    const rec = new SpeechRecognition();
    rec.continuous = false;
    rec.interimResults = false;
    rec.lang = 'en-US';

    rec.onstart = () => {
      setIsListening(true);
      setRecognizedText('Listening to speaker...');
      setMatchedSigns([]);
      if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(30);
    };

    rec.onresult = (event: any) => {
      const text = event.results[0][0].transcript;
      setRecognizedText(text);

      const cleanWords = text.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"").split(/\s+/);
      const vocabKeys = Object.keys(SIGN_TUTORIALS);
      const matches = vocabKeys.filter(key => cleanWords.includes(key));
      setMatchedSigns(matches);

      if (matches.length > 0) {
        addToast(`Parsed gestures: ${matches.join(', ')}`, 'success');
      } else {
        addToast('No vocabulary gestures detected in the speech.', 'info');
      }
    };

    rec.onerror = (err: any) => {
      console.error(err);
      setIsListening(false);
      addToast('Microphone error or speech not detected.', 'error');
    };

    rec.onend = () => {
      setIsListening(false);
    };

    recognitionRef.current = rec;
    rec.start();
  };

  const handleTypeTextSubmit = () => {
    if (!typedText.trim()) return;
    setRecognizedText(typedText);

    const cleanWords = typedText.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g,"").split(/\s+/);
    const vocabKeys = Object.keys(SIGN_TUTORIALS);
    const matches = vocabKeys.filter(key => cleanWords.includes(key));
    setMatchedSigns(matches);

    setTypedText('');
    if (matches.length > 0) {
      addToast(`Parsed gestures: ${matches.join(', ')}`, 'success');
    } else {
      addToast('No vocabulary gestures detected in the typed text.', 'info');
    }
  };

    return (
    <div className="flex-grow flex flex-col bg-[#050505] select-none font-sans max-w-7xl mx-auto w-full gap-8 px-6">
      {/* Segment Mode Selector */}
      <div className="flex border border-white/5 p-1.5 gap-1.5 justify-between shrink-0 apple-glass rounded-[1.75rem]">
        {[
          { id: 'sign2speech', label: 'Sign to Voice', icon: 'interpreter_mode' },
          { id: 'speech2sign', label: 'Voice to Sign', icon: 'sign_language' },
        ].map((m) => (
          <button
            key={m.id}
            onClick={() => {
              if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(10);
              setTranslateMode(m.id as any);
            }}
            className={`flex-1 py-3 rounded-2xl flex items-center justify-center gap-2 font-bold text-xs tracking-wider uppercase active:scale-95 transition-all cursor-pointer border-0 ${
              translateMode === m.id
                ? 'bg-white text-black shadow-lg'
                : 'text-on-surface-variant/70 hover:text-on-surface hover:bg-white/5'
            }`}
          >
            <span className="material-symbols-outlined text-[18px]">{m.icon}</span>
            {m.label}
          </button>
        ))}
      </div>

      {translateMode === 'sign2speech' ? (
        // Mode A: Sign-to-Speech (Deaf to Speaking)
        <>
          {/* Main Video Feed Container */}
          <section className="relative">
            <div className="w-full aspect-[4/3] md:aspect-video bg-surface-container-lowest rounded-[2.5rem] overflow-hidden video-container-floating relative group border border-white/5">
              {/* Webcam stream (always mounted to prevent null refs & ensure autoplay) */}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-300 ${
                  cameraOn ? 'opacity-85 pointer-events-auto' : 'opacity-0 pointer-events-none'
                }`}
              />

              {/* Offline fallback image preview */}
              {!cameraOn && (
                <>
                  <div 
                    className="absolute inset-0 w-full h-full opacity-50 bg-cover bg-center transition-all duration-300"
                    style={{ backgroundImage: "url('/male_signing.jpg')" }}
                  />
                  <div className="absolute inset-0 bg-black/40 flex flex-col items-center justify-center gap-3">
                    <button 
                      onClick={(e) => { e.stopPropagation(); handleToggleCamera(); }}
                      className="w-16 h-16 rounded-full bg-primary text-on-primary flex items-center justify-center shadow-[0_0_25px_rgba(184,200,223,0.4)] hover:scale-105 active:scale-95 transition-all cursor-pointer border-0"
                    >
                      <span className="material-symbols-outlined text-3xl font-bold">videocam</span>
                    </button>
                    <p className="text-on-surface font-syne font-bold text-xs uppercase tracking-widest">
                      Tap to Start Live Camera Stream
                    </p>
                  </div>
                </>
              )}

              {/* Viewfinder Overlays */}
              <div className="absolute top-6 left-6 viewfinder-corner border-t-2 border-l-2 border-primary/30"></div>
              <div className="absolute top-6 right-6 viewfinder-corner border-t-2 border-r-2 border-primary/30"></div>
              <div className="absolute bottom-6 left-6 viewfinder-corner border-b-2 border-l-2 border-primary/30"></div>
              <div className="absolute bottom-6 right-6 viewfinder-corner border-b-2 border-r-2 border-primary/30"></div>

              {/* HUD Overlays */}
              <div className="absolute top-6 left-6 z-10" onClick={(e) => { e.stopPropagation(); handleToggleCamera(); }}>
                <div className="glass-panel-heavy text-white px-4 py-2 rounded-full flex items-center gap-2.5 font-sans text-[11px] font-semibold uppercase tracking-[0.08em] cursor-pointer hover:bg-white/10 active:scale-95 transition-all">
                  <span className={`w-2 h-2 rounded-full ${cameraOn ? 'bg-red-500/80 shadow-[0_0_8px_rgba(239,68,68,0.5)]' : 'bg-outline'} live-indicator`}></span>
                  {cameraOn ? 'Live Stream Active' : 'Stream Paused (Tap to Start)'}
                </div>
              </div>

              <div className="absolute bottom-6 right-6 z-10">
                <div className="glass-panel-heavy text-on-surface/90 px-4 py-2 rounded-full flex items-center gap-2 font-sans text-[11px] font-semibold tracking-wider uppercase font-mono">
                  <span className="material-symbols-outlined text-[18px] text-primary">videocam</span>
                  {facingMode === 'user' ? 'Front Lens' : 'Rear Lens'}
                </div>
              </div>

              {/* Floating Camera Swap FAB */}
              {cameraOn && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handleSwapCamera();
                  }}
                  className="absolute top-4 right-4 z-40 w-11 h-11 rounded-full bg-surface/85 backdrop-blur-md border border-white/10 flex items-center justify-center text-primary active:scale-90 hover:text-secondary shadow-lg transition-all"
                  title="Switch Camera Source"
                >
                  <span className="material-symbols-outlined text-[20px]">flip_camera_ios</span>
                </button>
              )}
            </div>
          </section>

          {/* Bento HUD Dashboard */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Card 1: Recognition Engine */}
            <div className="glass-panel-heavy p-6 rounded-[2rem] flex flex-col justify-between group text-left border border-white/5">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <p className="text-on-surface-variant text-[11px] font-bold uppercase tracking-[0.15em] mb-1">Recognition</p>
                  <h3 className="text-headline-md text-on-surface font-syne font-bold text-lg">{currentGesture ? currentGesture.class : 'Awaiting Sign'}</h3>
                </div>
                <div className="p-3 glass-panel rounded-2xl">
                  <span className="material-symbols-outlined text-primary text-xl">neurology</span>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4 mb-8">
                <div>
                  <p className="text-on-surface-variant text-[10px] uppercase tracking-widest mb-1 opacity-70">Throughput</p>
                  <p className="font-sans text-2xl font-bold text-on-surface">
                    {isStreaming ? (bandwidthMode === 'high' ? '5' : bandwidthMode === 'standard' ? '4' : '3') : '0'}
                    <span className="text-sm font-medium text-on-surface-variant ml-1">FPS</span>
                  </p>
                </div>
                <div>
                  <p className="text-on-surface-variant text-[10px] uppercase tracking-widest mb-1 opacity-70">Latency</p>
                  <p className="font-sans text-2xl font-bold text-on-surface">
                    {isStreaming ? '18' : '--'}
                    <span className="text-sm font-medium text-on-surface-variant ml-1">ms</span>
                  </p>
                </div>
              </div>
              <div className="space-y-3">
                <div className="w-full bg-white/5 rounded-full h-1.5 overflow-hidden">
                  <div 
                    className="bg-primary h-full rounded-full transition-all duration-300 shadow-[0_0_15px_rgba(212,228,252,0.4)]"
                    style={{ width: `${currentGesture ? currentGesture.confidence * 100 : 0}%` }}
                  />
                </div>
                <div className="flex justify-between items-center text-[10px] font-mono font-bold">
                  <span className="text-on-surface-variant tracking-wider">CONFIDENCE</span>
                  <span className="text-primary font-mono">
                    {currentGesture ? (currentGesture.confidence * 100).toFixed(1) + '%' : '0.0%'}
                  </span>
                </div>
              </div>
            </div>

            {/* Card 2: NLP Refinement */}
            <div className="glass-panel-heavy p-6 rounded-[2rem] group text-left border border-white/5 flex flex-col justify-between">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <p className="text-on-surface-variant text-[11px] font-bold uppercase tracking-[0.15em] mb-1">NLP Refinement</p>
                  <h3 className="text-headline-md text-on-surface font-syne font-bold text-lg">ASL-Refine</h3>
                </div>
                <div className="p-3 glass-panel rounded-2xl">
                  <span className="material-symbols-outlined text-primary text-xl">translate</span>
                </div>
              </div>
              <div className="space-y-3.5">
                <div className="flex items-center justify-between py-1.5 border-b border-white/5">
                  <span className="text-[12px] text-on-surface-variant font-medium">Contextual Engine</span>
                  <span className="text-[12px] font-bold text-on-surface font-mono">{useOllama ? 'Llama 3' : 'ASL-Rules'}</span>
                </div>
                <div className="flex items-center justify-between py-1.5 border-b border-white/5">
                  <span className="text-[12px] text-on-surface-variant font-medium">Refinement Mode</span>
                  <span className={`px-2.5 py-0.5 rounded-full text-[9px] font-bold tracking-widest border ${
                    useOllama 
                      ? 'bg-[#d4e4fc]/10 text-primary-fixed border-[#d4e4fc]/20' 
                      : 'bg-primary/10 text-primary border-primary/20'
                  }`}>
                    {useOllama ? 'ACTIVE' : 'PASSTHROUGH'}
                  </span>
                </div>
                <div className="flex items-center justify-between py-1.5">
                  <span className="text-[12px] text-on-surface-variant font-medium">TTS Synthesis</span>
                  <span className="text-[12px] font-bold text-on-surface truncate max-w-[120px] font-mono" title={voiceId}>
                    {voiceId.split('-')[2] || 'System Default'}
                  </span>
                </div>
              </div>
            </div>

            {/* Card 3: Session Overview */}
            <div className="glass-panel-heavy p-6 rounded-[2rem] group text-left border border-white/5 flex flex-col justify-between">
              <div className="flex justify-between items-start mb-6">
                <div>
                  <p className="text-on-surface-variant text-[11px] font-bold uppercase tracking-[0.15em] mb-1">Session Data</p>
                  <h3 className="text-headline-md text-on-surface font-syne font-bold text-lg">Live Analytics</h3>
                </div>
                <div className="p-3 glass-panel rounded-2xl">
                  <span className="material-symbols-outlined text-primary text-xl">bar_chart</span>
                </div>
              </div>
              <div className="flex items-end justify-between">
                <div>
                  <p className="font-sans text-4xl font-extrabold text-on-surface tracking-tight">
                    {gestureHistory.length}
                  </p>
                  <p className="text-[10px] font-bold text-on-surface-variant tracking-wider mt-2">SIGNS DETECTED</p>
                </div>
                {/* Simulated Telemetry Audio/Data Wave */}
                <div className="h-16 flex items-end gap-1.5 pb-1">
                  <div className="w-2.5 h-6 bg-white/10 rounded-full transition-all group-hover:h-8"></div>
                  <div className="w-2.5 h-10 bg-white/10 rounded-full transition-all group-hover:h-12"></div>
                  <div className="w-2.5 h-14 bg-white/20 rounded-full transition-all group-hover:h-16"></div>
                  <div className={`w-2.5 h-12 rounded-full transition-all duration-300 shadow-[0_0_20px_rgba(212,228,252,0.4)] ${isStreaming ? 'bg-primary h-14' : 'bg-white/10'}`}></div>
                  <div className="w-2.5 h-9 bg-white/10 rounded-full transition-all group-hover:h-11"></div>
                </div>
              </div>
            </div>
            
          </div>

          {/* Premium Translation Stream Area */}
          <section className="space-y-4">
            <div className="flex items-center justify-between px-2">
              <h2 className="text-sm font-bold font-syne text-on-surface tracking-wider uppercase">Live Transcription</h2>
              <div className="flex gap-2">
                <button 
                  onClick={() => handleSpeak()}
                  disabled={!sentence}
                  className="p-3 rounded-full text-on-surface-variant bg-white/5 border border-white/5 active:scale-90 hover:text-white cursor-pointer transition-all disabled:opacity-40"
                  title="Speak Translation"
                >
                  <span className="material-symbols-outlined text-[20px]">volume_up</span>
                </button>
                <button 
                  onClick={() => {
                    if (sentence) {
                      navigator.clipboard.writeText(sentence);
                      addToast('Translation copied to clipboard', 'success');
                    }
                  }}
                  disabled={!sentence}
                  className="p-3 rounded-full text-on-surface-variant bg-white/5 border border-white/5 active:scale-90 hover:text-white cursor-pointer transition-all disabled:opacity-40"
                  title="Copy Transcription"
                >
                  <span className="material-symbols-outlined text-[20px]">content_copy</span>
                </button>
              </div>
            </div>

            <div className="glass-panel-heavy rounded-[2.5rem] p-8 md:p-10 border border-white/10 relative overflow-hidden group text-left">
              {/* Subtle soft inner glow accent */}
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent pointer-events-none"></div>
              <div className="absolute top-0 left-0 w-1.5 h-full bg-gradient-to-b from-primary/40 to-transparent"></div>
              
              <div className="min-h-[140px] flex items-center relative z-10">
                <p className="text-xl md:text-2xl font-semibold text-on-surface leading-normal tracking-tight font-syne">
                  {sentence ? `"${sentence}"` : "Begin signing to accumulate and compile translation..."}
                  {isStreaming && (
                    <span className="inline-block w-1.5 h-7 bg-primary animate-pulse ml-3 rounded-full align-middle shadow-[0_0_12px_rgba(212,228,252,1)]" />
                  )}
                </p>
              </div>

              {/* Transcription Area Pill Buttons */}
              <div className="mt-8 flex gap-3 overflow-x-auto pb-2 scrollbar-none relative z-10">
                <button 
                  onClick={handleClearStream}
                  disabled={!sentence && gestureHistory.length === 0}
                  className="glass-panel border border-white/5 text-on-surface/80 px-6 py-2.5 rounded-full text-xs font-bold whitespace-nowrap active:scale-95 transition-all hover:bg-white/10 cursor-pointer disabled:opacity-40"
                >
                  Clear Stream
                </button>
                <button 
                  onClick={handleSpeakAndSave}
                  disabled={!sentence}
                  className="glass-panel border border-white/5 text-on-surface/80 px-6 py-2.5 rounded-full text-xs font-bold whitespace-nowrap active:scale-95 transition-all hover:bg-white/10 cursor-pointer disabled:opacity-40"
                >
                  Speak & Save
                </button>
                <Link 
                  to="/live/settings"
                  className="glass-panel border border-white/5 text-on-surface/80 px-6 py-2.5 rounded-full text-xs font-bold whitespace-nowrap active:scale-95 transition-all hover:bg-white/10 cursor-pointer no-underline flex items-center"
                >
                  Configure Settings
                </Link>
              </div>
            </div>
          </section>

          {/* Primary Floating Action Button */}
          <button 
            onClick={handleSpeakAndSave}
            disabled={!sentence}
            className="fixed bottom-24 right-6 w-14 h-14 bg-white text-black rounded-full shadow-[0_20px_40px_-8px_rgba(255,255,255,0.35)] flex items-center justify-center z-40 active:scale-95 transition-all hover:brightness-110 cursor-pointer disabled:opacity-45 border border-white/10"
            title="Speak and Save Translation"
          >
            <span className="material-symbols-outlined text-[26px] font-bold">campaign</span>
          </button>
        </>
      ) : (
        // Mode B: Speech-to-Sign (Speaking to Deaf)
        <>
          <section className="relative w-full h-[40vh] min-h-[340px] bg-background flex flex-col items-center justify-center p-6 border border-white/5 rounded-[2rem] overflow-hidden shadow-inner">
            <div className="absolute inset-0 bg-gradient-to-b from-primary/5 to-transparent pointer-events-none" />
            
            <AnimatePresence mode="wait">
              {isListening ? (
                <motion.div 
                  key="listening"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                  className="flex flex-col items-center gap-6"
                >
                  {/* Glowing Pulse waveforms */}
                  <div className="flex items-center gap-1.5 h-16">
                    {[1, 2, 3, 4, 5, 6, 7].map((bar) => (
                      <motion.div
                        key={bar}
                        className="w-1.5 bg-primary rounded-full"
                        animate={{ height: [12, 56, 12] }}
                        transition={{
                          duration: 0.6,
                          repeat: Infinity,
                          delay: bar * 0.08,
                        }}
                      />
                    ))}
                  </div>
                  <span className="font-mono text-xs text-primary uppercase tracking-widest animate-pulse">
                    Listening to spoken audio...
                  </span>
                </motion.div>
              ) : (
                <motion.div 
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex flex-col items-center text-center w-full max-w-sm px-6 space-y-6"
                >
                  <div className="flex flex-col items-center space-y-3">
                    <button 
                      onClick={handleToggleVoiceListening}
                      className="w-20 h-20 rounded-full bg-primary/10 border-2 border-primary text-primary flex items-center justify-center shadow-lg hover:scale-105 active:scale-95 transition-all cursor-pointer border-0"
                    >
                      <span className="material-symbols-outlined text-4xl">mic</span>
                    </button>
                    <div>
                      <h3 className="font-syne font-bold text-on-surface text-sm">Tap Microphone to Speak</h3>
                      <p className="text-[11px] text-on-surface-variant mt-1">Speak words like "school", "home", "sorry", "friend", "good".</p>
                    </div>
                  </div>
                  
                  {/* Or Type Message Option */}
                  <div className="w-full border-t border-white/5 pt-4 flex flex-col space-y-2">
                    <span className="text-[10px] font-mono text-on-surface-variant uppercase tracking-wider text-center">Or write message</span>
                    <div className="flex gap-2 w-full">
                      <input 
                        type="text"
                        placeholder="Type sign words here..."
                        value={typedText}
                        onChange={(e) => setTypedText(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleTypeTextSubmit();
                        }}
                        className="flex-grow h-10 px-3.5 rounded-xl border border-white/10 bg-surface-container-low text-on-surface text-xs outline-none focus:border-primary focus:ring-1 focus:ring-primary shadow-inner"
                      />
                      <button 
                        onClick={handleTypeTextSubmit}
                        className="h-10 px-4 bg-primary hover:opacity-95 text-white text-xs font-bold rounded-xl active:scale-95 transition-all cursor-pointer shadow-sm border-0"
                      >
                        Show Signs
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>

          {/* Transcribed Text & Demonstrative Gesture Cards */}
          <section className="flex-grow flex flex-col gap-6 text-left">
            {/* Transcribed Text Output */}
            <div className="glass-panel-heavy rounded-[2rem] p-6 relative flex flex-col justify-center min-h-[100px] border border-white/5">
              <span className="text-[10px] font-mono uppercase text-on-surface-variant tracking-wider mb-1">Speaker Transcription</span>
              <p className="text-xl font-syne font-bold text-on-surface leading-snug">
                {recognizedText || 'Waiting for spoken voice input...'}
              </p>
            </div>

            {/* Sign Demonstration Board */}
            <div className="flex-1 flex flex-col">
              <span className="text-xs font-mono uppercase text-on-surface-variant tracking-wider mb-3">Demonstrative Sign Gestures</span>
              
              <div className="flex-grow overflow-x-auto pb-2 flex gap-4 items-start snap-x scrollbar-none">
                {matchedSigns.length > 0 ? (
                  matchedSigns.map((signKey) => {
                    const tutorial = SIGN_TUTORIALS[signKey];
                    if (!tutorial) return null;
                    
                    return (
                      <motion.div
                        key={signKey}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="w-[280px] shrink-0 glass-panel-heavy rounded-2xl p-5 shadow-sm snap-center flex flex-col space-y-4 border border-white/5"
                      >
                        <div className="flex items-center gap-3 border-b border-white/5 pb-3">
                          <span className="text-3xl">{tutorial.illustration}</span>
                          <div>
                            <h4 className="font-syne font-bold text-lg text-primary capitalize">{tutorial.title}</h4>
                            <span className="text-[10px] font-mono text-on-surface-variant uppercase">Vocabulary Sign</span>
                          </div>
                        </div>
                        
                        <div className="space-y-2.5 flex-1">
                          {tutorial.steps.map((step, idx) => (
                            <div key={idx} className="flex gap-2 text-xs text-on-surface">
                              <span className="font-mono text-primary font-bold">{idx + 1}.</span>
                              <p className="leading-normal">{step}</p>
                            </div>
                          ))}
                        </div>
                      </motion.div>
                    );
                  })
                ) : (
                  <div className="w-full h-40 border border-dashed border-white/5 rounded-2xl flex flex-col items-center justify-center p-8 text-center text-on-surface-variant">
                    <span className="material-symbols-outlined text-4xl mb-2 text-outline">gesture</span>
                    <p className="text-sm font-syne font-semibold">No active sign cards</p>
                    <p className="text-xs max-w-xs mt-1">Speak or write into the selector above to demonstrate signing cards.</p>
                  </div>
                )}
              </div>
            </div>
          </section>
        </>
      )}

      {/* Hidden Canvas for optimized video frame capturing */}
      <canvas 
        ref={canvasRef} 
        className="hidden" 
        width={bandwidthMode === 'high' ? 640 : bandwidthMode === 'standard' ? 480 : 320} 
        height={bandwidthMode === 'high' ? 480 : bandwidthMode === 'standard' ? 360 : 240} 
      />
    </div>
  );
}
