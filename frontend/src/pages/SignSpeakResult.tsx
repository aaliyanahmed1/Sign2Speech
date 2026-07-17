import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAppStore } from '../store';
import { useToast } from '../components/Toast';
import { apiClient } from '../lib/api';

export default function SignSpeakResult() {
  const navigate = useNavigate();
  const location = useLocation();
  const { addToast } = useToast();
  const { 
    currentResult, 
    setCurrentResult, 
    voiceSpeed, 
    voicePitch, 
    voiceId,
    togglePin,
    pinnedIds,
    vibrationEnabled
  } = useAppStore();

  const [isPlaying, setIsPlaying] = useState(false);
  const [waveformHeights, setWaveformHeights] = useState([16, 16, 16, 16, 16, 16]);
  const [isCleared, setIsCleared] = useState(false);

  // Animate the voice waveform
  useEffect(() => {
    if (isPlaying) {
      const interval = setInterval(() => {
        setWaveformHeights(
          Array.from({ length: 6 }, () => Math.floor(Math.random() * 24) + 8)
        );
      }, 100);
      return () => clearInterval(interval);
    } else {
      setWaveformHeights([16, 16, 16, 16, 16, 16]);
    }
  }, [isPlaying]);

  if (!currentResult) {
    return (
      <div className="flex-grow flex flex-col items-center justify-center p-8 bg-background text-center">
        <span className="material-symbols-outlined text-6xl text-outline mb-4">description</span>
        <p className="text-on-surface-variant text-body-md mb-6">No translation result to display</p>
        <button
          onClick={() => {
            const prefix = location.pathname.startsWith('/live') ? '/live' : '/signspeak';
            navigate(prefix);
          }}
          className="bg-primary text-white font-syne font-bold px-6 py-3 rounded-xl text-sm active:scale-95 transition-transform cursor-pointer shadow-md"
        >
          Go to Translate
        </button>
      </div>
    );
  }

  const handleSpeak = async () => {
    if (isPlaying) return;
    setIsPlaying(true);
    if (vibrationEnabled && 'vibrate' in navigator) {
      navigator.vibrate(50);
    }

    try {
      const data = await apiClient.speak(currentResult.sentence, voiceId);
      if (data.audio_url) {
        const audio = new Audio(`${apiClient.getApiBaseUrl()}${data.audio_url}`);
        audio.playbackRate = voiceSpeed;
        audio.onended = () => setIsPlaying(false);
        await audio.play();
      } else {
        const u = new SpeechSynthesisUtterance(currentResult.sentence);
        u.rate = voiceSpeed;
        u.pitch = voicePitch;
        u.onend = () => setIsPlaying(false);
        window.speechSynthesis.speak(u);
      }
    } catch {
      const u = new SpeechSynthesisUtterance(currentResult.sentence);
      u.rate = voiceSpeed;
      u.pitch = voicePitch;
      u.onend = () => setIsPlaying(false);
      window.speechSynthesis.speak(u);
    }

    // Backup safety fallback to turn off waveform
    setTimeout(() => {
      setIsPlaying(false);
    }, 3500);
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(currentResult.sentence);
      if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(20);
      addToast('Copied to clipboard', 'success');
    } catch {
      addToast('Failed to copy', 'error');
    }
  };

  const handleRepeat = () => {
    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate([30, 50, 30]);
    handleSpeak();
  };

  const handleSaveToggle = () => {
    togglePin(currentResult.id);
    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(20);
    const isPinnedNow = !pinnedIds.has(currentResult.id);
    addToast(isPinnedNow ? 'Translation saved to pinned' : 'Translation removed from pinned', 'info');
  };

  const handleClear = () => {
    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(100);
    setIsCleared(true);
    setTimeout(() => {
      setCurrentResult(null);
      addToast('Cleared translation output', 'info');
      const prefix = location.pathname.startsWith('/live') ? '/live' : '/signspeak';
      navigate(prefix);
    }, 400);
  };

  const isPinned = pinnedIds.has(currentResult.id);

  return (
    <div className="flex-grow flex flex-col p-margin-mobile gap-6 bg-background text-on-background max-w-2xl mx-auto w-full">
      {/* Viewfinder Context Card using Local Verified Asset */}
      <div className="relative w-full aspect-video rounded-3xl overflow-hidden border border-white/5 bg-surface shadow-md shrink-0">
        <div 
          className="absolute inset-0 bg-cover bg-center opacity-65" 
          style={{ backgroundImage: "url('/male_signing.jpg')" }}
        />
        <div className="absolute inset-0 bg-gradient-to-t from-background to-transparent" />
        <div className="absolute top-4 left-4 flex items-center gap-2 bg-surface/80 text-on-surface px-3.5 py-2 rounded-full backdrop-blur-md border border-white/10">
          <span className="material-symbols-outlined text-[16px]">history</span>
          <span className="font-label-caps text-label-caps uppercase">Last Input</span>
        </div>
      </div>

      {/* Translation Output Card */}
      <section className={`glass-card-dark rounded-[2rem] p-8 relative flex flex-col gap-4 overflow-hidden transition-all duration-300 ${
        isCleared ? 'opacity-0 scale-95' : 'opacity-100 scale-100'
      }`}>
        <div className="flex justify-between items-center">
          <span className="font-label-caps text-label-caps text-primary uppercase tracking-widest font-mono">Translation Output</span>
          <button 
            onClick={handleCopy}
            className="p-2.5 rounded-full hover:bg-white/5 text-primary transition-all active:scale-90 cursor-pointer border border-white/5 bg-transparent" 
            title="Copy text"
          >
            <span className="material-symbols-outlined text-primary text-lg">content_copy</span>
          </button>
        </div>
        <p className="text-2xl text-on-surface leading-normal py-2 text-left font-bold font-syne">
          "{currentResult.sentence}"
        </p>

        {/* Active Voice Waveform */}
        <div className={`h-10 flex items-center justify-center gap-1.5 transition-opacity duration-300 ${
          isPlaying ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}>
          {waveformHeights.map((h, i) => (
            <div 
              key={i} 
              className="waveform-bar w-1.5 bg-primary rounded-full transition-all duration-100" 
              style={{ height: `${h}px` }} 
            />
          ))}
        </div>
      </section>

      {/* Primary Play Button */}
      <button 
        onClick={handleSpeak}
        className="w-full h-14 bg-primary text-on-primary font-syne font-bold text-sm rounded-xl flex items-center justify-center gap-3.5 shadow-md active:scale-95 transition-all cursor-pointer shrink-0 border-0" 
      >
        <span className="material-symbols-outlined text-2xl" style={{ fontVariationSettings: "'FILL' 1" }}>volume_up</span>
        PLAY AUDIO
      </button>

      {/* Control Grid */}
      <div className="grid grid-cols-3 gap-3 shrink-0">
        <button 
          onClick={handleRepeat}
          className="flex flex-col items-center justify-center gap-2 p-4 bg-white/5 border border-white/5 hover:bg-white/10 transition-all active:scale-95 cursor-pointer text-on-surface rounded-xl shadow-sm"
        >
          <span className="material-symbols-outlined text-primary text-xl">replay</span>
          <span className="font-label-caps text-[10px] uppercase font-bold tracking-wider">Repeat</span>
        </button>
        
        <button 
          onClick={handleSaveToggle}
          className={`flex flex-col items-center justify-center gap-2 p-4 rounded-xl border transition-all active:scale-95 cursor-pointer shadow-sm ${
            isPinned 
              ? 'bg-primary/10 border-primary text-primary' 
              : 'bg-white/5 border-white/5 hover:bg-white/10 text-on-surface'
          }`}
        >
          <span className="material-symbols-outlined text-primary text-xl" style={{ fontVariationSettings: isPinned ? "'FILL' 1" : "'FILL' 0" }}>bookmark</span>
          <span className="font-label-caps text-[10px] uppercase font-bold tracking-wider">{isPinned ? 'Saved' : 'Save'}</span>
        </button>
        
        <button 
          onClick={handleClear}
          className="flex flex-col items-center justify-center gap-2 p-4 bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/15 transition-all active:scale-95 cursor-pointer rounded-xl shadow-sm"
        >
          <span className="material-symbols-outlined text-xl">delete_sweep</span>
          <span className="font-label-caps text-[10px] uppercase font-bold tracking-wider">Clear</span>
        </button>
      </div>

      {/* AI Refinement Status Box */}
      <div className="glass-card-dark rounded-[1.5rem] p-5 flex items-start gap-4 mt-auto shrink-0">
        <div className="bg-secondary-container/30 border border-secondary/20 p-2 rounded-xl text-secondary flex items-center justify-center">
          <span className="material-symbols-outlined text-[20px]">auto_awesome</span>
        </div>
        <div className="text-left">
          <h4 className="font-label-caps text-[10px] uppercase font-bold tracking-wider text-secondary mb-1">AI Refinement Active</h4>
          <p className="font-body-md text-xs text-on-surface-variant leading-snug">Grammar corrected and sentence flow optimized for clarity.</p>
        </div>
      </div>
    </div>
  );
}
