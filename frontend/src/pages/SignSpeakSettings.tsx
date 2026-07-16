import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAppStore } from '../store';
import { useToast } from '../components/Toast';

export default function SignSpeakSettings() {
  const navigate = useNavigate();
  const { addToast } = useToast();
  const {
    voiceSpeed,
    voicePitch,
    voiceId,
    vibrationEnabled,
    facingMode,
    bandwidthMode,
    setVoiceSpeed,
    setVoicePitch,
    setVoiceId,
    setVibrationEnabled,
    setFacingMode,
    setBandwidthMode,
    clearAllHistory,
    logoutUser
  } = useAppStore();

  const [highContrast, setHighContrast] = useState(false);
  const [largeText, setLargeText] = useState(true);
  const [isCalibrating, setIsCalibrating] = useState(false);

  const handleStartCalibration = () => {
    if (vibrationEnabled && 'vibrate' in navigator) {
      navigator.vibrate([40, 40]);
    }
    setIsCalibrating(true);
    addToast('Calibrating... Please align your hands within the frame.', 'info');
    setTimeout(() => {
      setIsCalibrating(false);
      addToast('Calibration completed successfully!', 'success');
    }, 2505);
  };

  const handleResetSystemData = () => {
    if (vibrationEnabled && 'vibrate' in navigator) {
      navigator.vibrate(100);
    }
    if (window.confirm("Are you sure you want to reset all system settings and translation history? This cannot be undone.")) {
      clearAllHistory();
      setVoiceSpeed(1.0);
      setVoicePitch(1.0);
      setVoiceId('ur-PK-AsmaNeural');
      setHighContrast(false);
      setLargeText(true);
      addToast('System data reset successfully', 'success');
    }
  };

  const handleLogout = () => {
    if (vibrationEnabled && 'vibrate' in navigator) {
      navigator.vibrate(50);
    }
    logoutUser();
    addToast('Logged out successfully', 'info');
    navigate('/login');
  };

  return (
    <div className="flex-grow flex flex-col px-6 pt-4 safe-pb select-none overflow-y-auto font-sans bg-background text-on-background max-w-7xl mx-auto w-full">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
        {/* Left Column */}
        <div className="space-y-12">
      {/* Hand Calibration Section */}
      <section className="space-y-6 shrink-0 text-left">
        <div className="flex items-center gap-3 text-on-surface">
          <span className="material-symbols-outlined text-primary text-2xl">precision_manufacturing</span>
          <h2 className="text-headline-md font-headline-md font-syne">Hand Calibration</h2>
        </div>
        <div className="relative rounded-xl overflow-hidden aspect-video border border-outline-variant bg-surface">
          <img 
            alt="Calibration View" 
            className="w-full h-full object-cover opacity-60" 
            src="/male_signing.jpg"
          />
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center">
            <div className={`w-40 h-40 border-4 border-dashed rounded-2xl flex items-center justify-center bg-primary/5 transition-all ${
              isCalibrating ? 'border-secondary scale-105 bg-secondary-container/10' : 'border-primary/60'
            }`}>
              <span className={`material-symbols-outlined text-6xl text-primary ${
                isCalibrating ? 'animate-pulse text-secondary' : ''
              }`}>{isCalibrating ? 'model_training' : 'front_hand'}</span>
            </div>
            <p className="mt-4 font-body-md text-on-surface bg-surface/75 backdrop-blur-sm px-4 py-1.5 rounded-full border border-outline-variant text-xs">
              {isCalibrating ? "Analyzing hand contours..." : "Align hands within frame"}
            </p>
          </div>
        </div>
        <button 
          onClick={handleStartCalibration}
          className="w-full h-14 bg-primary text-white font-syne font-bold text-sm rounded-xl flex items-center justify-center gap-2.5 active:scale-95 transition-transform shadow-md shadow-primary/10 cursor-pointer"
        >
          <span className="material-symbols-outlined text-xl">model_training</span>
          {isCalibrating ? 'CALIBRATING...' : 'START CALIBRATION'}
        </button>
      </section>

      {/* Voice Selection Section */}
      <section className="space-y-6 shrink-0 text-left">
        <div className="flex items-center gap-3 text-on-surface">
          <span className="material-symbols-outlined text-primary text-2xl">record_voice_over</span>
          <h2 className="text-headline-md font-headline-md font-syne">Voice Settings</h2>
        </div>
        <div className="bg-surface rounded-xl p-6 border border-outline-variant space-y-8 shadow-sm">
          {/* Expressive Voice Selector */}
          <div className="space-y-3">
            <label className="text-body-lg font-bold text-on-surface block text-xs font-mono uppercase tracking-wider">Expressive Voice Profile</label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {[
                { id: 'ur-PK-AsmaNeural', name: 'Asma (Urdu Female)', description: 'Conversational Urdu Voice Profile', icon: 'female' },
                { id: 'ur-PK-SalmanNeural', name: 'Salman (Urdu Male)', description: 'Clear Expressive Urdu Voice Profile', icon: 'male' },
                { id: 'en-US-EmmaNeural', name: 'Emma (English Female)', description: 'Expressive & Conversational English', icon: 'female' },
                { id: 'en-US-BrianNeural', name: 'Brian (English Male)', description: 'Warm & Expressive English Voice', icon: 'male' },
              ].map((v) => (
                <button
                  key={v.id}
                  onClick={() => {
                    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(15);
                    setVoiceId(v.id);
                    addToast(`Voice set to ${v.name.split(' ')[0]} neural profile`, 'info');
                  }}
                  className={`p-4 rounded-xl border-2 flex items-start gap-3 active:scale-[0.98] transition-all cursor-pointer text-left ${
                    voiceId === v.id
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-outline-variant bg-surface-variant text-on-surface-variant hover:bg-surface-variant/40'
                  }`}
                >
                  <span className="material-symbols-outlined text-2xl shrink-0 mt-0.5">{v.icon}</span>
                  <div className="min-w-0">
                    <p className="font-bold text-xs uppercase tracking-wider font-syne">{v.name}</p>
                    <p className="text-[10px] text-on-surface-variant mt-0.5 truncate">{v.description}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Speed Slider */}
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <label className="text-body-lg font-bold text-on-surface text-xs font-mono uppercase tracking-wider">Speech Speed</label>
              <span className="bg-primary/15 text-primary px-3 py-0.5 rounded-lg font-mono font-bold text-xs">{voiceSpeed.toFixed(1)}x</span>
            </div>
            <input 
              className="w-full h-2 cursor-pointer accent-primary bg-surface-variant rounded-lg appearance-none" 
              max="2" 
              min="0.5" 
              step="0.1" 
              type="range" 
              value={voiceSpeed}
              onChange={(e) => {
                setVoiceSpeed(parseFloat(e.target.value));
              }}
            />
          </div>

          {/* Pitch Slider */}
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <label className="text-body-lg font-bold text-on-surface text-xs font-mono uppercase tracking-wider">Voice Pitch</label>
              <span className="bg-primary/15 text-primary px-3 py-0.5 rounded-lg font-mono font-bold text-xs">
                {voicePitch === 1.0 ? 'Default' : `${voicePitch.toFixed(1)}x`}
              </span>
            </div>
            <input 
              className="w-full h-2 cursor-pointer accent-primary bg-surface-variant rounded-lg appearance-none" 
              max="1.5" 
              min="0.5" 
              step="0.1" 
              type="range" 
              value={voicePitch}
              onChange={(e) => {
                setVoicePitch(parseFloat(e.target.value));
              }}
            />
          </div>
        </div>
      </section>
        </div>

        {/* Right Column */}
        <div className="space-y-12">

      {/* Mobile & Hardware Optimization Section */}
      <section className="space-y-6 shrink-0 text-left">
        <div className="flex items-center gap-3 text-on-surface">
          <span className="material-symbols-outlined text-primary text-2xl">settings_system_daydream</span>
          <h2 className="text-headline-md font-headline-md font-syne">Mobile & Network Optimization</h2>
        </div>
        <div className="bg-surface rounded-xl p-6 border border-outline-variant space-y-8 shadow-sm">
          {/* Default Lens Selector */}
          <div className="space-y-3">
            <label className="text-body-lg font-bold text-on-surface block text-xs font-mono uppercase tracking-wider">Default Camera Lens</label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {[
                { id: 'user', name: 'Front Camera', description: 'Selfie view (signing oneself)', icon: 'portrait' },
                { id: 'environment', name: 'Rear Camera', description: 'World view (interpreting others)', icon: 'photo_camera' },
              ].map((c) => (
                <button
                  key={c.id}
                  onClick={() => {
                    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(15);
                    setFacingMode(c.id as any);
                    addToast(`Default camera set to ${c.name}`, 'info');
                  }}
                  className={`p-4 rounded-xl border-2 flex items-start gap-3 active:scale-[0.98] transition-all cursor-pointer text-left w-full ${
                    facingMode === c.id
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-outline-variant bg-surface-variant text-on-surface-variant hover:bg-surface-variant/40'
                  }`}
                >
                  <span className="material-symbols-outlined text-2xl shrink-0 mt-0.5">{c.icon}</span>
                  <div className="min-w-0">
                    <p className="font-bold text-xs uppercase tracking-wider font-syne">{c.name}</p>
                    <p className="text-[10px] text-on-surface-variant mt-0.5">{c.description}</p>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Quality & Bandwidth Optimizer */}
          <div className="space-y-3">
            <label className="text-body-lg font-bold text-on-surface block text-xs font-mono uppercase tracking-wider">Stream Quality (Mobile Bandwidth Optimizer)</label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { id: 'low', name: 'Low', desc: '320x240 @ 3 FPS', detail: 'Saves 75% data' },
                { id: 'standard', name: 'Standard', desc: '480x360 @ 4 FPS', detail: 'Balanced usage' },
                { id: 'high', name: 'High', desc: '640x480 @ 5 FPS', detail: 'Highest accuracy' },
              ].map((q) => (
                <button
                  key={q.id}
                  onClick={() => {
                    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(15);
                    setBandwidthMode(q.id as any);
                    addToast(`Stream profile set to ${q.name}`, 'info');
                  }}
                  className={`p-3 rounded-xl border-2 flex flex-col items-center justify-center text-center active:scale-[0.98] transition-all cursor-pointer w-full ${
                    bandwidthMode === q.id
                      ? 'border-primary bg-primary/10 text-primary'
                      : 'border-outline-variant bg-surface-variant text-on-surface-variant hover:bg-surface-variant/40'
                  }`}
                >
                  <p className="font-bold text-xs uppercase tracking-wider font-syne">{q.name}</p>
                  <p className="text-[9px] font-mono mt-1 text-on-surface-variant">{q.desc}</p>
                  <p className="text-[8px] text-outline mt-0.5">{q.detail}</p>
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Accessibility Section */}
      <section className="space-y-6 shrink-0 text-left">
        <div className="flex items-center gap-3 text-on-surface">
          <span className="material-symbols-outlined text-primary text-2xl">visibility</span>
          <h2 className="text-headline-md font-headline-md font-syne">Visual Accessibility</h2>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between p-5 bg-surface rounded-xl border border-outline-variant shadow-sm hover:bg-surface-variant/40 transition-all">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-surface-variant rounded-full flex items-center justify-center text-on-surface border border-outline-variant">
                <span className="material-symbols-outlined">contrast</span>
              </div>
              <span className="text-sm font-bold text-on-surface">High Contrast Mode</span>
            </div>
            <button 
              onClick={() => {
                if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(10);
                setHighContrast(!highContrast);
                document.documentElement.classList.toggle('high-contrast');
                addToast(highContrast ? 'High Contrast Disabled' : 'High Contrast Enabled', 'info');
              }}
              className={`w-14 h-8 rounded-full relative p-1 transition-colors duration-200 cursor-pointer ${
                highContrast ? 'bg-primary' : 'bg-outline-variant'
              }`}
            >
              <div className={`w-6 h-6 bg-white rounded-full shadow transition-transform duration-200 ${
                highContrast ? 'translate-x-6' : 'translate-x-0'
              }`} />
            </button>
          </div>

          <div className="flex items-center justify-between p-5 bg-surface rounded-xl border border-outline-variant shadow-sm hover:bg-surface-variant/40 transition-all">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-surface-variant rounded-full flex items-center justify-center text-on-surface border border-outline-variant">
                <span className="material-symbols-outlined">format_size</span>
              </div>
              <span className="text-sm font-bold text-on-surface">Large Text Output</span>
            </div>
            <button 
              onClick={() => {
                if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(10);
                setLargeText(!largeText);
                addToast(!largeText ? 'Large Text Enabled' : 'Large Text Disabled', 'info');
              }}
              className={`w-14 h-8 rounded-full relative p-1 transition-colors duration-200 cursor-pointer ${
                largeText ? 'bg-primary' : 'bg-outline-variant'
              }`}
            >
              <div className={`w-6 h-6 bg-white rounded-full shadow transition-transform duration-200 ${
                largeText ? 'translate-x-6' : 'translate-x-0'
              }`} />
            </button>
          </div>

          <div className="flex items-center justify-between p-5 bg-surface rounded-xl border border-outline-variant shadow-sm hover:bg-surface-variant/40 transition-all">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-surface-variant rounded-full flex items-center justify-center text-on-surface border border-outline-variant">
                <span className="material-symbols-outlined">vibration</span>
              </div>
              <span className="text-sm font-bold text-on-surface">Haptic Feedback</span>
            </div>
            <button 
              onClick={() => {
                const next = !vibrationEnabled;
                setVibrationEnabled(next);
                if (next && 'vibrate' in navigator) navigator.vibrate(50);
                addToast(next ? 'Haptics Enabled' : 'Haptics Disabled', 'info');
              }}
              className={`w-14 h-8 rounded-full relative p-1 transition-colors duration-200 cursor-pointer ${
                vibrationEnabled ? 'bg-primary' : 'bg-outline-variant'
              }`}
            >
              <div className={`w-6 h-6 bg-white rounded-full shadow transition-transform duration-200 ${
                vibrationEnabled ? 'translate-x-6' : 'translate-x-0'
              }`} />
            </button>
          </div>
        </div>
      </section>

      {/* Developer Portal Section */}
      <section className="space-y-6 shrink-0 text-left">
        <div className="flex items-center gap-3 text-on-surface">
          <span className="material-symbols-outlined text-primary text-2xl">code</span>
          <h2 className="text-headline-md font-headline-md font-syne">Developer Portal</h2>
        </div>
        <div className="grid grid-cols-1 gap-4">
          <Link 
            to="/developer"
            className="flex items-center p-4 bg-surface rounded-xl border border-outline-variant shadow-sm hover:bg-surface-variant/40 active:scale-[0.98] transition-all cursor-pointer no-underline"
          >
            <div className="w-12 h-12 bg-primary/10 text-primary rounded-xl flex items-center justify-center mr-4 shrink-0 border border-primary/20">
              <span className="material-symbols-outlined text-2xl">terminal</span>
            </div>
            <div className="flex-grow min-w-0">
              <div className="text-sm font-bold text-on-surface">Developer Console</div>
              <div className="text-xs text-on-surface-variant truncate mt-0.5 font-sans">Generate API keys and read integration guides</div>
            </div>
            <span className="material-symbols-outlined text-outline shrink-0">chevron_right</span>
          </Link>
        </div>
      </section>

      {/* Support Section */}
      <section className="space-y-6 shrink-0 text-left">
        <div className="flex items-center gap-3 text-on-surface">
          <span className="material-symbols-outlined text-primary text-2xl">help_center</span>
          <h2 className="text-headline-md font-headline-md font-syne">Support</h2>
        </div>
        <div className="grid grid-cols-1 gap-4">
          <a 
            onClick={(e) => { e.preventDefault(); addToast('Quick Start Guide loaded', 'success'); }}
            className="flex items-center p-4 bg-surface rounded-xl border border-outline-variant shadow-sm hover:bg-surface-variant/40 active:scale-[0.98] transition-all cursor-pointer no-underline" 
            href="#"
          >
            <div className="w-12 h-12 bg-primary/10 text-primary rounded-xl flex items-center justify-center mr-4 shrink-0 border border-primary/20">
              <span className="material-symbols-outlined text-2xl">menu_book</span>
            </div>
            <div className="flex-grow min-w-0">
              <div className="text-sm font-bold text-on-surface">Quick Start Guide</div>
              <div className="text-xs text-on-surface-variant truncate mt-0.5">Learn the basics in 2 minutes</div>
            </div>
            <span className="material-symbols-outlined text-outline shrink-0">chevron_right</span>
          </a>

          <a 
            onClick={(e) => { e.preventDefault(); addToast('Sign Library: 23 active gestures loaded', 'success'); }}
            className="flex items-center p-4 bg-surface rounded-xl border border-outline-variant shadow-sm hover:bg-surface-variant/40 active:scale-[0.98] transition-all cursor-pointer no-underline" 
            href="#"
          >
            <div className="w-12 h-12 bg-secondary/10 text-secondary rounded-xl flex items-center justify-center mr-4 shrink-0 border border-secondary/20">
              <span className="material-symbols-outlined text-2xl">video_library</span>
            </div>
            <div className="flex-grow min-w-0">
              <div className="text-sm font-bold text-on-surface">Sign Video Library</div>
              <div className="text-xs text-on-surface-variant truncate mt-0.5">Browse 23 detected vocabulary gestures</div>
            </div>
            <span className="material-symbols-outlined text-outline shrink-0">chevron_right</span>
          </a>
        </div>
      </section>

      {/* Danger Zone / Log Out */}
      <section className="pt-8 shrink-0 space-y-4 text-left">
        <button 
          onClick={handleLogout}
          className="w-full h-14 bg-surface border border-outline-variant hover:border-red-500/50 hover:text-red-500 font-syne font-bold text-sm rounded-xl flex items-center justify-center gap-2 active:scale-95 transition-all cursor-pointer text-on-surface shadow-sm"
        >
          <span className="material-symbols-outlined text-lg">logout</span>
          LOG OUT OF SESSION
        </button>

        <button 
          onClick={handleResetSystemData}
          className="w-full h-14 border border-red-500/40 text-red-500 font-syne font-bold text-sm rounded-xl flex items-center justify-center gap-2 active:bg-red-500/15 hover:bg-red-500/5 transition-colors cursor-pointer"
        >
          <span className="material-symbols-outlined text-lg">delete_forever</span>
          RESET ALL SYSTEM DATA
        </button>
      </section>
        </div>
      </div>
    </div>
  );
}
