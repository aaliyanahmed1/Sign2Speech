import { useState, useMemo, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAppStore } from '../store';
import { useToast } from '../components/Toast';
import { apiClient } from '../lib/api';

// Default mock history entries matching the design
const MOCK_ENTRIES = [
  {
    id: 'mock_1',
    sentence: "Where is the nearest hospital?",
    signs: ["WHERE", "NEAREST", "HOSPITAL"],
    confidence: 0.96,
    timestamp: new Date(Date.now() - 3600000 * 2).toISOString(), // 2 hours ago
    pinned: true,
  },
  {
    id: 'mock_2',
    sentence: "I would like to order a large coffee with milk.",
    signs: ["I", "WANT", "COFFEE", "MILK"],
    confidence: 0.94,
    timestamp: new Date().toISOString(), // Today
    pinned: false,
  },
  {
    id: 'mock_3',
    sentence: "Thank you for your help, I really appreciate it.",
    signs: ["THANK_YOU", "HELP", "APPRECIATE"],
    confidence: 0.98,
    timestamp: new Date(Date.now() - 3600000 * 24).toISOString(), // Yesterday
    pinned: false,
  },
  {
    id: 'mock_4',
    sentence: "Can you show me the way to the train station?",
    signs: ["SHOW", "WAY", "TRAIN", "STATION"],
    confidence: 0.89,
    timestamp: new Date(Date.now() - 3600000 * 48).toISOString(), // Oct 23
    pinned: false,
  },
  {
    id: 'mock_5',
    sentence: "Hello, my name is Alex. Nice to meet you.",
    signs: ["HELLO", "NAME", "A-L-E-X", "MEET_YOU"],
    confidence: 0.95,
    timestamp: new Date(Date.now() - 3600000 * 72).toISOString(), // Oct 22
    pinned: false,
  }
];

export default function SignSpeakHistory() {
  const navigate = useNavigate();
  const location = useLocation();
  const { addToast } = useToast();
  const {
    historyEntries,
    pinnedIds,
    voiceSpeed,
    voicePitch,
    togglePin,
    clearAllHistory,
    setCurrentResult,
    setHistoryEntries,
    vibrationEnabled
  } = useAppStore();

  const [searchQuery, setSearchQuery] = useState('');

  // Populate mock data if store history is empty
  useEffect(() => {
    if (historyEntries.length === 0) {
      setHistoryEntries(MOCK_ENTRIES);
      // Auto-pin the first mock item
      togglePin('mock_1');
    }
  }, [historyEntries, setHistoryEntries]);

  const handlePlay = async (text: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(20);
    try {
      const data = await apiClient.speak(text);
      if (data.audio_url) {
        const audio = new Audio(`${apiClient.getApiBaseUrl()}${data.audio_url}`);
        audio.playbackRate = voiceSpeed;
        await audio.play();
      } else {
        const u = new SpeechSynthesisUtterance(text);
        u.rate = voiceSpeed;
        u.pitch = voicePitch;
        window.speechSynthesis.speak(u);
      }
    } catch {
      const u = new SpeechSynthesisUtterance(text);
      u.rate = voiceSpeed;
      u.pitch = voicePitch;
      window.speechSynthesis.speak(u);
    }
  };

  const handleViewDetail = (entry: typeof historyEntries[0]) => {
    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(15);
    setCurrentResult(entry);
    const prefix = location.pathname.startsWith('/live') ? '/live' : '/signspeak';
    navigate(`${prefix}/result`);
  };

  const handleTogglePin = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    togglePin(id);
    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(20);
    addToast(pinnedIds.has(id) ? 'Removed from pinned' : 'Pinned phrase', 'info');
  };

  const handleClearAll = async () => {
    if (vibrationEnabled && 'vibrate' in navigator) navigator.vibrate(100);
    clearAllHistory();
    try { await apiClient.clearAllSessions(); } catch { /* offline ok */ }
    addToast('All history cleared', 'info');
  };

  // Filter history entries based on query
  const filteredEntries = useMemo(() => {
    const query = searchQuery.toLowerCase().trim();
    if (!query) return historyEntries;
    return historyEntries.filter(
      (e) =>
        e.sentence.toLowerCase().includes(query) ||
        e.signs.some((s) => s.toLowerCase().includes(query))
    );
  }, [historyEntries, searchQuery]);

  // Separate pinned and recent activity
  const pinnedEntries = useMemo(() => {
    return filteredEntries.filter((e) => pinnedIds.has(e.id));
  }, [filteredEntries, pinnedIds]);

  const recentEntries = useMemo(() => {
    return filteredEntries.filter((e) => !pinnedIds.has(e.id));
  }, [filteredEntries, pinnedIds]);

  const formatDate = (isoString: string) => {
    const d = new Date(isoString);
    // Simple custom date formatter
    const now = new Date();
    const isToday = d.toDateString() === now.toDateString();
    const isYesterday = new Date(now.setDate(now.getDate() - 1)).toDateString() === d.toDateString();
    
    const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    if (isToday) return `Today, ${timeStr}`;
    if (isYesterday) return `Yesterday, ${timeStr}`;
    
    const months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
    return `${months[d.getMonth()]} ${d.getDate()}, ${timeStr}`;
  };

  return (
    <div className="flex-grow flex flex-col px-margin-mobile py-6 history-list select-none">
      {/* Search Section */}
      <section className="mb-8 shrink-0">
        <div className="relative w-full">
          <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-on-surface-variant">search</span>
          <input 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full h-12 pl-12 pr-4 bg-surface-container-low border border-outline-variant focus:border-primary focus:ring-1 focus:ring-primary rounded-xl text-body-md transition-all outline-none text-on-surface placeholder:text-on-surface-variant/50" 
            placeholder="Search previous translations..." 
            type="text"
          />
        </div>
      </section>

      {/* Pinned Section */}
      {pinnedEntries.length > 0 && (
        <section className="mb-8 shrink-0">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-headline-md font-headline-md flex items-center gap-2 text-on-surface">
              <span className="material-symbols-outlined text-primary" style={{ fontVariationSettings: "'FILL' 1" }}>push_pin</span>
              Pinned Phrases
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {pinnedEntries.map((entry) => (
              <div 
                key={entry.id}
                onClick={() => handleViewDetail(entry)}
                className="bg-surface-container-high border border-primary/40 p-4 rounded-xl flex items-center gap-4 active-tap cursor-pointer shadow-lg shadow-primary/5 hover:border-primary/70 transition-colors"
              >
                <div className="flex-grow min-w-0">
                  <p className="text-body-lg font-semibold text-on-surface truncate">"{entry.sentence}"</p>
                  <p className="text-on-surface-variant text-label-caps mt-1 uppercase">{formatDate(entry.timestamp)}</p>
                </div>
                <button 
                  onClick={(e) => handlePlay(entry.sentence, e)}
                  className="w-12 h-12 bg-primary text-on-primary rounded-full flex items-center justify-center shadow-lg active:scale-90 transition-transform cursor-pointer shrink-0"
                >
                  <span className="material-symbols-outlined" style={{ fontVariationSettings: "'FILL' 1" }}>play_arrow</span>
                </button>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* History Feed Section */}
      <section className="flex-grow flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-headline-md font-headline-md text-on-surface">Recent Activity</h2>
          {historyEntries.length > 0 && (
            <button 
              onClick={handleClearAll}
              className="text-primary font-label-caps text-label-caps hover:opacity-80 transition-opacity cursor-pointer uppercase"
            >
              CLEAR ALL
            </button>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 flex-grow">
          {recentEntries.map((entry) => (
            <div 
              key={entry.id}
              onClick={() => handleViewDetail(entry)}
              className="bg-surface-container border border-white/5 p-4 rounded-xl flex items-center gap-4 active-tap cursor-pointer hover:border-white/10 transition-colors"
            >
              <div className="flex-grow min-w-0">
                <div className="flex items-center gap-2 mb-2 text-on-surface-variant">
                  <span className="material-symbols-outlined text-[16px]">schedule</span>
                  <span className="text-label-caps font-label-caps uppercase">{formatDate(entry.timestamp)}</span>
                </div>
                <p className="text-body-md text-on-surface truncate">"{entry.sentence}"</p>
              </div>
              <div className="flex flex-col gap-2 shrink-0">
                <button 
                  onClick={(e) => handleTogglePin(entry.id, e)}
                  className={`w-10 h-10 rounded-lg flex items-center justify-center active:scale-90 transition-all border cursor-pointer ${
                    pinnedIds.has(entry.id)
                      ? 'bg-primary-container/20 border-primary/50 text-primary'
                      : 'bg-surface-variant border-white/5 text-on-surface-variant hover:bg-surface-bright'
                  }`}
                >
                  <span className="material-symbols-outlined text-[20px]" style={{ fontVariationSettings: pinnedIds.has(entry.id) ? "'FILL' 1" : "'FILL' 0" }}>push_pin</span>
                </button>
                <button 
                  onClick={(e) => handlePlay(entry.sentence, e)}
                  className="w-10 h-10 bg-secondary text-on-secondary rounded-lg flex items-center justify-center active:scale-90 transition-all shadow-md cursor-pointer"
                >
                  <span className="material-symbols-outlined text-[20px]">volume_up</span>
                </button>
              </div>
            </div>
          ))}

          {recentEntries.length === 0 && pinnedEntries.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-center text-on-surface-variant">
              <span className="material-symbols-outlined text-5xl mb-2 text-outline">history</span>
              <p className="text-sm">No translations match your search</p>
            </div>
          )}

          {/* Atmospheric Banner */}
          <div className="relative w-full h-48 rounded-xl overflow-hidden mt-8 border border-white/10 group shrink-0 col-span-1 md:col-span-2">
            <div 
              className="w-full h-full bg-cover bg-center transition-transform duration-700 group-hover:scale-110" 
              style={{ backgroundImage: "url('https://lh3.googleusercontent.com/aida-public/AB6AXuCpSuXD36xN4dPtU0ATHGoWCSDdN2OT_6Bn00K41QuWmkpAzo_OMXcwXRsCp68DGpEEsh81nlz_jFxbiLSTTQjXKM5Hf8wKLlUIdEIi5E04rlL7gQ2hMhOVKXZdWHH6FRGWyR3tE21MwtP2UEig4nHLChgrjYDT90lFzGp1pzka6MF3YU8VddYGRHzBOo_sCJASM97qqDK5s0qgtcXBnl5f5mz9EmJ4HYGDi9hl_o6QCjFg0KnmiH4HZQ')" }}
            />
            <div className="absolute inset-0 bg-surface/40 backdrop-blur-[2px] flex items-end p-6">
              <div className="bg-primary/90 backdrop-blur-md px-5 py-3 rounded-xl shadow-2xl border border-white/20">
                <p className="text-headline-md font-headline-md text-on-primary">Your Journey In Words</p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
