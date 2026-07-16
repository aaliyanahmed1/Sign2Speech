import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Key, Copy, Trash2, Code, Terminal, Play, Eye, EyeOff, Plus, Volume2, Shield } from 'lucide-react';
import { useToast } from '../components/Toast';

interface APIKey {
  id: string;
  key: string;
  name: string;
  created_at: string;
  calls_count: number;
}

export default function Developer() {
  const { addToast } = useToast();
  const [keys, setKeys] = useState<APIKey[]>([]);
  const [newKeyName, setNewKeyName] = useState('');
  const [showKeyId, setShowKeyId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'curl' | 'js' | 'python'>('curl');
  
  // Interactive playground state
  const [playgroundGestures, setPlaygroundGestures] = useState('HELLO, FRIEND, HELP');
  const [playgroundOutput, setPlaygroundOutput] = useState<any>(null);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [playgroundLoading, setPlaygroundLoading] = useState(false);

  useEffect(() => {
    fetchKeys();
  }, []);

  const fetchKeys = async () => {
    try {
      const res = await fetch('/api/v1/developer/keys');
      if (res.ok) {
        const data = await res.json();
        setKeys(data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleCreateKey = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKeyName.trim()) return;

    try {
      const res = await fetch('/api/v1/developer/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: newKeyName })
      });
      if (res.ok) {
        const newKey = await res.json();
        setKeys([...keys, newKey]);
        setNewKeyName('');
        addToast(`API Key "${newKey.name}" generated successfully.`, 'success');
      }
    } catch (e) {
      addToast('Failed to generate API Key', 'error');
    }
  };

  const handleRevokeKey = async (id: string, name: string) => {
    if (!window.confirm(`Are you sure you want to revoke the key for "${name}"? This will immediately break any apps using it.`)) {
      return;
    }
    try {
      const res = await fetch(`/api/v1/developer/keys/${id}`, { method: 'DELETE' });
      if (res.ok) {
        setKeys(keys.filter(k => k.id !== id));
        addToast(`API Key for "${name}" revoked.`, 'info');
      }
    } catch (e) {
      addToast('Failed to revoke API Key', 'error');
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    addToast('API Key copied to clipboard.', 'success');
  };

  const handleTestAPI = async () => {
    if (keys.length === 0) {
      addToast('Please generate an API Key first.', 'info');
      return;
    }
    setPlaygroundLoading(true);
    setPlaygroundOutput(null);

    const activeKey = keys[0].key; // Use first active key for test
    const gesturesArray = playgroundGestures.split(',').map(g => g.trim().toUpperCase()).filter(Boolean);

    try {
      // 1. Call translation endpoint
      const translateRes = await fetch('/api/v1/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': activeKey
        },
        body: JSON.stringify({
          gestures: gesturesArray,
          use_context_nlp: true
        })
      });

      if (!translateRes.ok) {
        const err = await translateRes.json();
        throw new Error(err.detail || 'Translation query failed');
      }

      const translateData = await translateRes.json();

      // 2. Call voice synthesis endpoint
      const voiceRes = await fetch('/api/v1/synthesize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': activeKey
        },
        body: JSON.stringify({
          text: translateData.refined_sentence,
          voice_id: 'en-US-AriaNeural'
        })
      });

      if (!voiceRes.ok) {
        throw new Error('Speech synthesis query failed');
      }

      const voiceData = await voiceRes.json();

      setPlaygroundOutput({
        translation: translateData,
        voice: voiceData
      });
      fetchKeys(); // Refresh usage counter
      addToast('API Query completed successfully.', 'success');
    } catch (e: any) {
      addToast(e.message || 'API query failed. Check console.', 'error');
    } finally {
      setPlaygroundLoading(false);
    }
  };

  const playResponseAudio = (url: string) => {
    setIsPlayingAudio(true);
    const audio = new Audio(url);
    audio.play();
    audio.onended = () => setIsPlayingAudio(false);
    audio.onerror = () => {
      setIsPlayingAudio(false);
      addToast('Audio preview failed to load', 'error');
    };
  };

  const getCodeSnippet = () => {
    const exampleKey = keys.length > 0 ? keys[0].key : 'sk_live_your_api_key_here';
    const exampleGestures = JSON.stringify(playgroundGestures.split(',').map(g => g.trim().toUpperCase()));

    switch (activeTab) {
      case 'curl':
        return `# 1. Contextual Translation API
curl -X POST http://localhost:8001/api/v1/translate \\
  -H "X-API-Key: ${exampleKey}" \\
  -H "Content-Type: application/json" \\
  -d '{"gestures": ${exampleGestures}, "use_context_nlp": true}'

# 2. Neural Audio Synthesis API
curl -X POST http://localhost:8001/api/v1/synthesize \\
  -H "X-API-Key: ${exampleKey}" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello, my friend.", "voice_id": "en-US-AriaNeural"}'`;

      case 'js':
        return `// 1. Refine Gesture Array into Grammatical English
const refineRes = await fetch('http://localhost:8001/api/v1/translate', {
  method: 'POST',
  headers: {
    'X-API-Key': '${exampleKey}',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    gestures: ${exampleGestures},
    use_context_nlp: true
  })
});
const translation = await refineRes.json();
console.log(translation.refined_sentence); // Output: "Hello, my friend."

// 2. Synthesize to audio
const speechRes = await fetch('http://localhost:8001/api/v1/synthesize', {
  method: 'POST',
  headers: {
    'X-API-Key': '${exampleKey}',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: translation.refined_sentence,
    voice_id: 'en-US-AriaNeural'
  })
});
const audio = await speechRes.json();
console.log(audio.audio_url);`;

      case 'python':
        return `import requests

headers = {
    'X-API-Key': '${exampleKey}',
    'Content-Type': 'application/json'
}

# 1. Translate gesture array
translate_data = {
    'gestures': ${exampleGestures},
    'use_context_nlp': True
}
res = requests.post('http://localhost:8001/api/v1/translate', headers=headers, json=translate_data)
refined = res.json()
print("Refined Sentence:", refined['refined_sentence'])

# 2. Synthesize text into voice audio
speech_data = {
    'text': refined['refined_sentence'],
    'voice_id': 'en-US-AriaNeural'
}
audio_res = requests.post('http://localhost:8001/api/v1/synthesize', headers=headers, json=speech_data)
print("Audio Download URL:", audio_res.json()['audio_url'])`;
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 font-sans space-y-12">
      {/* Page Title Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between border-b border-outline-variant pb-6 gap-4">
        <div>
          <h1 className="text-3xl font-bold font-syne text-on-surface">Developer Console</h1>
          <p className="text-on-surface-variant text-sm mt-1.5">
            Manage authorization keys, inspect call logs, and integrate translation streams into developer clients.
          </p>
        </div>
        <div className="flex items-center gap-2.5">
          <span className="px-3.5 py-1.5 rounded-full border border-primary/20 bg-primary/5 text-primary text-xs font-mono font-bold tracking-widest uppercase">
            REST API v1 ACTIVE
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Side: Keys Management & Sandbox Test */}
        <div className="lg:col-span-7 space-y-8">
          
          {/* API Keys Card */}
          <div className="bg-surface border border-outline-variant rounded-2xl p-6 shadow-sm space-y-6">
            <div className="flex justify-between items-center border-b border-outline-variant pb-4">
              <div className="flex items-center gap-2 text-on-surface">
                <Key className="text-primary" size={20} />
                <h2 className="font-bold font-syne text-lg">Active Developer Keys</h2>
              </div>
            </div>

            {/* Create Key Form */}
            <form onSubmit={handleCreateKey} className="flex gap-3">
              <input
                type="text"
                value={newKeyName}
                onChange={(e) => setNewKeyName(e.target.value)}
                placeholder="Name (e.g. Healthcare iPad App)"
                className="flex-1 bg-surface-variant border border-outline-variant rounded-xl px-4 text-sm text-on-surface placeholder:text-on-surface-variant outline-none focus:border-primary transition-all"
              />
              <button
                type="submit"
                className="px-5 bg-primary text-white font-bold rounded-xl text-xs flex items-center gap-1.5 active:scale-95 transition-transform"
              >
                <Plus size={16} /> GENERATE KEY
              </button>
            </form>

            {/* Keys Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-outline-variant text-on-surface-variant uppercase font-mono tracking-wider">
                    <th className="py-3 font-semibold">Client Target</th>
                    <th className="py-3 font-semibold">API Secret Key</th>
                    <th className="py-3 font-semibold text-center">Calls</th>
                    <th className="py-3 font-semibold text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {keys.length > 0 ? (
                    keys.map((k) => (
                      <tr key={k.id} className="border-b border-outline-variant hover:bg-surface-variant/40 transition-colors">
                        <td className="py-4 font-bold text-on-surface">{k.name}</td>
                        <td className="py-4 font-mono text-on-surface-variant">
                          <div className="flex items-center gap-1.5">
                            <span>
                              {showKeyId === k.id ? k.key : `${k.key.substring(0, 12)}••••••••••••`}
                            </span>
                            <button
                              onClick={() => setShowKeyId(showKeyId === k.id ? null : k.id)}
                              className="text-on-surface-variant hover:text-on-surface"
                            >
                              {showKeyId === k.id ? <EyeOff size={13} /> : <Eye size={13} />}
                            </button>
                          </div>
                        </td>
                        <td className="py-4 text-center font-bold font-mono text-primary">{k.calls_count}</td>
                        <td className="py-4 text-right space-x-2">
                          <button
                            onClick={() => copyToClipboard(k.key)}
                            className="p-2 bg-surface-variant hover:bg-surface border border-outline-variant text-on-surface-variant hover:text-primary rounded-lg active:scale-95 transition-all"
                          >
                            <Copy size={13} />
                          </button>
                          <button
                            onClick={() => handleRevokeKey(k.id, k.name)}
                            className="p-2 bg-red-500/10 hover:bg-red-500 hover:text-white border border-red-500/20 text-red-500 rounded-lg active:scale-95 transition-all"
                          >
                            <Trash2 size={13} />
                          </button>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={4} className="py-8 text-center text-on-surface-variant italic">
                        No active developer tokens found. Enter a client target name above to generate one.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          {/* Sandbox Playground Card */}
          <div className="bg-surface border border-outline-variant rounded-2xl p-6 shadow-sm space-y-6">
            <div className="flex items-center gap-2 text-on-surface border-b border-outline-variant pb-4">
              <Terminal className="text-secondary" size={20} />
              <h2 className="font-bold font-syne text-lg">Interactive REST Sandbox</h2>
            </div>

            <div className="space-y-4">
              <div className="flex flex-col gap-1.5">
                <label className="text-xs font-mono uppercase tracking-wider text-on-surface-variant">Input Gesture Array</label>
                <div className="flex gap-3">
                  <input
                    type="text"
                    value={playgroundGestures}
                    onChange={(e) => setPlaygroundGestures(e.target.value)}
                    placeholder="HELLO, FRIEND, HELP"
                    className="flex-1 bg-surface-variant border border-outline-variant rounded-xl px-4 text-sm text-on-surface font-mono"
                  />
                  <button
                    onClick={handleTestAPI}
                    disabled={playgroundLoading}
                    className="px-6 bg-gradient-to-r from-primary to-secondary text-white font-bold rounded-xl text-xs flex items-center gap-1.5 hover:opacity-90 active:scale-95 disabled:opacity-50"
                  >
                    {playgroundLoading ? (
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    ) : (
                      <>
                        <Play size={14} /> QUERY API
                      </>
                    )}
                  </button>
                </div>
                <p className="text-[10px] text-on-surface-variant">Separate input gestures with commas (e.g. HELLO, SORRY, HELP, GOOD)</p>
              </div>

              {playgroundOutput && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-surface-variant border border-outline-variant rounded-xl p-5 space-y-4"
                >
                  <div className="flex justify-between items-center border-b border-outline-variant pb-2">
                    <span className="text-[10px] font-mono text-primary font-bold">API RESPONSE PAYLOAD</span>
                    <span className="text-[10px] font-mono text-on-surface-variant">STATUS 200 OK</span>
                  </div>

                  <div className="space-y-3.5">
                    <div>
                      <p className="text-[10px] text-on-surface-variant uppercase font-mono">Refined Intent Sentence:</p>
                      <p className="text-sm font-bold text-on-surface mt-1">"{playgroundOutput.translation.refined_sentence}"</p>
                    </div>

                    <div className="flex items-center justify-between border-t border-outline-variant pt-3 mt-3">
                      <div>
                        <p className="text-[10px] text-on-surface-variant uppercase font-mono">Synthesized Neural File:</p>
                        <p className="text-xs font-mono text-secondary mt-0.5">{playgroundOutput.voice.audio_url}</p>
                      </div>
                      <button
                        onClick={() => playResponseAudio(playgroundOutput.voice.audio_url)}
                        disabled={isPlayingAudio}
                        className="px-3 py-1.5 bg-primary text-white rounded-lg text-xs font-bold flex items-center gap-1 active:scale-95 transition-all disabled:opacity-70"
                      >
                        <Volume2 size={13} /> {isPlayingAudio ? 'PLAYING...' : 'PLAY'}
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </div>
          </div>
        </div>

        {/* Right Side: Interactive Docs */}
        <div className="lg:col-span-5 bg-surface border border-outline-variant rounded-2xl p-6 shadow-sm space-y-6">
          <div className="flex items-center justify-between border-b border-outline-variant pb-4">
            <div className="flex items-center gap-2 text-on-surface">
              <Code className="text-tertiary" size={20} />
              <h2 className="font-bold font-syne text-lg">API Documentation</h2>
            </div>
          </div>

          <div className="flex border-b border-outline-variant">
            {(['curl', 'js', 'python'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 pb-3 text-xs font-bold font-mono uppercase border-b-2 transition-colors cursor-pointer ${
                  activeTab === tab 
                    ? 'border-primary text-primary' 
                    : 'border-transparent text-on-surface-variant hover:text-on-surface'
                }`}
              >
                {tab === 'js' ? 'JavaScript' : tab}
              </button>
            ))}
          </div>

          <div className="relative">
            <pre className="bg-[#0b1326] text-slate-300 font-mono text-[11px] p-4.5 rounded-xl overflow-x-auto leading-relaxed border border-outline-variant select-all h-[340px] max-h-[340px]">
              <code>{getCodeSnippet()}</code>
            </pre>
            <button
              onClick={() => copyToClipboard(getCodeSnippet() || '')}
              className="absolute top-3 right-3 p-1.5 bg-white/5 border border-white/10 text-slate-400 hover:text-white rounded-md active:scale-95 transition-all text-xs"
              title="Copy Code"
            >
              <Copy size={13} />
            </button>
          </div>

          <div className="bg-surface-variant/40 border border-outline-variant rounded-xl p-4.5 space-y-3">
            <h4 className="text-xs font-bold text-on-surface flex items-center gap-1.5">
              <Shield size={14} className="text-primary" /> Request Headers
            </h4>
            <div className="text-[11px] text-on-surface-variant space-y-2">
              <p>Every REST API query requires passing the developer token in the header:</p>
              <div className="bg-surface-variant border border-outline-variant p-2 rounded font-mono text-on-surface">
                X-API-Key: sk_live_your_key_here
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
