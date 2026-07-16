import { useState, useRef, useEffect } from 'react';
import {
  UploadCloud,
  Image as ImageIcon,
  CheckCircle,
  Volume2,
  X,
  Loader2,
  AlertCircle,
  RefreshCw,
} from 'lucide-react';
import { apiClient } from '../lib/api';
import { useToast } from '../components/Toast';
import type { UploadResponse } from '../types/api';

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10 MB

export default function UploadAnalyze() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const previewUrlRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const { addToast } = useToast();

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  const handleFileSelection = (selectedFile: File) => {
    if (selectedFile.size > MAX_FILE_SIZE) {
      setError('File exceeds 10 MB limit.');
      return;
    }
    if (!selectedFile.type.startsWith('image/')) {
      setError('Only image files are supported.');
      return;
    }
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const url = URL.createObjectURL(selectedFile);
    previewUrlRef.current = url;
    setFile(selectedFile);
    setPreview(url);
    setResult(null);
    setError(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files[0];
    if (dropped) handleFileSelection(dropped);
  };

  const handleClear = () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setProcessing(true);
    setError(null);
    abortRef.current = new AbortController();

    try {
      const data: UploadResponse = await apiClient.uploadImage(file, abortRef.current.signal);
      if (data.error) {
        setError(data.error);
        return;
      }
      setResult(data);
      addToast('Image analyzed successfully', 'success');
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        setError('Failed to connect to the backend. Ensure the server is running.');
      }
    } finally {
      setProcessing(false);
    }
  };

  const handleSpeak = async () => {
    if (!result?.sentence) return;
    try {
      const res = await fetch('/api/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence: result.sentence })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.audio_url) {
          const audio = new Audio(data.audio_url);
          audio.play();
          addToast('Synthesized neural speech output', 'success');
          return;
        }
      }
    } catch (e) {
      console.error(e);
    }
    // Offline local fallback
    const utterance = new SpeechSynthesisUtterance(result.sentence);
    window.speechSynthesis.speak(utterance);
    addToast('Playing local voice offline fallback', 'info');
  };

    return (
    <div className="max-w-7xl mx-auto p-6 md:p-8 pt-10 font-sans space-y-8 px-margin-mobile">
      <div className="text-left">
        <h2 className="text-3xl md:text-4xl font-syne font-bold text-on-surface">Image Detection Engine</h2>
        <p className="text-on-surface-variant mt-1.5 text-sm leading-relaxed">
          Upload a static sign language frame. The AI pipeline will infer the gesture, refine the context, and synthesize speech.
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`w-full aspect-video border-2 border-dashed ${
          result ? 'border-white/5 opacity-50 pointer-events-none' : file ? 'border-primary' : 'border-white/10'
        } rounded-[2rem] apple-glass-dark flex flex-col items-center justify-center relative overflow-hidden transition-all hover:bg-white/5 shadow-inner`}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        {!result && (
          <input
            type="file"
            accept="image/*"
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
            onChange={(e) => e.target.files?.[0] && handleFileSelection(e.target.files[0])}
            disabled={processing}
          />
        )}

        {preview ? (
          <div className="absolute inset-0 p-4 flex items-center justify-center">
            <img src={preview} alt="Upload preview" className="max-h-full max-w-full rounded-2xl object-contain shadow-lg border border-white/5" />
          </div>
        ) : (
          <div className="text-center pointer-events-none p-6 space-y-4">
            <div className="w-16 h-16 rounded-[1.25rem] apple-glass flex items-center justify-center mx-auto border border-white/10">
              <UploadCloud className="w-8 h-8 text-primary" />
            </div>
            <div>
              <h3 className="text-lg font-bold font-syne text-on-surface">Drag & Drop Image Here</h3>
              <p className="text-xs text-on-surface-variant mt-1">JPG, PNG, WEBP &middot; Max 10 MB</p>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/40 rounded-xl flex items-center gap-2 text-red-600 text-sm animate-fade-in">
          <AlertCircle size={16} className="shrink-0" />
          <p>{error}</p>
        </div>
      )}

      {/* File info bar */}
      {!result && (
        <div className="flex justify-between items-center apple-glass-dark border border-white/5 p-4 rounded-[1.5rem] shadow-md">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-10 h-10 rounded-lg bg-secondary/15 flex items-center justify-center shrink-0 border border-secondary/20">
              <ImageIcon className="text-secondary" size={18} />
            </div>
            <div className="min-w-0 text-left">
              <p className="font-bold text-sm text-on-surface truncate">{file ? file.name : 'No file selected'}</p>
              <p className="text-xs text-on-surface-variant mt-0.5">{file ? `${(file.size / 1024).toFixed(1)} KB` : '-'}</p>
            </div>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {file && !processing && (
              <button
                onClick={handleClear}
                className="p-2 text-on-surface-variant hover:text-red-500 hover:bg-red-500/10 rounded-lg transition-colors cursor-pointer"
                title="Remove image"
              >
                <X size={18} />
              </button>
            )}
            <button
              onClick={handleUpload}
              disabled={!file || processing}
              className={`px-6 py-3 rounded-xl font-bold text-xs flex items-center gap-1.5 transition-all cursor-pointer border-0 ${
                !file || processing
                  ? 'bg-white/5 text-on-surface-variant/40 border border-white/5 cursor-not-allowed'
                  : 'bg-white text-black hover:brightness-110 active:scale-95 shadow-lg'
              }`}
            >
              {processing ? (
                <>
                  <Loader2 size={14} className="animate-spin" /> Processing...
                </>
              ) : (
                <>
                  <CheckCircle size={14} /> Process Image
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="animate-in slide-in-from-bottom-4 fade-in duration-500">
          <div className="apple-glass border border-white/10 rounded-[2rem] p-8 shadow-lg flex flex-col md:flex-row gap-8 items-center">
            <div className="flex-1 text-center md:text-left space-y-4">
              <div>
                <span className="text-primary font-mono text-xs uppercase tracking-widest block mb-1">
                  Detection Result
                </span>
                <h2 className="text-3xl md:text-4xl font-syne font-bold text-on-surface">
                  {result.gestures[0] || 'None'}
                </h2>
              </div>
              
              {result.gestures.length > 1 && (
                <div className="flex flex-wrap gap-1.5 justify-center md:justify-start">
                  {result.gestures.map((g) => (
                    <span key={g} className="px-2 py-0.5 bg-primary/10 border border-primary/30 rounded text-primary text-xs font-mono">
                      {g}
                    </span>
                  ))}
                </div>
              )}
              
              <div className="apple-glass-dark border border-white/5 border-l-4 border-l-primary p-4 text-on-surface text-sm italic rounded-r-2xl inline-block text-left shadow-inner">
                &ldquo;{result.sentence}&rdquo;
              </div>
            </div>

            <div className="shrink-0 flex flex-col items-center gap-3">
              <button
                onClick={handleSpeak}
                className="w-16 h-16 rounded-full bg-white text-black flex items-center justify-center hover:scale-105 active:scale-95 transition-all shadow-lg cursor-pointer border-0"
                aria-label="Speak result"
              >
                <Volume2 strokeWidth={2} size={26} fill="currentColor" />
              </button>
              <button
                onClick={handleClear}
                className="px-5 py-2.5 border border-white/10 hover:border-white/20 text-on-surface rounded-xl hover:bg-white/5 transition-all flex items-center gap-1.5 text-xs font-bold cursor-pointer active:scale-95 bg-transparent"
              >
                <RefreshCw size={12} /> Upload New
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}