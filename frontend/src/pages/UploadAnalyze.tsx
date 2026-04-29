import React, { useState } from 'react';
import { UploadCloud, Image as ImageIcon, CheckCircle, Volume2, X, Loader2, AlertCircle, RefreshCw } from 'lucide-react';

export default function UploadAnalyze() {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [processing, setProcessing] = useState(false);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile && droppedFile.type.startsWith('image/')) {
            handleFileSelection(droppedFile);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelection(e.target.files[0]);
        }
    };

    const handleFileSelection = (selectedFile: File) => {
        setFile(selectedFile);
        const objectUrl = URL.createObjectURL(selectedFile);
        setPreview(objectUrl);
        setResult(null);
        setError(null);
    };

    const handleClear = () => {
        setFile(null);
        setPreview(null);
        setResult(null);
        setError(null);
    };

    const handleUpload = async () => {
        if (!file) return;
        setProcessing(true);
        setError(null);

        // Create form data
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch('http://localhost:8000/api/upload', {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();

            if(data.error) {
                setError(data.error);
                return;
            }

            setResult({
                gestures: data.gestures || ["DETECTED_SIGN"],
                sentence: data.sentence || "Assumed sentence generated from the sign",
            });
        } catch (err) {
            console.error("Upload failed", err);
            setError("Failed to connect to the backend API. Please ensure the server is running.");
        } finally {
            setProcessing(false);
        }
    };

    const handleSpeak = () => {
        if (result && result.sentence) {
            const utterance = new SpeechSynthesisUtterance(result.sentence);
            window.speechSynthesis.speak(utterance);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-8 pt-12">
            <h2 className="text-4xl font-syne font-bold mb-2">Image Detection Engine</h2>
            <p className="text-gray-400 mb-8 font-sans">Upload a static sign language frame, and the underlying AI pipeline will infer the gesture and utter it natively.</p>

            <div
                className={`w-full aspect-video border-2 border-dashed ${file && !result ? 'border-[#00E5CC]' : 'border-[#6C3FC8]'} rounded-2xl bg-[#0A0A0F]/50 glass-panel flex flex-col items-center justify-center relative overflow-hidden transition-all hover:bg-white/5 ${result ? 'pointer-events-none opacity-50' : ''}`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
            >
                {!result && (
                    <input
                        type="file"
                        accept="image/*"
                        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                        onChange={handleFileChange}
                        disabled={processing}
                    />
                )}

                {preview ? (
                    <div className="absolute inset-0 w-full h-full p-4 flex items-center justify-center">
                        <img src={preview} alt="Upload preview" className="max-h-full max-w-full rounded-xl object-contain shadow-2xl" />
                    </div>
                ) : (
                    <div className="text-center pointer-events-none p-6">
                        <UploadCloud className="w-16 h-16 text-[#00E5CC] mx-auto mb-4" />
                        <h3 className="text-xl font-bold font-syne mb-2">Drag & Drop Image Here</h3>
                        <p className="text-sm text-gray-500 font-sans">Supported formats: JPG, PNG, WEBP</p>
                    </div>
                )}
            </div>

            {error && (
                <div className="mt-4 p-4 bg-red-500/10 border border-red-500/50 rounded-lg flex items-center gap-3 text-red-500 animate-in fade-in">
                    <AlertCircle size={20} />
                    <p>{error}</p>
                </div>
            )}

            <div className={`mt-6 flex justify-between items-center bg-[#0A0A0F] border border-gray-800 p-4 rounded-xl shadow-lg transition-all ${result ? 'hidden' : 'block'}`}>
                <div className="flex items-center gap-3">
                    <ImageIcon className="text-[#6C3FC8]" size={24} />
                    <div>
                        <p className="font-bold text-sm text-gray-200">{file ? file.name : "No file selected"}</p>
                        <p className="text-xs text-gray-500">{file ? (file.size / 1024).toFixed(1) + " KB" : "-"}</p>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    {file && !processing && (
                        <button 
                            onClick={handleClear} 
                            className="p-3 text-gray-400 hover:text-red-500 hover:bg-red-500/10 rounded-lg transition-colors"
                            title="Remove image"
                        >
                            <X size={20} />
                        </button>
                    )}
                    <button
                        onClick={handleUpload}
                        disabled={!file || processing}
                        className={`px-8 py-3 rounded-lg font-bold flex items-center gap-2 transition-all ${!file || processing ? 'bg-gray-800 text-gray-500 cursor-not-allowed' : 'bg-[#00E5CC] text-black shadow-[0_0_15px_rgba(0,229,204,0.3)] hover:bg-teal-400'}`}
                    >
                        {processing ? (
                            <span className="flex items-center gap-2"><Loader2 size={18} className="animate-spin"/> Processing...</span>
                        ) : (
                            <span className="flex items-center gap-2"><CheckCircle size={18} /> Process Image</span>
                        )}
                    </button>
                </div>
            </div>

            {result && (
                <div className="mt-8 animate-in slide-in-from-bottom-4 fade-in duration-500">
                    <div className="bg-[#6C3FC8]/10 border border-[#6C3FC8] rounded-xl p-8 shadow-[0_0_30px_rgba(108,63,200,0.15)] flex flex-col md:flex-row gap-8 items-center">

                        <div className="flex-1 text-center md:text-left">
                            <h3 className="text-[#00E5CC] font-mono text-sm uppercase tracking-widest mb-2 shadow-black">Detection Result</h3>
                            <h2 className="text-4xl font-syne font-bold mb-4">{result.gestures[0]}</h2>

                            <div className="bg-[#0A0A0F] border-l-4 border-[#00E5CC] p-4 text-gray-300 font-sans italic text-lg rounded-r-lg inline-block">
                                "{result.sentence}"
                            </div>
                        </div>

                        <div className="shrink-0 flex flex-col items-center gap-4">
                            <button onClick={handleSpeak} className="w-20 h-20 rounded-full bg-[#00E5CC] text-black flex items-center justify-center hover:scale-110 transition-transform shadow-[0_0_20px_rgba(0,229,204,0.3)] group relative overflow-hidden">
                                <div className="absolute inset-0 group-hover:bg-white/20 transition-colors" />
                                <Volume2 strokeWidth={2.5} size={32} fill="currentColor" />
                            </button>
                            
                            <button onClick={handleClear} className="mt-4 px-6 py-2 border border-blue-400/30 text-blue-400 rounded-lg hover:bg-blue-400/10 transition-colors flex items-center gap-2 text-sm font-bold w-full justify-center">
                                <RefreshCw size={16} /> Upload New
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
