import React from 'react';
import { motion } from 'framer-motion';
import { Camera, MessageSquareText, Volume2, Target, History } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Landing() {
    return (
        <div className="flex flex-col min-h-screen">
            {/* Hero Section */}
            <section className="relative flex flex-col items-center justify-center min-h-[85vh] px-4 text-center overflow-hidden">
                {/* Background glow overlay */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] bg-[#00E5CC]/10 rounded-full blur-[120px] pointer-events-none" />

                <motion.h1
                    className="text-5xl md:text-7xl lg:text-8xl font-bold font-syne mb-6 z-10"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    Bridging silence and speech — <br />
                    <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#00E5CC] to-[#6C3FC8]">
                        in real time
                    </span>
                </motion.h1>

                <motion.p
                    className="text-gray-400 text-lg md:text-2xl mb-12 max-w-2xl z-10"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.1 }}
                >
                    Sign2Speech is a Python-powered AI pipeline that converts sign language from real-time video into natural spoken output.
                </motion.p>

                <motion.div
                    className="flex flex-wrap items-center justify-center gap-6 z-10"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.2 }}
                >
                    <Link to="/live" className="group relative px-8 py-4 bg-[#0A0A0F] rounded-lg font-bold transition-all hover:scale-105">
                        <div className="absolute inset-0 bg-gradient-to-r from-[#00E5CC] to-[#6C3FC8] rounded-lg animate-pulse opacity-75 blur-md transition-opacity group-hover:opacity-100" />
                        <div className="absolute inset-[1px] bg-[#0A0A0F] rounded-lg" />
                        <span className="relative z-10 flex items-center gap-2 text-[#00E5CC]">
                            <Camera size={20} /> Try Live Demo
                        </span>
                    </Link>
                    <Link to="/upload" className="px-8 py-4 bg-transparent border border-gray-600 text-white rounded-lg hover:border-[#6C3FC8] hover:text-[#00E5CC] transition-all hover:scale-105 shadow-xl glass-panel">
                        Upload an Image
                    </Link>
                </motion.div>
            </section>

            {/* 3-Column Feature Strip */}
            <section className="py-20 px-8 bg-[#0A0A0F]/80 backdrop-blur-sm border-t border-b border-gray-900">
                <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
                    <FeatureCard
                        icon={<Target className="text-[#00E5CC]" size={32} />}
                        title="YOLO12 Detection"
                        desc="Lightning-fast and highly accurate 22-class gesture recognition trained directly on diverse datasets."
                    />
                    <FeatureCard
                        icon={<History className="text-[#6C3FC8]" size={32} />}
                        title="DeepSORT Tracking"
                        desc="Continuous temporal hand-tracking allows assembling fluid gestures instead of static disjoint signs."
                    />
                    <FeatureCard
                        icon={<Volume2 className="text-[#00E5CC]" size={32} />}
                        title="Natural Speech Output"
                        desc="Smart grammar reconstruction with Ollama generates natural sentences before pyttsx3 speaks them aloud."
                    />
                </div>
            </section>

            {/* How It Works Flow */}
            <section className="py-24 px-8 max-w-7xl mx-auto text-center">
                <h2 className="text-4xl font-syne font-bold mb-16">How It Works</h2>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-8 relative items-start">
                    <div className="hidden md:block absolute top-[28%] left-[12%] right-[12%] h-[2px] bg-gradient-to-r from-[#00E5CC]/20 via-[#6C3FC8] to-[#00E5CC]/20 -z-10" />

                    <StepCard num={1} icon={<Camera size={24} />} label="Sign Input" />
                    <StepCard num={2} icon={<Target size={24} />} label="Gesture Detection" />
                    <StepCard num={3} icon={<MessageSquareText size={24} />} label="Text Generation" />
                    <StepCard num={4} icon={<Volume2 size={24} />} label="Speech Output" />
                </div>
            </section>
        </div>
    );
}

function FeatureCard({ icon, title, desc }: { icon: React.ReactNode, title: string, desc: string }) {
    return (
        <div className="p-8 glass-panel rounded-xl glowing-box transition-transform hover:-translate-y-2">
            <div className="w-14 h-14 bg-gray-900 rounded-lg flex items-center justify-center mb-6 border border-gray-800">
                {icon}
            </div>
            <h3 className="text-xl font-bold font-syne mb-3">{title}</h3>
            <p className="text-gray-400 leading-relaxed font-sans">{desc}</p>
        </div>
    );
}

function StepCard({ num, icon, label }: { num: number, icon: React.ReactNode, label: string }) {
    return (
        <div className="flex flex-col items-center">
            <div className="w-16 h-16 rounded-full bg-[#0A0A0F] border-2 border-[#6C3FC8] shadow-[0_0_20px_rgba(108,63,200,0.5)] flex items-center justify-center text-[#6C3FC8] text-xl font-bold mb-6 mx-auto hover:bg-[#6C3FC8] hover:text-white transition-colors duration-300">
                {icon}
            </div>
            <h4 className="text-lg font-bold font-syne text-[#ECECEC]">{label}</h4>
            <span className="text-[#00E5CC] text-sm font-mono mt-2 block">Step 0{num}</span>
        </div>
    );
}
