import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { useGSAP } from '@gsap/react';

gsap.registerPlugin(ScrollTrigger);
export default function Landing() {
  const [mockSentence, setMockSentence] = useState('Awaiting sign input...');
  const [activeFaq, setActiveFaq] = useState<number | null>(null);
  const [simulatedSign, setSimulatedSign] = useState<string>('HELLO');
  const [simulatedSentence, setSimulatedSentence] = useState('Hello.');
  const containerRef = useRef<HTMLDivElement>(null);

  // Simulate live translation preview on the landing page for visual engagement
  useEffect(() => {
    const steps = [
      { signs: ['HELLO'], sentence: 'Hello.' },
      { signs: ['HELLO', 'FRIEND'], sentence: 'Hello, my friend.' },
      { signs: ['HELLO', 'FRIEND', 'HELP'], sentence: 'Hello, my friend, do you need help?' },
    ];
    let index = 0;
    const interval = setInterval(() => {
      index = (index + 1) % steps.length;
      setMockSentence(steps[index].sentence);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  // GSAP Kinetic scrolling, Parallax zooming, and Staggered reveals
  useGSAP(() => {
    // 1. Background image parallax scale zoom
    gsap.to(".parallax-bg", {
      scale: 1.25,
      ease: "none",
      scrollTrigger: {
        trigger: containerRef.current,
        start: "top top",
        end: "bottom bottom",
        scrub: true,
      }
    });

    // 2. Features Card Stagger Slide-up
    gsap.from(".feature-card", {
      y: 50,
      opacity: 0,
      duration: 0.8,
      stagger: 0.15,
      ease: "power2.out",
      scrollTrigger: {
        trigger: ".features-trigger",
        start: "top 80%",
        toggleActions: "play none none reverse",
      }
    });

    // 3. Statistics Stagger slide-up
    gsap.from(".stat-card", {
      y: 40,
      opacity: 0,
      duration: 0.8,
      stagger: 0.15,
      ease: "power2.out",
      scrollTrigger: {
        trigger: ".stats-trigger",
        start: "top 85%",
        toggleActions: "play none none reverse",
      }
    });

    // Tween the actual number values in statistics cards dynamically on scroll
    const counters = document.querySelectorAll(".stat-counter");
    counters.forEach((c) => {
      const targetVal = parseFloat(c.getAttribute("data-target") || "0");
      const isPercent = c.getAttribute("data-suffix") === "%";
      const obj = { val: 0 };
      gsap.to(obj, {
        val: targetVal,
        duration: 1.5,
        ease: "power2.out",
        scrollTrigger: {
          trigger: c,
          start: "top 90%",
        },
        onUpdate: () => {
          c.textContent = isPercent ? `${Math.round(obj.val)}%` : `${obj.val}`;
        }
      });
    });

    // 4. Vocabulary grid categories staggered pop-in
    gsap.from(".vocab-category-card", {
      scale: 0.9,
      opacity: 0,
      duration: 0.7,
      stagger: 0.2,
      ease: "back.out(1.2)",
      scrollTrigger: {
        trigger: ".vocab-trigger",
        start: "top 80%",
        toggleActions: "play none none reverse",
      }
    });
  }, { scope: containerRef });

  return (
    <div ref={containerRef} className="min-h-screen text-on-surface antialiased overflow-x-hidden selection:bg-primary/30 font-sans relative bg-black">
      {/* Zoomable Parallax Background Layer */}
      <div 
        className="parallax-bg absolute inset-0 bg-cover bg-center z-0 pointer-events-none opacity-40" 
        style={{ backgroundImage: "url('/male_signing.jpg')" }} 
      />
      {/* Dark frosted overlay to make text pop while keeping the background image beautifully visible */}
      <div className="absolute inset-0 bg-[#050505]/85 backdrop-blur-[4px] pointer-events-none z-0" />
      
      <main className="pt-2 pb-20 min-h-screen relative z-10">
        {/* Hero Section */}
        {/* Hero Section */}
        <section className="relative px-margin-mobile pt-2 md:pt-4 max-w-7xl mx-auto overflow-hidden z-10">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 lg:gap-16 xl:gap-20 items-center text-center lg:text-left">
            
            {/* Left Column: Text Content & Actions */}
            <div className="space-y-6">
              {/* Neural Badge */}
              <motion.div 
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/5 border border-white/10 backdrop-blur-xl mx-auto lg:mx-0"
              >
                <span className="w-1.5 h-1.5 rounded-full bg-primary shadow-[0_0_12px_rgba(184,200,223,0.8)]"></span>
                <span className="text-[11px] font-bold tracking-[0.2em] text-on-surface-variant uppercase font-mono">PSL Engine v1.0</span>
              </motion.div>

              {/* Headline */}
              <motion.h1 
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="font-display-lg text-3xl sm:text-4xl md:text-5xl lg:text-6xl max-w-xl mx-auto lg:mx-0 text-on-surface tracking-tight font-extrabold leading-tight font-syne"
              >
                Bridging the Silence.<br/>
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-secondary to-tertiary">Helping Pakistan's Deaf Communicate.</span>
              </motion.h1>

              {/* Subheadline */}
              <motion.p 
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="font-body-lg text-on-surface-variant max-w-xl mx-auto lg:mx-0 opacity-80 leading-relaxed text-xs md:text-sm"
              >
                A free accessibility tool translating Pakistan Sign Language (PSL) to spoken Urdu and English, 100% locally and privately.
              </motion.p>

              {/* Quick Metrics Counters Grid (Compact 2x2 inside left column) */}
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.25 }}
                className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-2 gap-4 border border-white/5 py-4 px-4 bg-black/45 rounded-2xl backdrop-blur-sm shadow-inner max-w-xl mx-auto lg:mx-0"
              >
                {[
                  { value: '23+', label: 'Active PSL Signs', desc: 'Real-time vocab classes supported' },
                  { value: '18ms', label: 'Local Neural Latency', desc: 'Instant frame-by-frame inference' },
                  { value: '100%', label: 'On-Device Execution', desc: 'Zero cloud dependencies, edge secure' },
                  { value: 'Dual', label: 'Translation Modes', desc: 'Two-way audio & visual interpreter' }
                ].map((c) => (
                  <div key={c.label} className="space-y-0.5 text-left border-l-2 border-primary/20 pl-3">
                    <p className="font-syne text-xl font-extrabold text-primary leading-none">{c.value}</p>
                    <p className="text-[9px] font-bold text-on-surface tracking-wider uppercase leading-tight">{c.label}</p>
                    <p className="text-[8px] text-on-surface-variant opacity-75 leading-tight">{c.desc}</p>
                  </div>
                ))}
              </motion.div>

              {/* Quick Specs / Telemetry Badges (inside left column) */}
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.28 }}
                className="flex flex-wrap items-center justify-center lg:justify-start gap-2 max-w-xl mx-auto lg:mx-0"
              >
                {[
                  { label: 'Latency', val: '< 18ms', icon: 'speed', color: 'text-emerald-400' },
                  { label: 'Inference', val: 'Local Neural', icon: 'neurology', color: 'text-cyan-400' },
                  { label: 'Vocabulary', val: '23 PSL signs', icon: 'gesture', color: 'text-amber-400' },
                  { label: 'Data Privacy', val: '100% Local', icon: 'lock', color: 'text-blue-400' }
                ].map((s) => (
                  <div key={s.label} className="apple-glass px-3 py-1.5 rounded-full flex items-center gap-1.5 border border-white/5 shadow-sm text-[9px] font-semibold font-mono text-on-surface-variant">
                    <span className={`material-symbols-outlined text-xs ${s.color}`}>{s.icon}</span>
                    <span className="uppercase tracking-wider">{s.label}:</span>
                    <span className="text-on-surface font-sans font-bold">{s.val}</span>
                  </div>
                ))}
              </motion.div>

              {/* Active Status Callout Banner */}
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 }}
                className="inline-flex items-center gap-2.5 px-4 py-2 rounded-xl bg-emerald-500/5 border border-emerald-500/10 shadow-sm mx-auto lg:mx-0"
              >
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
                </span>
                <span className="text-[9px] font-bold tracking-wider text-emerald-400 uppercase font-mono font-semibold">Open Accessibility Project &mdash; Free & Local</span>
              </motion.div>

              {/* CTAs */}
              <motion.div 
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.32 }}
                className="pt-2 space-y-4 mx-auto lg:mx-0"
              >
                <div className="flex flex-col sm:flex-row items-center gap-4">
                  <Link
                    to="/live"
                    className="w-full sm:w-auto bg-gradient-to-r from-primary via-secondary to-primary text-on-primary-fixed font-bold px-10 py-4 rounded-full shadow-[0_0_30px_rgba(184,200,223,0.25)] flex items-center justify-center gap-3 transition-all hover:scale-[1.03] active:scale-[0.97] no-underline cursor-pointer border-0"
                  >
                    Launch Free Interpreter
                    <span className="material-symbols-outlined text-xl">arrow_forward</span>
                  </Link>
                  <Link
                    to="/about"
                    className="w-full sm:w-auto apple-glass-dark text-on-surface border border-white/10 hover:border-white/20 font-bold px-10 py-4 rounded-full transition-all hover:bg-white/10 active:scale-[0.97] no-underline cursor-pointer flex items-center justify-center"
                  >
                    Read Architecture
                  </Link>
                </div>
                
                {/* Conversion Trust Badges */}
                <div className="flex flex-wrap items-center justify-center lg:justify-start gap-4 text-[9px] font-mono text-on-surface-variant opacity-80 pt-1">
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-[11px] text-emerald-400 font-bold">check_circle</span> 
                    Free & Open-Source
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-[11px] text-emerald-400 font-bold">check_circle</span> 
                    100% Local Privacy
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="material-symbols-outlined text-[11px] text-emerald-400 font-bold">check_circle</span> 
                    No Account Required
                  </span>
                </div>
              </motion.div>
            </div>

            {/* Right Column: Visual Mockup / Interface Simulator */}
            <motion.div 
              initial={{ opacity: 0, y: 35 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4, duration: 0.7 }}
              className="relative w-full z-10"
            >
              <div className="rounded-[2rem] p-2.5 apple-glass transition-transform duration-700 hover:scale-[1.01] max-w-2xl mx-auto lg:mx-0">
                <Link 
                  to="/live" 
                  className="relative rounded-[1.5rem] overflow-hidden aspect-video bg-surface-container-lowest border border-white/5 flex items-center justify-center bg-black group/preview block w-full h-full cursor-pointer"
                >
                  <img 
                    alt="User signing mockup" 
                    className="w-full h-full object-cover opacity-50 transition-opacity duration-300 group-hover/preview:opacity-40" 
                    src="/male_signing.jpg"
                  />
                  
                  {/* Large Pulsing Central Play Button */}
                  <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
                    <div className="w-16 h-16 rounded-full bg-gradient-to-r from-primary/80 to-secondary/80 text-on-primary-fixed flex items-center justify-center shadow-[0_0_40px_rgba(184,200,223,0.3)] group-hover/preview:scale-110 group-hover/preview:shadow-[0_0_55px_rgba(184,200,223,0.4)] transition-all duration-300">
                      <span className="material-symbols-outlined text-3xl font-extrabold ml-1">play_arrow</span>
                    </div>
                    <div className="apple-glass px-5 py-2 rounded-full border border-white/10 text-on-surface font-syne font-bold text-[10px] uppercase tracking-widest group-hover/preview:border-primary/30 transition-all">
                      Launch Free Interpreter
                    </div>
                  </div>

                  {/* Simulated Hand Tracking Overlays */}
                  <div className="absolute inset-0 pointer-events-none">
                    <div className="absolute top-[48%] left-[45%] w-1.5 h-1.5 bg-primary rounded-full shadow-[0_0_12px_rgba(184,200,223,0.8)] animate-pulse" />
                    <div className="absolute top-[42%] left-[52%] w-2 h-2 bg-secondary rounded-full shadow-[0_0_12px_rgba(200,198,200,0.8)]" />
                    <div className="absolute top-[55%] left-[40%] w-1.5 h-1.5 bg-primary rounded-full shadow-[0_0_12px_rgba(184,200,223,0.8)]" />
                    <div className="absolute top-[38%] left-[48%] w-1.5 h-1.5 bg-tertiary rounded-full shadow-[0_0_12px_rgba(173,198,255,0.8)]" />
                    <svg className="absolute inset-0 w-full h-full stroke-primary/30 stroke-[1.5]" fill="none">
                      <line x1="45%" y1="48%" x2="52%" y2="42%" />
                      <line x1="52%" y1="42%" x2="48%" y2="38%" />
                      <line x1="45%" y1="48%" x2="40%" y2="55%" />
                    </svg>
                  </div>

                  {/* Box tag overlay */}
                  <div className="absolute top-[32%] left-[32%] border border-primary bg-primary/10 backdrop-blur-md text-primary text-[9px] font-mono px-2 py-0.5 rounded shadow-lg uppercase tracking-wider">
                    PSL Active
                  </div>

                  {/* Overlay HUD Elements */}
                  <div className="absolute top-4 left-4">
                    <div className="flex gap-1.5 px-4 py-2 apple-glass-dark rounded-full border border-white/10">
                      <span className="w-2 h-2 rounded-full bg-red-500/80 shadow-[0_0_8px_rgba(239,68,68,0.5)] animate-pulse"></span>
                      <span className="w-2 h-2 rounded-full bg-yellow-500/80"></span>
                      <span className="w-2 h-2 rounded-full bg-green-500/80"></span>
                      <span className="ml-2 text-white/80 text-[9px] uppercase tracking-[0.15em] font-bold font-mono">PSL Pipeline</span>
                    </div>
                  </div>

                  {/* Real-time Dynamic Output HUD */}
                  <div className="absolute top-4 right-4 max-w-[200px] z-20">
                    <div className="flex flex-col gap-0.5 px-3.5 py-1.5 apple-glass rounded-xl border border-white/10 text-left bg-black/60 backdrop-blur-md">
                      <span className="text-[7px] font-mono text-primary uppercase tracking-widest block">Live Output</span>
                      <p className="text-[10px] font-bold text-on-surface font-syne truncate">
                        "{mockSentence}"
                      </p>
                    </div>
                  </div>
                  
                  <div className="absolute bottom-4 right-4">
                    <div className="flex items-center gap-2.5 px-4 py-2 apple-glass-dark rounded-2xl text-primary border border-primary/20">
                      <span className="w-2 h-2 rounded-full bg-primary animate-pulse shadow-[0_0_12px_rgba(184,200,223,0.8)]"></span>
                      <span className="font-bold text-[9px] uppercase tracking-widest font-mono">Streaming</span>
                    </div>
                  </div>
                </Link>
              </div>
            </motion.div>

          </div>
        </section>
        {/* Features Grid */}
        <section className="features-trigger px-margin-mobile md:px-margin-desktop max-w-7xl mx-auto mb-20 relative z-10">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            {/* Spatial Logic */}
            <div className="feature-card apple-glass-dark p-8 rounded-[2rem] group hover:scale-[1.02] transition-all duration-500 text-left">
              <div className="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center mb-8 border border-white/10 group-hover:bg-primary/10 group-hover:border-primary/20 transition-all">
                <span className="material-symbols-outlined text-primary text-3xl">gesture</span>
              </div>
              <h3 className="font-syne font-bold text-lg text-on-surface mb-3">Spatial Logic</h3>
              <p className="text-xs text-on-surface-variant opacity-70 leading-relaxed">Advanced hand-skeletal mapping that recognizes nuances in gesture depth and speed.</p>
            </div>

            {/* Edge Privacy */}
            <div className="feature-card relative overflow-hidden p-8 rounded-[2rem] bg-primary text-on-primary-fixed group hover:scale-[1.02] transition-all duration-500 shadow-[0_30px_60px_rgba(184,200,223,0.1)] text-left">
              <div className="absolute inset-0 bg-gradient-to-br from-white/30 to-transparent pointer-events-none" />
              <div className="w-14 h-14 rounded-2xl bg-white/20 flex items-center justify-center mb-8 border border-white/30 relative z-10">
                <span className="material-symbols-outlined text-on-primary-fixed text-3xl">security</span>
              </div>
              <h3 className="font-syne font-bold text-lg mb-3 relative z-10">Edge Privacy</h3>
              <p className="text-xs opacity-90 leading-relaxed font-medium relative z-10">All neural processing happens on your local machine. No video stream ever leaves your browser.</p>
            </div>

            {/* Natural Syntax */}
            <div className="feature-card apple-glass-dark p-8 rounded-[2rem] group hover:scale-[1.02] transition-all duration-500 text-left">
              <div className="w-14 h-14 rounded-2xl bg-white/5 flex items-center justify-center mb-8 border border-white/10 group-hover:bg-primary/10 group-hover:border-primary/20 transition-all">
                <span className="material-symbols-outlined text-primary text-3xl">text_fields</span>
              </div>
              <h3 className="font-syne font-bold text-lg text-on-surface mb-3">Natural Syntax</h3>
              <p className="text-xs text-on-surface-variant opacity-70 leading-relaxed">Converts ASL/BSL syntax structures into grammatically perfect spoken sentences.</p>
            </div>

          </div>
        </section>

        {/* Real-World Impact Statistics Section */}
        <section className="stats-trigger px-margin-mobile md:px-margin-desktop max-w-7xl mx-auto mb-20 relative z-10 border-t border-white/5 pt-20">
          <div className="flex flex-col md:flex-row justify-between items-start gap-8 mb-12 text-center md:text-left mx-auto md:mx-0">
            <div className="max-w-xl">
              
              <h2 className="text-3xl md:text-4xl font-syne font-bold leading-tight">Why Assistive AI Translation Matters</h2>
            </div>
            <p className="text-on-surface-variant text-xs md:text-sm leading-relaxed max-w-md">
              Communication barriers for Deaf and Hard of Hearing (DHH) individuals restrict autonomy in critical daily settings. Sign2Speech bridges this divide locally and privately.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
            {[
              { 
                stat: '62%', 
                label: 'Healthcare Barriers', 
                desc: 'Over 62% of Deaf individuals encounter severe communication gaps in medical settings, resulting in treatment delays or misunderstandings.',
                glow: 'border-red-500/10 hover:border-red-500/30 hover:shadow-[0_0_30px_rgba(239,68,68,0.15)]',
                color: 'text-red-400'
              },
              { 
                stat: '66%', 
                label: 'Employment Impact', 
                desc: '66% of DHH professionals report that structural communication barriers heavily impact career decisions and professional growth opportunities.',
                glow: 'border-amber-500/10 hover:border-amber-500/30 hover:shadow-[0_0_30px_rgba(245,158,11,0.15)]',
                color: 'text-amber-400'
              },
              { 
                stat: '24%', 
                label: 'Emergency Concerns', 
                desc: 'Nearly 24% of DHH respondents express concern that they would be unable to communicate effectively in sudden emergency situations.',
                glow: 'border-purple-500/10 hover:border-purple-500/30 hover:shadow-[0_0_30px_rgba(139,92,246,0.15)]',
                color: 'text-purple-400'
              }
            ].map((item, idx) => (
              <div key={idx} className={`stat-card apple-glass p-8 rounded-[2rem] border flex flex-col justify-between group transition-all duration-300 ${item.glow}`}>
                <div className={`font-syne text-5xl font-extrabold mb-6 group-hover:scale-105 transition-transform duration-300 ${item.color}`}>
                  <span className="stat-counter" data-target={item.stat.replace("%", "")} data-suffix={item.stat.includes("%") ? "%" : ""}>{item.stat}</span>
                </div>
                <div>
                  <h4 className="font-syne font-bold text-sm text-on-surface mb-2">{item.label}</h4>
                  <p className="text-[11px] text-on-surface-variant leading-relaxed opacity-85">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Supported Gesture Vocabulary Showcase */}
        <section className="vocab-trigger px-margin-mobile md:px-margin-desktop max-w-7xl mx-auto mb-20 relative z-10 border-t border-white/5 pt-20 text-left">
          <div className="max-w-xl mb-12 text-center md:text-left mx-auto md:mx-0">
            
            <h2 className="text-3xl md:text-4xl font-syne font-bold leading-tight">Supported Sign Vocabulary</h2>
            <p className="text-on-surface-variant text-xs md:text-sm leading-relaxed mt-2">
              Sign2Speech supports 23 active gesture vocabulary classifications locally on-device.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[
              { 
                category: 'Basic Conversation', 
                cardClass: 'vocab-category-card', 
                color: 'from-emerald-500/20 to-emerald-500/5 border-emerald-500/20 text-emerald-400 hover:shadow-[0_0_15px_rgba(16,185,129,0.1)]',
                items: ['school', 'sorry', 'help', 'easy', 'work', 'age', 'effort', 'respect'] 
              },
              { 
                category: 'Locations & Directions', 
                color: 'from-cyan-500/20 to-cyan-500/5 border-cyan-500/20 text-cyan-400 hover:shadow-[0_0_15px_rgba(6,182,212,0.1)]',
                items: ['near', 'home', 'village', 'washroom'] 
              },
              { 
                category: 'Social Interactions', 
                color: 'from-amber-500/20 to-amber-500/5 border-amber-500/20 text-amber-400 hover:shadow-[0_0_15px_rgba(245,158,11,0.1)]',
                items: ['friend', 'teacher', 'message', 'good'] 
              },
              { 
                category: 'Actions & Verbs', 
                color: 'from-orange-500/20 to-orange-500/5 border-orange-500/20 text-orange-400 hover:shadow-[0_0_15px_rgba(249,115,22,0.1)]',
                items: ['eating', 'drinking', 'pass', 'fail'] 
              },
              { 
                category: 'System Triggers', 
                color: 'from-purple-500/20 to-purple-500/5 border-purple-500/20 text-purple-400 hover:shadow-[0_0_15px_rgba(139,92,246,0.1)]',
                items: ['preset', 'dress'] 
              }
            ].map((group) => (
              <div key={group.category} className="apple-glass rounded-[2rem] p-6 space-y-4 border border-white/5 flex flex-col justify-between">
                <div>
                  <h3 className="font-syne font-bold text-xs uppercase tracking-wider text-on-surface border-b border-white/5 pb-3 flex items-center justify-between">
                    <span>{group.category}</span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-white/5 border border-white/10 text-on-surface-variant font-normal">
                      {group.items.length} signs
                    </span>
                  </h3>
                  <div className="flex flex-wrap gap-2 pt-4">
                    {group.items.map((g) => (
                      <span 
                        key={g} 
                        className={`px-3 py-1 bg-gradient-to-br ${group.color} border rounded-xl text-xs font-mono font-semibold transition-all hover:scale-105 active:scale-95 duration-200 cursor-pointer`}
                      >
                        {g}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Unified pipeline description */}
        <section className="py-20 px-6 max-w-7xl mx-auto text-center relative z-10 border-t border-white/5">
          <h2 className="text-3xl font-syne font-bold mb-4">Unified Translation Pipeline</h2>
          <p className="text-on-surface-variant max-w-xl mx-auto text-xs leading-relaxed mb-12 opacity-85">
            Sign2Speech aggregates camera feed capturing, contextual sentence refining, and voice synthesis into a single local loop.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 text-left">
            {[
              { num: '01', title: 'Video Stream', desc: 'Streams camera frames via optimized local WebSockets directly to the local model.' },
              { num: '02', title: 'Sentence Refinement', desc: 'Lightweight transformers structure raw sign sequences into natural grammatical intent.' },
              { num: '03', title: 'Acoustic Voice', desc: 'Synthesizes neural voice outputs instantly for responsive spoken feedback.' }
            ].map((p) => (
              <div key={p.num} className="apple-glass-dark p-6 rounded-2xl border border-white/5">
                <div className="w-10 h-10 bg-primary/15 text-primary rounded-xl flex items-center justify-center font-mono font-bold text-xs mb-4 border border-primary/20">{p.num}</div>
                <h4 className="font-bold text-sm mb-2">{p.title}</h4>
                <p className="text-xs text-on-surface-variant leading-relaxed opacity-75">{p.desc}</p>
              </div>
            ))}
          </div>
        </section>
        {/* Interactive Inference Sandbox */}
        <section className="px-margin-mobile md:px-margin-desktop max-w-7xl mx-auto mb-20 relative z-10 border-t border-white/5 pt-20 text-left">
          <div className="flex flex-col md:flex-row justify-between items-start gap-8 mb-12 text-center md:text-left mx-auto md:mx-0">
            <div className="max-w-xl">
              
              <h2 className="text-3xl md:text-4xl font-syne font-bold leading-tight">Inference Simulator</h2>
              <p className="text-on-surface-variant text-xs md:text-sm leading-relaxed mt-2">
                Click any vocabulary gesture below to simulate hand keypoint tracking and grammar translation in real-time.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 items-stretch">
            {/* Control card: List of selectable signs */}
            <div className="lg:col-span-1 apple-glass rounded-[2rem] p-6 border border-white/5 flex flex-col justify-between space-y-6">
              <div className="space-y-4">
                <span className="text-[10px] font-mono text-on-surface-variant uppercase tracking-widest block border-b border-white/5 pb-2">Select Gesture</span>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    { id: 'HELLO', text: 'HELLO', sentence: 'Hello.' },
                    { id: 'HELP', text: 'HELP', sentence: 'Do you need help?' },
                    { id: 'SCHOOL', text: 'SCHOOL', sentence: 'I am going to school.' },
                    { id: 'FRIEND', text: 'FRIEND', sentence: 'You are my friend.' },
                    { id: 'HOME', text: 'HOME', sentence: 'Let us go home.' },
                    { id: 'SORRY', text: 'SORRY', sentence: 'I am extremely sorry.' }
                  ].map((s) => (
                    <button
                      key={s.id}
                      onClick={() => {
                        setSimulatedSign(s.id);
                        setSimulatedSentence(s.sentence);
                      }}
                      className={`py-3 rounded-xl font-mono text-xs font-semibold tracking-wider transition-all border cursor-pointer active:scale-95 ${
                        simulatedSign === s.id
                          ? 'bg-primary border-primary text-on-primary-fixed shadow-md'
                          : 'bg-white/5 border-white/5 text-on-surface-variant hover:bg-white/10 hover:text-on-surface'
                      }`}
                    >
                      {s.text}
                    </button>
                  ))}
                </div>
              </div>
              <div className="p-4 bg-primary/5 rounded-2xl border border-primary/10 text-[11px] text-on-surface-variant leading-relaxed font-mono">
                <span className="font-bold text-primary">Simulation Active:</span> Keypoint extraction matrices are drawn programmatically.
              </div>
            </div>

            {/* Viewport card: Simulated skeletal hands tracking visualizer */}
            <div className="lg:col-span-2 apple-glass-dark rounded-[2.5rem] p-8 border border-white/5 flex flex-col justify-between relative overflow-hidden bg-black">
              {/* Subtle grid lines background */}
              <div className="absolute inset-0 opacity-[0.02] pointer-events-none" style={{ backgroundImage: 'linear-gradient(to right, white 1px, transparent 1px), linear-gradient(to bottom, white 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>
              
              <div className="flex justify-between items-start z-10 border-b border-white/5 pb-4">
                <div>
                  <span className="text-[9px] font-mono text-primary uppercase tracking-widest block">Camera Viewfinder (Simulated)</span>
                  <h4 className="font-syne font-bold text-lg text-on-surface uppercase mt-1">Telemetry Canvas</h4>
                </div>
                <div className="px-3.5 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/25 text-emerald-400 text-[10px] font-mono font-semibold uppercase tracking-wider animate-pulse">
                  Engine Ready
                </div>
              </div>

              {/* Skeletal Hands Canvas Simulation */}
              <div className="h-48 flex items-center justify-center relative my-4">
                {/* Simulated Hand Skeletal structure mapping */}
                <div className="relative w-72 h-40 border border-white/5 rounded-2xl bg-white/5 flex items-center justify-center overflow-hidden">
                  <div className="absolute top-[45%] left-[40%] w-2 h-2 bg-primary rounded-full shadow-[0_0_12px_rgba(184,200,223,0.8)] animate-pulse" />
                  <div className="absolute top-[38%] left-[48%] w-2.5 h-2.5 bg-secondary rounded-full shadow-[0_0_12px_rgba(200,198,200,0.8)]" />
                  <div className="absolute top-[52%] left-[35%] w-2 h-2 bg-primary rounded-full shadow-[0_0_12px_rgba(184,200,223,0.8)]" />
                  <div className="absolute top-[32%] left-[42%] w-2 h-2 bg-tertiary rounded-full shadow-[0_0_12px_rgba(173,198,255,0.8)] animate-bounce" />
                  
                  {/* Dynamic sign indicator details */}
                  <div className="absolute bottom-2 left-2 text-[9px] font-mono text-on-surface-variant uppercase">Sign Detected: {simulatedSign}</div>
                  
                  <svg className="absolute inset-0 w-full h-full stroke-primary/30 stroke-[1.5]" fill="none">
                    <line x1="40%" y1="45%" x2="48%" y2="38%" />
                    <line x1="48%" y1="38%" x2="42%" y2="32%" />
                    <line x1="40%" y1="45%" x2="35%" y2="52%" />
                  </svg>
                </div>
              </div>

              {/* Grammar Translation HUD Output */}
              <div className="glass-card rounded-2xl p-5 border border-white/10 text-left z-10 space-y-1 bg-[#0a0c0c]/85">
                <span className="text-[9px] font-mono text-primary uppercase tracking-widest block">Grammar Refined output</span>
                <p className="text-base font-bold text-on-surface font-syne leading-relaxed">
                  "{simulatedSentence}"
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Interactive FAQ Section */}
        <section className="px-margin-mobile md:px-margin-desktop max-w-7xl mx-auto mb-28 relative z-10 border-t border-white/5 pt-20 text-left">
          <div className="text-center max-w-xl mx-auto mb-12">
            <span className="text-primary font-mono text-xs uppercase tracking-widest block mb-2">Frequently Asked</span>
            <h2 className="text-3xl font-syne font-bold leading-tight">Got Questions?</h2>
          </div>

          <div className="space-y-4">
            {[
              {
                q: "Does Sign2Speech process my video on the cloud?",
                a: "No. Sign2Speech is designed as a local-first application. All video capturing and skeletal tracking processes run directly in your local browser environment. No video stream is ever uploaded or stored, ensuring total data privacy."
              },
              {
                q: "What sign languages are supported?",
                a: "The core model is trained on a 23-gesture subset of American Sign Language (ASL) and British Sign Language (BSL), including location directions, conversation greetings, action verbs, and system triggers."
              },
              {
                q: "How do I download and install the app on iOS?",
                a: "Simply open your deployment link in Safari on iOS, tap the Share icon, and select 'Add to Home Screen'. The app will install with its high-res icon and open in native fullscreen standalone mode."
              },
              {
                q: "Is a dedicated GPU required to run this engine?",
                a: "No. The neural networks and deep keypoint architectures are highly optimized and run efficiently on standard CPU architectures (including mobile phone processors), delivering smooth real-time performance."
              }
            ].map((faq, idx) => (
              <div key={idx} className="apple-glass rounded-[1.5rem] border border-white/5 overflow-hidden transition-all duration-300">
                <button
                  onClick={() => setActiveFaq(activeFaq === idx ? null : idx)}
                  className="w-full p-6 text-left flex justify-between items-center bg-transparent border-0 cursor-pointer active:bg-white/5 text-on-surface font-syne font-bold text-sm"
                >
                  <span>{faq.q}</span>
                  <span className={`material-symbols-outlined transition-transform duration-300 ${activeFaq === idx ? 'rotate-180 text-primary' : 'text-on-surface-variant'}`}>
                    expand_more
                  </span>
                </button>
                {activeFaq === idx && (
                  <div className="px-6 pb-6 text-xs text-on-surface-variant leading-relaxed animate-fade-in border-t border-white/5 pt-4">
                    {faq.a}
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-white/5 py-8 px-6 text-on-surface-variant/70 text-[11px] font-mono relative z-10 flex flex-col sm:flex-row items-center justify-between max-w-7xl mx-auto gap-4">
        <span>Sign2Speech v1.0.0 &mdash; Real-time AI Sign Language Translation</span>
        <div className="flex gap-6">
          <Link to="/developer" className="hover:text-primary transition-colors hover:underline no-underline">Developer Console</Link>
          <Link to="/about" className="hover:text-primary transition-colors hover:underline no-underline">About</Link>
        </div>
      </footer>
    </div>
  );
}
