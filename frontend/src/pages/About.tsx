import { motion } from 'framer-motion';
import { Shield, Sparkles, Accessibility, Heart, MessageSquare, BookOpen } from 'lucide-react';

const FEATURES = [
  { 
    icon: <Accessibility className="text-primary" size={24} />, 
    title: 'Two-Way Translation', 
    desc: 'Empowers smooth, real-time two-way dialogue between signers and speakers using advanced vision pipelines.' 
  },
  { 
    icon: <Sparkles className="text-secondary" size={24} />, 
    title: 'Natural Language Refinement', 
    desc: 'Refines isolated raw sign vocabulary tokens into grammatically rich, contextual English sentences.' 
  },
  { 
    icon: <Heart className="text-tertiary" size={24} />, 
    title: 'Expressive Neural Voice', 
    desc: 'Converts refined text outputs into warm, human-like voice synthesis using realistic speech profiles.' 
  }
] as const;

const GESTURES = [
  { category: 'Basic Conversation', items: ['school', 'sorry', 'help', 'easy', 'work', 'age', 'effort', 'respect'] },
  { category: 'Locations & Directions', items: ['near', 'home', 'village', 'washroom'] },
  { category: 'Social Interactions', items: ['friend', 'teacher', 'message', 'good'] },
  { category: 'Actions & Verbs', items: ['eating', 'drinking', 'pass', 'fail'] },
  { category: 'System Triggers', items: ['preset', 'dress'] }
] as const;

export default function About() {
  return (
    <div className="max-w-7xl mx-auto p-6 md:p-8 pt-10 font-sans space-y-12 text-left bg-background text-on-background">
      {/* Hero Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="space-y-4"
      >
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 text-primary text-xs font-semibold uppercase tracking-wider font-mono">
          <Shield size={12} /> Empowering Connection
        </div>
        <h1 className="text-4xl md:text-5xl font-syne font-bold leading-tight">
          About{' '}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary to-secondary">
            Sign2Speech
          </span>
        </h1>
        <p className="text-on-surface-variant text-base md:text-lg max-w-3xl leading-relaxed">
          Sign2Speech is a real-time assistive translation suite designed to bridge the communication gap for deaf and mute individuals. By combining real-time gesture recognition with contextual language engines, the platform translates sign language into speech and spoken voice back into sign demonstrations.
        </p>
      </motion.div>

      {/* Core Mission Cards */}
      <section className="space-y-6">
        <h2 className="text-2xl font-syne font-bold text-on-surface">Core Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {FEATURES.map((item, idx) => (
            <motion.div 
              key={item.title}
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="glass-card rounded-2xl p-6 flex flex-col items-start text-left space-y-4"
            >
              <div className="w-12 h-12 rounded-xl bg-surface-variant flex items-center justify-center border border-outline-variant">
                {item.icon}
              </div>
              <div>
                <h3 className="font-syne font-bold text-base text-on-surface">{item.title}</h3>
                <p className="text-on-surface-variant text-xs mt-1.5 leading-relaxed">{item.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Supported vocabulary */}
      <section className="space-y-6">
        <div className="flex items-center gap-2.5">
          <BookOpen className="text-primary" size={22} />
          <h2 className="text-2xl font-syne font-bold text-on-surface">Supported Vocabulary (23 Signs)</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {GESTURES.map((group) => (
            <div key={group.category} className="glass-card rounded-2xl p-5 space-y-3.5">
              <h3 className="font-syne font-bold text-xs uppercase tracking-wider text-secondary border-b border-outline-variant/60 pb-2">{group.category}</h3>
              <div className="flex flex-wrap gap-1.5">
                {group.items.map((g) => (
                  <span key={g} className="px-2.5 py-1 bg-surface-variant border border-outline-variant rounded-lg text-on-surface text-xs font-mono font-medium hover:border-primary transition-colors">
                    {g}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* How it works pipeline */}
      <section className="space-y-6">
        <div className="flex items-center gap-2.5">
          <MessageSquare className="text-primary" size={22} />
          <h2 className="text-2xl font-syne font-bold text-on-surface">How It Works</h2>
        </div>
        <div className="glass-card rounded-2xl p-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 relative">
            {[
              { step: '1', title: 'Capture Sign', desc: 'Signer performs gestures in front of the camera, instantly tracked frame-by-frame.' },
              { step: '2', title: 'Refine Context', desc: 'Disjointed vocabulary tokens are assembled and refined into correct English sentences.' },
              { step: '3', title: 'Synthesize Speech', desc: 'The polished sentence is read aloud in a warm, expressive voice profile.' },
            ].map((step) => (
              <div key={step.step} className="flex gap-4 items-start relative text-left">
                <div className="w-10 h-10 rounded-full bg-primary text-white flex items-center justify-center font-mono font-bold text-sm shrink-0 shadow-md shadow-primary/10">
                  {step.step}
                </div>
                <div className="space-y-1">
                  <h4 className="font-syne font-bold text-sm text-on-surface">{step.title}</h4>
                  <p className="text-on-surface-variant text-xs leading-relaxed">{step.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
