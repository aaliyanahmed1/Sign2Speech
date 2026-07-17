import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Mail, Lock, Eye, EyeOff, User, ArrowRight } from 'lucide-react';
import { useAppStore } from '../store';
import { useToast } from '../components/Toast';

export default function Login() {
  const navigate = useNavigate();
  const { addToast } = useToast();
  const loginUser = useAppStore((state) => state.loginUser);

  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Validation errors
  const [errors, setErrors] = useState<{ email?: string; password?: string; name?: string }>({});

  const validateForm = () => {
    const nextErrors: typeof errors = {};
    if (!email) {
      nextErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(email)) {
      nextErrors.email = 'Invalid email address';
    }

    if (!password) {
      nextErrors.password = 'Password is required';
    } else if (password.length < 6) {
      nextErrors.password = 'Password must be at least 6 characters';
    }

    if (isSignUp && !name) {
      nextErrors.name = 'Name is required';
    }

    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validateForm()) return;

    setIsLoading(true);
    // Simulate API round-trip delay
    setTimeout(() => {
      setIsLoading(false);
      loginUser(email, isSignUp ? name : 'User');
      addToast(
        isSignUp ? 'Account created successfully! Welcome.' : 'Signed in successfully. Welcome back!',
        'success'
      );
      navigate('/live');
    }, 1200);
  };

  const handleDemoAccess = () => {
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      loginUser('demo@sign2speech.ai', 'Guest Practitioner');
      addToast('Logged in as Guest Practitioner via Quick Bypass.', 'info');
      navigate('/live');
    }, 800);
  };

  return (
    <div className="min-h-[90vh] w-full flex items-center justify-center px-4 relative overflow-hidden font-sans bg-background text-on-background">
      {/* Background Ambient Glows */}
      <div className="absolute top-1/4 left-1/4 w-[500px] h-[500px] bg-primary/10 rounded-full blur-[140px] pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-[400px] h-[400px] bg-tertiary/10 rounded-full blur-[120px] pointer-events-none" />
      
      {/* Mesh Grid Overlay */}
      <div 
        className="absolute inset-0 opacity-[0.02] pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(to right, var(--color-primary) 1px, transparent 1px),
            linear-gradient(to bottom, var(--color-primary) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px'
        }}
      />

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="w-full max-w-md relative z-10"
      >
        {/* Logo Banner */}
        <div className="flex flex-col items-center mb-8 text-center">
          <img src="/logo.jpg" alt="Sign2Speech Logo" className="w-16 h-16 rounded-2xl border border-outline-variant shadow-lg mb-3 object-cover animate-pulse" />
          <h2 className="text-3xl font-bold font-syne text-on-surface tracking-wide">
            Sign2Speech Portal
          </h2>
          <p className="text-sm text-on-surface-variant mt-1.5 font-mono">
            Verify credentials to access live translation feeds
          </p>
        </div>

        {/* Auth Card */}
        <div className="glass-card-dark rounded-3xl p-8 relative overflow-hidden">
          {/* Top subtle glow line */}
          <div className="absolute top-0 left-0 w-full h-[2px] bg-gradient-to-r from-transparent via-primary to-transparent" />
          
          <AnimatePresence mode="wait">
            <motion.form
              key={isSignUp ? 'signup' : 'login'}
              initial={{ opacity: 0, x: isSignUp ? 20 : -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: isSignUp ? -20 : 20 }}
              transition={{ duration: 0.25 }}
              onSubmit={handleSubmit}
              className="space-y-5"
            >
              {isSignUp && (
                <div className="space-y-1.5 text-left">
                  <label className="block text-xs font-mono text-on-surface-variant uppercase tracking-widest">
                    Your Name
                  </label>
                  <div className="relative">
                    <User className="absolute left-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant" size={18} />
                    <input
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Jane Doe"
                      className={`w-full h-12 bg-surface-variant border rounded-xl pl-11 pr-4 text-sm text-on-surface placeholder:text-on-surface-variant/60 outline-none transition-all ${
                        errors.name ? 'border-red-500/50 focus:border-red-500' : 'border-outline-variant focus:border-primary focus:ring-1 focus:ring-primary'
                      }`}
                    />
                  </div>
                  {errors.name && (
                    <p className="text-xs text-red-500 font-mono pl-1">{errors.name}</p>
                  )}
                </div>
              )}

              <div className="space-y-1.5 text-left">
                <label className="block text-xs font-mono text-on-surface-variant uppercase tracking-widest">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant" size={18} />
                  <input
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="user@signspeak.ai"
                    className={`w-full h-12 bg-surface-variant border rounded-xl pl-11 pr-4 text-sm text-on-surface placeholder:text-on-surface-variant/60 outline-none transition-all ${
                      errors.email ? 'border-red-500/50 focus:border-red-500' : 'border-outline-variant focus:border-primary focus:ring-1 focus:ring-primary'
                    }`}
                  />
                </div>
                {errors.email && (
                  <p className="text-xs text-red-500 font-mono pl-1">{errors.email}</p>
                )}
              </div>

              <div className="space-y-1.5 text-left">
                <label className="block text-xs font-mono text-on-surface-variant uppercase tracking-widest">
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant" size={18} />
                  <input
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="••••••••"
                    className={`w-full h-12 bg-surface-variant border rounded-xl pl-11 pr-11 text-sm text-on-surface placeholder:text-on-surface-variant/60 outline-none transition-all ${
                      errors.password ? 'border-red-500/50 focus:border-red-500' : 'border-outline-variant focus:border-primary focus:ring-1 focus:ring-primary'
                    }`}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3.5 top-1/2 -translate-y-1/2 text-on-surface-variant hover:text-on-surface transition-colors"
                  >
                    {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
                {errors.password && (
                  <p className="text-xs text-red-500 font-mono pl-1">{errors.password}</p>
                )}
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                disabled={isLoading}
                className="w-full h-12 bg-gradient-to-r from-primary to-secondary text-white font-bold rounded-xl flex items-center justify-center gap-2 hover:opacity-90 active:scale-[0.98] transition-all disabled:opacity-50 disabled:pointer-events-none mt-2 shadow-md shadow-primary/10 cursor-pointer"
              >
                {isLoading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                ) : (
                  <>
                    {isSignUp ? 'CREATE ACCOUNT' : 'SECURE SIGN IN'}
                    <ArrowRight size={16} />
                  </>
                )}
              </button>
            </motion.form>
          </AnimatePresence>

          {/* Toggle form link */}
          <div className="mt-6 text-center text-xs">
            <span className="text-on-surface-variant">
              {isSignUp ? 'Already have an account? ' : "Don't have an account? "}
            </span>
            <button
              onClick={() => {
                setIsSignUp(!isSignUp);
                setErrors({});
              }}
              className="text-primary font-semibold hover:underline bg-transparent border-0 cursor-pointer"
            >
              {isSignUp ? 'Sign In' : 'Register Now'}
            </button>
          </div>
        </div>

        {/* Quick Demo access bypass */}
        <div className="mt-5 text-center">
          <p className="text-xs text-on-surface-variant font-mono mb-2">DEVELOPER / TESTER QUICK BYPASS</p>
          <button
            onClick={handleDemoAccess}
            disabled={isLoading}
            className="px-6 py-2.5 bg-surface border border-dashed border-primary/30 hover:border-primary rounded-full text-xs font-semibold text-primary hover:bg-primary/5 transition-all active:scale-95 disabled:opacity-50 cursor-pointer"
          >
            Instant Guest Session Access
          </button>
        </div>
      </motion.div>
    </div>
  );
}
