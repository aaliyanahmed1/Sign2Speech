import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate } from 'react-router-dom';
import { useState } from 'react';
import { Menu, X } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';
import { useAppStore } from './store';
import { useToast } from './components/Toast';
import Landing from './pages/Landing';
import UploadAnalyze from './pages/UploadAnalyze';
import About from './pages/About';
import NotFound from './pages/NotFound';
import Login from './pages/Login';
import SignSpeakTranslate from './pages/SignSpeakTranslate';
import SignSpeakResult from './pages/SignSpeakResult';
import SignSpeakHistory from './pages/SignSpeakHistory';
import SignSpeakSettings from './pages/SignSpeakSettings';
import Developer from './pages/Developer';

const NAV_LINKS = [
  { to: '/live', label: 'Live Demo' },
  { to: '/upload', label: 'Upload' },
  { to: '/about', label: 'About' },
] as const;


function NavBar() {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const isSignSpeak = location.pathname.startsWith('/signspeak') || location.pathname.startsWith('/live');
  const { addToast } = useToast();
  const { isAuthenticated, user, logoutUser } = useAppStore();

  const isActive = (path: string) => location.pathname === path;

  const navClass = isSignSpeak ? "hidden" : "block";
  const visibleLinks = isAuthenticated ? NAV_LINKS : [];

  return (
    <nav className={`border-b border-outline-variant bg-background/80 backdrop-blur-md sticky top-0 z-50 ${navClass}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center gap-2 text-primary font-bold text-xl tracking-wider uppercase font-syne no-underline">
            <img src="/logo.jpg" alt="Sign2Speech Logo" className="w-8 h-8 rounded-lg object-cover border border-outline-variant" />
            <span>Sign2Speech</span>
          </Link>

          <div className="hidden md:flex items-center space-x-2">
            {visibleLinks.map((link) => (
              <Link
                key={link.to}
                to={link.to}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(link.to)
                    ? 'text-primary bg-primary-container'
                    : 'text-on-surface-variant hover:text-primary hover:bg-surface-variant'
                }`}
              >
                {link.label}
              </Link>
            ))}

            {isAuthenticated ? (
              <div className="flex items-center gap-3 border-l border-outline-variant pl-4 ml-2">
                <span className="text-xs text-on-surface-variant font-mono">
                  {user?.name || 'User'}
                </span>
                <button
                  onClick={() => {
                    logoutUser();
                    addToast('Logged out successfully', 'info');
                  }}
                  className="px-3 py-1.5 border border-red-500/30 hover:border-red-500 bg-transparent text-red-600 text-xs font-semibold rounded-lg active:scale-95 transition-all cursor-pointer"
                >
                  Log Out
                </button>
              </div>
            ) : (
              <Link
                to="/login"
                className="px-5 py-2 ml-4 border border-white/10 hover:border-white/20 text-on-surface font-bold text-xs rounded-full hover:bg-white/5 active:scale-95 transition-all bg-transparent no-underline"
              >
                Sign In
              </Link>
            )}
          </div>

          {/* Mobile toggle */}
          <button
            className="md:hidden p-2 text-on-surface-variant hover:text-on-surface transition-colors"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label={mobileOpen ? 'Close menu' : 'Open menu'}
          >
            {mobileOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="md:hidden overflow-hidden border-t border-outline-variant bg-background/95 backdrop-blur-md"
          >
            <div className="px-4 py-3 space-y-1">
              {visibleLinks.map((link) => (
                <Link
                  key={link.to}
                  to={link.to}
                  onClick={() => setMobileOpen(false)}
                  className={`block px-3 py-2.5 rounded-md text-sm font-medium transition-colors ${
                    isActive(link.to)
                      ? 'text-primary bg-primary-container'
                      : 'text-on-surface-variant hover:text-primary hover:bg-surface-variant'
                  }`}
                >
                  {link.label}
                </Link>
              ))}

              <div className="border-t border-outline-variant pt-3 mt-3">
                {isAuthenticated ? (
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-on-surface-variant font-mono">
                      {user?.name || 'User'}
                    </span>
                    <button
                      onClick={() => {
                        setMobileOpen(false);
                        logoutUser();
                        addToast('Logged out successfully', 'info');
                      }}
                      className="px-3 py-1.5 border border-red-500/30 hover:border-red-500 bg-transparent text-red-600 text-xs font-semibold rounded-lg active:scale-95 transition-all cursor-pointer"
                    >
                      Log Out
                    </button>
                  </div>
                ) : (
                  <Link
                    to="/login"
                    onClick={() => setMobileOpen(false)}
                    className="block w-full py-2.5 text-center border border-white/10 hover:border-white/20 text-on-surface font-bold text-xs rounded-full hover:bg-white/5 active:scale-95 transition-all bg-transparent no-underline animate-fade-in"
                  >
                    Sign In
                  </Link>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </nav>
  );
}

function SignSpeakHeader() {
  const { user } = useAppStore();
  return (
    <header className="bg-surface/40 backdrop-blur-3xl fixed top-0 w-full z-50 border-b border-white/5">
      <div className="flex justify-between items-center px-margin-mobile h-16 w-full max-w-7xl mx-auto">
        <div className="flex items-center gap-4">
          <Link to="/" className="text-on-surface/85 flex items-center justify-center cursor-pointer active:scale-90 duration-200">
            <span className="material-symbols-outlined">menu</span>
          </Link>
          <Link to="/" className="font-syne font-extrabold text-xl tracking-tight text-on-surface no-underline">
            Sign2Speech
          </Link>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-on-surface-variant font-mono hidden sm:inline">{user?.name || 'Demo User'}</span>
          <div className="w-8 h-8 rounded-full overflow-hidden border border-white/10 cursor-pointer active:scale-90 duration-200">
            <img alt="Profile" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/a/default-user=s120-c"/>
          </div>
        </div>
      </div>
    </header>
  );
}

function SignSpeakBottomNav() {
  const location = useLocation();
  const currentPath = location.pathname;
  const prefix = currentPath.startsWith('/live') ? '/live' : '/signspeak';
  const isTranslateActive = currentPath === prefix || currentPath === `${prefix}/result`;
  const isHistoryActive = currentPath === `${prefix}/history`;
  const isSettingsActive = currentPath === `${prefix}/settings`;

  return (
    <div className="fixed bottom-8 left-0 right-0 flex justify-center z-50">
      <nav className="apple-glass-dark w-[90%] max-w-md rounded-full px-6 py-3 flex justify-around items-center shadow-[0_20px_50px_rgba(0,0,0,0.6)] border border-white/10">
        <Link 
          to={prefix} 
          className={`flex items-center justify-center p-3 rounded-full transition-all duration-300 active:scale-90 ${
            isTranslateActive ? 'bg-white text-black shadow-lg' : 'text-on-surface/65 hover:text-on-surface hover:bg-white/5'
          }`}
          title="Live Translation"
        >
          <span className="material-symbols-outlined" style={{ fontVariationSettings: isTranslateActive ? "'FILL' 1" : "'FILL' 0" }}>translate</span>
        </Link>
        <Link 
          to={`${prefix}/history`} 
          className={`flex items-center justify-center p-3 rounded-full transition-all duration-300 active:scale-90 ${
            isHistoryActive ? 'bg-white text-black shadow-lg' : 'text-on-surface/65 hover:text-on-surface hover:bg-white/5'
          }`}
          title="History Log"
        >
          <span className="material-symbols-outlined" style={{ fontVariationSettings: isHistoryActive ? "'FILL' 1" : "'FILL' 0" }}>history</span>
        </Link>
        <Link 
          to={`${prefix}/settings`} 
          className={`flex items-center justify-center p-3 rounded-full transition-all duration-300 active:scale-90 ${
            isSettingsActive ? 'bg-white text-black shadow-lg' : 'text-on-surface/65 hover:text-on-surface hover:bg-white/5'
          }`}
          title="Engine Settings"
        >
          <span className="material-symbols-outlined" style={{ fontVariationSettings: isSettingsActive ? "'FILL' 1" : "'FILL' 0" }}>settings</span>
        </Link>
        <Link 
          to="/" 
          className="flex items-center justify-center p-3 rounded-full text-on-surface/65 hover:text-on-surface hover:bg-white/5 transition-all duration-300 active:scale-90"
          title="Log Out"
        >
          <span className="material-symbols-outlined">logout</span>
        </Link>
      </nav>
    </div>
  );
}

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

function SignSpeakSidebar({ collapsed, onToggle }: SidebarProps) {
  const location = useLocation();
  const currentPath = location.pathname;
  const prefix = currentPath.startsWith('/live') ? '/live' : '/signspeak';
  const isTranslateActive = currentPath === prefix || currentPath === `${prefix}/result`;
  const isHistoryActive = currentPath === `${prefix}/history`;
  const isSettingsActive = currentPath === `${prefix}/settings`;
  const { user, logoutUser } = useAppStore();
  const { addToast } = useToast();

  const handleLogout = () => {
    logoutUser();
    addToast('Logged out successfully', 'info');
  };

  const navItems = [
    { to: prefix, label: 'Live Interpreter', icon: 'translate', active: isTranslateActive },
    { to: `${prefix}/history`, label: 'Translation History', icon: 'history', active: isHistoryActive },
    { to: `${prefix}/settings`, label: 'Engine Settings', icon: 'settings', active: isSettingsActive },
  ];

  return (
    <aside className={`hidden md:flex flex-col ${collapsed ? 'w-20' : 'w-68'} h-screen fixed left-0 top-0 bg-[#08080a]/80 backdrop-blur-2xl border-r border-white/5 z-45 p-4 py-6 justify-between select-none transition-all duration-300`}>
      <div className="space-y-8 relative">
        {/* Collapse Toggle Chevron */}
        <button
          onClick={onToggle}
          className="absolute -right-7 top-4 w-6.5 h-6.5 rounded-full bg-white text-black border border-outline-variant flex items-center justify-center cursor-pointer shadow-md hover:scale-105 active:scale-95 transition-all z-50"
          title={collapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          <span className="material-symbols-outlined text-sm font-bold">
            {collapsed ? "chevron_right" : "chevron_left"}
          </span>
        </button>

        {/* Brand Logo */}
        <Link to="/" className={`flex items-center gap-3 text-primary font-bold text-lg tracking-wider uppercase font-syne no-underline pl-2 mt-2 ${collapsed ? 'justify-center pl-0' : ''}`}>
          <img src="/logo.jpg" alt="Sign2Speech Logo" className="w-8 h-8 rounded-lg object-cover border border-white/10" />
          {!collapsed && <span className="transition-opacity duration-300">Sign2Speech</span>}
        </Link>

        {/* Links */}
        <nav className="space-y-2.5 pt-4">
          {navItems.map((item) => (
            <Link
              key={item.to}
              to={item.to}
              className={`flex items-center gap-4 ${collapsed ? 'justify-center px-0' : 'px-4'} py-3.5 rounded-2xl transition-all duration-300 no-underline ${
                item.active 
                  ? 'bg-white text-black font-semibold shadow-lg shadow-white/5' 
                  : 'text-on-surface/65 hover:text-on-surface hover:bg-white/5'
              }`}
              title={collapsed ? item.label : undefined}
            >
              <span className="material-symbols-outlined" style={{ fontVariationSettings: item.active ? "'FILL' 1" : "'FILL' 0" }}>{item.icon}</span>
              {!collapsed && <span className="text-sm font-medium transition-opacity duration-300">{item.label}</span>}
            </Link>
          ))}
        </nav>
      </div>

      {/* User profile & Log out block */}
      <div className="space-y-4 border-t border-white/5 pt-6 mb-2">
        <div className={`flex items-center gap-3 ${collapsed ? 'justify-center px-0' : 'px-2'}`}>
          <div className="w-9 h-9 rounded-full overflow-hidden border border-white/10 shrink-0">
            <img alt="Profile" className="w-full h-full object-cover" src="https://lh3.googleusercontent.com/a/default-user=s120-c"/>
          </div>
          {!collapsed && (
            <div className="flex flex-col text-left min-w-0 transition-opacity duration-300">
              <span className="text-xs text-on-surface font-semibold truncate">{user?.name || 'Demo User'}</span>
              <span className="text-[10px] text-on-surface-variant font-mono opacity-65">FYP Practitioner</span>
            </div>
          )}
        </div>
        
        <button
          onClick={handleLogout}
          className={`w-full flex items-center gap-4 ${collapsed ? 'justify-center px-0' : 'px-4'} py-3.5 rounded-2xl text-red-400 hover:text-red-300 hover:bg-red-500/5 transition-all duration-200 border-0 bg-transparent cursor-pointer text-left`}
          title={collapsed ? "Log Out Portal" : undefined}
        >
          <span className="material-symbols-outlined">logout</span>
          {!collapsed && <span className="text-sm font-medium transition-opacity duration-300">Log Out Portal</span>}
        </button>
      </div>
    </aside>
  );
}

function SignSpeakLayout() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="min-h-screen w-full bg-[#050505] text-on-background relative overflow-hidden flex flex-col md:flex-row font-sans select-none pb-28 md:pb-0">
      {/* Ambient background glows */}
      <div className="absolute top-0 left-1/4 w-[600px] h-[600px] bg-primary/5 rounded-full blur-[160px] pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-[500px] h-[500px] bg-secondary/5 rounded-full blur-[140px] pointer-events-none" />
      
      {/* Left Sidebar (Desktop only) */}
      <SignSpeakSidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} />

      {/* Mobile Top Header (Mobile only) */}
      <div className="md:hidden w-full">
        <SignSpeakHeader />
      </div>

      {/* Page Content Container */}
      <div className={`flex-grow pt-24 md:pt-8 ${sidebarCollapsed ? 'md:pl-24' : 'md:pl-72'} px-margin-mobile w-full overflow-x-hidden transition-all duration-300`}>
          <Routes>
            <Route index element={<SignSpeakTranslate />} />
            <Route path="result" element={<SignSpeakResult />} />
            <Route path="history" element={<SignSpeakHistory />} />
            <Route path="settings" element={<SignSpeakSettings />} />
          </Routes>
      </div>

      {/* Mobile Floating Bottom Nav (Mobile only) */}
      <div className="md:hidden">
        <SignSpeakBottomNav />
      </div>
    </div>
  );
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const isAuthenticated = useAppStore((state) => state.isAuthenticated);
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />;
}

function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/login" element={<Login />} />
      <Route
        path="/live/*"
        element={
          <ProtectedRoute>
            <SignSpeakLayout />
          </ProtectedRoute>
        }
      />
      <Route
        path="/upload"
        element={
          <ProtectedRoute>
            <UploadAnalyze />
          </ProtectedRoute>
        }
      />

      <Route
        path="/developer"
        element={
          <ProtectedRoute>
            <Developer />
          </ProtectedRoute>
        }
      />
      <Route path="/about" element={<About />} />
      <Route
        path="/signspeak/*"
        element={
          <ProtectedRoute>
            <SignSpeakLayout />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-background text-on-background font-sans selection:bg-primary selection:text-on-primary">
        <NavBar />
        <main className="min-h-[calc(100vh-64px)]">
          <AppRoutes />
        </main>
      </div>
    </Router>
  );
}
