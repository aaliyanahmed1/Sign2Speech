import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Landing from './pages/Landing';
import LiveDetection from './pages/LiveDetection';
import UploadAnalyze from './pages/UploadAnalyze';
import Analytics from './pages/Analytics';
import About from './pages/About';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-[#0A0A0F] text-gray-200 font-sans selection:bg-[#00E5CC] selection:text-black">
        <nav className="border-b border-gray-800 bg-[#0A0A0F]/80 backdrop-blur-md sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center">
                <Link to="/" className="text-[#00E5CC] font-bold text-xl tracking-wider uppercase font-syne">
                  Sign2Speech
                </Link>
                <div className="hidden md:block ml-10">
                  <div className="flex items-baseline space-x-4">
                    <Link to="/live" className="hover:text-[#00E5CC] px-3 py-2 rounded-md transition-colors">Live Demo</Link>
                    <Link to="/upload" className="hover:text-[#00E5CC] px-3 py-2 rounded-md transition-colors">Upload</Link>
                    <Link to="/analytics" className="hover:text-[#00E5CC] px-3 py-2 rounded-md transition-colors">Analytics</Link>
                    <Link to="/about" className="hover:text-[#00E5CC] px-3 py-2 rounded-md transition-colors">About</Link>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </nav>

        <main className="min-h-[calc(100vh-64px)]">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/live" element={<LiveDetection />} />
            <Route path="/upload" element={<UploadAnalyze />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
