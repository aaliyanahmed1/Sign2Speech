import { Link } from 'react-router-dom';
import { Home, ArrowLeft } from 'lucide-react';

export default function NotFound() {
  return (
    <div className="min-h-[calc(100vh-64px)] flex items-center justify-center p-8">
      <div className="text-center">
        <div className="text-8xl font-syne font-bold text-transparent bg-clip-text bg-gradient-to-r from-[#00E5CC] to-[#6C3FC8] mb-4">
          404
        </div>
        <h1 className="text-2xl font-syne font-bold mb-2">Page Not Found</h1>
        <p className="text-gray-500 mb-8">The page you're looking for doesn't exist or has been moved.</p>
        <div className="flex items-center justify-center gap-3">
          <Link
            to="/"
            className="px-5 py-2.5 bg-[#00E5CC] text-black font-bold rounded-lg hover:bg-[#00E5CC]/80 transition-colors flex items-center gap-2 text-sm"
          >
            <Home size={16} /> Go Home
          </Link>
          <button
            onClick={() => window.history.back()}
            className="px-5 py-2.5 border border-gray-600 text-gray-300 rounded-lg hover:border-[#6C3FC8] hover:text-[#00E5CC] transition-colors flex items-center gap-2 text-sm"
          >
            <ArrowLeft size={16} /> Go Back
          </button>
        </div>
      </div>
    </div>
  );
}
