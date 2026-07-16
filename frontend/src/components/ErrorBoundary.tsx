import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-background flex items-center justify-center p-8">
          <div className="glass-panel rounded-2xl p-10 max-w-lg text-center">
            <AlertTriangle className="w-16 h-16 text-red-500 mx-auto mb-6" />
            <h1 className="text-2xl font-syne font-bold mb-4">
              Something went wrong
            </h1>
            <p className="text-gray-400 mb-2">
              An unexpected error occurred while rendering the application.
            </p>
            {this.state.error && (
              <pre className="text-xs text-red-400 bg-black/40 rounded-lg p-4 mt-4 text-left overflow-x-auto font-mono">
                {this.state.error.message}
              </pre>
            )}
            <button
              onClick={this.handleReset}
              className="mt-8 px-6 py-3 bg-[#00E5CC] text-black font-bold rounded-lg hover:bg-[#00E5CC]/80 transition-colors inline-flex items-center gap-2"
            >
              <RefreshCw size={18} />
              Try Again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
