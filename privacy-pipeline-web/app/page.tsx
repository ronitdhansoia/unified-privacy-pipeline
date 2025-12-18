'use client';

import { useState, useEffect } from 'react';
import ControlPanel from '@/components/ControlPanel';
import ProgressDisplay from '@/components/ProgressDisplay';
import MetricsDisplay from '@/components/MetricsDisplay';
import LogsDisplay from '@/components/LogsDisplay';
import Navbar from '@/components/Navbar';

export default function Home() {
  const [status, setStatus] = useState('idle');
  const [progress, setProgress] = useState(0);
  const [metrics, setMetrics] = useState<any>({});
  const [logs, setLogs] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState('dashboard');

  // Poll for status updates
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/backend/status');
        const data = await res.json();
        setStatus(data.status);
        setProgress(data.progress);
        setMetrics(data.metrics);
        setLogs(data.logs || []);
      } catch (error) {
        console.error('Failed to fetch status:', error);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <main className="min-h-screen bg-black">
      {/* Background gradient effect */}
      <div className="fixed inset-0 gradient-bg pointer-events-none" />

      {/* Content */}
      <div className="relative z-10">
        <Navbar status={status} activeTab={activeTab} onTabChange={setActiveTab} />

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          {activeTab === 'dashboard' && (
            <>
              {/* Hero Section */}
              <div className="text-center mb-16 animate-fade-in">
                <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                  Train models with differential privacy, perform machine unlearning,
                  and evaluate privacy attacks — all in real-time.
                </p>
              </div>

              {/* Control Panel */}
              <ControlPanel />

              {/* Progress Display */}
              {status !== 'idle' && <ProgressDisplay status={status} progress={progress} />}

              {/* Logs Display */}
              <LogsDisplay logs={logs} />
            </>
          )}

          {activeTab === 'metrics' && (
            <>
              <div className="text-center mb-12">
                <h2 className="text-3xl font-bold mb-4">Performance Metrics</h2>
                <p className="text-gray-400">View detailed performance and privacy metrics</p>
              </div>
              {Object.keys(metrics).length > 0 ? (
                <MetricsDisplay metrics={metrics} />
              ) : (
                <div className="text-center py-20">
                  <p className="text-gray-500">No metrics available. Run a training session first.</p>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="relative z-10 border-t border-gray-900 mt-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex justify-between items-center">
            <p className="text-gray-500 text-sm">
              <span style={{ fontFamily: "'Helvetica Neue', Helvetica, Arial, sans-serif", fontWeight: 700 }}>PRVCYPPLN</span> © 2025 | GDPR & HIPAA Compliant
            </p>
            <div className="flex gap-6">
              <a href="#" className="text-gray-500 hover:text-white text-sm transition-colors">
                Documentation
              </a>
              <a href="#" className="text-gray-500 hover:text-white text-sm transition-colors">
                GitHub
              </a>
              <a href="#" className="text-gray-500 hover:text-white text-sm transition-colors">
                About
              </a>
            </div>
          </div>
        </div>
      </footer>
    </main>
  );
}
