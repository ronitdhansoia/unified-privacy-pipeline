'use client';

import { useEffect, useRef, useState } from 'react';

interface LogsDisplayProps {
  logs: any[];
}

export default function LogsDisplay({ logs }: LogsDisplayProps) {
  const logsEndRef = useRef<HTMLDivElement>(null);
  const logsContainerRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const isUserScrollingRef = useRef(false);

  useEffect(() => {
    // Only auto-scroll if user is at bottom
    if (!logsContainerRef.current) return;

    const container = logsContainerRef.current;
    const { scrollTop, scrollHeight, clientHeight } = container;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 100;

    // If user is at bottom or hasn't scrolled yet, scroll to bottom
    if (isAtBottom || (!isUserScrollingRef.current && logs.length > 0)) {
      container.scrollTop = container.scrollHeight;
    }
  }, [logs]);

  const handleScroll = () => {
    if (!logsContainerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = logsContainerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 100;

    // Mark that user has scrolled
    if (!isAtBottom) {
      isUserScrollingRef.current = true;
    } else {
      isUserScrollingRef.current = false;
    }

    setAutoScroll(isAtBottom);
  };

  const getLogIcon = (level: string) => {
    switch (level) {
      case 'success':
        return (
          <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
        );
      case 'error':
        return (
          <svg className="w-4 h-4 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        );
      case 'warning':
        return (
          <svg className="w-4 h-4 text-yellow-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        );
      default:
        return (
          <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        );
    }
  };

  const getLogColor = (level: string) => {
    switch (level) {
      case 'success':
        return 'text-green-400';
      case 'error':
        return 'text-red-400';
      case 'warning':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="mb-12">
      <div className="bg-card border border-gray-900 rounded-2xl p-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">System Logs</h3>
          <button
            onClick={() => window.location.reload()}
            className="text-xs text-gray-500 hover:text-white transition-colors"
          >
            Clear
          </button>
        </div>

        <div
          ref={logsContainerRef}
          onScroll={handleScroll}
          className="bg-black rounded-lg p-4 max-h-96 overflow-y-auto font-mono text-sm"
        >
          {logs.length === 0 ? (
            <div className="text-gray-500 text-center py-8">
              No logs yet. Start training to see activity.
            </div>
          ) : (
            <div className="space-y-1">
              {logs.map((log, index) => {
                // Check if this is a section header (contains only "=" or starts with spaces and "=")
                const isSectionHeader = log.message.trim().startsWith('=======');
                const isSectionTitle = index > 0 && logs[index - 1]?.message.trim().startsWith('=======');

                if (isSectionHeader) {
                  return null; // Skip separator lines, we'll style the titles instead
                }

                return (
                  <div
                    key={index}
                    className={`flex items-start gap-3 py-1 px-2 rounded transition-colors ${
                      isSectionTitle
                        ? 'bg-white/5 border-l-2 border-white mt-3 mb-1 font-semibold'
                        : 'hover:bg-gray-900/50'
                    }`}
                  >
                    {!isSectionTitle && (
                      <span className="text-gray-600 text-xs mt-0.5 min-w-[60px]">
                        {log.timestamp}
                      </span>
                    )}
                    {!isSectionTitle && (
                      <div className="mt-0.5">{getLogIcon(log.level)}</div>
                    )}
                    <span className={`flex-1 ${isSectionTitle ? 'text-white text-base' : getLogColor(log.level)}`}>
                      {log.message.trim()}
                    </span>
                  </div>
                );
              })}
              <div ref={logsEndRef} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
