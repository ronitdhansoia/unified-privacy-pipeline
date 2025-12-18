'use client';

interface MetricsDisplayProps {
  metrics: any;
}

export default function MetricsDisplay({ metrics }: MetricsDisplayProps) {
  const formatValue = (key: string, value: number) => {
    if (key.includes('accuracy') || key.includes('quality') || key.includes('protection')) {
      return `${(value * 100).toFixed(1)}%`;
    }
    return value.toFixed(4);
  };

  const getMetricIcon = (key: string) => {
    if (key.includes('accuracy')) {
      return (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <polyline points="20 6 9 17 4 12" />
        </svg>
      );
    }
    if (key.includes('forget')) {
      return (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <circle cx="12" cy="12" r="10" />
          <line x1="15" y1="9" x2="9" y2="15" />
          <line x1="9" y1="9" x2="15" y2="15" />
        </svg>
      );
    }
    if (key.includes('privacy') || key.includes('protection')) {
      return (
        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
          <path d="M7 11V7a5 5 0 0 1 10 0v4" />
        </svg>
      );
    }
    return (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    );
  };

  const getMetricColor = (key: string) => {
    // Black and white theme - using white/gray gradients only
    return 'from-white to-gray-200';
  };

  const formatKey = (key: string) => {
    return key
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  return (
    <div className="mb-12 animate-slide-up">
      <h3 className="text-lg font-semibold mb-4">Metrics</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(metrics).map(([key, value]) => (
          <div
            key={key}
            className="bg-card border border-gray-900 rounded-2xl p-6 hover:bg-card-hover transition-all group"
          >
            <div className="flex items-start justify-between mb-4">
              <div className={`p-3 rounded-xl bg-gradient-to-br ${getMetricColor(key)}`}>
                <div className="text-black">{getMetricIcon(key)}</div>
              </div>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-gray-400">{formatKey(key)}</p>
              <p className="text-2xl font-bold">
                {typeof value === 'number' ? formatValue(key, value) : value}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
