'use client';

interface NavbarProps {
  status: string;
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export default function Navbar({ status, activeTab, onTabChange }: NavbarProps) {
  const getStatusColor = () => {
    switch (status) {
      case 'training':
      case 'unlearning':
      case 'evaluating':
        return 'bg-blue-500';
      case 'completed':
        return 'bg-green-500';
      case 'error':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  const getStatusText = () => {
    switch (status) {
      case 'training':
        return 'Training';
      case 'unlearning':
        return 'Unlearning';
      case 'evaluating':
        return 'Evaluating';
      case 'completed':
        return 'Completed';
      case 'error':
        return 'Error';
      default:
        return 'Ready';
    }
  };

  return (
    <nav className="border-b border-gray-900 backdrop-blur-sm bg-black/50 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <div className="flex items-center gap-8">
            <span className="text-2xl font-bold tracking-tight" style={{ fontFamily: "'Helvetica Neue', Helvetica, Arial, sans-serif", fontWeight: 700 }}>
              PRVCYPPLN
            </span>

            {/* Tabs */}
            <div className="flex gap-1">
              <button
                onClick={() => onTabChange('dashboard')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'dashboard'
                    ? 'bg-white text-black'
                    : 'text-gray-400 hover:text-white hover:bg-gray-900'
                }`}
              >
                Dashboard
              </button>
              <button
                onClick={() => onTabChange('metrics')}
                className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                  activeTab === 'metrics'
                    ? 'bg-white text-black'
                    : 'text-gray-400 hover:text-white hover:bg-gray-900'
                }`}
              >
                Metrics
              </button>
            </div>
          </div>

          {/* Status Indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-gray-900 rounded-full">
            <div className={`w-2 h-2 rounded-full ${getStatusColor()} animate-pulse`} />
            <span className="text-sm text-gray-300">{getStatusText()}</span>
          </div>
        </div>
      </div>
    </nav>
  );
}
