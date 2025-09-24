interface StatusIndicatorProps {
  isConnected: boolean;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

export const StatusIndicator = ({ 
  isConnected, 
  size = 'md', 
  showLabel = false 
}: StatusIndicatorProps) => {
  const sizeClasses = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4'
  };

  return (
    <div className="flex items-center gap-2">
      <div 
        className={`status-indicator ${sizeClasses[size]} ${
          isConnected ? 'status-connected' : 'status-disconnected'
        }`}
      />
      {showLabel && (
        <span className={`text-sm ${isConnected ? 'text-success' : 'text-danger'}`}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      )}
    </div>
  );
};