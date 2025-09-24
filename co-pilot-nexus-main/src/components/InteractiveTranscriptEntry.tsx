import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { toast } from 'sonner';
import { 
  MessageSquare, 
  Mic, 
  FileText, 
  Settings,
  Clock,
  Flag,
  CheckSquare,
  MoreHorizontal
} from 'lucide-react';

interface TranscriptEntry {
  id: string;
  speaker: string;
  message: string;
  timestamp: Date;
  type: 'chat' | 'voice' | 'file' | 'system';
  avatar?: string;
  highlighted?: 'action' | 'decision' | null;
}

interface InteractiveTranscriptEntryProps {
  entry: TranscriptEntry;
  onHighlight: (id: string, type: 'action' | 'decision' | null) => void;
}

const entryIcons = {
  chat: MessageSquare,
  voice: Mic,
  file: FileText,
  system: Settings
};

const entryColors = {
  chat: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  voice: 'bg-green-500/10 text-green-400 border-green-500/20',
  file: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  system: 'bg-gray-500/10 text-gray-400 border-gray-500/20'
};

export const InteractiveTranscriptEntry = ({ 
  entry, 
  onHighlight 
}: InteractiveTranscriptEntryProps) => {
  const [showActions, setShowActions] = useState(false);
  const Icon = entryIcons[entry.type];

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const handleHighlight = (type: 'action' | 'decision') => {
    const newHighlight = entry.highlighted === type ? null : type;
    onHighlight(entry.id, newHighlight);
    
    if (newHighlight) {
      toast.success(`Marked as ${type} item`);
    } else {
      toast.success('Highlight removed');
    }
    setShowActions(false);
  };

  const getHighlightClass = () => {
    if (entry.highlighted === 'action') return 'highlight-action';
    if (entry.highlighted === 'decision') return 'highlight-decision';
    return '';
  };

  return (
    <div 
      className={`transcript-entry flex gap-3 group ${getHighlightClass()}`}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      <div className="flex flex-col items-center">
        {/* Speaker Avatar or Icon */}
        {entry.type !== 'system' ? (
          <Avatar className="h-7 w-7">
            <AvatarImage src={entry.avatar} alt={entry.speaker} />
            <AvatarFallback className="text-xs">
              {entry.speaker.split(' ').map(n => n[0]).join('')}
            </AvatarFallback>
          </Avatar>
        ) : (
          <div className="p-1.5 rounded bg-card border border-border">
            <Icon className="h-3 w-3 text-muted-foreground" />
          </div>
        )}
        
        <div className="w-px bg-border flex-1 mt-2 group-last:hidden" />
      </div>

      <div className="flex-1 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <Badge 
            variant="outline" 
            className={`text-xs ${entryColors[entry.type]}`}
          >
            {entry.type.toUpperCase()}
          </Badge>
          
          <span className="text-sm font-medium text-card-foreground">
            {entry.speaker}
          </span>
          
          <div className="flex items-center gap-1 text-xs text-muted-foreground ml-auto">
            <Clock className="h-3 w-3" />
            {formatTime(entry.timestamp)}
          </div>
        </div>
        
        <p className="text-sm text-card-foreground leading-relaxed mb-2">
          {entry.message}
        </p>

        {/* Highlight Indicators */}
        {entry.highlighted && (
          <div className="flex items-center gap-2 mb-2">
            {entry.highlighted === 'action' && (
              <Badge variant="outline" className="text-xs bg-orange-500/10 text-orange-400 border-orange-500/20">
                <Flag className="h-3 w-3 mr-1" />
                Action Item
              </Badge>
            )}
            {entry.highlighted === 'decision' && (
              <Badge variant="outline" className="text-xs bg-blue-500/10 text-blue-400 border-blue-500/20">
                <CheckSquare className="h-3 w-3 mr-1" />
                Decision
              </Badge>
            )}
          </div>
        )}

        {/* Interactive Actions */}
        {showActions && entry.type !== 'system' && (
          <div className="flex items-center gap-2 animate-fadeIn">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => handleHighlight('action')}
              className={`h-6 px-2 text-xs ${
                entry.highlighted === 'action' 
                  ? 'bg-orange-500/20 text-orange-400' 
                  : 'hover:bg-orange-500/10 hover:text-orange-400'
              }`}
            >
              <Flag className="h-3 w-3 mr-1" />
              Action
            </Button>
            
            <Button
              size="sm"
              variant="ghost"
              onClick={() => handleHighlight('decision')}
              className={`h-6 px-2 text-xs ${
                entry.highlighted === 'decision' 
                  ? 'bg-blue-500/20 text-blue-400' 
                  : 'hover:bg-blue-500/10 hover:text-blue-400'
              }`}
            >
              <CheckSquare className="h-3 w-3 mr-1" />
              Decision
            </Button>
            
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-1 text-xs hover:bg-muted"
            >
              <MoreHorizontal className="h-3 w-3" />
            </Button>
          </div>
        )}
      </div>
    </div>
  );
};