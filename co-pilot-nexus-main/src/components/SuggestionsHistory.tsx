import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Check, 
  X, 
  MessageSquare, 
  Mail, 
  CheckSquare, 
  FileText,
  History
} from 'lucide-react';

interface AISuggestion {
  id: string;
  type: 'ASK' | 'EMAIL' | 'TASK' | 'SUMMARY';
  content: string;
  approved?: boolean;
  timestamp?: Date;
}

interface SuggestionsHistoryProps {
  suggestions: AISuggestion[];
}

const suggestionIcons = {
  ASK: MessageSquare,
  EMAIL: Mail,
  TASK: CheckSquare,
  SUMMARY: FileText
};

const suggestionColors = {
  ASK: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
  EMAIL: 'bg-green-500/10 text-green-400 border-green-500/20',
  TASK: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
  SUMMARY: 'bg-purple-500/10 text-purple-400 border-purple-500/20'
};

export const SuggestionsHistory = ({ suggestions }: SuggestionsHistoryProps) => {
  const completedSuggestions = suggestions.filter(s => s.approved !== undefined);

  const formatTime = (date?: Date) => {
    if (!date) return '';
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <Card className="ai-card">
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <History className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Suggestions History</h3>
            <p className="text-sm text-muted-foreground">
              {completedSuggestions.length} completed suggestions
            </p>
          </div>
        </div>

        <ScrollArea className="h-80 w-full">
          <div className="space-y-3">
            {completedSuggestions.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                <History className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No completed suggestions yet</p>
                <p className="text-xs mt-1">Approved and rejected suggestions will appear here</p>
              </div>
            ) : (
              completedSuggestions.map((suggestion) => {
                const Icon = suggestionIcons[suggestion.type];
                const isApproved = suggestion.approved === true;
                
                return (
                  <div
                    key={suggestion.id}
                    className={`p-3 rounded-lg border transition-all ${
                      isApproved 
                        ? 'border-success/30 bg-success/5' 
                        : 'border-danger/30 bg-danger/5'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="p-1.5 rounded bg-primary/10">
                        <Icon className="h-4 w-4 text-primary" />
                      </div>
                      
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge 
                            variant="outline" 
                            className={`text-xs ${suggestionColors[suggestion.type]}`}
                          >
                            {suggestion.type}
                          </Badge>
                          
                          {suggestion.timestamp && (
                            <span className="text-xs text-muted-foreground">
                              {formatTime(suggestion.timestamp)}
                            </span>
                          )}
                        </div>
                        
                        <p className="text-sm text-card-foreground leading-relaxed mb-2">
                          {suggestion.content}
                        </p>
                        
                        <div className="flex items-center gap-2">
                          {isApproved ? (
                            <div className="flex items-center gap-1 text-success text-xs">
                              <Check className="h-3 w-3" />
                              <span>Approved</span>
                            </div>
                          ) : (
                            <div className="flex items-center gap-1 text-danger text-xs">
                              <X className="h-3 w-3" />
                              <span>Rejected</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </ScrollArea>
      </div>
    </Card>
  );
};