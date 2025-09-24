import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { SuggestionsHistory } from '@/components/SuggestionsHistory';
import { toast } from 'sonner';
import { 
  Check, 
  X, 
  MessageSquare, 
  Mail, 
  CheckSquare, 
  FileText,
  Sparkles 
} from 'lucide-react';

interface AISuggestion {
  id: string;
  type: 'ASK' | 'EMAIL' | 'TASK' | 'SUMMARY';
  content: string;
  approved?: boolean;
  timestamp?: Date;
}

interface AISuggestionsProps {
  suggestions: AISuggestion[];
  onSuggestionAction: (id: string, approved: boolean) => void;
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

export const AISuggestions = ({ suggestions, onSuggestionAction }: AISuggestionsProps) => {
  const [activeTab, setActiveTab] = useState('active');
  
  const handleSuggestionAction = (id: string, approved: boolean) => {
    onSuggestionAction(id, approved);
    toast.success(
      approved ? 'Suggestion approved' : 'Suggestion rejected'
    );
  };

  const activeSuggestions = suggestions.filter(s => s.approved === undefined);
  const completedSuggestions = suggestions.filter(s => s.approved !== undefined);

  return (
    <Card className="ai-card h-fit">
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">AI Suggestions</h3>
            <p className="text-sm text-muted-foreground">
              Smart recommendations for your meeting
            </p>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="active" className="text-sm">
              Active ({activeSuggestions.length})
            </TabsTrigger>
            <TabsTrigger value="history" className="text-sm">
              History ({completedSuggestions.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="active" className="mt-4">
            <div className="space-y-4 max-h-96 overflow-y-auto">
              {activeSuggestions.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  <Sparkles className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No active suggestions</p>
                  <p className="text-xs mt-1">New AI suggestions will appear here</p>
                </div>
              ) : (
                activeSuggestions.map((suggestion) => {
                  const Icon = suggestionIcons[suggestion.type];
                  
                  return (
                    <div
                      key={suggestion.id}
                      className="suggestion-card suggestion-appear"
                    >
                      <div className="flex items-start gap-3 mb-3">
                        <div className="p-1.5 rounded bg-primary/10">
                          <Icon className="h-4 w-4 text-primary" />
                        </div>
                        <div className="flex-1">
                          <Badge 
                            variant="outline" 
                            className={`mb-2 ${suggestionColors[suggestion.type]}`}
                          >
                            {suggestion.type}
                          </Badge>
                          <p className="text-sm text-card-foreground leading-relaxed">
                            {suggestion.content}
                          </p>
                        </div>
                      </div>

                      <div className="flex gap-2">
                        <Button
                          size="sm"
                          onClick={() => handleSuggestionAction(suggestion.id, true)}
                          className="flex-1 bg-success hover:bg-success/90 text-success-foreground border-0"
                        >
                          <Check className="h-3 w-3 mr-1" />
                          Approve
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => handleSuggestionAction(suggestion.id, false)}
                          className="flex-1 border-danger/50 text-danger hover:bg-danger/10"
                        >
                          <X className="h-3 w-3 mr-1" />
                          Reject
                        </Button>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </TabsContent>

          <TabsContent value="history" className="mt-4">
            <SuggestionsHistory suggestions={completedSuggestions} />
          </TabsContent>
        </Tabs>
      </div>
    </Card>
  );
};