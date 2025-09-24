import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { InteractiveTranscriptEntry } from '@/components/InteractiveTranscriptEntry';
import { MessageSquare } from 'lucide-react';

interface TranscriptEntry {
  id: string;
  speaker: string;
  message: string;
  timestamp: Date;
  type: 'chat' | 'voice' | 'file' | 'system';
  avatar?: string;
  highlighted?: 'action' | 'decision' | null;
}

interface LiveTranscriptProps {
  entries: TranscriptEntry[];
  onHighlight: (id: string, type: 'action' | 'decision' | null) => void;
}

export const LiveTranscript = ({ entries, onHighlight }: LiveTranscriptProps) => {
  return (
    <Card className="ai-card">
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-primary/10">
            <MessageSquare className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h3 className="text-lg font-semibold">Live Transcript</h3>
            <p className="text-sm text-muted-foreground">
              Real-time conversation and activity log
            </p>
          </div>
        </div>

        <ScrollArea className="h-64 w-full rounded-lg border border-border bg-muted/20 p-4">
          <div className="space-y-4">
            {entries.map((entry) => (
              <InteractiveTranscriptEntry
                key={entry.id}
                entry={entry}
                onHighlight={onHighlight}
              />
            ))}

            {entries.length === 0 && (
              <div className="text-center text-muted-foreground py-8">
                <MessageSquare className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No transcript entries yet</p>
                <p className="text-xs mt-1">Start a meeting to see real-time transcription</p>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>
    </Card>
  );
};