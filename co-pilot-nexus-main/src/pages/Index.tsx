import { useState } from 'react';
import { Toaster } from '@/components/ui/sonner';
import { MeetingControls } from '@/components/MeetingControls';
import { AISuggestions } from '@/components/AISuggestions';
import { LiveTranscript } from '@/components/LiveTranscript';
import { MeetingHeader } from '@/components/MeetingHeader';
import { Bot, Sparkles } from 'lucide-react';

interface TranscriptEntry {
  id: string;
  speaker: string;
  message: string;
  timestamp: Date;
  type: 'chat' | 'voice' | 'file' | 'system';
  avatar?: string;
  highlighted?: 'action' | 'decision' | null;
}

interface AISuggestion {
  id: string;
  type: 'ASK' | 'EMAIL' | 'TASK' | 'SUMMARY';
  content: string;
  approved?: boolean;
  timestamp?: Date;
}

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
}

interface MeetingParticipant {
  id: string;
  name: string;
  avatar?: string;
  status: 'online' | 'offline';
}

const Index = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [transcriptEntries, setTranscriptEntries] = useState<TranscriptEntry[]>([
    {
      id: '1',
      speaker: 'System',
      message: 'AI Meeting Assistant initialized. Ready to help you with your meeting.',
      timestamp: new Date(),
      type: 'system'
    }
  ]);
  const [aiSuggestions, setAiSuggestions] = useState<AISuggestion[]>([
    {
      id: '1',
      type: 'ASK',
      content: 'Ask about the project timeline and key milestones for Q4.',
      timestamp: new Date()
    },
    {
      id: '2',
      type: 'EMAIL',
      content: 'Draft follow-up email with action items and next steps.',
      timestamp: new Date()
    },
    {
      id: '3',
      type: 'TASK',
      content: 'Create calendar event for next sprint planning meeting.',
      timestamp: new Date()
    }
  ]);

  // Mock meeting data
  const meetingData = {
    title: "Q4 Strategic Planning Meeting",
    startTime: new Date(Date.now() - 25 * 60 * 1000), // Started 25 minutes ago
    participants: [
      { id: '1', name: 'Alex Thompson', avatar: '', status: 'online' as const },
      { id: '2', name: 'Sarah Chen', avatar: '', status: 'online' as const },
      { id: '3', name: 'Marcus Johnson', avatar: '', status: 'online' as const },
      { id: '4', name: 'Emily Rodriguez', avatar: '', status: 'offline' as const },
      { id: '5', name: 'David Kim', avatar: '', status: 'online' as const }
    ]
  };

  const addTranscriptEntry = (entry: Omit<TranscriptEntry, 'id' | 'timestamp'>) => {
    const newEntry: TranscriptEntry = {
      ...entry,
      id: Date.now().toString(),
      timestamp: new Date(),
      avatar: entry.speaker === 'You' ? '' : undefined
    };
    setTranscriptEntries(prev => [...prev, newEntry]);
  };

  const addAISuggestion = (suggestion: Omit<AISuggestion, 'id' | 'timestamp'>) => {
    const newSuggestion: AISuggestion = {
      ...suggestion,
      id: Date.now().toString(),
      timestamp: new Date()
    };
    setAiSuggestions(prev => [...prev, newSuggestion]);
  };

  const handleSuggestionAction = (id: string, approved: boolean) => {
    setAiSuggestions(prev => 
      prev.map(s => s.id === id ? { ...s, approved, timestamp: new Date() } : s)
    );
  };

  const handleTranscriptHighlight = (id: string, highlightType: 'action' | 'decision' | null) => {
    setTranscriptEntries(prev =>
      prev.map(entry => 
        entry.id === id ? { ...entry, highlighted: highlightType } : entry
      )
    );
  };

  const calculateDuration = () => {
    const now = new Date();
    const diff = now.getTime() - meetingData.startTime.getTime();
    const minutes = Math.floor(diff / 60000);
    const seconds = Math.floor((diff % 60000) / 1000);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  };

  return (
    <div className="min-h-screen bg-background">
      <Toaster />
      
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center gap-4 mb-2">
            <div className="p-3 rounded-xl bg-primary/10 backdrop-blur-sm">
              <Bot className="h-8 w-8 text-primary" />
            </div>
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-primary-glow bg-clip-text text-transparent">
                AI Meeting Assistant
              </h1>
              <p className="text-muted-foreground text-lg">
                Your intelligent co-pilot for productive meetings
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Sparkles className="h-4 w-4" />
            <span>Real-time transcription • Smart suggestions • Seamless collaboration</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {/* Meeting Header */}
        <MeetingHeader
          title={meetingData.title}
          startTime={meetingData.startTime}
          participants={meetingData.participants}
          duration={calculateDuration()}
        />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Meeting Controls */}
          <MeetingControls
            isConnected={isConnected}
            isRecording={isRecording}
            uploadedFiles={uploadedFiles}
            onConnectionToggle={() => setIsConnected(!isConnected)}
            onRecordingToggle={setIsRecording}
            onFileUpload={setUploadedFiles}
            onFileRemove={(id) => setUploadedFiles(prev => prev.filter(f => f.id !== id))}
            onChatMessage={(message) => addTranscriptEntry({
              speaker: 'You',
              message,
              type: 'chat'
            })}
            onVoiceTranscript={(message) => addTranscriptEntry({
              speaker: 'You',
              message,
              type: 'voice'
            })}
          />

          {/* AI Suggestions */}
          <AISuggestions
            suggestions={aiSuggestions}
            onSuggestionAction={handleSuggestionAction}
          />
        </div>

        {/* Live Transcript */}
        <LiveTranscript 
          entries={transcriptEntries} 
          onHighlight={handleTranscriptHighlight}
        />
      </main>
    </div>
  );
};

export default Index;