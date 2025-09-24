import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { FileUpload } from '@/components/FileUpload';
import { StatusIndicator } from '@/components/StatusIndicator';
import { VoiceRecording } from '@/components/VoiceRecording';
import { toast } from 'sonner';
import { 
  Send, 
  Play, 
  Square,
  Loader2 
} from 'lucide-react';

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
}

interface MeetingControlsProps {
  isConnected: boolean;
  isRecording: boolean;
  uploadedFiles: UploadedFile[];
  onConnectionToggle: () => void;
  onRecordingToggle: (isRecording: boolean) => void;
  onFileUpload: (files: UploadedFile[]) => void;
  onFileRemove: (id: string) => void;
  onChatMessage: (message: string) => void;
  onVoiceTranscript: (message: string) => void;
}

export const MeetingControls = ({
  isConnected,
  isRecording,
  uploadedFiles,
  onConnectionToggle,
  onRecordingToggle,
  onFileUpload,
  onFileRemove,
  onChatMessage,
  onVoiceTranscript
}: MeetingControlsProps) => {
  const [chatMessage, setChatMessage] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);

  const handleConnectionToggle = async () => {
    setIsConnecting(true);
    
    // Simulate WebSocket connection
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    onConnectionToggle();
    setIsConnecting(false);
    
    toast.success(
      isConnected ? 'Disconnected from meeting' : 'Connected to meeting successfully'
    );
  };

  const handleSendMessage = () => {
    if (!chatMessage.trim()) return;
    if (!isConnected) {
      toast.error('Please connect to meeting first');
      return;
    }

    onChatMessage(chatMessage);
    setChatMessage('');
    toast.success('Message sent');
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="space-y-6">
      {/* Meeting Status & Controls */}
      <Card className="ai-card">
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <StatusIndicator isConnected={isConnected} />
              <div>
                <h3 className="text-lg font-semibold">Meeting Status</h3>
                <p className="text-sm text-muted-foreground">
                  {isConnected ? 'Connected to AI assistant' : 'Not connected'}
                </p>
              </div>
            </div>
          </div>

          <div className="flex gap-3">
            <Button
              onClick={handleConnectionToggle}
              disabled={isConnecting}
              className="ai-button-primary flex-1"
            >
              {isConnecting ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Connecting...
                </>
              ) : (
                <>
                  {isConnected ? <Square className="mr-2 h-4 w-4" /> : <Play className="mr-2 h-4 w-4" />}
                  {isConnected ? 'End Meeting' : 'Start Meeting'}
                </>
              )}
            </Button>

            {/* Enhanced Voice Recording */}
            <VoiceRecording
              isConnected={isConnected}
              onRecordingChange={onRecordingToggle}
              onTranscriptAdd={onVoiceTranscript}
            />
          </div>
        </div>
      </Card>

      {/* File Upload */}
      <Card className="ai-card">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Document Upload</h3>
          <FileUpload
            uploadedFiles={uploadedFiles}
            onFileUpload={onFileUpload}
            onFileRemove={onFileRemove}
          />
        </div>
      </Card>

      {/* Chat Input */}
      <Card className="ai-card">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Chat Input</h3>
          <div className="flex gap-2">
            <Input
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message or question..."
              className="ai-input flex-1"
              disabled={!isConnected}
            />
            <Button
              onClick={handleSendMessage}
              disabled={!isConnected || !chatMessage.trim()}
              className="ai-button-primary px-4"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};