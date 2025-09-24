import { Button } from '@/components/ui/button';
import { useVoiceRecording } from '@/hooks/useVoiceRecording';
import { toast } from 'sonner';
import { Mic, MicOff, Volume2 } from 'lucide-react';
import { useEffect } from 'react';

interface VoiceRecordingProps {
  isConnected: boolean;
  onRecordingChange: (isRecording: boolean) => void;
  onTranscriptAdd: (message: string) => void;
}

export const VoiceRecording = ({ 
  isConnected, 
  onRecordingChange, 
  onTranscriptAdd 
}: VoiceRecordingProps) => {
  const { isRecording, audioLevel, startRecording, stopRecording, error } = useVoiceRecording();

  useEffect(() => {
    onRecordingChange(isRecording);
  }, [isRecording, onRecordingChange]);

  useEffect(() => {
    if (error) {
      toast.error(error);
    }
  }, [error]);

  const handleToggleRecording = async () => {
    if (!isConnected) {
      toast.error('Please connect to meeting first');
      return;
    }

    if (isRecording) {
      stopRecording();
      toast.success('Voice recording stopped');
      
      // Simulate transcription result
      setTimeout(() => {
        onTranscriptAdd("I think we should focus on the key deliverables for next quarter.");
      }, 1000);
    } else {
      await startRecording();
      if (!error) {
        toast.success('Voice recording started');
      }
    }
  };

  // Generate waveform bars based on audio level
  const generateWaveformBars = () => {
    const bars = [];
    for (let i = 0; i < 5; i++) {
      const height = isRecording 
        ? Math.max(4, audioLevel * 20 + Math.random() * 8) 
        : 4;
      
      bars.push(
        <div
          key={i}
          className="waveform-bar bg-primary rounded-full w-1 transition-all duration-100"
          style={{ height: `${height}px` }}
        />
      );
    }
    return bars;
  };

  return (
    <div className="flex items-center gap-3">
      <Button
        onClick={handleToggleRecording}
        disabled={!isConnected}
        variant={isRecording ? "destructive" : "secondary"}
        className={`px-4 ${isRecording ? 'recording-pulse' : ''}`}
      >
        {isRecording ? <MicOff className="h-4 w-4" /> : <Mic className="h-4 w-4" />}
      </Button>

      {/* Waveform Visualization */}
      {isRecording && (
        <div className="flex items-center gap-1 px-3 py-2 rounded-lg bg-card border border-border">
          <Volume2 className="h-3 w-3 text-muted-foreground mr-2" />
          <div className="flex items-center gap-0.5 h-4">
            {generateWaveformBars()}
          </div>
          <span className="text-xs text-muted-foreground ml-2">
            {Math.round(audioLevel * 100)}%
          </span>
        </div>
      )}
    </div>
  );
};