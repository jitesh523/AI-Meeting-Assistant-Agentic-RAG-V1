import { Card } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Calendar, Users, Clock } from 'lucide-react';

interface MeetingParticipant {
  id: string;
  name: string;
  avatar?: string;
  status: 'online' | 'offline';
}

interface MeetingHeaderProps {
  title: string;
  startTime: Date;
  participants: MeetingParticipant[];
  duration?: string;
}

export const MeetingHeader = ({ 
  title, 
  startTime, 
  participants, 
  duration = "00:00" 
}: MeetingHeaderProps) => {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: true
    });
  };

  const onlineParticipants = participants.filter(p => p.status === 'online');

  return (
    <Card className="ai-card mb-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        {/* Meeting Info */}
        <div className="space-y-2">
          <h2 className="text-xl font-bold text-card-foreground">{title}</h2>
          
          <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-1">
              <Calendar className="h-4 w-4" />
              <span>{formatTime(startTime)}</span>
            </div>
            
            <div className="flex items-center gap-1">
              <Clock className="h-4 w-4" />
              <span>{duration}</span>
            </div>
            
            <div className="flex items-center gap-1">
              <Users className="h-4 w-4" />
              <span>{onlineParticipants.length} online</span>
            </div>
          </div>
        </div>

        {/* Participants */}
        <div className="flex items-center gap-3">
          <span className="text-sm text-muted-foreground hidden md:block">
            Participants:
          </span>
          
          <div className="flex items-center -space-x-2">
            {participants.slice(0, 4).map((participant) => (
              <div key={participant.id} className="relative">
                <Avatar className="h-8 w-8 border-2 border-background">
                  <AvatarImage src={participant.avatar} alt={participant.name} />
                  <AvatarFallback className="text-xs">
                    {participant.name.split(' ').map(n => n[0]).join('')}
                  </AvatarFallback>
                </Avatar>
                
                {/* Status indicator */}
                <div className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-background ${
                  participant.status === 'online' 
                    ? 'bg-success' 
                    : 'bg-muted-foreground'
                }`} />
              </div>
            ))}
            
            {participants.length > 4 && (
              <Badge variant="secondary" className="ml-2 text-xs">
                +{participants.length - 4}
              </Badge>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
};