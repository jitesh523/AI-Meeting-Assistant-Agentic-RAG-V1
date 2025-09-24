import { useState } from 'react'
import { Mic, MicOff, Play, Square, Users, Clock } from 'lucide-react'

interface MeetingPanelProps {
  isListening: boolean
  onStartMeeting: () => void
  onEndMeeting: () => void
  meetingId: string | null
}

export function MeetingPanel({ isListening, onStartMeeting, onEndMeeting, meetingId }: MeetingPanelProps) {
  const [participants] = useState(['You', 'AI Assistant'])

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">Meeting Controls</h2>
      
      {/* Meeting Status */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">Status</span>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isListening ? 'bg-green-500' : 'bg-gray-400'}`} />
            <span className="text-sm text-gray-600">
              {isListening ? 'Active' : 'Inactive'}
            </span>
          </div>
        </div>
        
        {meetingId && (
          <div className="text-xs text-gray-500 font-mono">
            ID: {meetingId}
          </div>
        )}
      </div>

      {/* Control Buttons */}
      <div className="space-y-3">
        {!isListening ? (
          <button
            onClick={onStartMeeting}
            className="w-full flex items-center justify-center space-x-2 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
          >
            <Play className="h-4 w-4" />
            <span>Start Meeting</span>
          </button>
        ) : (
          <button
            onClick={onEndMeeting}
            className="w-full flex items-center justify-center space-x-2 bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700 transition-colors"
          >
            <Square className="h-4 w-4" />
            <span>End Meeting</span>
          </button>
        )}
      </div>

      {/* Participants */}
      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3 flex items-center">
          <Users className="h-4 w-4 mr-2" />
          Participants
        </h3>
        <div className="space-y-2">
          {participants.map((participant, index) => (
            <div key={index} className="flex items-center space-x-2">
              <div className="w-2 h-2 rounded-full bg-green-500" />
              <span className="text-sm text-gray-600">{participant}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Privacy Mode */}
      <div className="mt-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Privacy Mode</h3>
        <div className="bg-gray-50 rounded-md p-3">
          <div className="text-sm text-gray-600">
            <strong>Transcript + Notes</strong>
            <p className="text-xs text-gray-500 mt-1">
              AI can transcribe and take notes, but requires approval for actions
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
