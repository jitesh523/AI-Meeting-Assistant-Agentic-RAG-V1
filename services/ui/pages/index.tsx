import { useState, useEffect } from 'react'
import { Mic, MicOff, Settings, Users, MessageSquare, CheckCircle, XCircle } from 'lucide-react'
import { MeetingPanel } from '../components/MeetingPanel'
import { SuggestionsPanel } from '../components/SuggestionsPanel'
import { TranscriptPanel } from '../components/TranscriptPanel'

interface Suggestion {
  id: string
  kind: string
  text: string
  confidence: number
  reasons: string[]
  citations: string[]
  status: string
}

interface Utterance {
  speaker: string
  text: string
  timestamp: string
  confidence: number
}

export default function Home() {
  const [isListening, setIsListening] = useState(false)
  const [meetingId, setMeetingId] = useState<string | null>(null)
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [transcript, setTranscript] = useState<Utterance[]>([])
  const [ws, setWs] = useState<WebSocket | null>(null)

  useEffect(() => {
    // Generate a meeting ID for this session
    const newMeetingId = `meeting_${Date.now()}`
    setMeetingId(newMeetingId)

    // Connect to WebSocket
    const websocket = new WebSocket(`ws://localhost:8001/ws/audio/${newMeetingId}`)
    setWs(websocket)

    websocket.onopen = () => {
      console.log('Connected to meeting service')
    }

    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'transcript') {
        setTranscript(prev => [...prev, data.utterance])
      } else if (data.type === 'suggestion') {
        setSuggestions(prev => [...prev, data.suggestion])
      }
    }

    websocket.onclose = () => {
      console.log('Disconnected from meeting service')
    }

    return () => {
      websocket.close()
    }
  }, [])

  const startMeeting = async () => {
    if (!meetingId) return

    try {
      const response = await fetch(`/api/ingestion/meetings/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          meeting_id: meetingId,
          title: 'AI Meeting Assistant Demo',
          platform: 'web',
          start_time: Date.now() / 1000,
          privacy_mode: 'transcript+notes',
          participants: ['user@example.com']
        })
      })

      if (response.ok) {
        setIsListening(true)
      }
    } catch (error) {
      console.error('Error starting meeting:', error)
    }
  }

  const endMeeting = async () => {
    if (!meetingId) return

    try {
      await fetch(`/api/ingestion/meetings/${meetingId}/end`, {
        method: 'POST'
      })
      setIsListening(false)
    } catch (error) {
      console.error('Error ending meeting:', error)
    }
  }

  const approveSuggestion = async (suggestionId: string) => {
    try {
      const response = await fetch(`/api/agent/suggestions/${suggestionId}/approve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          approved_by: 'user@example.com'
        })
      })

      if (response.ok) {
        setSuggestions(prev => 
          prev.map(s => 
            s.id === suggestionId 
              ? { ...s, status: 'approved' }
              : s
          )
        )
      }
    } catch (error) {
      console.error('Error approving suggestion:', error)
    }
  }

  const rejectSuggestion = async (suggestionId: string) => {
    try {
      const response = await fetch(`/api/agent/suggestions/${suggestionId}/reject`, {
        method: 'POST'
      })

      if (response.ok) {
        setSuggestions(prev => 
          prev.map(s => 
            s.id === suggestionId 
              ? { ...s, status: 'rejected' }
              : s
          )
        )
      }
    } catch (error) {
      console.error('Error rejecting suggestion:', error)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold text-gray-900">
                AI Meeting Assistant
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <button className="p-2 text-gray-400 hover:text-gray-500">
                <Settings className="h-5 w-5" />
              </button>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isListening ? 'bg-green-500' : 'bg-gray-400'}`} />
                <span className="text-sm text-gray-600">
                  {isListening ? 'Listening' : 'Offline'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Meeting Controls */}
          <div className="lg:col-span-1">
            <MeetingPanel
              isListening={isListening}
              onStartMeeting={startMeeting}
              onEndMeeting={endMeeting}
              meetingId={meetingId}
            />
          </div>

          {/* Transcript */}
          <div className="lg:col-span-1">
            <TranscriptPanel transcript={transcript} />
          </div>

          {/* Suggestions */}
          <div className="lg:col-span-1">
            <SuggestionsPanel
              suggestions={suggestions}
              onApprove={approveSuggestion}
              onReject={rejectSuggestion}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
