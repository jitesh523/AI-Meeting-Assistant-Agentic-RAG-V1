import { useEffect, useRef } from 'react'
import { MessageSquare, User, Bot } from 'lucide-react'

interface Utterance {
  speaker: string
  text: string
  timestamp: string
  confidence: number
}

interface TranscriptPanelProps {
  transcript: Utterance[]
}

export function TranscriptPanel({ transcript }: TranscriptPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [transcript])

  const getSpeakerIcon = (speaker: string) => {
    if (speaker === 'unknown' || speaker === 'AI Assistant') {
      return <Bot className="h-4 w-4 text-blue-500" />
    }
    return <User className="h-4 w-4 text-green-500" />
  }

  const getSpeakerName = (speaker: string) => {
    if (speaker === 'unknown') return 'Speaker'
    if (speaker === 'AI Assistant') return 'AI Assistant'
    return speaker
  }

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-4 border-b">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center">
          <MessageSquare className="h-5 w-5 mr-2" />
          Live Transcript
        </h2>
      </div>
      
      <div 
        ref={scrollRef}
        className="h-96 overflow-y-auto p-4 space-y-4"
      >
        {transcript.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <MessageSquare className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>No transcript yet. Start speaking to see the live transcript here.</p>
          </div>
        ) : (
          transcript.map((utterance, index) => (
            <div key={index} className="flex space-x-3">
              <div className="flex-shrink-0">
                {getSpeakerIcon(utterance.speaker)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center space-x-2 mb-1">
                  <span className="text-sm font-medium text-gray-900">
                    {getSpeakerName(utterance.speaker)}
                  </span>
                  <span className="text-xs text-gray-500">
                    {formatTimestamp(utterance.timestamp)}
                  </span>
                  <div className="flex items-center space-x-1">
                    <div className={`w-2 h-2 rounded-full ${
                      utterance.confidence > 0.8 ? 'bg-green-500' : 
                      utterance.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                    <span className="text-xs text-gray-400">
                      {Math.round(utterance.confidence * 100)}%
                    </span>
                  </div>
                </div>
                <p className="text-sm text-gray-700 leading-relaxed">
                  {utterance.text}
                </p>
              </div>
            </div>
          ))
        )}
      </div>
      
      {transcript.length > 0 && (
        <div className="p-4 border-t bg-gray-50">
          <div className="text-xs text-gray-500 text-center">
            {transcript.length} utterances • Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>
      )}
    </div>
  )
}
