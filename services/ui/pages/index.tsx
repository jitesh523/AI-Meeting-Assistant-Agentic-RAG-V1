import { useEffect, useRef, useState } from 'react'
import { Settings, Upload, Search } from 'lucide-react'
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
  const [meetingId, setMeetingId] = useState<string>('')
  const [tenantId] = useState<string>('tenant_demo')
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [transcript, setTranscript] = useState<Utterance[]>([])
  const [message, setMessage] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchHits, setSearchHits] = useState<any[]>([])
  const pollingRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    // Generate a meeting ID on client to avoid SSR hydration mismatch
    setMeetingId(`meeting_${Date.now()}`)
    // start suggestions polling when meeting active
    if (isListening && meetingId) {
      if (pollingRef.current) clearInterval(pollingRef.current)
      pollingRef.current = setInterval(async () => {
        try {
          const res = await fetch(`/api/agent/meetings/${meetingId}/suggestions`)
          if (res.ok) {
            const data = await res.json()
            setSuggestions(data.suggestions || [])
          }
        } catch (e) {
          // ignore transient errors
        }
      }, 2000)
    }
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current)
    }
  }, [isListening, meetingId])

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

  const sendUtterance = async () => {
    if (!message.trim() || !meetingId) return
    const utter: Utterance = {
      speaker: 'You',
      text: message.trim(),
      timestamp: new Date().toISOString(),
      confidence: 1.0,
    }
    // append locally for immediate feedback
    setTranscript(prev => [...prev, utter])
    setMessage('')
    // send to ingestion -> agent
    try {
      await fetch(`/api/ingestion/meetings/${meetingId}/utterances`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speaker: 'You', text: utter.text, timestamp: Date.now() / 1000 }),
      })
      // also directly request suggestions from agent for snappy UX
      const res = await fetch(`/api/agent/meetings/${meetingId}/suggest-from-text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speaker: 'You', text: utter.text, timestamp: Date.now() / 1000 }),
      })
      if (res.ok) {
        const data = await res.json()
        if (data.generated && Array.isArray(data.generated)) {
          setSuggestions(prev => [...prev, ...data.generated])
        }
      }
    } catch (e) {
      console.error('Failed to send utterance', e)
    }
  }

  const uploadFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    for (const file of Array.from(files)) {
      const form = new FormData()
      form.append('file', file)
      form.append('tenant_id', tenantId)
      form.append('source', 'upload')
      try {
        const res = await fetch(`/api/rag/upload`, { method: 'POST', body: form })
        const data = await res.json()
        if (data.status !== 'success') {
          console.warn('Upload failed', data)
        }
      } catch (e) {
        console.error('Upload error', e)
      }
    }
  }

  const performSearch = async () => {
    if (!searchQuery.trim()) return
    try {
      const res = await fetch(`/api/rag/search?q=${encodeURIComponent(searchQuery)}&tenant_id=${encodeURIComponent(tenantId)}&k=5`)
      if (res.ok) {
        const data = await res.json()
        setSearchHits(data.hits || [])
      }
    } catch (e) {
      console.error('Search error', e)
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

            {/* Chat input */}
            <div className="mt-4 bg-white border rounded p-3">
              <div className="flex gap-2">
                <input
                  className="flex-1 border rounded px-3 py-2"
                  placeholder="Type a message..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  disabled={!isListening}
                />
                <button onClick={sendUtterance} disabled={!isListening} className="px-3 py-2 bg-indigo-600 text-white rounded">
                  Send
                </button>
              </div>
            </div>
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

            {/* Document upload + search */}
            <div className="mt-4 bg-white border rounded p-3 space-y-3">
              <div>
                <label className="block text-sm font-medium mb-1">Upload documents</label>
                <input type="file" multiple onChange={(e) => uploadFiles(e.target.files)} />
              </div>
              <div>
                <label className="block text-sm font-medium mb-1">Search</label>
                <div className="flex gap-2">
                  <input className="flex-1 border rounded px-3 py-2" placeholder="Search docs..." value={searchQuery} onChange={(e)=>setSearchQuery(e.target.value)} />
                  <button onClick={performSearch} className="px-3 py-2 bg-gray-800 text-white rounded flex items-center gap-1">
                    <Search className="h-4 w-4"/> Search
                  </button>
                </div>
                {searchHits.length > 0 && (
                  <div className="mt-3 max-h-48 overflow-auto space-y-2">
                    {searchHits.map((h, idx) => (
                      <div key={idx} className="border rounded p-2">
                        <div className="text-xs text-gray-500">{h.source}</div>
                        <div className="text-sm">{h.text?.slice(0,200)}{h.text && h.text.length>200?'...':''}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
