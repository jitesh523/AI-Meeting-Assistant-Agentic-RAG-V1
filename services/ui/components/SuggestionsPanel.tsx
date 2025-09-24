import { useState } from 'react'
import { CheckCircle, XCircle, Lightbulb, Mail, Calendar, FileText, Users, ExternalLink } from 'lucide-react'

interface Suggestion {
  id: string
  kind: string
  text: string
  confidence: number
  reasons: string[]
  citations: string[]
  status: string
}

interface SuggestionsPanelProps {
  suggestions: Suggestion[]
  onApprove: (suggestionId: string) => void
  onReject: (suggestionId: string) => void
}

export function SuggestionsPanel({ suggestions, onApprove, onReject }: SuggestionsPanelProps) {
  const [expandedSuggestion, setExpandedSuggestion] = useState<string | null>(null)

  const getSuggestionIcon = (kind: string) => {
    switch (kind) {
      case 'ask':
        return <Lightbulb className="h-4 w-4 text-blue-500" />
      case 'email':
        return <Mail className="h-4 w-4 text-green-500" />
      case 'calendar':
        return <Calendar className="h-4 w-4 text-purple-500" />
      case 'task':
        return <FileText className="h-4 w-4 text-orange-500" />
      case 'fact':
        return <Users className="h-4 w-4 text-indigo-500" />
      default:
        return <Lightbulb className="h-4 w-4 text-gray-500" />
    }
  }

  const getSuggestionColor = (kind: string) => {
    switch (kind) {
      case 'ask':
        return 'border-blue-200 bg-blue-50'
      case 'email':
        return 'border-green-200 bg-green-50'
      case 'calendar':
        return 'border-purple-200 bg-purple-50'
      case 'task':
        return 'border-orange-200 bg-orange-50'
      case 'fact':
        return 'border-indigo-200 bg-indigo-50'
      default:
        return 'border-gray-200 bg-gray-50'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'approved':
        return 'text-green-600 bg-green-100'
      case 'rejected':
        return 'text-red-600 bg-red-100'
      default:
        return 'text-yellow-600 bg-yellow-100'
    }
  }

  const pendingSuggestions = suggestions.filter(s => s.status === 'pending')
  const processedSuggestions = suggestions.filter(s => s.status !== 'pending')

  return (
    <div className="bg-white rounded-lg shadow-sm border">
      <div className="p-4 border-b">
        <h2 className="text-lg font-semibold text-gray-900 flex items-center">
          <Lightbulb className="h-5 w-5 mr-2" />
          AI Suggestions
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          {pendingSuggestions.length} pending, {processedSuggestions.length} processed
        </p>
      </div>
      
      <div className="max-h-96 overflow-y-auto">
        {suggestions.length === 0 ? (
          <div className="text-center text-gray-500 py-8">
            <Lightbulb className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>No suggestions yet. AI will provide suggestions as the meeting progresses.</p>
          </div>
        ) : (
          <div className="p-4 space-y-3">
            {/* Pending Suggestions */}
            {pendingSuggestions.map((suggestion) => (
              <div
                key={suggestion.id}
                className={`border rounded-lg p-4 ${getSuggestionColor(suggestion.kind)}`}
              >
                <div className="flex items-start space-x-3">
                  <div className="flex-shrink-0 mt-1">
                    {getSuggestionIcon(suggestion.kind)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-gray-900 capitalize">
                        {suggestion.kind}
                      </span>
                      <div className="flex items-center space-x-2">
                        <div className={`w-2 h-2 rounded-full ${
                          suggestion.confidence > 0.8 ? 'bg-green-500' : 
                          suggestion.confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                        }`} />
                        <span className="text-xs text-gray-500">
                          {Math.round(suggestion.confidence * 100)}%
                        </span>
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-700 mb-3">
                      {suggestion.text}
                    </p>
                    
                    {suggestion.reasons.length > 0 && (
                      <div className="mb-3">
                        <button
                          onClick={() => setExpandedSuggestion(
                            expandedSuggestion === suggestion.id ? null : suggestion.id
                          )}
                          className="text-xs text-gray-500 hover:text-gray-700"
                        >
                          {expandedSuggestion === suggestion.id ? 'Hide' : 'Show'} reasons
                        </button>
                        {expandedSuggestion === suggestion.id && (
                          <div className="mt-2 space-y-1">
                            {suggestion.reasons.map((reason, index) => (
                              <div key={index} className="text-xs text-gray-600">
                                • {reason}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                    
                    {suggestion.citations.length > 0 && (
                      <div className="mb-3">
                        <div className="text-xs text-gray-500 mb-1">Sources:</div>
                        <div className="space-y-1">
                          {suggestion.citations.map((citation, index) => (
                            <div key={index} className="flex items-center space-x-1 text-xs text-blue-600">
                              <ExternalLink className="h-3 w-3" />
                              <span>{citation}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    <div className="flex space-x-2">
                      <button
                        onClick={() => onApprove(suggestion.id)}
                        className="flex items-center space-x-1 px-3 py-1 bg-green-600 text-white text-xs rounded-md hover:bg-green-700 transition-colors"
                      >
                        <CheckCircle className="h-3 w-3" />
                        <span>Approve</span>
                      </button>
                      <button
                        onClick={() => onReject(suggestion.id)}
                        className="flex items-center space-x-1 px-3 py-1 bg-red-600 text-white text-xs rounded-md hover:bg-red-700 transition-colors"
                      >
                        <XCircle className="h-3 w-3" />
                        <span>Reject</span>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            
            {/* Processed Suggestions */}
            {processedSuggestions.length > 0 && (
              <div className="pt-4 border-t">
                <h3 className="text-sm font-medium text-gray-700 mb-3">Processed</h3>
                <div className="space-y-2">
                  {processedSuggestions.map((suggestion) => (
                    <div
                      key={suggestion.id}
                      className="flex items-center justify-between p-3 bg-gray-50 rounded-md"
                    >
                      <div className="flex items-center space-x-3">
                        {getSuggestionIcon(suggestion.kind)}
                        <span className="text-sm text-gray-700 truncate">
                          {suggestion.text}
                        </span>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(suggestion.status)}`}>
                        {suggestion.status}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
