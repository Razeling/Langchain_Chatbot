'use client'

import React from 'react'
import SourceCitations from './SourceCitations'

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'
  const isError = message.isError
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} fade-in`}>
      <div className={`message-bubble rounded-lg px-4 py-3 max-w-[85%] ${
        isUser 
          ? 'message-user bg-blue-600 text-white' 
          : isError
          ? 'bg-red-50 border border-red-200 text-red-800'
          : 'message-assistant bg-white border border-gray-200 text-gray-800'
      }`}>
        {/* Message Content */}
        <div className="whitespace-pre-wrap break-words">
          {message.content}
        </div>
        
        {/* Sources - Only show for assistant messages that have sources */}
        {!isUser && !isError && (message.sources || message.webSources || message.learnedDocuments || message.pure_knowledge_sources || message.web_learned_sources || message.web_sources) && (
          <SourceCitations 
            pureKnowledgeSources={message.pure_knowledge_sources || []}
            webLearnedSources={message.web_learned_sources || []}
            webSources={message.web_sources || []}
            learnedDocuments={message.learnedDocuments || []}
            // Legacy fallback fields
            knowledgeSources={message.sources || []} 
            legacyWebSources={message.webSources || []}
          />
        )}
        
        {/* Timestamp */}
        <div className={`text-xs mt-2 ${
          isUser ? 'text-blue-100' : 'text-gray-500'
        }`}>
          {formatTimestamp(message.timestamp)}
        </div>
      </div>
    </div>
  )
}

function formatTimestamp(timestamp) {
  const date = new Date(timestamp)
  const now = new Date()
  
  // If same day, show time only
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString('en-US', { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }
  
  // Otherwise show date and time
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  })
} 