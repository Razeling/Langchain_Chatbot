'use client'

import React from 'react'

export default function TypingIndicator() {
  return (
    <div className="message-bubble bg-white border border-gray-200 rounded-lg px-4 py-3 max-w-[200px]">
      <div className="typing-indicator flex items-center space-x-1">
        <div className="text-gray-500 text-sm mr-2">Analyzing...</div>
        <div className="typing-dot w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
        <div className="typing-dot w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
        <div className="typing-dot w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
      </div>
    </div>
  )
} 